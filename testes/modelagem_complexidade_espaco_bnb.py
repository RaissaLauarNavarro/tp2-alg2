"""
Modelagem de Complexidade Espacial Empírica do Branch and Bound (DFS vs. Best-First).

Utiliza os arquivos gerados com `--memoria` (`resultados_bnb_<inicial>_<estrategia>_memoria.csv`)
sem filtrar apenas instâncias ótimas (sem viés de seleção), aplicando:
  1. Pico de memória (MiB) vs. N para DFS e Best-First (AICc, R², α em log-log com IC 95%).
  2. Comparação direta DFS vs. Best-First em escala log-log (M(N) = c · N^α).
  3. Custo espacial por estado ativo (`max_fila` no Best-First; `prof_max` no DFS) vs. N.
  4. Best-First: predição do pico de memória (MiB) pelo tamanho da fila (`max_fila`)
     via regressão linear (bytes/nó na fila com IC 95%) e lei de potência (log-log) com AICc e R²
     — tanto com todas as inicializações quanto excluindo a inicialização `trivial` da regressão.
  5. Best-First: predição do tamanho da fila (`max_fila`) pelo número de vértices N
     via ajuste polinomial e lei de potência (log-log) com AICc, R² e IC 95% de α.

Uso:
    python modelagem_memoria_bnb.py
    python modelagem_memoria_bnb.py --pasta-resultados ./resultados --pasta-saida graficos_memoria_bnb
"""

import argparse
import glob
import os
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as student_t

# Configuração visual global
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10

CORES_INICIALIZACOES_BNB = {
    "trivial": "#7f8c8d",
    "vizinho": "#e67e22",
    "fecho": "#8e44ad",
    "fecho_graham": "#8e44ad",
    "fecho_gift": "#9b59b6",
    "twice": "#2980b9",
    "christofides": "#27ae60",
}

LABELS_ESTRATEGIAS_BNB = {
    "profundidade": "Busca em Profundidade (DFS)",
    "melhor": "Best-First Search (Melhor Primeiro)",
}

# Curvas polinomiais candidatas em função de N
MODELOS_POLINOMIAIS = {
    "O(1)": lambda n, a, b: a * np.ones_like(n, dtype=float) + b,
    "O(N)": lambda n, a, b: a * n + b,
    "O(N log N)": lambda n, a, b: a * n * np.log2(np.maximum(n, 2)) + b,
    "O(N²)": lambda n, a, b: a * (n ** 2) + b,
    "O(N² log N)": lambda n, a, b: a * (n ** 2) * np.log2(np.maximum(n, 2)) + b,
    "O(N³)": lambda n, a, b: a * (n ** 3) + b,
}


def _formatar_eixo_inteiro(val, _pos):
    """Formata contagens grandes de nós (ex.: 150k, 20.0M) sem offset text oculto."""
    if val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if val >= 1_000:
        return f"{val / 1_000:.0f}k"
    return f"{int(round(val))}"


# ---------------------------------------------------------------------------
# 1. CARREGAMENTO DOS DADOS DE MEMÓRIA DO BRANCH AND BOUND (*_memoria.csv)
# ---------------------------------------------------------------------------

def _inferir_estrategia_nome_arquivo(caminho_csv):
    """Extrai a estratégia (`profundidade` ou `melhor`) de `resultados_bnb_<inicial>_<estrategia>_memoria.csv`."""
    base = os.path.basename(caminho_csv)
    sem_ext = base[:-4] if base.endswith(".csv") else base
    if sem_ext.endswith("_memoria"):
        sem_ext = sem_ext[:-len("_memoria")]
    partes = sem_ext.split("_")
    if partes and partes[-1] in ("profundidade", "melhor"):
        return partes[-1]
    return None


def carregar_dados_bnb_memoria(pasta_resultados):
    """Carrega exclusivamente os CSVs gerados com `--memoria` (`resultados_bnb_*_memoria.csv`)."""
    padroes = [
        os.path.join(pasta_resultados, "resultados_bnb_*_memoria.csv"),
        "resultados_bnb_*_memoria.csv",
    ]
    arquivos = []
    for p in padroes:
        arquivos = sorted(glob.glob(p))
        if arquivos:
            break

    if not arquivos:
        print("[Aviso] Nenhum CSV de memória do Branch and Bound (resultados_bnb_*_memoria.csv) encontrado.")
        return None

    lista_mem = []
    for arq in arquivos:
        est_arq = _inferir_estrategia_nome_arquivo(arq)
        df = pd.read_csv(arq, na_values=["NA"])

        if "estrategia" not in df.columns and est_arq is not None:
            df["estrategia"] = est_arq
        elif est_arq is not None:
            df["estrategia"] = df["estrategia"].fillna(est_arq)

        lista_mem.append(df)

    return pd.concat(lista_mem, ignore_index=True) if lista_mem else None


# ---------------------------------------------------------------------------
# 2. MÉTRICAS ESTATÍSTICAS (AICc, IC 95%, R², LOG-LOG E CURVE_FIT)
# ---------------------------------------------------------------------------

def calcular_aicc(n_obs, rss, k_params=3):
    """Calcula AIC e AICc sob verossimilhança gaussiana."""
    if n_obs <= k_params + 1 or not np.isfinite(rss) or rss <= 0:
        return np.nan, np.nan
    aic = n_obs * np.log(rss / n_obs) + 2.0 * k_params
    correcao = (2.0 * k_params * (k_params + 1)) / (n_obs - k_params - 1)
    return float(aic), float(aic + correcao)


def calcular_metricas_ajuste(y_real, y_pred, k_params=3):
    """Calcula R², RMSE, RSS, AIC e AICc entre os valores observados e preditos."""
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n_obs = len(y_real)
    residuos = y_real - y_pred
    ss_res = float(np.sum(residuos ** 2))
    ss_tot = float(np.sum((y_real - np.mean(y_real)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-18 else 0.0
    rmse = float(np.sqrt(np.mean(residuos ** 2)))
    aic, aicc = calcular_aicc(n_obs, ss_res, k_params=k_params)
    return {
        "r2": r2,
        "rmse": rmse,
        "rss": ss_res,
        "aic": aic,
        "aicc": aicc,
    }


def regressao_linear_ols(x_vals, y_vals, nivel_conf=0.95):
    """Regressão linear simples y = beta0 + beta1 * x com IC (t-Student) para beta1, R² e AICc."""
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n_obs = len(x)
    if n_obs < 4:
        return None

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx <= 1e-18:
        return None

    beta1 = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
    beta0 = float(y_mean - beta1 * x_mean)
    y_pred = beta0 + beta1 * x
    residuos = y - y_pred
    rss = float(np.sum(residuos ** 2))

    df = n_obs - 2
    s2 = rss / df
    se_beta1 = float(np.sqrt(s2 / sxx))
    t_crit = float(student_t.ppf(0.5 + nivel_conf / 2.0, df=df))
    ic_inf = beta1 - t_crit * se_beta1
    ic_sup = beta1 + t_crit * se_beta1

    metricas = calcular_metricas_ajuste(y, y_pred, k_params=3)
    return {
        "n_obs": n_obs,
        "beta0": beta0,
        "beta1": beta1,
        "se_beta1": se_beta1,
        "ic_beta1": (float(ic_inf), float(ic_sup)),
        "s2": float(s2),
        "x_mean": x_mean,
        "sxx": sxx,
        "t_crit": t_crit,
        **metricas,
    }


def ajustar_lei_potencia_loglog(x_vals, y_vals, nivel_conf=0.95):
    """Estima y = c * x^alpha via regressão linear em ln(y) = ln(c) + alpha * ln(x), com AICc, R² e IC 95%."""
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0) & (y_arr > 0)
    if np.sum(mask) < 4:
        return None

    ols = regressao_linear_ols(np.log(x_arr[mask]), np.log(y_arr[mask]), nivel_conf=nivel_conf)
    if ols is None:
        return None

    return {
        "modelo": "Potência c·x^α (log-log)",
        "c": float(np.exp(ols["beta0"])),
        "alpha": ols["beta1"],
        "ic_alpha": ols["ic_beta1"],
        "r2_log": ols["r2"],
        "rmse_log": ols["rmse"],
        "aicc_log": ols["aicc"],
        "ols": ols,
    }


def ajustar_curvas_polinomiais(x_vals, y_vals, candidatos=None):
    """Ajusta curvas polinomiais candidatas via `curve_fit` e ranqueia por AICc."""
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]

    if len(x_arr) < 4:
        return []

    nomes = candidatos if candidatos is not None else list(MODELOS_POLINOMIAIS.keys())
    resultados = []
    for nome in nomes:
        func = MODELOS_POLINOMIAIS[nome]
        k_params = 2 if nome == "O(1)" else 3
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    func,
                    x_arr,
                    y_arr,
                    p0=[1e-6, float(np.min(y_arr))],
                    bounds=([0.0, -np.inf], [np.inf, np.inf]),
                    maxfev=10000,
                )
            y_pred = func(x_arr, *popt)
            m = calcular_metricas_ajuste(y_arr, y_pred, k_params=k_params)
            resultados.append({
                "modelo": nome,
                "a": float(popt[0]),
                "b": float(popt[1]),
                "r2": m["r2"],
                "rmse": m["rmse"],
                "aic": m["aic"],
                "aicc": m["aicc"],
                "func": func,
            })
        except Exception:
            continue

    resultados.sort(key=lambda r: (np.inf if np.isnan(r["aicc"]) else r["aicc"]))
    if resultados and np.isfinite(resultados[0]["aicc"]):
        min_aicc = resultados[0]["aicc"]
        for r in resultados:
            r["delta_aicc"] = float(r["aicc"] - min_aicc) if np.isfinite(r["aicc"]) else np.nan
    return resultados


def _resumir_grupo_bestfirst_fila(rotulo, grp_raw, grp_agg):
    """Calcula estatísticas de Memória ~ max_fila e max_fila ~ N para um subconjunto do Best-First."""
    ols_mf = regressao_linear_ols(grp_raw["max_fila"].values, grp_raw["mem_mib"].values)
    log_mf = ajustar_lei_potencia_loglog(grp_raw["max_fila"].values, grp_raw["mem_mib"].values)

    med_n = grp_agg.groupby("n", as_index=False)["max_fila"].median().sort_values("n")
    fits_fn = ajustar_curvas_polinomiais(
        med_n["n"].values,
        med_n["max_fila"].values,
        candidatos=["O(N)", "O(N log N)", "O(N²)", "O(N² log N)", "O(N³)"],
    )
    log_fn = ajustar_lei_potencia_loglog(med_n["n"].values, med_n["max_fila"].values)
    melhor_fn = fits_fn[0] if fits_fn else None

    return {
        "inicial": rotulo,
        "n_instancias": len(grp_raw),
        "bytes_por_no_fila": ols_mf["beta1"] * (1024.0 ** 2) if ols_mf else np.nan,
        "bytes_no_ic95_inf": ols_mf["ic_beta1"][0] * (1024.0 ** 2) if ols_mf else np.nan,
        "bytes_no_ic95_sup": ols_mf["ic_beta1"][1] * (1024.0 ** 2) if ols_mf else np.nan,
        "mem_base_mib": ols_mf["beta0"] if ols_mf else np.nan,
        "mem_fila_r2_lin": ols_mf["r2"] if ols_mf else np.nan,
        "mem_fila_aicc_lin": ols_mf["aicc"] if ols_mf else np.nan,
        "mem_fila_alpha_log": log_mf["alpha"] if log_mf else np.nan,
        "mem_fila_r2_log": log_mf["r2_log"] if log_mf else np.nan,
        "mem_fila_aicc_log": log_mf["aicc_log"] if log_mf else np.nan,
        "fila_n_melhor_poli": melhor_fn["modelo"] if melhor_fn else "-",
        "fila_n_r2_poli": melhor_fn["r2"] if melhor_fn else np.nan,
        "fila_n_aicc_poli": melhor_fn["aicc"] if melhor_fn else np.nan,
        "fila_n_alpha_log": log_fn["alpha"] if log_fn else np.nan,
        "fila_n_alpha_ic95_inf": log_fn["ic_alpha"][0] if log_fn else np.nan,
        "fila_n_alpha_ic95_sup": log_fn["ic_alpha"][1] if log_fn else np.nan,
        "fila_n_r2_log": log_fn["r2_log"] if log_fn else np.nan,
        "fila_n_aicc_log": log_fn["aicc_log"] if log_fn else np.nan,
    }


def analisar_memoria_bnb(df_m):
    """Agrega e ajusta os modelos de memória pico, memória por nó ativo e predições do Best-First."""
    resultados = {
        "raw": None,
        "agregado": None,
        "tabela_memoria": None,
        "tabela_bestfirst_fila": None,
    }

    if df_m is None or df_m.empty or "mem_pico_bytes" not in df_m.columns:
        return resultados

    df_valid = df_m.copy()
    for col in ["n", "mem_pico_bytes", "nos_expandidos", "nos_gerados", "prof_max", "max_fila"]:
        if col in df_valid.columns:
            df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")

    df_valid = df_valid.dropna(subset=["n", "mem_pico_bytes"])
    df_valid = df_valid[df_valid["mem_pico_bytes"] > 0]
    if df_valid.empty:
        return resultados

    # Estrutura ativa simultânea: max_fila no Best-First ('melhor') vs. prof_max no DFS ('profundidade')
    nos_ativos_inst = np.where(
        df_valid["estrategia"] == "melhor",
        np.maximum(df_valid.get("max_fila", 1).fillna(1), 1),
        np.maximum(df_valid.get("prof_max", 1).fillna(1), 1),
    )
    df_valid["nos_ativos"] = nos_ativos_inst
    df_valid["mem_mib"] = df_valid["mem_pico_bytes"] / (1024.0 ** 2)
    df_valid["kib_por_no_ativo"] = (df_valid["mem_pico_bytes"] / 1024.0) / df_valid["nos_ativos"]
    resultados["raw"] = df_valid

    agg_m = (
        df_valid.groupby(["estrategia", "inicial", "n"], as_index=False)
        .agg(
            mem_mib=("mem_mib", "median"),
            kib_por_no_ativo=("kib_por_no_ativo", "median"),
            nos_ativos=("nos_ativos", "median"),
            max_fila=("max_fila", "median"),
            prof_max=("prof_max", "median"),
            nos_expandidos=("nos_expandidos", "median"),
            n_runs=("mem_mib", "count"),
            taxa_otimo=("status", lambda s: float(np.mean(s.astype(str).str.lower() == "otimo"))),
        )
        .sort_values("n")
    )
    resultados["agregado"] = agg_m

    # 1. Tabela geral por (estrategia, inicial): Memória Pico vs N e Memória/Nó Ativo vs N
    linhas = []
    for (est, ini), grp in agg_m.groupby(["estrategia", "inicial"]):
        grp_ord = grp.sort_values("n")
        n_vals = grp_ord["n"].values

        fits_mem = ajustar_curvas_polinomiais(n_vals, grp_ord["mem_mib"].values)
        fit_log_mem = ajustar_lei_potencia_loglog(n_vals, grp_ord["mem_mib"].values)
        melhor_mem = fits_mem[0] if fits_mem else None

        fits_no = ajustar_curvas_polinomiais(n_vals, grp_ord["kib_por_no_ativo"].values)
        fit_log_no = ajustar_lei_potencia_loglog(n_vals, grp_ord["kib_por_no_ativo"].values)
        melhor_no = fits_no[0] if fits_no else None

        linhas.append({
            "estrategia": est,
            "inicial": ini,
            "n_pts": len(grp_ord),
            "mem_melhor_aicc": melhor_mem["modelo"] if melhor_mem else "-",
            "mem_aicc": melhor_mem["aicc"] if melhor_mem else np.nan,
            "mem_r2": melhor_mem["r2"] if melhor_mem else np.nan,
            "mem_rmse_mib": melhor_mem["rmse"] if melhor_mem else np.nan,
            "mem_alpha_loglog": fit_log_mem["alpha"] if fit_log_mem else np.nan,
            "mem_alpha_ic95_inf": fit_log_mem["ic_alpha"][0] if fit_log_mem else np.nan,
            "mem_alpha_ic95_sup": fit_log_mem["ic_alpha"][1] if fit_log_mem else np.nan,
            "mem_r2_loglog": fit_log_mem["r2_log"] if fit_log_mem else np.nan,
            "mem_aicc_loglog": fit_log_mem["aicc_log"] if fit_log_mem else np.nan,
            "no_melhor_aicc": melhor_no["modelo"] if melhor_no else "-",
            "no_aicc": melhor_no["aicc"] if melhor_no else np.nan,
            "no_r2": melhor_no["r2"] if melhor_no else np.nan,
            "no_alpha_loglog": fit_log_no["alpha"] if fit_log_no else np.nan,
            "no_alpha_ic95_inf": fit_log_no["ic_alpha"][0] if fit_log_no else np.nan,
            "no_alpha_ic95_sup": fit_log_no["ic_alpha"][1] if fit_log_no else np.nan,
            "no_r2_loglog": fit_log_no["r2_log"] if fit_log_no else np.nan,
        })

    if linhas:
        resultados["tabela_memoria"] = pd.DataFrame(linhas)

    # 2. Predições específicas do Best-First:
    #    (A) Memória Pico (MiB) ~ max_fila (com e sem `trivial`)
    #    (B) Tamanho da fila (max_fila) ~ N
    bf_raw = df_valid[(df_valid["estrategia"] == "melhor") & (df_valid["max_fila"] > 0)].copy()
    bf_agg = agg_m[(agg_m["estrategia"] == "melhor") & (agg_m["max_fila"] > 0)].copy()
    linhas_bf = []

    if not bf_raw.empty:
        linhas_bf.append(_resumir_grupo_bestfirst_fila("GERAL (todas)", bf_raw, bf_agg))

        bf_raw_sem_triv = bf_raw[bf_raw["inicial"] != "trivial"]
        bf_agg_sem_triv = bf_agg[bf_agg["inicial"] != "trivial"]
        if not bf_raw_sem_triv.empty:
            linhas_bf.append(
                _resumir_grupo_bestfirst_fila("GERAL (sem trivial)", bf_raw_sem_triv, bf_agg_sem_triv)
            )

        for ini, grp_raw in bf_raw.groupby("inicial"):
            grp_agg = bf_agg[bf_agg["inicial"] == ini]
            linhas_bf.append(_resumir_grupo_bestfirst_fila(ini, grp_raw, grp_agg))

    if linhas_bf:
        resultados["tabela_bestfirst_fila"] = pd.DataFrame(linhas_bf)

    return resultados


def imprimir_resumo_memoria(resultados):
    """Exibe no terminal as estatísticas de AICc, R² e IC 95% para memória e fila do Best-First."""
    df_tab = resultados.get("tabela_memoria")
    if df_tab is not None and not df_tab.empty:
        print("=" * 115)
        print("BRANCH AND BOUND - COMPLEXIDADE ESPACIAL (DFS vs. BEST-FIRST) [AICc + R² + IC 95%]")
        print("=" * 115)
        for _, row in df_tab.iterrows():
            est = row["estrategia"]
            ini = row["inicial"]

            if np.isfinite(row["mem_alpha_loglog"]):
                str_mem = (
                    f"Poli={row['mem_melhor_aicc']} (R²={row['mem_r2']:.3f}, AICc={row['mem_aicc']:.1f}) | "
                    f"α={row['mem_alpha_loglog']:.2f} [IC95%: {row['mem_alpha_ic95_inf']:.2f}, {row['mem_alpha_ic95_sup']:.2f}] "
                    f"(R²_log={row['mem_r2_loglog']:.3f})"
                )
            else:
                str_mem = f"Poli={row['mem_melhor_aicc']} (R²={row['mem_r2']:.3f})"

            if np.isfinite(row["no_alpha_loglog"]):
                str_no = (
                    f"Poli={row['no_melhor_aicc']} (R²={row['no_r2']:.3f}, AICc={row['no_aicc']:.1f}) | "
                    f"α={row['no_alpha_loglog']:.2f} [IC95%: {row['no_alpha_ic95_inf']:.2f}, {row['no_alpha_ic95_sup']:.2f}] "
                    f"(R²_log={row['no_r2_loglog']:.3f})"
                )
            else:
                str_no = f"Poli={row['no_melhor_aicc']}"

            print(f"[{est:<12} | {ini:<12}] Memória Pico (MiB): {str_mem}")
            print(f"{'':<29} Memória/Nó Ativo:   {str_no}")

    df_bf = resultados.get("tabela_bestfirst_fila")
    if df_bf is not None and not df_bf.empty:
        print("\n" + "=" * 115)
        print("BEST-FIRST SEARCH - PREDIÇÃO: MEMÓRIA ~ MAX_FILA  E  MAX_FILA ~ N [AICc + R² + IC 95%]")
        print("=" * 115)
        for _, row in df_bf.iterrows():
            ini = row["inicial"]
            if np.isfinite(row["bytes_por_no_fila"]):
                str_mf = (
                    f"Linear: {row['bytes_por_no_fila']:.1f} B/nó "
                    f"[IC95%: {row['bytes_no_ic95_inf']:.1f}, {row['bytes_no_ic95_sup']:.1f}], "
                    f"base={row['mem_base_mib']:.2f} MiB (R²={row['mem_fila_r2_lin']:.3f}, AICc={row['mem_fila_aicc_lin']:.1f}) | "
                    f"Log-Log: α={row['mem_fila_alpha_log']:.2f} (R²_log={row['mem_fila_r2_log']:.3f}, AICc={row['mem_fila_aicc_log']:.1f})"
                )
            else:
                str_mf = "insuficiente"

            if np.isfinite(row["fila_n_alpha_log"]):
                str_fn = (
                    f"Poli={row['fila_n_melhor_poli']} (R²={row['fila_n_r2_poli']:.3f}, AICc={row['fila_n_aicc_poli']:.1f}) | "
                    f"Log-Log: α={row['fila_n_alpha_log']:.2f} "
                    f"[IC95%: {row['fila_n_alpha_ic95_inf']:.2f}, {row['fila_n_alpha_ic95_sup']:.2f}] "
                    f"(R²_log={row['fila_n_r2_log']:.3f})"
                )
            else:
                str_fn = f"Poli={row['fila_n_melhor_poli']}"

            print(f"[Best-First | {ini:<19}] Memória ~ Fila: {str_mf}")
            print(f"{'':<36} Fila ~ N:       {str_fn}")


# ---------------------------------------------------------------------------
# 3. VISUALIZAÇÕES (MANTIDOS + NOVOS GRÁFICOS DO BEST-FIRST)
# ---------------------------------------------------------------------------

def plotar_memoria_pico_por_estrategia(resultados, estrategia, pasta_saida="."):
    """Gráfico 1/2 (mantido): Memória Pico (MiB) vs. N com R², AICc e expoente α (IC 95%)."""
    df_agg = resultados.get("agregado")
    if df_agg is None or df_agg.empty:
        return

    sub = df_agg[df_agg["estrategia"] == estrategia].copy()
    if sub.empty:
        print(f"[Aviso] Sem dados de memória para estratégia '{estrategia}'.")
        return

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    titulo_est = LABELS_ESTRATEGIAS_BNB.get(estrategia, estrategia)

    for ini, grp in sub.groupby("inicial"):
        grp_ord = grp.sort_values("n")
        cor = CORES_INICIALIZACOES_BNB.get(ini, None)
        fit_log_ini = ajustar_lei_potencia_loglog(grp_ord["n"].values, grp_ord["mem_mib"].values)
        if fit_log_ini is not None:
            label = (
                f"{ini}: α={fit_log_ini['alpha']:.2f} "
                f"[{fit_log_ini['ic_alpha'][0]:.2f}, {fit_log_ini['ic_alpha'][1]:.2f}] "
                f"(R²_log={fit_log_ini['r2_log']:.3f})"
            )
        else:
            label = f"Inicialização: {ini}"

        ax.plot(
            grp_ord["n"],
            grp_ord["mem_mib"],
            marker="s",
            markersize=4.5,
            linewidth=1.3,
            alpha=0.8,
            color=cor,
            label=label,
        )

    med_m = sub.groupby("n", as_index=False)["mem_mib"].median().sort_values("n")
    ax.plot(
        med_m["n"],
        med_m["mem_mib"],
        color="#000000",
        marker="D",
        markersize=5.5,
        linewidth=2.0,
        label="Mediana Geral por N",
    )

    fits = ajustar_curvas_polinomiais(med_m["n"].values, med_m["mem_mib"].values)
    fit_log = ajustar_lei_potencia_loglog(med_m["n"].values, med_m["mem_mib"].values)
    if fits:
        n_grid = np.linspace(med_m["n"].min(), med_m["n"].max(), 150)
        f0 = fits[0]
        sufixo_alpha = (
            f", α={fit_log['alpha']:.2f} [{fit_log['ic_alpha'][0]:.2f}, {fit_log['ic_alpha'][1]:.2f}] "
            f"(R²_log={fit_log['r2_log']:.3f})"
            if fit_log is not None
            else ""
        )
        ax.plot(
            n_grid,
            f0["func"](n_grid, f0["a"], f0["b"]),
            color="#c0392b",
            linestyle="--",
            linewidth=2.0,
            label=f"Melhor AICc: {f0['modelo']} (R²={f0['r2']:.3f}, AICc={f0['aicc']:.1f}{sufixo_alpha})",
        )

    ax.set_xlabel("Número de Vértices (N)")
    ax.set_ylabel("Memória Pico - tracemalloc (MiB)")
    ax.set_title(f"B&B ({titulo_est}): Complexidade Espacial Empírica vs. N", fontweight="bold")
    ax.legend(fontsize=7.8, loc="upper left")
    fig.tight_layout()

    caminho = os.path.join(pasta_saida, f"01_bnb_memoria_pico_{estrategia}.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Salvo: {caminho}")


def plotar_comparacao_loglog_dfs_vs_bestfirst(resultados, pasta_saida="."):
    """Gráfico 3 (mantido): Comparação direta DFS vs. Best-First em escala log-log (M(N) = c · N^α)."""
    df_agg = resultados.get("agregado")
    if df_agg is None or df_agg.empty:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    estilos = {
        "profundidade": {"cor": "#2980b9", "marker": "o", "nome": "DFS (profundidade)"},
        "melhor": {"cor": "#e74c3c", "marker": "s", "nome": "Best-First (melhor)"},
    }

    for est, cfg in estilos.items():
        sub = df_agg[df_agg["estrategia"] == est]
        if sub.empty:
            continue
        med_m = sub.groupby("n", as_index=False)["mem_mib"].median().sort_values("n")
        mask = (med_m["n"] > 0) & (med_m["mem_mib"] > 0)
        n_vals = med_m.loc[mask, "n"].values
        y_vals = med_m.loc[mask, "mem_mib"].values

        ax.loglog(
            n_vals,
            y_vals,
            marker=cfg["marker"],
            markersize=5.5,
            linestyle="none",
            color=cfg["cor"],
            label=f"{cfg['nome']} — Mediana por N",
        )

        fit_log = ajustar_lei_potencia_loglog(n_vals, y_vals)
        if fit_log is not None:
            ols = fit_log["ols"]
            n_grid = np.linspace(n_vals.min(), n_vals.max(), 150)
            ln_n = np.log(n_grid)
            ln_pred = ols["beta0"] + ols["beta1"] * ln_n
            se_fit = np.sqrt(ols["s2"] * (1.0 / ols["n_obs"] + ((ln_n - ols["x_mean"]) ** 2) / ols["sxx"]))
            ln_inf = ln_pred - ols["t_crit"] * se_fit
            ln_sup = ln_pred + ols["t_crit"] * se_fit

            ax.loglog(
                n_grid,
                np.exp(ln_pred),
                color=cfg["cor"],
                linestyle="--",
                linewidth=2.0,
                label=(
                    f"{cfg['nome']}: α={fit_log['alpha']:.2f} "
                    f"[{fit_log['ic_alpha'][0]:.2f}, {fit_log['ic_alpha'][1]:.2f}] "
                    f"(R²_log={fit_log['r2_log']:.3f}, AICc={fit_log['aicc_log']:.1f})"
                ),
            )
            ax.fill_between(n_grid, np.exp(ln_inf), np.exp(ln_sup), color=cfg["cor"], alpha=0.14)

    ax.set_xlabel("Número de Vértices N (escala log)")
    ax.set_ylabel("Memória Pico MiB (escala log)")
    ax.set_title("B&B DFS vs. Best-First: Expoente Espacial Empírico M(N) = c · N^α (Log-Log)", fontweight="bold")
    ax.legend(fontsize=8.2, loc="upper left")
    fig.tight_layout()

    caminho = os.path.join(pasta_saida, "02_bnb_memoria_loglog_dfs_vs_bestfirst.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Salvo: {caminho}")


def plotar_memoria_por_no_ativo_por_estrategia(resultados, estrategia, pasta_saida="."):
    """Gráfico 4/5 (mantido): Memória por nó ativo na fronteira/pilha (KiB / nó ativo) vs. N."""
    df_agg = resultados.get("agregado")
    if df_agg is None or df_agg.empty:
        return

    sub = df_agg[df_agg["estrategia"] == estrategia].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    titulo_est = LABELS_ESTRATEGIAS_BNB.get(estrategia, estrategia)
    denom_label = "max_fila" if estrategia == "melhor" else "prof_max"

    for ini, grp in sub.groupby("inicial"):
        grp_ord = grp.sort_values("n")
        cor = CORES_INICIALIZACOES_BNB.get(ini, None)
        ax.plot(
            grp_ord["n"],
            grp_ord["kib_por_no_ativo"],
            marker="o",
            markersize=4.5,
            linewidth=1.3,
            alpha=0.75,
            color=cor,
            label=f"Inicialização: {ini}",
        )

    med_no = sub.groupby("n", as_index=False)["kib_por_no_ativo"].median().sort_values("n")
    ax.plot(
        med_no["n"],
        med_no["kib_por_no_ativo"],
        color="#000000",
        marker="D",
        markersize=5.5,
        linewidth=2.0,
        label=f"Mediana Geral por N (KiB / {denom_label})",
    )

    fits = ajustar_curvas_polinomiais(med_no["n"].values, med_no["kib_por_no_ativo"].values)
    fit_log = ajustar_lei_potencia_loglog(med_no["n"].values, med_no["kib_por_no_ativo"].values)
    if fits:
        n_grid = np.linspace(med_no["n"].min(), med_no["n"].max(), 150)
        f0 = fits[0]
        sufixo_alpha = (
            f", α={fit_log['alpha']:.2f} [{fit_log['ic_alpha'][0]:.2f}, {fit_log['ic_alpha'][1]:.2f}] "
            f"(R²_log={fit_log['r2_log']:.3f})"
            if fit_log is not None
            else ""
        )
        ax.plot(
            n_grid,
            f0["func"](n_grid, f0["a"], f0["b"]),
            color="#c0392b",
            linestyle="--",
            linewidth=2.0,
            label=f"Melhor AICc: {f0['modelo']} (R²={f0['r2']:.3f}, AICc={f0['aicc']:.1f}{sufixo_alpha})",
        )

    ax.set_xlabel("Número de Vértices (N)")
    ax.set_ylabel(f"Memória Pico por Nó Ativo (KiB / {denom_label})")
    ax.set_title(f"B&B ({titulo_est}): Pegada de Memória por Estado Ativo vs. N", fontweight="bold")
    ax.legend(fontsize=8.0, loc="upper left")
    fig.tight_layout()

    caminho = os.path.join(pasta_saida, f"03_bnb_memoria_por_no_ativo_{estrategia}.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Salvo: {caminho}")


def plotar_bestfirst_memoria_por_tamanho_fila(resultados, pasta_saida=".", excluir_trivial=False):
    """Gráfico 6/8: Best-First — Predição da memória pico (MiB) pelo tamanho da fila (`max_fila`).
    Se `excluir_trivial=True`, remove `trivial` dos pontos e da regressão.
    """
    df_raw = resultados.get("raw")
    if df_raw is None or df_raw.empty:
        return

    bf = df_raw[(df_raw["estrategia"] == "melhor") & (df_raw["max_fila"] > 0)].copy()
    if excluir_trivial:
        bf = bf[bf["inicial"] != "trivial"].copy()

    if bf.empty:
        print(f"[Aviso] Sem dados do Best-First (`melhor`) (excluir_trivial={excluir_trivial}).")
        return

    fig, ax = plt.subplots(figsize=(9.8, 5.8))

    for ini, grp in bf.groupby("inicial"):
        cor = CORES_INICIALIZACOES_BNB.get(ini, None)
        ax.scatter(
            grp["max_fila"],
            grp["mem_mib"],
            s=32,
            alpha=0.72,
            color=cor,
            label=f"Inicialização: {ini}",
        )

    x_vals = bf["max_fila"].values
    y_vals = bf["mem_mib"].values

    # Ajuste 1: Modelo linear M(fila) = beta0 + beta1 * max_fila (bytes marginais por nó na fila)
    ols = regressao_linear_ols(x_vals, y_vals)
    # Ajuste 2: Lei de potência em log-log M(fila) = c * max_fila^alpha
    fit_log = ajustar_lei_potencia_loglog(x_vals, y_vals)

    x_grid = np.linspace(np.min(x_vals), np.max(x_vals), 200)

    if ols is not None:
        y_lin = ols["beta0"] + ols["beta1"] * x_grid
        se_fit = np.sqrt(ols["s2"] * (1.0 / ols["n_obs"] + ((x_grid - ols["x_mean"]) ** 2) / ols["sxx"]))
        y_inf = y_lin - ols["t_crit"] * se_fit
        y_sup = y_lin + ols["t_crit"] * se_fit

        bytes_no = ols["beta1"] * (1024.0 ** 2)
        bytes_inf = ols["ic_beta1"][0] * (1024.0 ** 2)
        bytes_sup = ols["ic_beta1"][1] * (1024.0 ** 2)

        ax.plot(
            x_grid,
            y_lin,
            color="#c0392b",
            linestyle="--",
            linewidth=2.0,
            label=(
                f"Linear: {bytes_no:.1f} B/nó [IC95%: {bytes_inf:.1f}, {bytes_sup:.1f}] + {ols['beta0']:.2f} MiB "
                f"(R²={ols['r2']:.3f}, AICc={ols['aicc']:.1f})"
            ),
        )
        ax.fill_between(x_grid, y_inf, y_sup, color="#c0392b", alpha=0.14)

    if fit_log is not None:
        y_pot = fit_log["c"] * (x_grid ** fit_log["alpha"])
        ax.plot(
            x_grid,
            y_pot,
            color="#1f618d",
            linestyle=":",
            linewidth=1.9,
            label=(
                f"Potência (log-log): α={fit_log['alpha']:.2f} "
                f"[{fit_log['ic_alpha'][0]:.2f}, {fit_log['ic_alpha'][1]:.2f}] "
                f"(R²_log={fit_log['r2_log']:.3f}, AICc={fit_log['aicc_log']:.1f})"
            ),
        )

    ax.xaxis.set_major_formatter(FuncFormatter(_formatar_eixo_inteiro))
    ax.set_xlabel("Tamanho Máximo da Fila de Prioridade (`max_fila` - nós)")
    ax.set_ylabel("Memória Pico - tracemalloc (MiB)")
    sufixo_titulo = " (Sem Inicialização Trivial)" if excluir_trivial else " (Todas as Inicializações)"
    ax.set_title(
        f"Best-First Search: Predição da Memória Pico pelo Tamanho da Fila{sufixo_titulo}",
        fontweight="bold",
    )
    ax.legend(fontsize=7.8, loc="upper left")
    fig.tight_layout()

    nome_arq = (
        "06_bnb_bestfirst_memoria_vs_fila_sem_trivial.png"
        if excluir_trivial
        else "04_bnb_bestfirst_memoria_vs_fila.png"
    )
    caminho = os.path.join(pasta_saida, nome_arq)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Salvo: {caminho}")


def plotar_bestfirst_fila_por_vertices(resultados, pasta_saida="."):
    """Gráfico 7 (mantido): Best-First — Predição do tamanho máximo da fila (`max_fila`) pelo número de vértices N."""
    df_agg = resultados.get("agregado")
    if df_agg is None or df_agg.empty:
        return

    bf = df_agg[(df_agg["estrategia"] == "melhor") & (df_agg["max_fila"] > 0)].copy()
    if bf.empty:
        print("[Aviso] Sem dados agregados do Best-First (`melhor`) com max_fila > 0.")
        return

    fig, ax = plt.subplots(figsize=(9.8, 5.8))

    for ini, grp in bf.groupby("inicial"):
        grp_ord = grp.sort_values("n")
        cor = CORES_INICIALIZACOES_BNB.get(ini, None)
        fit_log_ini = ajustar_lei_potencia_loglog(grp_ord["n"].values, grp_ord["max_fila"].values)
        if fit_log_ini is not None:
            label = (
                f"{ini}: α={fit_log_ini['alpha']:.2f} "
                f"[{fit_log_ini['ic_alpha'][0]:.2f}, {fit_log_ini['ic_alpha'][1]:.2f}] "
                f"(R²_log={fit_log_ini['r2_log']:.3f})"
            )
        else:
            label = f"Inicialização: {ini}"

        ax.plot(
            grp_ord["n"],
            grp_ord["max_fila"],
            marker="s",
            markersize=4.5,
            linewidth=1.3,
            alpha=0.8,
            color=cor,
            label=label,
        )

    med_fila = bf.groupby("n", as_index=False)["max_fila"].median().sort_values("n")
    ax.plot(
        med_fila["n"],
        med_fila["max_fila"],
        color="#000000",
        marker="D",
        markersize=5.5,
        linewidth=2.0,
        label="Mediana Geral por N",
    )

    fits = ajustar_curvas_polinomiais(
        med_fila["n"].values,
        med_fila["max_fila"].values,
        candidatos=["O(N)", "O(N log N)", "O(N²)", "O(N² log N)", "O(N³)"],
    )
    fit_log = ajustar_lei_potencia_loglog(med_fila["n"].values, med_fila["max_fila"].values)
    if fits:
        n_grid = np.linspace(med_fila["n"].min(), med_fila["n"].max(), 150)
        f0 = fits[0]
        sufixo_alpha = (
            f", α={fit_log['alpha']:.2f} [{fit_log['ic_alpha'][0]:.2f}, {fit_log['ic_alpha'][1]:.2f}] "
            f"(R²_log={fit_log['r2_log']:.3f})"
            if fit_log is not None
            else ""
        )
        ax.plot(
            n_grid,
            f0["func"](n_grid, f0["a"], f0["b"]),
            color="#c0392b",
            linestyle="--",
            linewidth=2.0,
            label=f"Melhor AICc: {f0['modelo']} (R²={f0['r2']:.3f}, AICc={f0['aicc']:.1f}{sufixo_alpha})",
        )

    ax.yaxis.set_major_formatter(FuncFormatter(_formatar_eixo_inteiro))
    ax.set_xlabel("Número de Vértices (N)")
    ax.set_ylabel("Tamanho Máximo da Fila (`max_fila` - nós)")
    ax.set_title("Best-First Search: Predição do Tamanho da Fila pelo Número de Vértices N", fontweight="bold")
    ax.legend(fontsize=7.8, loc="upper left")
    fig.tight_layout()

    caminho = os.path.join(pasta_saida, "05_bnb_bestfirst_fila_vs_vertices.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Salvo: {caminho}")


# ---------------------------------------------------------------------------
# 4. EXECUÇÃO PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Modelagem de Complexidade Espacial Empírica do Branch and Bound (DFS vs. Best-First)"
    )
    parser.add_argument(
        "--pasta-resultados",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resultados"),
        help="Pasta contendo os CSVs de memória do Branch and Bound (resultados_bnb_*_memoria.csv)",
    )
    parser.add_argument(
        "--pasta-saida",
        default="graficos_memoria_bnb",
        help="Diretório de saída para tabelas e gráficos",
    )
    args = parser.parse_args()

    os.makedirs(args.pasta_saida, exist_ok=True)

    print("1. Carregando CSVs de memória do Branch and Bound (*_memoria.csv)...")
    df_bnb_mem = carregar_dados_bnb_memoria(args.pasta_resultados)

    print("2. Ajustando modelos espaciais (DFS vs. Best-First) e predições de fila via AICc, R² e IC 95%...")
    resultados = analisar_memoria_bnb(df_bnb_mem)

    df_tab = resultados.get("tabela_memoria")
    if df_tab is not None and not df_tab.empty:
        caminho_csv = os.path.join(args.pasta_saida, "metricas_memoria_bnb.csv")
        df_tab.to_csv(caminho_csv, index=False)
        print(f"Tabela de métricas espaciais salva em: {caminho_csv}")

    df_bf = resultados.get("tabela_bestfirst_fila")
    if df_bf is not None and not df_bf.empty:
        caminho_bf_csv = os.path.join(args.pasta_saida, "metricas_bestfirst_fila_memoria.csv")
        df_bf.to_csv(caminho_bf_csv, index=False)
        print(f"Tabela de predição Best-First (Memória ~ Fila e Fila ~ N) salva em: {caminho_bf_csv}")

    imprimir_resumo_memoria(resultados)

    print("\n" + "=" * 115)
    print("3. GERANDO GRÁFICOS DE COMPLEXIDADE ESPACIAL E PREDIÇÃO DA FILA (BEST-FIRST)")
    print("=" * 115)
    for estrategia in ["profundidade", "melhor"]:
        plotar_memoria_pico_por_estrategia(resultados, estrategia, pasta_saida=args.pasta_saida)
        plotar_memoria_por_no_ativo_por_estrategia(resultados, estrategia, pasta_saida=args.pasta_saida)

    plotar_comparacao_loglog_dfs_vs_bestfirst(resultados, pasta_saida=args.pasta_saida)
    plotar_bestfirst_memoria_por_tamanho_fila(resultados, pasta_saida=args.pasta_saida, excluir_trivial=False)
    plotar_bestfirst_fila_por_vertices(resultados, pasta_saida=args.pasta_saida)
    plotar_bestfirst_memoria_por_tamanho_fila(resultados, pasta_saida=args.pasta_saida, excluir_trivial=True)


if __name__ == "__main__":
    main()