"""
Modelagem de Complexidade Empírica do Branch and Bound por Nó:
  T(N) = nós(N) × custo_por_nó(N)

  - nós(N): ajustado ao modelo exponencial a · b^N via regressão linear em ln(nós),
    com seleção de modelo via AICc, R² (log) e intervalo de confiança (IC 95%) para b.
  - custo_por_nó(N): calculado sobre TODAS as execuções (ótimas + timeouts com 1800 s)
    para eliminar viés de seleção, com ajuste polinomial e log-log ranqueados por AICc.

Regras de carregamento dos CSVs (conforme `rodar_bnb.py`):
  - Padrão de nome B&B: `resultados_bnb_<inicial>_<estrategia>[_memoria].csv`
    onde `<estrategia>` é `profundidade` ou `melhor`.
  - Tempo: arquivos executados SEM `--memoria` (não terminam com `_memoria.csv`).

Uso:
    python modelagem_complexidade_bnb.py
    python modelagem_complexidade_bnb.py --pasta-resultados ./resultados --pasta-saida graficos_complexidade --timeout-s 1800
"""

import argparse
import glob
import os
import warnings

import matplotlib.pyplot as plt
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

# Curvas polinomiais candidatas para o custo por nó em função de N
MODELOS_CUSTO_NO = {
    "O(N)": lambda n, a, b: a * n + b,
    "O(N log N)": lambda n, a, b: a * n * np.log2(np.maximum(n, 2)) + b,
    "O(N²)": lambda n, a, b: a * (n ** 2) + b,
    "O(N² log N)": lambda n, a, b: a * (n ** 2) * np.log2(np.maximum(n, 2)) + b,
    "O(N³)": lambda n, a, b: a * (n ** 3) + b,
}


# ---------------------------------------------------------------------------
# 1. CARREGAMENTO DOS DADOS DO BRANCH AND BOUND (TEMPO)
# ---------------------------------------------------------------------------

def _inferir_estrategia_nome_arquivo(caminho_csv):
    """Extrai a estratégia (`profundidade` ou `melhor`) do padrão `resultados_bnb_<inicial>_<estrategia>.csv`."""
    base = os.path.basename(caminho_csv)
    sem_ext = base[:-4] if base.endswith(".csv") else base
    partes = sem_ext.split("_")
    if partes and partes[-1] in ("profundidade", "melhor"):
        return partes[-1]
    return None


def carregar_dados_bnb_tempo(pasta_resultados):
    """Carrega apenas os CSVs de tempo do Branch and Bound (ignora `*_memoria.csv`)."""
    padroes = [
        os.path.join(pasta_resultados, "resultados_bnb_*.csv"),
        "resultados_bnb_*.csv",
    ]
    arquivos = []
    for p in padroes:
        arquivos = sorted(glob.glob(p))
        if arquivos:
            break

    arquivos_tempo = [arq for arq in arquivos if not arq.endswith("_memoria.csv")]
    if not arquivos_tempo:
        print("[Aviso] Nenhum CSV de tempo do Branch and Bound encontrado.")
        return None

    lista_tempo = []
    for arq in arquivos_tempo:
        est_arq = _inferir_estrategia_nome_arquivo(arq)
        df = pd.read_csv(arq, na_values=["NA"])

        if "estrategia" not in df.columns and est_arq is not None:
            df["estrategia"] = est_arq
        elif est_arq is not None:
            df["estrategia"] = df["estrategia"].fillna(est_arq)

        lista_tempo.append(df)

    return pd.concat(lista_tempo, ignore_index=True) if lista_tempo else None


# ---------------------------------------------------------------------------
# 2. MÉTRICAS ESTATÍSTICAS (AICc, IC 95%, R², REGRESSÃO EM LOG E CURVE_FIT)
# ---------------------------------------------------------------------------

def calcular_aicc(n_obs, rss, k_params=3):
    """Calcula AIC e AICc sob verossimilhança gaussiana (k_params inclui intercepto, inclinação e variância)."""
    if n_obs <= k_params + 1 or not np.isfinite(rss) or rss <= 0:
        return np.nan, np.nan
    aic = n_obs * np.log(rss / n_obs) + 2.0 * k_params
    correcao = (2.0 * k_params * (k_params + 1)) / (n_obs - k_params - 1)
    return float(aic), float(aic + correcao)


def calcular_metricas_ajuste(y_real, y_pred, k_params=3):
    """Calcula R², RMSE, RSS, AIC e AICc entre os valores observados e os preditos."""
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


def ajustar_lei_potencia_loglog(n_vals, y_vals, nivel_conf=0.95):
    """Estima y = c * N^alpha via regressão linear em log(y) = log(c) + alpha * log(N), com AICc, R² e IC de alpha."""
    n_arr = np.asarray(n_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(n_arr) & np.isfinite(y_arr) & (n_arr > 0) & (y_arr > 0)
    if np.sum(mask) < 4:
        return None

    ols = regressao_linear_ols(np.log(n_arr[mask]), np.log(y_arr[mask]), nivel_conf=nivel_conf)
    if ols is None:
        return None

    return {
        "modelo": "Potência c·N^α (log-log)",
        "c": float(np.exp(ols["beta0"])),
        "alpha": ols["beta1"],
        "ic_alpha": ols["ic_beta1"],
        "r2_log": ols["r2"],
        "rmse_log": ols["rmse"],
        "aicc_log": ols["aicc"],
        "ols": ols,
    }


def ajustar_nos_exponencial_bnb(n_vals, nos_vals, nivel_conf=0.95):
    """Ajusta nós(N) = a · b^N via regressão linear em ln(nós) = ln(a) + N · ln(b).

    Calcula IC 95% de b, R² (log) e compara o AICc na escala ln(nós) contra modelos concorrentes:
      - Exponencial:     ln(nós) = ln(a) + N · ln(b)          [a · b^N]
      - Subexponencial:  ln(nós) = c0 + γ · √N                [exp(c0 + γ√N)]
      - Lei de potência: ln(nós) = ln(c) + α · ln(N)          [c · N^α]
    """
    n_arr = np.asarray(n_vals, dtype=float)
    y_arr = np.asarray(nos_vals, dtype=float)
    mask = np.isfinite(n_arr) & np.isfinite(y_arr) & (n_arr > 0) & (y_arr > 0)
    n_arr, y_arr = n_arr[mask], y_arr[mask]

    if len(n_arr) < 4:
        return None

    ln_y = np.log(y_arr)

    # 1. Modelo principal: Exponencial a · b^N
    ols_exp = regressao_linear_ols(n_arr, ln_y, nivel_conf=nivel_conf)
    if ols_exp is None:
        return None

    a_hat = float(np.exp(ols_exp["beta0"]))
    b_hat = float(np.exp(ols_exp["beta1"]))
    b_ic_inf = float(np.exp(ols_exp["ic_beta1"][0]))
    b_ic_sup = float(np.exp(ols_exp["ic_beta1"][1]))

    # 2. Modelos concorrentes na mesma escala ln(nós) para comparação justa de AICc
    ols_subexp = regressao_linear_ols(np.sqrt(n_arr), ln_y, nivel_conf=nivel_conf)
    ols_pot = regressao_linear_ols(np.log(n_arr), ln_y, nivel_conf=nivel_conf)

    candidatos = [
        {"modelo": "Exponencial (a·b^N)", "aicc": ols_exp["aicc"], "r2_log": ols_exp["r2"]},
    ]
    if ols_subexp is not None:
        candidatos.append(
            {"modelo": "Subexponencial (c·e^{γ√N})", "aicc": ols_subexp["aicc"], "r2_log": ols_subexp["r2"]}
        )
    if ols_pot is not None:
        candidatos.append(
            {"modelo": "Polinomial (c·N^α)", "aicc": ols_pot["aicc"], "r2_log": ols_pot["r2"]}
        )

    validos = [c for c in candidatos if np.isfinite(c["aicc"])]
    validos.sort(key=lambda c: c["aicc"])
    melhor_aicc = validos[0]["aicc"] if validos else np.nan
    for c in validos:
        c["delta_aicc"] = float(c["aicc"] - melhor_aicc)

    delta_aicc_exp = next((c["delta_aicc"] for c in validos if c["modelo"] == "Exponencial (a·b^N)"), np.nan)
    melhor_modelo = validos[0]["modelo"] if validos else "Exponencial (a·b^N)"

    return {
        "n_obs": ols_exp["n_obs"],
        "a": a_hat,
        "b": b_hat,
        "b_ic95": (b_ic_inf, b_ic_sup),
        "ln_b": ols_exp["beta1"],
        "ln_b_ic95": ols_exp["ic_beta1"],
        "r2_log": ols_exp["r2"],
        "rmse_log": ols_exp["rmse"],
        "aicc_exp": ols_exp["aicc"],
        "delta_aicc_exp": delta_aicc_exp,
        "melhor_modelo_aicc": melhor_modelo,
        "comparacao_aicc": validos,
        "ols_exp": ols_exp,
    }


def ajustar_curvas_custo_no(n_vals, y_vals):
    """Ajusta curvas polinomiais candidatas para o custo por nó via `curve_fit` e ranqueia por AICc."""
    n_arr = np.asarray(n_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(n_arr) & np.isfinite(y_arr)
    n_arr, y_arr = n_arr[mask], y_arr[mask]

    if len(n_arr) < 4:
        return []

    resultados = []
    for nome, func in MODELOS_CUSTO_NO.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    func,
                    n_arr,
                    y_arr,
                    p0=[1e-6, float(np.min(y_arr))],
                    bounds=([0.0, -np.inf], [np.inf, np.inf]),
                    maxfev=10000,
                )
            y_pred = func(n_arr, *popt)
            m = calcular_metricas_ajuste(y_arr, y_pred, k_params=3)
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


def analisar_decomposicao_bnb(df_t, timeout_s=1800.0):
    """Decompõe T(N) = nós(N) × custo_por_nó(N) no Branch and Bound:
      - `nos`: ajustado sobre instâncias ótimas (sem censura de timeout) para a · b^N via ln(nós).
      - `custo_no`: calculado sobre TODAS as instâncias (ótimas + timeouts com tempo/1800s) sem viés de seleção.
    """
    resultados_bnb = {
        "nos": None,
        "custo_no": None,
        "tabela_decomposicao": None,
    }

    if df_t is None or df_t.empty:
        return resultados_bnb

    df_valid = df_t.copy()
    for col in ["n", "tempo_busca_s", "tempo_decorrido_s", "nos_expandidos", "nos_gerados"]:
        if col in df_valid.columns:
            df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")

    df_valid = df_valid.dropna(subset=["n", "nos_expandidos"])
    eh_otimo = df_valid["status"].astype(str).str.lower() == "otimo"

    # 1. Custo por nó SEM viés de seleção: inclui ótimas e timeouts (nós / 1800 s nos timeouts)
    t_busca = df_valid["tempo_busca_s"] if "tempo_busca_s" in df_valid.columns else pd.Series(np.nan, index=df_valid.index)
    t_dec = df_valid["tempo_decorrido_s"] if "tempo_decorrido_s" in df_valid.columns else pd.Series(np.nan, index=df_valid.index)
    tempo_base = t_busca.fillna(t_dec)

    df_valid["tempo_efetivo_s"] = np.where(
        eh_otimo,
        tempo_base,
        np.where(tempo_base.notna() & (tempo_base > 0), tempo_base, float(timeout_s)),
    )
    df_valid = df_valid.dropna(subset=["tempo_efetivo_s"])
    df_valid["us_por_no"] = (
        df_valid["tempo_efetivo_s"] / np.maximum(df_valid["nos_expandidos"], 1)
    ) * 1e6

    agg_custo_no = (
        df_valid.groupby(["estrategia", "inicial", "n"], as_index=False)
        .agg(
            us_por_no=("us_por_no", "median"),
            tempo_efetivo_s=("tempo_efetivo_s", "median"),
            nos_todos=("nos_expandidos", "median"),
            n_runs=("nos_expandidos", "count"),
            taxa_otimo=("status", lambda s: float(np.mean(s.astype(str).str.lower() == "otimo"))),
        )
        .sort_values("n")
    )
    resultados_bnb["custo_no"] = agg_custo_no

    # 2. Número de nós nós(N) = a · b^N apenas em instâncias ótimas (evita truncamento artificial de b nos timeouts)
    otimos_t = df_valid[eh_otimo].copy()
    agg_nos = None
    if not otimos_t.empty:
        agg_nos = (
            otimos_t.groupby(["estrategia", "inicial", "n"], as_index=False)
            .agg(
                nos_expandidos=("nos_expandidos", "median"),
                nos_gerados=("nos_gerados", "median"),
                tempo_busca_s=("tempo_busca_s", "median"),
            )
            .sort_values("n")
        )
        resultados_bnb["nos"] = agg_nos

    # 3. Tabela consolidada com AICc, R² e IC 95% de b (nós) e AICc/R² (custo por nó)
    linhas_decomp = []
    for (est, ini), grp_custo in agg_custo_no.groupby(["estrategia", "inicial"]):
        fit_custo_poli = ajustar_curvas_custo_no(grp_custo["n"].values, grp_custo["us_por_no"].values)
        fit_custo_log = ajustar_lei_potencia_loglog(grp_custo["n"].values, grp_custo["us_por_no"].values)
        melhor_custo = fit_custo_poli[0] if fit_custo_poli else None

        fit_nos = None
        if agg_nos is not None:
            grp_nos = agg_nos[(agg_nos["estrategia"] == est) & (agg_nos["inicial"] == ini)]
            if len(grp_nos) >= 4:
                fit_nos = ajustar_nos_exponencial_bnb(
                    grp_nos["n"].values, grp_nos["nos_expandidos"].values
                )

        linhas_decomp.append({
            "estrategia": est,
            "inicial": ini,
            # Modelo exponencial nós(N) = a · b^N (regressão em ln)
            "n_pts_nos": fit_nos["n_obs"] if fit_nos else 0,
            "nos_a": fit_nos["a"] if fit_nos else np.nan,
            "nos_b": fit_nos["b"] if fit_nos else np.nan,
            "nos_b_ic95_inf": fit_nos["b_ic95"][0] if fit_nos else np.nan,
            "nos_b_ic95_sup": fit_nos["b_ic95"][1] if fit_nos else np.nan,
            "nos_r2_log": fit_nos["r2_log"] if fit_nos else np.nan,
            "nos_aicc_exp": fit_nos["aicc_exp"] if fit_nos else np.nan,
            "nos_delta_aicc_exp": fit_nos["delta_aicc_exp"] if fit_nos else np.nan,
            "nos_melhor_aicc": fit_nos["melhor_modelo_aicc"] if fit_nos else "-",
            # Modelo custo por nó (µs/nó) sem viés de seleção
            "n_pts_custo": len(grp_custo),
            "custo_melhor_aicc": melhor_custo["modelo"] if melhor_custo else "-",
            "custo_aicc": melhor_custo["aicc"] if melhor_custo else np.nan,
            "custo_r2": melhor_custo["r2"] if melhor_custo else np.nan,
            "custo_alpha_loglog": fit_custo_log["alpha"] if fit_custo_log else np.nan,
            "custo_alpha_ic95_inf": fit_custo_log["ic_alpha"][0] if fit_custo_log else np.nan,
            "custo_alpha_ic95_sup": fit_custo_log["ic_alpha"][1] if fit_custo_log else np.nan,
            "custo_r2_loglog": fit_custo_log["r2_log"] if fit_custo_log else np.nan,
            "custo_aicc_loglog": fit_custo_log["aicc_log"] if fit_custo_log else np.nan,
        })

    if linhas_decomp:
        resultados_bnb["tabela_decomposicao"] = pd.DataFrame(linhas_decomp)

    return resultados_bnb


def imprimir_resumo_decomposicao(resultados_bnb):
    """Exibe no terminal as tabelas de AICc, R² e IC 95% para nós(N) e custo_por_nó(N)."""
    df_decomp = resultados_bnb.get("tabela_decomposicao")
    if df_decomp is None or df_decomp.empty:
        print("[Aviso] Sem dados suficientes para gerar o resumo da decomposição.")
        return

    print("=" * 115)
    print("BRANCH AND BOUND - DECOMPOSIÇÃO T(N) = nós(N) × custo_por_nó(N) [AICc + R² + IC 95%]")
    print("=" * 115)
    for _, row in df_decomp.iterrows():
        est = row["estrategia"]
        ini = row["inicial"]
        if np.isfinite(row["nos_b"]):
            str_nos = (
                f"b={row['nos_b']:.4f} [IC95%: {row['nos_b_ic95_inf']:.4f}, {row['nos_b_ic95_sup']:.4f}] "
                f"(a={row['nos_a']:.2e}, R²_log={row['nos_r2_log']:.3f}, AICc={row['nos_aicc_exp']:.1f}, "
                f"ΔAICc={row['nos_delta_aicc_exp']:.1f}, melhor={row['nos_melhor_aicc']})"
            )
        else:
            str_nos = "insuficiente (<4 pts ótimos)"

        if np.isfinite(row["custo_alpha_loglog"]):
            str_custo = (
                f"Poli={row['custo_melhor_aicc']} (R²={row['custo_r2']:.3f}, AICc={row['custo_aicc']:.1f}) | "
                f"α={row['custo_alpha_loglog']:.2f} [IC95%: {row['custo_alpha_ic95_inf']:.2f}, {row['custo_alpha_ic95_sup']:.2f}] "
                f"(R²_log={row['custo_r2_loglog']:.3f})"
            )
        else:
            str_custo = f"Poli={row['custo_melhor_aicc']}"

        print(f"[{est:<12} | {ini:<12}] nós(N): {str_nos}")
        print(f"{'':<29} custo/nó (sem viés): {str_custo}")


# ---------------------------------------------------------------------------
# 3. VISUALIZAÇÃO POR NÓ (UM MÉTODO POR GRÁFICO)
# ---------------------------------------------------------------------------

def plotar_nos_exponencial_bnb_por_estrategia(resultados_bnb, estrategia, pasta_saida="."):
    """Gráfico 1/2 (por estratégia): Crescimento exponencial nós(N) = a·b^N (escala semi-log) com R², AICc e IC 95% de b."""
    df_nos = resultados_bnb.get("nos")
    if df_nos is None or df_nos.empty:
        return

    sub = df_nos[df_nos["estrategia"] == estrategia].copy()
    if sub.empty:
        print(f"[Aviso] Sem dados ótimos de nós para estratégia '{estrategia}' no B&B.")
        return

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    titulo_est = LABELS_ESTRATEGIAS_BNB.get(estrategia, estrategia)

    for ini, grp in sub.groupby("inicial"):
        grp_ord = grp.sort_values("n")
        cor = CORES_INICIALIZACOES_BNB.get(ini, None)
        fit_ini = ajustar_nos_exponencial_bnb(grp_ord["n"].values, grp_ord["nos_expandidos"].values)
        if fit_ini is not None:
            label = (
                f"{ini}: b={fit_ini['b']:.3f} "
                f"[{fit_ini['b_ic95'][0]:.3f}, {fit_ini['b_ic95'][1]:.3f}] "
                f"(R²={fit_ini['r2_log']:.3f}, AICc={fit_ini['aicc_exp']:.1f})"
            )
        else:
            label = f"Inicialização: {ini}"

        ax.semilogy(
            grp_ord["n"],
            grp_ord["nos_expandidos"],
            marker="o",
            markersize=4.5,
            linewidth=1.3,
            alpha=0.8,
            color=cor,
            label=label,
        )

    med_n = sub.groupby("n", as_index=False)["nos_expandidos"].median().sort_values("n")
    ax.semilogy(
        med_n["n"],
        med_n["nos_expandidos"],
        color="#000000",
        marker="D",
        markersize=5.5,
        linewidth=2.0,
        label="Mediana Geral por N",
    )

    fit_geral = ajustar_nos_exponencial_bnb(med_n["n"].values, med_n["nos_expandidos"].values)
    if fit_geral is not None:
        ols = fit_geral["ols_exp"]
        n_grid = np.linspace(med_n["n"].min(), med_n["n"].max(), 150)
        ln_pred = ols["beta0"] + ols["beta1"] * n_grid
        se_fit = np.sqrt(ols["s2"] * (1.0 / ols["n_obs"] + ((n_grid - ols["x_mean"]) ** 2) / ols["sxx"]))
        ln_inf = ln_pred - ols["t_crit"] * se_fit
        ln_sup = ln_pred + ols["t_crit"] * se_fit

        ax.semilogy(
            n_grid,
            np.exp(ln_pred),
            color="#c0392b",
            linestyle="--",
            linewidth=2.0,
            label=(
                f"Ajuste Exponencial: {fit_geral['a']:.2e}·{fit_geral['b']:.3f}^N "
                f"(IC95% b=[{fit_geral['b_ic95'][0]:.3f}, {fit_geral['b_ic95'][1]:.3f}], "
                f"R²={fit_geral['r2_log']:.3f}, AICc={fit_geral['aicc_exp']:.1f})"
            ),
        )
        ax.fill_between(n_grid, np.exp(ln_inf), np.exp(ln_sup), color="#c0392b", alpha=0.14)

    ax.set_xlabel("Número de Vértices (N)")
    ax.set_ylabel("Nós Expandidos - escala log")
    ax.set_title(f"B&B ({titulo_est}): Fator de Ramificação Empírico nós(N) = a·b^N", fontweight="bold")
    ax.legend(fontsize=7.8, loc="upper left")
    fig.tight_layout()

    caminho = os.path.join(pasta_saida, f"01_bnb_nos_exponencial_{estrategia}.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Salvo: {caminho}")


def plotar_custo_por_no_bnb_por_estrategia(resultados_bnb, estrategia, pasta_saida="."):
    """Gráfico 3/4 (por estratégia): Custo temporal por nó (µs/nó) SEM viés de seleção vs. N com R² e AICc."""
    df_custo = resultados_bnb.get("custo_no")
    if df_custo is None or df_custo.empty:
        return

    sub = df_custo[df_custo["estrategia"] == estrategia].copy()
    if sub.empty:
        print(f"[Aviso] Sem dados de custo por nó para estratégia '{estrategia}' no B&B.")
        return

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    titulo_est = LABELS_ESTRATEGIAS_BNB.get(estrategia, estrategia)

    for ini, grp in sub.groupby("inicial"):
        grp_ord = grp.sort_values("n")
        cor = CORES_INICIALIZACOES_BNB.get(ini, None)
        ax.plot(
            grp_ord["n"],
            grp_ord["us_por_no"],
            marker="o",
            markersize=4.5,
            linewidth=1.3,
            alpha=0.75,
            color=cor,
            label=f"Inicialização: {ini}",
        )

    med_n = sub.groupby("n", as_index=False)["us_por_no"].median().sort_values("n")
    ax.plot(
        med_n["n"],
        med_n["us_por_no"],
        color="#000000",
        marker="D",
        markersize=5.5,
        linewidth=2.0,
        label="Mediana Geral por N (ótimas + timeouts)",
    )

    fits = ajustar_curvas_custo_no(med_n["n"].values, med_n["us_por_no"].values)
    fit_log = ajustar_lei_potencia_loglog(med_n["n"].values, med_n["us_por_no"].values)
    if fits:
        n_grid = np.linspace(med_n["n"].min(), med_n["n"].max(), 150)
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
    ax.set_ylabel("Tempo por Nó Expandido (µs / nó)")
    ax.set_title(f"B&B ({titulo_est}): Custo por Nó sem Viés de Seleção vs. N", fontweight="bold")
    ax.legend(fontsize=8.0, loc="upper left")
    fig.tight_layout()

    caminho = os.path.join(pasta_saida, f"02_bnb_custo_por_no_{estrategia}.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Salvo: {caminho}")


# ---------------------------------------------------------------------------
# 4. EXECUÇÃO PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Modelagem de Complexidade Empírica do Branch and Bound por Nó"
    )
    parser.add_argument(
        "--pasta-resultados",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resultados"),
        help="Pasta contendo os CSVs de tempo do Branch and Bound (resultados_bnb_*.csv)",
    )
    parser.add_argument(
        "--pasta-saida",
        default="graficos_complexidade",
        help="Diretório de saída para tabelas e gráficos",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=1800.0,
        help="Tempo limite em segundos usado como denominador nos timeouts para custo/nó (padrão: 1800.0)",
    )
    args = parser.parse_args()

    os.makedirs(args.pasta_saida, exist_ok=True)

    print("1. Carregando CSVs de tempo do Branch and Bound...")
    df_bnb_tempo = carregar_dados_bnb_tempo(args.pasta_resultados)

    print("2. Ajustando decomposição T(N) = nós(N) × custo_por_nó(N) via AICc, R² e IC 95%...")
    resultados_bnb = analisar_decomposicao_bnb(df_bnb_tempo, timeout_s=args.timeout_s)

    df_decomp = resultados_bnb.get("tabela_decomposicao")
    if df_decomp is not None and not df_decomp.empty:
        caminho_bnb_csv = os.path.join(args.pasta_saida, "metricas_decomposicao_bnb.csv")
        df_decomp.to_csv(caminho_bnb_csv, index=False)
        print(f"Tabela de decomposição B&B (AICc + R² + IC 95% de b) salva em: {caminho_bnb_csv}")

    imprimir_resumo_decomposicao(resultados_bnb)

    print("\n" + "=" * 115)
    print("3. GERANDO GRÁFICOS POR NÓ (nós(N) E custo_por_nó(N))")
    print("=" * 115)
    for estrategia in ["profundidade", "melhor"]:
        plotar_nos_exponencial_bnb_por_estrategia(resultados_bnb, estrategia, pasta_saida=args.pasta_saida)
        plotar_custo_por_no_bnb_por_estrategia(resultados_bnb, estrategia, pasta_saida=args.pasta_saida)


if __name__ == "__main__":
    main()