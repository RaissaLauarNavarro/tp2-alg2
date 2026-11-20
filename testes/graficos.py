#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graficos_aproximativos.py - Figuras e modelos dos algoritmos aproximativos do
TP2 (Twice-around-the-tree e Christofides), lendo direto a pasta resultados_aprox/.

Arquivos esperados (formato atual do grupo: uma linha por instância, mediana
e IQR das repetições já calculados):
    resultados_twice.csv              tempos SEM tracemalloc
    resultados_twice_memoria.csv      mesma execução COM tracemalloc (memória)
    resultados_christofides.csv
    resultados_christofides_memoria.csv

Por padrão os tempos vêm dos arquivos SEM tracemalloc (o rastreamento de
alocações acrescenta custo em cada malloc e distorce o tempo) e a memória vem
dos arquivos *_memoria.csv. Use --tempo-de memoria para inverter.

Figuras geradas (PDF vetorial + PNG 300 dpi):
    A01  razão custo/ótimo por instância, com as garantias 2 e 1,5
    A02  gap (%) vs N
    A03  decomposição do custo: MST/ótimo (limite inferior), passeio
         euleriano/ótimo e custo final/ótimo (ganho dos atalhos)
    A04  tempo por subrotina vs N (log-log) com o melhor modelo (AICc)
    A05  composição percentual do tempo por subrotina
    A06  ajuste do tempo total: modelos candidatos + resíduos
    A07  tempo do matching vs nº de vértices ímpares k (variável certa
         para o blossom, O(k^3)), comparado com o ajuste em função de N
    A08  memória de pico por subrotina vs N, com ajuste
    A09  gap vs características disponíveis nos CSVs (Spearman)
    A10  tempo com x sem tracemalloc (diagnóstico da medição)
Tabelas (CSV + LaTeX): qualidade, ajustes de tempo e de memória, Spearman.

Uso:
    python graficos_aproximativos.py --dados resultados_aprox --saida analise_aprox
Dependências: numpy, scipy, pandas, matplotlib.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator
from scipy import optimize, stats

logging.getLogger("fontTools").setLevel(logging.ERROR)

# ----------------------------------------------------------------------------
# Estilo (mesma identidade visual do gerar_graficos.py)
# ----------------------------------------------------------------------------
LARGURA = 6.0  # polegadas ~ largura do texto no modelo SBC
COR = {"twice": "#2a78d6", "christofides": "#eb6834"}
# cor fixa por subrotina (a MST é azul nos dois algoritmos)
COR_SUB = {"mst": "#2a78d6", "impares": "#eb6834", "matching": "#1baf7a",
           "euler": "#eda100", "atalho": "#e87ba4", "preordem": "#6250d6"}
TINTA, TINTA_2, GRADE, NEUTRO, FUNDO_NM = "#0b0b0b", "#52514e", "#dcdbd6", "#8a8984", "#f0efec"
ROTULO = {"twice": "Twice-around-the-tree", "christofides": "Christofides"}
SUBROTINAS = {  # ordem de execução do algoritmo
    "twice": ["mst", "preordem"],
    "christofides": ["mst", "impares", "matching", "euler", "atalho"],
}
ROTULO_SUB = {"mst": "Árvore geradora mínima", "preordem": "Pré-ordem (DFS)",
              "impares": "Vértices ímpares", "matching": "Matching perfeito mínimo",
              "euler": "Circuito euleriano", "atalho": "Atalhos", "total": "Total"}
MODELOS = {
    r"constante": None,           # sem crescimento mensurável na faixa medida
    r"$N$": lambda n: n,
    r"$N\log N$": lambda n: n * np.log2(n),
    r"$N^2$": lambda n: n ** 2,
    r"$N^2\log N$": lambda n: n ** 2 * np.log2(n),
    r"$N^3$": lambda n: n ** 3,
}
FORMATOS = ["pdf", "png"]


def estilo():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 9.5,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRADE, "grid.linewidth": 0.5,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "lines.linewidth": 1.6, "hatch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def salvar(fig, pasta: Path, nome: str):
    pasta.mkdir(parents=True, exist_ok=True)
    for ext in FORMATOS:
        fig.savefig(pasta / f"{nome}.{ext}")
    plt.close(fig)
    print(f"  figura: {nome}")


def xticks_instancias(ax, x, df):
    ax.set_xticks(x, [f"{i} ({int(n)})" for i, n in zip(df["instancia"], df["n"])],
                  rotation=35, ha="right", rotation_mode="anchor")


def ticks_log_x(ax, valores):
    v = sorted(set(int(a) for a in valores))
    marcas = [m for m in [5, 6, 8, 10, 15, 20, 30, 40, 50, 70, 100] if min(v) * 0.9 <= m <= max(v) * 1.1]
    ax.set_xticks(marcas, [str(m) for m in marcas])
    ax.minorticks_off()


def ticks_log_y(ax, escala=1.0, unidade=""):
    """Rótulos legíveis no eixo log (1, 2, 5 x 10^k) com unidade opcional."""
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * escala:g}{unidade}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))


# ----------------------------------------------------------------------------
# Leitura
# ----------------------------------------------------------------------------
def ler(pasta: Path, alg: str, tempo_de: str) -> pd.DataFrame:
    base = pd.read_csv(pasta / f"resultados_{alg}.csv")
    mem = pd.read_csv(pasta / f"resultados_{alg}_memoria.csv")
    for d in (base, mem):
        d["instancia"] = d["instancia"].astype(str).str.strip()
    t_cols = [c for c in base.columns if c.startswith("tempo_")]
    if tempo_de == "memoria":
        base = base.drop(columns=t_cols).merge(mem[["instancia"] + t_cols], on="instancia")
    # tempos da outra execução, para o diagnóstico (A10)
    outro = mem if tempo_de != "memoria" else pd.read_csv(pasta / f"resultados_{alg}.csv")
    base["tempo_total_outra_s"] = base["instancia"].map(outro.set_index("instancia")["tempo_total_s"])
    m_cols = [c for c in mem.columns if c.startswith("mem_pico_")]
    df = base.merge(mem[["instancia"] + m_cols], on="instancia", how="left")
    if not (df["custo"].values == mem.set_index("instancia").loc[df["instancia"], "custo"].values).all():
        print(f"  [aviso] custos de {alg} diferem entre as duas execuções")
    df["algoritmo"] = alg
    df["nao_metrica"] = df["classe_metrica"].astype(str).str.lower().eq("nao_metrica")
    df["pares_total"] = df["n"] * (df["n"] - 1) / 2
    df["frac_pares_violam"] = df["pares_violam_triangular"] / df["pares_total"]
    return df.sort_values(["n", "instancia"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Ajuste de modelos
# ----------------------------------------------------------------------------
def ajustar(x, y, f):
    """y = a·f(x) + c, a,c >= 0, mínimos quadrados em escala log.

    A constante c absorve o custo fixo (chamada de função, alocação inicial),
    que domina quando x é pequeno. Métricas: R² e RMSE na escala original,
    R² e AICc na escala log (onde o ajuste foi feito).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ly = np.log(y)
    if f is None:  # modelo constante: y = c (1 parâmetro); média geométrica
        c = float(np.exp(ly.mean()))
        yh = np.full_like(y, c)
        rl = ly - np.log(yh)
        m, k = len(y), 1
        ss_res, ss_tot = np.sum((y - yh) ** 2), np.sum((y - y.mean()) ** 2)
        aic = m * np.log(max(np.sum(rl ** 2), 1e-300) / m) + 2 * k
        return dict(prev=lambda xx: np.full(np.shape(xx), c, dtype=float), a=0.0, c=c,
                    r2=1 - ss_res / ss_tot, rmse=math.sqrt(ss_res / m), r2_log=0.0,
                    aicc=aic + 2 * k * (k + 1) / (m - k - 1), frac_constante=1.0)

    def prev(p, xx):
        return np.exp(p[0]) * f(xx) + np.exp(p[1])

    p0 = [np.log(np.median(y / f(x)) * 0.5), np.log(y.min() * 0.5)]
    r = optimize.least_squares(lambda p: np.log(prev(p, x)) - ly, p0, max_nfev=20000)
    yh = prev(r.x, x)
    ss_res, ss_tot = np.sum((y - yh) ** 2), np.sum((y - y.mean()) ** 2)
    rl = ly - np.log(yh)
    m, k = len(y), 2
    aic = m * np.log(max(np.sum(rl ** 2), 1e-300) / m) + 2 * k
    return dict(
        prev=lambda xx: prev(r.x, xx),
        a=float(np.exp(r.x[0])), c=float(np.exp(r.x[1])),
        r2=1 - ss_res / ss_tot, rmse=math.sqrt(ss_res / m),
        r2_log=1 - np.sum(rl ** 2) / np.sum((ly - ly.mean()) ** 2),
        aicc=aic + 2 * k * (k + 1) / (m - k - 1),
        frac_constante=float(np.median(np.exp(r.x[1]) / yh)),
    )


def expoente_livre(x, y):
    r = stats.linregress(np.log(x), np.log(y))
    tc = stats.t.ppf(0.975, len(x) - 2)
    return r.slope, r.slope - tc * r.stderr, r.slope + tc * r.stderr


def ajustar_todos(x, y, modelos=MODELOS):
    """Seleciona pelo AICc, com uma regra de parcimônia: se o IC 95% do
    expoente livre b contém 0, não há crescimento detectável na faixa medida
    e o modelo reportado é o constante (o custo fixo domina). Sem essa regra,
    o AICc premia um a·N^3 + c com 'a' minúsculo que só acompanha o ruído."""
    res = {nome: ajustar(x, y, f) for nome, f in modelos.items()}
    melhor = min(res, key=lambda k: res[k]["aicc"])
    ex = expoente_livre(x, y)
    if "constante" in res and ex[1] <= 0 <= ex[2]:
        melhor = "constante"
    return res, melhor, ex


def linhas_tabela(alg, sub, var, xnome, res, melhor, exp_):
    return [dict(algoritmo=alg, subrotina=sub, variavel=var, x=xnome, modelo=nome,
                 r2=r["r2"], r2_log=r["r2_log"], rmse=r["rmse"], aicc=r["aicc"],
                 delta_aicc=r["aicc"] - res[melhor]["aicc"], a=r["a"], c=r["c"],
                 frac_tempo_constante=r["frac_constante"], melhor=nome == melhor,
                 crescimento_detectado=not (exp_[1] <= 0 <= exp_[2]),
                 expoente_livre=exp_[0], exp_ic_inf=exp_[1], exp_ic_sup=exp_[2])
            for nome, r in res.items()]


def tabela_latex(tab: pd.DataFrame, caminho: Path, legenda: str, rotulo: str, unidade: str):
    t = tab[tab["melhor"]]
    L = [r"\begin{table}[ht]", r"\centering", r"\small", f"\\caption{{{legenda}}}",
         f"\\label{{{rotulo}}}", r"\begin{tabular}{llcrrrc}", r"\hline",
         r"Algoritmo & Subrotina & Modelo & $R^2$ & $R^2_{\log}$ & RMSE " + unidade
         + r" & $b$ [IC 95\%] \\", r"\hline"]
    for _, r in t.iterrows():
        sub = ROTULO_SUB.get(r.subrotina, r.subrotina) + (" (vs.\\ $k$)" if r.x == "k" else "")
        L.append(f"{ROTULO[r.algoritmo]} & {sub} & {r.modelo.replace('N', 'k') if r.x == 'k' else r.modelo} & "
                 f"{r.r2:.3f} & {r.r2_log:.3f} & {r.rmse:.2e} & "
                 f"{r.expoente_livre:.2f} [{r.exp_ic_inf:.2f}, {r.exp_ic_sup:.2f}] \\\\")
    L += [r"\hline", r"\end{tabular}", r"\end{table}"]
    caminho.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------
# Figuras de qualidade
# ----------------------------------------------------------------------------
def fig_qualidade(dados, pasta, tab):
    ordem = dados["christofides"][["instancia", "n", "nao_metrica"]]
    x = np.arange(len(ordem))
    w = 0.4
    fig, ax = plt.subplots(figsize=(LARGURA, 3.0))
    for i, nm in enumerate(ordem["nao_metrica"]):
        if nm:
            ax.axvspan(i - 0.5, i + 0.5, color=FUNDO_NM, lw=0, zorder=0)
    for k, alg in enumerate(["twice", "christofides"]):
        d = dados[alg].set_index("instancia").loc[ordem["instancia"]]
        ax.bar(x + (k - 0.5) * w, d["razao"] - 1, bottom=1, width=w * 0.92,
               color=COR[alg], edgecolor="white", lw=0.6, zorder=3, label=ROTULO[alg])
    ax.axhline(1, color=TINTA, lw=0.8, zorder=4)
    ax.axhline(2.0, color=COR["twice"], lw=1.0, ls=(0, (4, 2)))
    ax.axhline(1.5, color=COR["christofides"], lw=1.0, ls=(0, (1.5, 1.5)))
    ax.set_ylim(0.95, 2.08)
    ax.set_xlim(-0.6, len(ordem) - 0.4)
    ax.set_ylabel("custo / ótimo")
    xticks_instancias(ax, x, ordem)
    ax.grid(axis="x", visible=False)
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color=COR["twice"], ls=(0, (4, 2)), lw=1),
          Line2D([], [], color=COR["christofides"], ls=(0, (1.5, 1.5)), lw=1),
          Patch(facecolor=FUNDO_NM)]
    l += ["Garantia Twice ($\\leq 2$)", "Garantia Christofides ($\\leq 1.5$)",
          "Não métrica (garantia não vale)"]
    ax.legend(h, l, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7.5)
    salvar(fig, pasta, "A01_razao_aproximacao")

    # A02: gap vs N
    fig, ax = plt.subplots(figsize=(LARGURA * 0.62, 2.7))
    for alg in ["twice", "christofides"]:
        d = dados[alg]
        nm = d["nao_metrica"].values
        ax.scatter(d.loc[~nm, "n"], d.loc[~nm, "gap_pct"], s=28, color=COR[alg],
                   edgecolor="white", lw=0.8, zorder=3, label=ROTULO[alg])
        ax.scatter(d.loc[nm, "n"], d.loc[nm, "gap_pct"], s=28, facecolor="white",
                   edgecolor=COR[alg], lw=1.2, zorder=3)
        med = d["gap_pct"].median()
        ax.axhline(med, color=COR[alg], lw=0.8, ls=":", zorder=2)
        ax.text(d["n"].max() + 1.5, med, f"mediana {med:.1f}%", color=COR[alg],
                fontsize=7, va="center")
    ax.scatter([], [], s=28, facecolor="white", edgecolor=NEUTRO, label="não métrica")
    ax.set_xlabel("número de vértices $N$")
    ax.set_ylabel("gap em relação ao ótimo (%)")
    ax.set_ylim(0, None)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7)
    salvar(fig, pasta, "A02_gap_vs_n")

    # A03: decomposição do custo em relação ao ótimo
    fig, axs = plt.subplots(1, 2, figsize=(LARGURA, 3.0), sharey=True)
    c = dados["christofides"]
    t = dados["twice"]
    xi = np.arange(len(c))
    for ax, d, alg in [(axs[0], t, "twice"), (axs[1], c, "christofides")]:
        cor = COR[alg]
        for i, nm in enumerate(d["nao_metrica"]):
            if nm:
                ax.axvspan(i - 0.5, i + 0.5, color=FUNDO_NM, lw=0, zorder=0)
        ax.bar(xi, d["custo_mst"] / d["otimo"], width=0.7, color=cor + "55",
               edgecolor="white", lw=0.6, zorder=2, label="MST (limite inferior)")
        if alg == "christofides":
            ax.bar(xi, d["custo_matching"] / d["otimo"], bottom=d["custo_mst"] / d["otimo"],
                   width=0.7, color=cor + "aa", edgecolor="white", lw=0.6, zorder=2,
                   label="+ matching = passeio euleriano")
            topo = d["custo_euleriano"] / d["otimo"]
        else:
            ax.bar(xi, d["custo_mst"] / d["otimo"], bottom=d["custo_mst"] / d["otimo"],
                   width=0.7, color=cor + "aa", edgecolor="white", lw=0.6, zorder=2,
                   label="+ MST = caminhada (2·MST)")
            topo = d["custo_caminhada"] / d["otimo"]
        ax.scatter(xi, d["razao"], marker="_", s=120, color=TINTA, lw=2, zorder=4,
                   label="custo final (após atalhos)")
        ax.vlines(xi, d["razao"], topo, color=TINTA, lw=0.7, zorder=3)
        ax.axhline(1, color=TINTA, lw=0.8, zorder=3)
        ax.set_title(ROTULO[alg], loc="left")
        xticks_instancias(ax, xi, d)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", fontsize=6.5)
    axs[0].set_ylabel("custo / ótimo")
    axs[0].set_ylim(0, 2.25)
    salvar(fig, pasta, "A03_decomposicao_custo")

    q = pd.concat([d[["algoritmo", "instancia", "n", "classe_metrica", "otimo", "custo",
                      "razao", "gap_pct", "custo_mst"]] for d in dados.values()])
    q["mst_sobre_otimo"] = q["custo_mst"] / q["otimo"]
    q.to_csv(tab / "qualidade_aproximativos.csv", index=False)
    resumo = q.groupby("algoritmo")["gap_pct"].describe()[["min", "50%", "mean", "max"]]
    resumo.to_csv(tab / "qualidade_resumo.csv")
    print(resumo.round(2).to_string())


# ----------------------------------------------------------------------------
# Profiling de tempo
# ----------------------------------------------------------------------------
def barra_iqr(ax, x, med, iqr, cor):
    """Barra vertical de comprimento IQR centrada na mediana (os CSVs trazem
    só a largura do IQR, não Q1 e Q3). Em escala log, corta em 1% da mediana."""
    lo = np.maximum(med - iqr / 2, med * 0.01)
    ax.vlines(x, lo, med + iqr / 2, color=cor, lw=0.8, alpha=0.6, zorder=2)


def fig_tempo(dados, pasta, tab):
    linhas = []
    for alg, d in dados.items():
        subs = SUBROTINAS[alg]
        n = d["n"].values.astype(float)

        # A04 - tempo por subrotina vs N
        fig, ax = plt.subplots(figsize=(LARGURA * 0.66, 3.0))
        for k, sub in enumerate(subs + ["total"]):
            y = d[f"tempo_{sub}_s"].values
            iq = d[f"tempo_{sub}_iqr_s"].values
            cor = TINTA if sub == "total" else COR_SUB[sub]
            barra_iqr(ax, n, y, iq, cor)
            ax.scatter(n, y, s=18, color=cor, edgecolor="white", lw=0.5, zorder=3,
                       label=ROTULO_SUB[sub])
            res, melhor, ex = ajustar_todos(n, y)
            linhas += linhas_tabela(alg, sub, "tempo_s", "N", res, melhor, ex)
            nn = np.linspace(n.min(), n.max(), 200)
            ax.plot(nn, res[melhor]["prev"](nn), color=cor, lw=1.0, alpha=0.75, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks_log_x(ax, n)
        ticks_log_y(ax, 1e3, " ms")
        ax.set_xlabel("número de vértices $N$ (escala log)")
        ax.set_ylabel("tempo (mediana, escala log)")
        ax.set_title(f"{ROTULO[alg]}: tempo por subrotina", loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.text(1.03, 0.0, "linhas: modelo selecionado\n$a\\cdot f(N)+c$ (AICc; constante\nse o IC de $b$ contém 0)\nbarras: IQR",
                transform=ax.transAxes, fontsize=6.5, color=TINTA_2, va="bottom")
        salvar(fig, pasta, f"A04_tempo_subrotinas_{alg}")

        # A05 - composição percentual
        fig, ax = plt.subplots(figsize=(LARGURA, 2.6))
        soma = sum(d[f"tempo_{s}_s"] for s in subs)
        base = np.zeros(len(d))
        x = np.arange(len(d))
        for k, sub in enumerate(subs):
            v = (d[f"tempo_{sub}_s"] / soma * 100).values
            ax.bar(x, v, bottom=base, width=0.75, color=COR_SUB[sub], edgecolor="white",
                   lw=0.8, label=ROTULO_SUB[sub])
            base += v
        ax.set_ylim(0, 100)
        ax.set_ylabel("fração do tempo (%)")
        xticks_instancias(ax, x, d)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{ROTULO[alg]}: composição do tempo (soma das subrotinas)",
                     loc="left", pad=20)
        ax.legend(ncol=len(subs), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7,
                  handlelength=1.2, columnspacing=1.0)
        salvar(fig, pasta, f"A05_composicao_tempo_{alg}")

        # A06 - ajuste do tempo total + resíduos
        y = d["tempo_total_s"].values
        res, melhor, ex = ajustar_todos(n, y)
        fig_ajuste(n, y, d["tempo_total_iqr_s"].values, res, melhor, ex,
                   f"{ROTULO[alg]} - tempo total", "número de vértices $N$",
                   pasta, f"A06_ajuste_tempo_total_{alg}", COR[alg])

    # A07 - matching do Christofides em função de k = nº de vértices ímpares
    c = dados["christofides"]
    k_ = c["n_impares"].values.astype(float)
    y = c["tempo_matching_s"].values
    modelos_k = dict(MODELOS)
    res_k, mk, exk = ajustar_todos(k_, y, modelos_k)
    linhas += linhas_tabela("christofides", "matching", "tempo_s", "k", res_k, mk, exk)
    res_n, mn, exn = ajustar_todos(c["n"].values.astype(float), y)
    fig, axs = plt.subplots(1, 2, figsize=(LARGURA, 2.9), sharey=True)
    for ax, xv, res, mel, ex, rot in [
            (axs[0], c["n"].values.astype(float), res_n, mn, exn, "$N$ (vértices)"),
            (axs[1], k_, res_k, mk, exk, "$k$ (vértices de grau ímpar na MST)")]:
        barra_iqr(ax, xv, y, c["tempo_matching_iqr_s"].values, COR["christofides"])
        ax.scatter(xv, y, s=22, color=COR["christofides"], edgecolor="white", lw=0.6, zorder=3)
        xx = np.linspace(xv.min(), xv.max(), 200)
        nome_mel = mel.replace("N", "k") if ax is axs[1] else mel
        ax.plot(xx, res[mel]["prev"](xx), color=TINTA, lw=1.1,
                label=f"melhor: {nome_mel}, $R^2$={res[mel]['r2']:.3f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks_log_x(ax, xv)
        ticks_log_y(ax, 1e3, " ms")
        ax.set_xlabel(rot + " (log)")
        ax.set_title(f"expoente livre $b$ = {ex[0]:.2f} [{ex[1]:.2f}, {ex[2]:.2f}]",
                     loc="left", fontsize=8)
        ax.legend(loc="upper left", fontsize=7)
    axs[0].set_ylabel("tempo do matching (log)")
    fig.suptitle("Christofides: matching perfeito mínimo em função de $N$ e de $k$",
                 x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    salvar(fig, pasta, "A07_matching_vs_k")

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(tab / "ajustes_tempo.csv", index=False)
    tabela_latex(tabela, tab / "ajustes_tempo.tex",
                 "Modelo de complexidade selecionado para o tempo de cada subrotina, "
                 "ajustado como $t = a\\,f(x) + c$ em escala log e escolhido pelo AICc; "
                 "quando o IC 95\\% do expoente livre $b$ (de $t = a x^b$) contém 0, "
                 "reporta-se o modelo constante.", "tab:ajustes-tempo", "(s)")


def fig_ajuste(x, y, iqr, res, melhor, ex, titulo, xlabel, pasta, nome, cor):
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(LARGURA * 0.66, 3.7), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1.2], hspace=0.08))
    barra_iqr(ax, x, y, iqr, TINTA)
    ax.scatter(x, y, s=22, color=TINTA, edgecolor="white", lw=0.6, zorder=4, label="mediana (IQR)")
    xx = np.linspace(x.min() * 0.97, x.max() * 1.03, 300)
    estilos = [(0, (1, 1)), (0, (3, 1.5)), (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (6, 2, 1, 2))]
    for k, (nm, r) in enumerate(res.items()):
        e = nm == melhor
        ax.plot(xx, r["prev"](xx), color=cor if e else NEUTRO, lw=1.7 if e else 0.9,
                ls="-" if e else estilos[k % 5], zorder=3 if e else 1,
                label=f"{nm}  $R^2$={r['r2']:.3f}" + ("  (selecionado)" if e else ""))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks_log_y(ax, 1e3, " ms")
    ax.set_ylabel("tempo (log)")
    ax.set_title(f"{titulo}\nexpoente livre $b$ = {ex[0]:.2f} [{ex[1]:.2f}, {ex[2]:.2f}] (IC 95%)",
                 loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    resid = (y - res[melhor]["prev"](x)) / y * 100
    axr.axhline(0, color=TINTA_2, lw=0.7)
    axr.scatter(x, resid, s=16, color=cor, zorder=3)
    axr.set_ylabel("resíduo (%)")
    axr.set_xlabel(xlabel + " (escala log)")
    ticks_log_x(axr, x)
    salvar(fig, pasta, nome)


# ----------------------------------------------------------------------------
# Memória
# ----------------------------------------------------------------------------
def fig_memoria(dados, pasta, tab):
    linhas = []
    fig, axs = plt.subplots(1, 2, figsize=(LARGURA, 3.0), sharey=True)
    for ax, (alg, d) in zip(axs, dados.items()):
        n = d["n"].values.astype(float)
        for k, sub in enumerate(SUBROTINAS[alg]):
            col = f"mem_pico_{sub}_bytes"
            if col not in d or d[col].isna().all():
                continue
            y = d[col].values.astype(float) / 1024
            ax.scatter(n, y, s=18, color=COR_SUB[sub], edgecolor="white", lw=0.5, zorder=3,
                       label=ROTULO_SUB[sub])
            res, mel, ex = ajustar_todos(n, y)
            linhas += linhas_tabela(alg, sub, "memoria_kib", "N", res, mel, ex)
            nn = np.linspace(n.min(), n.max(), 200)
            ax.plot(nn, res[mel]["prev"](nn), color=COR_SUB[sub], lw=1.0, alpha=0.75)
            ax.annotate(f"{mel}  $b$={ex[0]:.2f}", (nn[-1], res[mel]["prev"](nn[-1])),
                        xytext=(3, 0), textcoords="offset points", fontsize=6.5,
                        color=COR_SUB[sub], va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks_log_x(ax, n)
        ticks_log_y(ax, 1, " KiB")
        ax.set_title(ROTULO[alg], loc="left")
        ax.set_xlabel("número de vértices $N$ (log)")
    axs[0].set_ylabel("pico de memória alocada\n(tracemalloc, log)")
    h = [Line2D([], [], marker="o", ls="", color=COR_SUB[s_], mec="white", ms=5,
                label=ROTULO_SUB[s_]) for s_ in ["mst", "preordem", "impares", "matching",
                                                 "euler", "atalho"]]
    fig.legend(handles=h, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=7)
    fig.tight_layout(w_pad=4)
    salvar(fig, pasta, "A08_memoria_subrotinas")
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(tab / "ajustes_memoria.csv", index=False)
    tabela_latex(tabela, tab / "ajustes_memoria.tex",
                 "Melhor modelo (AICc) para o pico de memória alocada por subrotina "
                 "(tracemalloc), $m = a\\,f(N) + c$.", "tab:ajustes-memoria", "(KiB)")


# ----------------------------------------------------------------------------
# Sensibilidade com as características já presentes nos CSVs
# ----------------------------------------------------------------------------
FEATURES = {
    "n": "número de vértices $N$",
    "frac_pares_violam": "fração de pares que violam\na desigualdade triangular",
    "grau_max_mst": "grau máximo na MST",
    "frac_impares": "fração de vértices ímpares\nna MST ($k/N$)",
    "frac_folhas": "fração de folhas na MST",
}


def fig_sensibilidade(dados, pasta, tab):
    c, t = dados["christofides"].copy(), dados["twice"].copy()
    imp = c.set_index("instancia")["n_impares"]
    fol = t.set_index("instancia")["n_folhas_mst"]
    linhas = []
    feats = list(FEATURES)
    fig, axs = plt.subplots(1, len(feats), figsize=(LARGURA * 1.25, 2.4), sharey=True)
    for ax, f in zip(axs, feats):
        txt = []
        for alg, d in [("twice", t), ("christofides", c)]:
            d = d.copy()
            d["frac_impares"] = d["instancia"].map(imp) / d["n"]
            d["frac_folhas"] = d["instancia"].map(fol) / d["n"]
            nm = d["nao_metrica"].values
            xj = d[f].values.astype(float)
            if f == "grau_max_mst":  # valores inteiros repetidos: pequeno deslocamento
                xj = xj + (-0.08 if alg == "twice" else 0.08)
            ax.scatter(xj[~nm], d["gap_pct"].values[~nm], s=16, color=COR[alg],
                       edgecolor="white", lw=0.5, zorder=3)
            ax.scatter(xj[nm], d["gap_pct"].values[nm], s=16, facecolor="white",
                       edgecolor=COR[alg], lw=1.0, zorder=3)
            rho, p = stats.spearmanr(d[f], d["gap_pct"])
            linhas.append(dict(algoritmo=alg, feature=f, spearman_rho=rho, p_valor=p, n=len(d)))
            txt.append(f"{'T' if alg == 'twice' else 'C'}: {rho:+.2f}{'*' if p < 0.05 else ''}")
        ax.set_xlabel(FEATURES[f] + "\n$\\rho$ " + "   ".join(txt), fontsize=7)
        ax.tick_params(labelsize=7)
    axs[0].set_ylabel("gap (%)")
    h = [Line2D([], [], marker="o", ls="", color=COR[a], mec="white", ms=5, label=ROTULO[a])
         for a in COR]
    h.append(Line2D([], [], marker="o", ls="", mfc="white", mec=NEUTRO, ms=5, label="não métrica"))
    fig.legend(handles=h, ncol=3, loc="lower left", bbox_to_anchor=(0.05, 0.98))
    fig.text(0.99, -0.16, "$\\rho$ de Spearman (T = Twice, C = Christofides); * p < 0,05; "
             "13 instâncias, baixo poder estatístico", ha="right", fontsize=6.5, color=TINTA_2)
    fig.tight_layout(w_pad=0.6)
    salvar(fig, pasta, "A09_sensibilidade_preliminar")
    pd.DataFrame(linhas).to_csv(tab / "sensibilidade_spearman_preliminar.csv", index=False)


# ----------------------------------------------------------------------------
# Diagnóstico: as duas execuções deveriam concordar (ou a com tracemalloc ser mais lenta)
# ----------------------------------------------------------------------------
def fig_diagnostico(dados, pasta, tab, tempo_de):
    fig, ax = plt.subplots(figsize=(LARGURA * 0.62, 2.9))
    lin = []
    for alg, d in dados.items():
        a, b = d["tempo_total_s"].values, d["tempo_total_outra_s"].values
        sem, com = (a, b) if tempo_de != "memoria" else (b, a)
        ax.scatter(sem, com, s=24, color=COR[alg], edgecolor="white", lw=0.6, zorder=3,
                   label=ROTULO[alg])
        for i, s, cc in zip(d["instancia"], sem, com):
            lin.append(dict(algoritmo=alg, instancia=i, tempo_sem_tracemalloc_s=s,
                            tempo_com_tracemalloc_s=cc, razao_com_sem=cc / s))
    v = np.array([[r["tempo_sem_tracemalloc_s"], r["tempo_com_tracemalloc_s"]] for r in lin])
    lim = [v.min() * 0.7, v.max() * 1.4]
    ax.plot(lim, lim, color=TINTA_2, lw=0.9, ls="--")
    ax.fill_between(lim, lim, [lim[0]] * 2, color="#e66767", alpha=0.08, lw=0)
    ax.text(0.97, 0.04, "abaixo da diagonal:\ncom tracemalloc mais RÁPIDO\n(não deveria ocorrer)",
            transform=ax.transAxes, ha="right", fontsize=6.5, color="#a32020")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    for eixo in (ax.xaxis, ax.yaxis):
        eixo.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        eixo.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 1e3:g}"))
        eixo.set_minor_locator(LogLocator(base=10, subs=()))
    ax.set_xlabel("tempo total sem tracemalloc (ms)")
    ax.set_ylabel("tempo total com tracemalloc (ms)")
    ax.legend(loc="upper left", fontsize=7)
    salvar(fig, pasta, "A10_diagnostico_tracemalloc")
    df = pd.DataFrame(lin)
    df.to_csv(tab / "diagnostico_tracemalloc.csv", index=False)
    frac = (df["razao_com_sem"] < 1).mean()
    print(f"  [diagnóstico] com tracemalloc foi mais rápido em {frac:.0%} das execuções; "
          f"razão mediana com/sem = {df['razao_com_sem'].median():.2f}")


def main():
    global FORMATOS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", type=Path, default=Path("resultados_aprox"))
    ap.add_argument("--saida", type=Path, default=Path("analise_aprox"))
    ap.add_argument("--tempo-de", choices=["sem", "memoria"], default="sem",
                    help="arquivo de onde vêm os tempos (padrão: execução sem tracemalloc)")
    ap.add_argument("--formato", default="pdf,png")
    a = ap.parse_args()
    FORMATOS = [f.strip() for f in a.formato.split(",") if f.strip()]
    estilo()
    figs, tabs = a.saida / "figuras", a.saida / "tabelas"
    tabs.mkdir(parents=True, exist_ok=True)
    dados = {alg: ler(a.dados, alg, a.tempo_de) for alg in ["twice", "christofides"]}
    print("Qualidade")
    fig_qualidade(dados, figs, tabs)
    print("Tempo")
    fig_tempo(dados, figs, tabs)
    print("Memória")
    fig_memoria(dados, figs, tabs)
    print("Sensibilidade (preliminar)")
    fig_sensibilidade(dados, figs, tabs)
    print("Diagnóstico da medição")
    fig_diagnostico(dados, figs, tabs, a.tempo_de)
    print(f"Pronto: {figs} e {tabs}")


if __name__ == "__main__":
    main()