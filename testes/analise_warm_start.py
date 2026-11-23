import glob
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

# Configuração visual global
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10

ORDEM_INICIAL = [
    "trivial",
    "vizinho",
    "fecho",
    "fecho_graham",
    "fecho_gift",
    "twice",
    "christofides",
]

LABELS_INICIAL = {
    "trivial": "Trivial (Cold)",
    "vizinho": "Vizinho Mais Próximo",
    "fecho": "Fecho (Graham)",
    "fecho_graham": "Fecho (Graham)",
    "fecho_gift": "Fecho (Gift)",
    "twice": "Twice-Around",
    "christofides": "Christofides",
}

CORES = {
    "trivial": "#7f8c8d",
    "vizinho": "#e67e22",
    "fecho": "#000000",         # Preto (quando a coluna 'inicial' vem como 'fecho')
    "fecho_graham": "#000000",  # Preto (quando normalizado com algoritmo_fecho)
    "fecho_gift": "#8e44ad",
    "twice": "#2980b9",
    "christofides": "#27ae60",
}


def carregar_dados(padrao="./resultados/resultados_bnb_*_melhor.csv"):
    """Carrega e concatena os CSVs de resultados, ordenando os warm-starts."""
    arquivos = sorted(glob.glob(padrao))
    if not arquivos:
        arquivos = sorted(glob.glob("resultados_bnb_*_melhor.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV encontrado com o padrão: {padrao}")

    # Ignora eventuais CSVs de profiling de memória (_memoria.csv) que distorcem tempo/nós
    arquivos = [a for a in arquivos if not a.endswith("_memoria.csv")]
    if not arquivos:
        raise FileNotFoundError(
            "Apenas arquivos _memoria.csv encontrados; execute sem --memoria para comparar tempo/nós."
        )

    dfs = []
    for arq in arquivos:
        df_tmp = pd.read_csv(arq, na_values=["NA"])
        # Diferencia graham vs gift caso ambos estejam presentes como inicial == 'fecho'
        if "algoritmo_fecho" in df_tmp.columns:
            mask_fecho = df_tmp["inicial"] == "fecho"
            alg = df_tmp.loc[mask_fecho, "algoritmo_fecho"].fillna("graham").astype(str).str.strip()
            df_tmp.loc[mask_fecho, "inicial"] = "fecho_" + alg
        dfs.append(df_tmp)

    dados = pd.concat(dfs, ignore_index=True)

    presentes = [x for x in ORDEM_INICIAL if x in dados["inicial"].unique()]
    extras = [x for x in dados["inicial"].unique() if x not in presentes]
    ordem_final = presentes + sorted(extras)

    dados["inicial"] = pd.Categorical(
        dados["inicial"], categories=ordem_final, ordered=True
    )
    return dados, ordem_final


def imprimir_analise(df):
    """Imprime no terminal as tabelas resumo para os 4 tópicos de análise."""
    piv_status = df.pivot(index=["instancia", "n"], columns="inicial", values="status")

    print("=" * 85)
    print("1. ALCANCE DO ÓTIMO POR WARM-START")
    print("=" * 85)
    print(piv_status.to_string())

    divergentes = piv_status[piv_status.nunique(axis=1) > 1]
    print("\nInstâncias onde houve divergência de status:")
    print(divergentes.to_string())

    print("\n" + "=" * 85)
    print("2. NÓS EXPANDIDOS E PODADOS (INSTÂNCIAS RESOLVIDAS POR TODOS)")
    print("=" * 85)
    resolvidas_todos = piv_status[(piv_status == "otimo").all(axis=1)].index.get_level_values("instancia")
    df_comum = df[df["instancia"].isin(resolvidas_todos)]

    print("\nNós Expandidos (comum):")
    print(df_comum.pivot(index=["instancia", "n"], columns="inicial", values="nos_expandidos").to_string())

    print("\nNós Podados por Limite (comum):")
    print(df_comum.pivot(index=["instancia", "n"], columns="inicial", values="nos_podados_limite").to_string())

    print("\n" + "=" * 85)
    print("3. TEMPO DECORRIDO (s)")
    print("=" * 85)
    print(df.pivot(index=["instancia", "n"], columns="inicial", values="tempo_decorrido_s").round(3).to_string())

    print("\n" + "=" * 85)
    print("4. INSTÂNCIAS NÃO RESOLVIDAS POR TODOS (CUSTO FINAL E GAP %)")
    print("=" * 85)
    nao_comum = df[~df["instancia"].isin(resolvidas_todos)]
    print("\nMelhor Custo Final:")
    print(nao_comum.pivot(index=["instancia", "n", "otimo_publicado"], columns="inicial", values="melhor_custo").to_string())

    print("\nGap Final em relação ao Ótimo Publicado (%):")
    print(nao_comum.pivot(index=["instancia", "n"], columns="inicial", values="gap_melhor_pct").round(2).to_string())


def plotar_alcance_otimo(df, ordem, pasta_saida="."):
    """Gráfico 1: Heatmap categórico mostrando quais instâncias atingiram o ótimo."""
    piv_status = df.pivot(index=["instancia", "n"], columns="inicial", values="status")
    instancias_ord = df[["instancia", "n"]].drop_duplicates().sort_values("n")

    mapa_status = {"otimo": 1, "tempo_esgotado": 0, "memoria_esgotada": -1}
    pares_idx = list(zip(instancias_ord["instancia"], instancias_ord["n"]))
    mat_status = piv_status.loc[pares_idx].replace(mapa_status).astype(float)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = ListedColormap(["#e74c3c", "#2ecc71"])
    ax.imshow(mat_status.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(ordem)))
    ax.set_xticklabels([LABELS_INICIAL.get(c, c) for c in ordem], rotation=15)
    ax.set_yticks(range(len(instancias_ord)))
    ax.set_yticklabels([f"{r.instancia} (n={r.n})" for r in instancias_ord.itertuples()])

    for i in range(mat_status.shape[0]):
        for j in range(mat_status.shape[1]):
            txt = "Ótimo" if mat_status.values[i, j] == 1 else "Timeout"
            ax.text(j, i, txt, ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    ax.set_title("Alcance do Ótimo dentro do Limite de Tempo (Best-First)", fontweight="bold")
    fig.tight_layout()

    caminho = os.path.join(pasta_saida, "01_alcance_otimo.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Salvo: {caminho}")


def plotar_nos_expandidos(df, ordem, pasta_saida="."):
    """Gráfico 2: Barras agrupadas de nós expandidos em escala logarítmica."""
    instancias_ord = df[["instancia", "n"]].drop_duplicates().sort_values("n")
    inst_plot = instancias_ord["instancia"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(inst_plot))
    largura = 0.8 / len(ordem)

    for idx, ini in enumerate(ordem):
        sub = df[df["inicial"] == ini].set_index("instancia").reindex(inst_plot)
        vals = sub["nos_expandidos"].values
        deslocamento = (idx - len(ordem) / 2 + 0.5) * largura

        bars = ax.bar(
            x + deslocamento,
            vals,
            width=largura,
            label=LABELS_INICIAL.get(ini, ini),
            color=CORES.get(ini, None),
            edgecolor="black",
            linewidth=0.4,
        )
        for b, st in zip(bars, sub["status"]):
            if st != "otimo":
                b.set_alpha(0.35)
                b.set_hatch("//")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(inst_plot, rotation=45, ha="right")
    ax.set_ylabel("Nós Expandidos (escala log)")
    ax.set_title("Nós Expandidos por Instância (hachurado = tempo esgotado)", fontweight="bold")
    ax.legend()
    fig.tight_layout()

    caminho = os.path.join(pasta_saida, "02_nos_expandidos.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Salvo: {caminho}")


def plotar_nos_podados(df, ordem, pasta_saida="."):
    """Gráfico 3: Barras agrupadas de nós podados por limite em escala logarítmica."""
    instancias_ord = df[["instancia", "n"]].drop_duplicates().sort_values("n")
    inst_plot = instancias_ord["instancia"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(inst_plot))
    largura = 0.8 / len(ordem)

    for idx, ini in enumerate(ordem):
        sub = df[df["inicial"] == ini].set_index("instancia").reindex(inst_plot)
        # Substitui 0 por NaN para não quebrar a escala log e sinaliza no gráfico
        vals = sub["nos_podados_limite"].replace(0, np.nan).values
        deslocamento = (idx - len(ordem) / 2 + 0.5) * largura

        bars = ax.bar(
            x + deslocamento,
            vals,
            width=largura,
            label=LABELS_INICIAL.get(ini, ini),
            color=CORES.get(ini, None),
            edgecolor="black",
            linewidth=0.4,
        )
        for b, st in zip(bars, sub["status"]):
            if st != "otimo":
                b.set_alpha(0.35)
                b.set_hatch("//")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(inst_plot, rotation=45, ha="right")
    ax.set_ylabel("Nós Podados por Limite (escala log)")
    ax.set_title("Nós Podados por Limite (ausente = 0 podas; hachurado = timeout)", fontweight="bold")
    ax.legend()
    fig.tight_layout()

    caminho = os.path.join(pasta_saida, "03_nos_podados.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Salvo: {caminho}")


def plotar_tempo_execucao(df, ordem, pasta_saida="."):
    """Gráfico 4: Evolução do tempo decorrido por instância em escala logarítmica."""
    instancias_ord = df[["instancia", "n"]].drop_duplicates().sort_values("n")
    inst_plot = instancias_ord["instancia"].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for ini in ordem:
        sub = df[df["inicial"] == ini].set_index("instancia").reindex(inst_plot)
        otimos = sub["status"] == "otimo"

        ax.plot(
            inst_plot,
            sub["tempo_decorrido_s"],
            marker="o",
            label=LABELS_INICIAL.get(ini, ini),
            color=CORES.get(ini, None),
            linewidth=1.8,
        )
        ax.scatter(
            np.array(inst_plot)[~otimos],
            sub.loc[~otimos, "tempo_decorrido_s"],
            marker="x",
            s=85,
            color="red",
            zorder=5,
        )

    ax.axhline(1800, color="red", linestyle="--", linewidth=1, alpha=0.6, label="Limite (1800s)")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Tempo Decorrido (s) - escala log")
    ax.set_title("Tempo de Execução por Instância (X vermelho = timeout)", fontweight="bold")
    ax.legend()
    fig.tight_layout()

    caminho = os.path.join(pasta_saida, "04_tempo_execucao.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Salvo: {caminho}")


def plotar_gap_dificeis(df, ordem, pasta_saida="."):
    """Gráfico 5: Gap final (%) nas instâncias onde nem todos atingiram o ótimo."""
    piv_status = df.pivot(index=["instancia", "n"], columns="inicial", values="status")
    instancias_ord = df[["instancia", "n"]].drop_duplicates().sort_values("n")
    inst_plot = instancias_ord["instancia"].tolist()

    divergentes = piv_status[piv_status.nunique(axis=1) > 1].index.get_level_values("instancia").tolist()
    todas_timeout = piv_status[(piv_status != "otimo").all(axis=1)].index.get_level_values("instancia").tolist()
    inst_dificeis = [i for i in inst_plot if i in set(divergentes + todas_timeout)]

    if not inst_dificeis:
        print("Nenhuma instância com timeout ou divergência encontrada para plotar gap.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(inst_dificeis))
    largura = 0.8 / len(ordem)

    for idx, ini in enumerate(ordem):
        sub = df[df["inicial"] == ini].set_index("instancia").reindex(inst_dificeis)
        deslocamento = (idx - len(ordem) / 2 + 0.5) * largura

        ax.bar(
            x + deslocamento,
            sub["gap_melhor_pct"],
            width=largura,
            label=LABELS_INICIAL.get(ini, ini),
            color=CORES.get(ini, None),
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(inst_dificeis, rotation=30, ha="right")
    ax.set_ylabel("Gap Final em relação ao Ótimo (%)")
    ax.set_title("Qualidade da Solução Final nas Instâncias Difíceis", fontweight="bold")
    ax.legend()
    fig.tight_layout()

    caminho = os.path.join(pasta_saida, "05_gap_instancias_dificeis.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"Salvo: {caminho}")


def main():
    pasta_graficos = "graficos_warm_start"
    os.makedirs(pasta_graficos, exist_ok=True)

    df_dados, ordem_ws = carregar_dados()
    imprimir_analise(df_dados)

    print("\n" + "=" * 85)
    print("GERANDO GRÁFICOS INDIVIDUAIS")
    print("=" * 85)
    plotar_alcance_otimo(df_dados, ordem_ws, pasta_saida=pasta_graficos)
    plotar_nos_expandidos(df_dados, ordem_ws, pasta_saida=pasta_graficos)
    plotar_nos_podados(df_dados, ordem_ws, pasta_saida=pasta_graficos)
    plotar_tempo_execucao(df_dados, ordem_ws, pasta_saida=pasta_graficos)
    plotar_gap_dificeis(df_dados, ordem_ws, pasta_saida=pasta_graficos)


if __name__ == "__main__":
    main()