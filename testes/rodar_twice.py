"""
Roda o Twice-around-the-tree em todas as instâncias TSPLIB da pasta data/ e
salva as métricas num CSV (parte da "base de dados com as métricas coletadas").

Uso (a partir da pasta testes/; a pasta data/ fica em ../data e resultados em ../resultados):
    python rodar_twice.py
    python rodar_twice.py --reps 5 --memoria
    python rodar_twice.py --nmin 10 --nmax 75     # filtra por tamanho
    python rodar_twice.py --so-euc2d               # só EUC_2D
    python rodar_twice.py outra_pasta/ --saida ../resultados/outro.csv

Por padrão roda TODAS as instâncias .tsp / .tsp.gz da pasta, sem filtro de n.

Tempos: com --reps k, o algoritmo roda k vezes e cada tempo salvo é a MEDIANA
das k execuções (na casa dos milissegundos, uma execução só é muito ruidosa
para ajustar curvas de complexidade).
Memória: com --memoria, faz UMA execução extra com tracemalloc ligado, separada
das execuções cronometradas, para o tracemalloc não inflar os tempos.
"""

import argparse
import csv
import gc
import os
import statistics
import sys

RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_DATA_PADRAO = os.path.join(RAIZ_PROJETO, "data")
PASTA_RESULTADOS_PADRAO = os.path.join(RAIZ_PROJETO, "resultados")

# as implementações ficam em ../implementacoes
sys.path.insert(0, os.path.join(RAIZ_PROJETO, "implementacoes"))

from twice_around import twice_around_the_tree
from leitor_tsp import (ler_instancia, listar_instancias,
                        violacoes_desigualdade_triangular)
from utils import custo_tour

ETAPAS = ["mst", "preordem", "total"]


def avaliar(inst, reps, medir_memoria):
    twice_around_the_tree(inst.D)  # aquecimento: descartado (caches, imports tardios, alocador)
    gc.collect()
    gc.disable()  # evita pausas do coletor de lixo dentro das medições
    try:
        execucoes = [twice_around_the_tree(inst.D) for _ in range(reps)]
    finally:
        gc.enable()
    tour, custo, st = execucoes[0]

    # auditoria do resultado: permutação válida e custo recalculado
    assert sorted(tour) == list(range(inst.n)), f"{inst.nome}: tour inválido"
    assert custo == custo_tour(inst.D, tour), f"{inst.nome}: custo divergente"

    nviol, vmax = violacoes_desigualdade_triangular(inst.D)
    linha = {
        "instancia": inst.nome,
        "n": inst.n,
        "tipo_peso": inst.tipo_peso,
        "formato_peso": inst.formato_peso or "",
        "tem_coordenadas": inst.pontos is not None,
        "otimo": inst.otimo if inst.otimo is not None else "NA",
        "custo": custo,
        "custo_mst": st["custo_mst"],
        "custo_caminhada": st["custo_caminhada"],
        # quanto os atalhos economizaram em relação à volta completa 2c(T);
        # negativo só é possível se D violar a desigualdade triangular
        "economia_atalhos": st["custo_caminhada"] - custo,
        "grau_max_mst": st["grau_max_mst"],
        "n_folhas_mst": st["n_folhas_mst"],
        "pares_violam_triangular": nviol,
        "maior_violacao_triangular": vmax,
        # a garantia da aula 13 exige que D seja métrica; violações de 1 unidade
        # vêm só do arredondamento da TSPLIB (nint)
        "classe_metrica": ("metrica" if nviol == 0 else
                           "arredondamento" if vmax <= 1 else "nao_metrica"),
    }
    if inst.otimo:
        razao = custo / inst.otimo
        linha["razao"] = round(razao, 6)
        linha["gap_pct"] = round(100 * (razao - 1), 4)
        linha["razao_abaixo_2"] = razao <= 2
    else:
        linha["razao"] = linha["gap_pct"] = linha["razao_abaixo_2"] = "NA"

    for e in ETAPAS:
        amostras = [s[f"tempo_{e}"] for _, _, s in execucoes]
        linha[f"tempo_{e}_s"] = statistics.median(amostras)
        if len(amostras) >= 2:
            q1, _, q3 = statistics.quantiles(amostras, n=4)
            linha[f"tempo_{e}_iqr_s"] = q3 - q1  # dispersão: detecta medições ruidosas

    if medir_memoria:
        _, _, sm = twice_around_the_tree(inst.D, medir_memoria=True)
        for e in ETAPAS[:-1]:
            linha[f"mem_pico_{e}_bytes"] = sm[f"mem_pico_{e}"]

    return linha


def main():
    ap = argparse.ArgumentParser(description="Twice-around-the-tree nas instâncias TSPLIB")
    ap.add_argument("pasta", nargs="?", default=PASTA_DATA_PADRAO,
                    help="pasta com os .tsp/.tsp.gz (padrão: ../data)")
    ap.add_argument("--nmin", type=int, default=0)
    ap.add_argument("--nmax", type=int, default=10**9)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--memoria", action="store_true")
    ap.add_argument("--so-euc2d", action="store_true")
    ap.add_argument("--saida", default=None,
                    help="padrão: ../resultados/resultados_twice[_memoria].csv")
    args = ap.parse_args()
    if args.saida is None:
        nome_arquivo = ("resultados_twice_memoria.csv" if args.memoria
                        else "resultados_twice.csv")
        args.saida = os.path.join(PASTA_RESULTADOS_PADRAO, nome_arquivo)

    if not os.path.isdir(args.pasta):
        sys.exit(f"Pasta '{args.pasta}' não encontrada. Rode a partir de "
                 f"testes/ ou passe o caminho: python rodar_twice.py caminho/")

    caminhos = listar_instancias(args.pasta, args.nmin, args.nmax)
    if not caminhos:
        sys.exit(f"Nenhum .tsp ou .tsp.gz em '{args.pasta}' com {args.nmin} <= n <= {args.nmax}.")

    linhas, puladas = [], []
    print(f"{'instância':<11}{'n':>5}  {'tipo':<9}{'custo':>9}{'ótimo':>9}"
          f"{'gap %':>8}{'2c(T)':>9}{'mst':>10}{'total':>10}")
    for caminho in caminhos:
        arq = os.path.basename(caminho)
        try:
            inst = ler_instancia(caminho)
        except (ValueError, KeyError) as e:
            puladas.append((arq, str(e)))
            continue
        if args.so_euc2d and inst.tipo_peso != "EUC_2D":
            continue
        if inst.n < 4:
            puladas.append((arq, f"n={inst.n} é pequeno demais"))
            continue

        r = avaliar(inst, args.reps, args.memoria)
        linhas.append(r)
        print(f"{r['instancia']:<11}{r['n']:>5}  {r['tipo_peso']:<9}{r['custo']:>9}"
              f"{str(r['otimo']):>9}{str(r['gap_pct']):>8}{r['custo_caminhada']:>9}"
              f"{r['tempo_mst_s'] * 1e3:>8.2f}ms{r['tempo_total_s'] * 1e3:>8.2f}ms")

    for arq, motivo in puladas:
        print(f"PULADA {arq}: {motivo}")

    if linhas:
        pasta_saida = os.path.dirname(os.path.abspath(args.saida))
        if pasta_saida:
            os.makedirs(pasta_saida, exist_ok=True)
        with open(args.saida, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
            w.writeheader()
            w.writerows(linhas)
        print(f"{len(linhas)} instâncias -> {args.saida}")


if __name__ == "__main__":
    main()
