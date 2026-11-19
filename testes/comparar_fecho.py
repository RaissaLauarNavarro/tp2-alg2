"""
Compara os dois algoritmos de fecho convexo da aula 05 e salva as métricas num
CSV, para a análise do trade-off O(n log n) x O(nh):

    graham  Varredura de Graham, O(n log n), dominada pela ordenação polar
    gift    Gift Wrapping (Jarvis), O(nh), h = vértices do fecho

Uso (a partir da pasta testes/; data/ em ../data, resultados em ../resultados):
    python comparar_fecho.py                 # completo (alguns minutos)
    python comparar_fecho.py --rapido        # tamanhos menores, para conferir
    python comparar_fecho.py --reps 7 --saida ../resultados/outro.csv

Por que conjuntos sintéticos além da TSPLIB: nas instâncias do trabalho,
n <= 76 e h fica entre 4 e 11. Nessa faixa os dois algoritmos rodam em
microssegundos e o trade-off não aparece. Para ver onde O(nh) passa a perder
para O(n log n), é preciso variar n e h de forma independente:

    quadrado      inteiros uniformes num quadrado: h esperado = Theta(log n)
    disco         inteiros uniformes num disco:    h esperado = Theta(n^(1/3))
    h_controlado  h pontos sobre um círculo + (n - h) pontos estritamente
                  dentro do polígono que eles formam: h EXATO, escolhido
                  (h = n é o pior caso do Gift Wrapping, O(n^2))
    tsplib        as instâncias da pasta data/ (coordenadas de pontos_para_fecho)

Para cada caso o CSV guarda:
  - tempo mediano por chamada de cada algoritmo (timeit, com o coletor de lixo
    desligado durante a medição) e todos os tempos repetidos, em JSON;
  - o número de testes de orientação (chamadas a _cruz), contado numa execução
    SEPARADA da cronometrada: é uma medida de custo que não depende da máquina
    nem do ruído, boa para ajustar as curvas n log n e nh;
  - n log2 n e n*h, já prontos como variáveis explicativas da regressão.
Cada caso também confere que os dois algoritmos devolvem a mesma lista e, no
h_controlado, que o h obtido é o h pedido.
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
import timeit

import numpy as np

RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_DATA_PADRAO = os.path.join(RAIZ_PROJETO, "data")
PASTA_RESULTADOS_PADRAO = os.path.join(RAIZ_PROJETO, "resultados")
sys.path.insert(0, os.path.join(RAIZ_PROJETO, "implementacoes"))

from fecho_convexo import (ALGORITMOS_FECHO, _cruz, pontos_para_fecho)  # noqa: E402
from leitor_tsp import ler_instancia, listar_instancias                 # noqa: E402

CAMPOS = [
    "origem", "distribuicao", "instancia", "n", "h", "h_pedido",
    "n_log2_n", "n_h",
    "tempo_graham_s", "tempo_gift_s", "razao_tempo_gift_graham",
    "orientacoes_graham", "orientacoes_gift", "razao_orientacoes_gift_graham",
    "chamadas_por_medicao_graham", "chamadas_por_medicao_gift",
    "tempos_graham_s", "tempos_gift_s",
]


# ---------------------------------------------------------------------------
# Geradores de pontos (semente fixa: o experimento é reprodutível)
# ---------------------------------------------------------------------------
LADO = 10**6     # inteiros até 1e6: o teste de orientação é exato em float


def gerar_quadrado(n, rng):
    return rng.integers(0, LADO, size=(n, 2)).astype(float)


def gerar_disco(n, rng):
    pts = []
    while len(pts) < n:
        c = rng.integers(-LADO, LADO, size=(2 * n, 2))
        dentro = c[(c ** 2).sum(axis=1) < LADO ** 2]
        pts.extend(dentro.tolist())
    return np.asarray(pts[:n], dtype=float)


def gerar_h_controlado(n, h, rng):
    """h vértices de um polígono regular inscrito no círculo unitário (girado
    ao acaso) + (n - h) pontos no disco de raio 0,9 cos(pi/h), que fica
    estritamente dentro do polígono (cos(pi/h) é o raio do círculo inscrito).
    A ordem dos pontos é embaralhada para não favorecer nenhum algoritmo."""
    rot = rng.uniform(0, 2 * math.pi)
    ang = rot + 2 * math.pi * np.arange(h) / h
    borda = np.column_stack([np.cos(ang), np.sin(ang)])
    m = n - h
    r_int = 0.9 * math.cos(math.pi / h)
    raio = r_int * np.sqrt(rng.uniform(0, 1, m))
    teta = rng.uniform(0, 2 * math.pi, m)
    dentro = np.column_stack([raio * np.cos(teta), raio * np.sin(teta)])
    pts = np.vstack([borda, dentro])
    return pts[rng.permutation(n)]


# ---------------------------------------------------------------------------
# Medição
# ---------------------------------------------------------------------------
def contar_orientacoes(f, P):
    cont = [0]

    def orient(o, a, b):
        cont[0] += 1
        return _cruz(o, a, b)

    return f(P, orient=orient), cont[0]


def cronometrar(f, P, reps):
    """Tempo mediano por chamada. timeit desliga o coletor de lixo durante a
    medição. autorange escolhe quantas chamadas por medição para passar de
    0,2 s (casos rápidos repetem muitas vezes; o pior caso roda uma vez)."""
    t = timeit.Timer(lambda: f(P))
    numero, _ = t.autorange()
    tempos = [x / numero for x in t.repeat(repeat=reps, number=numero)]
    return statistics.median(tempos), tempos, numero


def avaliar(P, reps, origem, distribuicao, instancia="", h_pedido=""):
    n = len(P)
    res, linha = {}, {}
    for nome, f in ALGORITMOS_FECHO.items():
        H, ops = contar_orientacoes(f, P)
        med, tempos, numero = cronometrar(f, P, reps)
        res[nome] = H
        linha[f"tempo_{nome}_s"] = med
        linha[f"tempos_{nome}_s"] = json.dumps([round(x, 9) for x in tempos])
        linha[f"orientacoes_{nome}"] = ops
        linha[f"chamadas_por_medicao_{nome}"] = numero
    assert res["graham"] == res["gift"], f"{distribuicao} n={n}: algoritmos divergem"
    h = len(res["graham"])
    if h_pedido != "":
        assert h == h_pedido, f"h obtido {h} != h pedido {h_pedido}"
    linha.update({
        "origem": origem, "distribuicao": distribuicao, "instancia": instancia,
        "n": n, "h": h, "h_pedido": h_pedido,
        "n_log2_n": round(n * math.log2(n), 3), "n_h": n * h,
        "razao_tempo_gift_graham": round(linha["tempo_gift_s"] / linha["tempo_graham_s"], 4),
        "razao_orientacoes_gift_graham": round(
            linha["orientacoes_gift"] / linha["orientacoes_graham"], 4),
    })
    return linha


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Graham x Gift Wrapping")
    ap.add_argument("pasta", nargs="?", default=PASTA_DATA_PADRAO,
                    help="pasta com as instâncias TSPLIB (padrão: ../data)")
    ap.add_argument("--reps", type=int, default=5, help="repetições cronometradas")
    ap.add_argument("--semente", type=int, default=2026)
    ap.add_argument("--rapido", action="store_true", help="tamanhos menores")
    ap.add_argument("--saida", default=os.path.join(PASTA_RESULTADOS_PADRAO,
                                                    "comparacao_fecho.csv"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.semente)

    if args.rapido:
        tamanhos, hs = [100, 200, 400, 800], [4, 16, 64]
    else:
        tamanhos, hs = [125, 250, 500, 1000, 2000, 4000], [4, 8, 16, 32, 64, 128, 256, 512]

    casos = []
    for n in tamanhos:
        casos.append(("sintetico", "quadrado", "", gerar_quadrado(n, rng), ""))
        casos.append(("sintetico", "disco", "", gerar_disco(n, rng), ""))
        for h in [x for x in hs if x < n] + [n]:      # h = n: pior caso do gift
            casos.append(("sintetico", "h_controlado", "", gerar_h_controlado(n, h, rng), h))

    if os.path.isdir(args.pasta):
        for c in listar_instancias(args.pasta):
            try:
                inst = ler_instancia(c)
            except (ValueError, KeyError):
                continue
            pts, origem_coords, _ = pontos_para_fecho(inst)
            casos.append(("tsplib", origem_coords, inst.nome, pts, ""))

    os.makedirs(os.path.dirname(os.path.abspath(args.saida)), exist_ok=True)
    print(f"{len(casos)} casos, {args.reps} repetições -> {args.saida}")
    print(f"{'origem':<10}{'distribuição':<14}{'inst.':<11}{'n':>6}{'h':>6}"
          f"{'graham':>12}{'gift':>12}{'gift/graham':>13}{'orient. g/gw':>18}")
    with open(args.saida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        for origem, dist, nome, P, h_pedido in casos:
            r = avaliar(P, args.reps, origem, dist, nome, h_pedido)
            w.writerow(r)
            f.flush()
            print(f"{origem:<10}{dist:<14}{nome:<11}{r['n']:>6}{r['h']:>6}"
                  f"{r['tempo_graham_s'] * 1e3:>10.3f}ms{r['tempo_gift_s'] * 1e3:>10.3f}ms"
                  f"{r['razao_tempo_gift_graham']:>13.3f}"
                  f"{r['orientacoes_graham']:>9}/{r['orientacoes_gift']:<9}", flush=True)
    print(f"\nResultados em {args.saida}")


if __name__ == "__main__":
    main()