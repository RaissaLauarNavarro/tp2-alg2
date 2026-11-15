"""
Algoritmo Twice-around-the-tree (2-aproximado) para o TSP métrico/euclidiano.

Implementação fiel ao APPROX-TSP-TOUR do slide da Aula 13 (DCC207, CLRS 35.2):
  1. Selecione um vértice r de G.V para ser a "raiz"
  2. Compute uma árvore geradora mínima T de G a partir da raiz r
     usando MST-PRIM(G, c, r)
  3. Seja H a lista de vértices ordenada pela primeira visita num
     caminhamento em pré-ordem de T
  4. Retorne o ciclo hamiltoniano H

Nenhuma biblioteca de grafos é usada. O Prim é o MESMO do christofides.py
(importado de lá), então os dois algoritmos partem exatamente da mesma árvore T
numa mesma instância. Qualquer diferença de custo entre eles vem só do que cada
um faz depois da AGM (duplicar as arestas vs. emparelhar os vértices ímpares).

Entrada: matriz de distâncias D (numpy n x n, simétrica), a mesma que o
leitor_tsp.py devolve.
"""

import time
import tracemalloc

import numpy as np

# Mesma AGM, mesmas funções de custo e mesma instrumentação do Christofides,
# para que tempos, memória e custos dos dois algoritmos sejam comparáveis.
from christofides import prim_mst, custo_tour, custo_arestas, _Medidor


# ---------------------------------------------------------------------------
# Passo 3: caminhamento em pré-ordem de T
# ---------------------------------------------------------------------------
def lista_de_filhos(n, arestas):
    """Converte as arestas (pai, filho) devolvidas pelo Prim em listas de filhos.

    A ordem dos filhos é a ordem em que o Prim os inseriu na árvore. O slide não
    fixa essa ordem, e a garantia de fator 2 vale para qualquer uma; mas ela muda
    o tour retornado, então fica registrada aqui para ser reprodutível."""
    filhos = [[] for _ in range(n)]
    for pai, filho in arestas:
        filhos[pai].append(filho)
    return filhos


def preordem(filhos, raiz):
    """Pré-ordem iterativa (pilha explícita), O(n).

    Os filhos são empilhados em ordem reversa para que o primeiro filho seja
    visitado primeiro. A pilha explícita evita o limite de recursão do Python,
    que seria atingido por uma AGM em forma de caminho (altura n - 1)."""
    H = []
    pilha = [raiz]
    while pilha:
        u = pilha.pop()
        H.append(u)
        pilha.extend(reversed(filhos[u]))
    return H


def caminhada_completa(filhos, raiz):
    """A volta "duas vezes ao redor da árvore": percorre cada aresta de T uma vez
    descendo e uma vez subindo, e devolve a sequência de vértices visitados
    (2(n-1) + 1 vértices, custo exatamente 2c(T)).

    Não é usada pelo algoritmo, que vai direto para a pré-ordem. Serve para
    auditar o argumento da prova: eliminar os vértices repetidos desta caminhada
    produz exatamente a pré-ordem."""
    W = [raiz]
    pilha = [(raiz, 0)]                 # (vértice, índice do próximo filho)
    while pilha:
        u, i = pilha[-1]
        if i < len(filhos[u]):
            pilha[-1] = (u, i + 1)
            v = filhos[u][i]
            W.append(v)                 # desce pela aresta u-v
            pilha.append((v, 0))
        else:
            pilha.pop()
            if pilha:
                W.append(pilha[-1][0])  # sobe pela aresta v-u
    return W


# ---------------------------------------------------------------------------
# Algoritmo completo
# ---------------------------------------------------------------------------
def twice_around_the_tree(D, inicio=0, medir_memoria=False):
    """Retorna (tour, custo, stats).

    `inicio` é a raiz r do passo 1. medir_memoria=True liga o tracemalloc, que
    deixa o código bem mais lento: meça tempo e memória em execuções SEPARADAS.
    """
    D = np.asarray(D)
    n = D.shape[0]
    if n <= 3:
        tour = list(range(n))
        return tour, (custo_tour(D, tour) if n > 1 else 0), {}

    ligou_aqui = medir_memoria and not tracemalloc.is_tracing()
    if ligou_aqui:
        tracemalloc.start()
    med = _Medidor(medir_memoria)
    t_total = time.perf_counter()

    try:
        with med.etapa("mst"):                       # passos 1 e 2
            T = prim_mst(D, raiz=inicio)

        with med.etapa("preordem"):                  # passo 3
            filhos = lista_de_filhos(n, T)
            H = preordem(filhos, inicio)
    finally:
        if ligou_aqui:
            tracemalloc.stop()

    custo = custo_tour(D, H)                         # passo 4: ciclo H + volta a r
    c_T = custo_arestas(D, T)
    grau = np.zeros(n, dtype=np.int64)
    for u, v in T:
        grau[u] += 1
        grau[v] += 1

    med.stats.update({
        "tempo_total": time.perf_counter() - t_total,
        "n": n,
        "grau_max_mst": int(grau.max()),
        "n_folhas_mst": int((grau == 1).sum()),
        "custo_mst": c_T,               # limite inferior: c(T) <= OPT
        "custo_caminhada": 2 * c_T,     # volta completa; >= c(H) se D for métrica
        "custo_tour": custo,
    })
    return H, custo, med.stats


# ---------------------------------------------------------------------------
# Verificações
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from itertools import permutations

    from christofides import atalhar, matriz_euc2d

    # Exemplo do slide (mesmo grafo usado no Christofides), a..e -> 0..4
    nomes = "abcde"
    pesos = {("a", "b"): 4, ("a", "c"): 8, ("a", "d"): 9, ("a", "e"): 12,
             ("b", "c"): 6, ("b", "d"): 8, ("b", "e"): 9,
             ("c", "d"): 10, ("c", "e"): 11, ("d", "e"): 7}
    D = np.zeros((5, 5), dtype=np.int64)
    for (x, y), w in pesos.items():
        i, j = nomes.index(x), nomes.index(y)
        D[i, j] = D[j, i] = w

    T = prim_mst(D)
    filhos = lista_de_filhos(5, T)
    W = caminhada_completa(filhos, 0)
    tour, custo, st = twice_around_the_tree(D)
    print("Exemplo do slide")
    print("  T  =", [nomes[u] + "-" + nomes[v] for u, v in T], "custo", st["custo_mst"])
    print("  W  =", "-".join(nomes[v] for v in W), "custo", custo_tour(D, W[:-1]),
          "(= 2c(T))")
    print("  H  =", "-".join(nomes[v] for v in tour + [tour[0]]), "custo", custo)

    def otimo_forca_bruta(D):
        n = D.shape[0]
        return min(custo_tour(D, [0, *p]) for p in permutations(range(1, n)))

    # Instâncias euclidianas aleatórias:
    #   (a) tour é permutação válida
    #   (b) a caminhada completa custa exatamente 2c(T)
    #   (c) atalhar a caminhada completa dá exatamente a pré-ordem (passo da prova)
    #   (d) razão custo/ótimo < 2
    rng = np.random.default_rng(0)
    pior = 1.0
    for n in range(4, 10):
        for _ in range(30):
            D = matriz_euc2d(rng.uniform(0, 1000, size=(n, 2)))
            raiz = int(rng.integers(n))
            tour, custo, st = twice_around_the_tree(D, inicio=raiz)
            assert sorted(tour) == list(range(n))
            W = caminhada_completa(lista_de_filhos(n, prim_mst(D, raiz)), raiz)
            assert len(W) == 2 * (n - 1) + 1
            assert custo_tour(D, W[:-1]) == st["custo_caminhada"]
            assert atalhar(W, n) == tour
            opt = otimo_forca_bruta(D)
            pior = max(pior, custo / opt)
    print(f"\n180 instâncias aleatórias (n=4..9, raiz sorteada): "
          f"pior razão custo/ótimo = {pior:.4f}")

    # Profiling por sub-rotina numa instância maior
    D = matriz_euc2d(rng.uniform(0, 1000, size=(75, 2)))
    _, _, st = twice_around_the_tree(D)
    print("\nn=75, tempos (ms):",
          {k[6:]: round(v * 1e3, 3) for k, v in st.items() if k.startswith("tempo_")})
    _, _, st = twice_around_the_tree(D, medir_memoria=True)
    print("n=75, pico de memória (KiB):",
          {k[9:]: round(v / 1024, 1) for k, v in st.items() if k.startswith("mem_pico_")})