"""
Algoritmo de Christofides (1.5-aproximado) para o TSP métrico/euclidiano.

Implementação fiel ao slide da Aula 13 (DCC207):
  1. Compute T, uma árvore geradora mínima de G
  2. Seja I o conjunto de vértices de grau ímpar de T. Compute M, um matching
     perfeito de peso mínimo no subgrafo induzido por I
  3. Seja G' o multigrafo com os vértices de V e as arestas de M e T.
     Compute um circuito euleriano em G' (DFS)
  4. Elimine vértices duplicados, substituindo subcaminhos u-w-v por arestas u-v

O networkx é usado SOMENTE no passo 2 (Blossom de Edmonds). Todo o resto
(Prim, graus, circuito euleriano, atalhos) é implementado aqui.

Entrada: matriz de distâncias D (numpy n x n, simétrica). Assim a mesma função
serve para qualquer tipo de instância, desde que D seja uma métrica.
"""

import time
import tracemalloc

import numpy as np
import networkx as nx

from Medidor import Medidor
from utils import custo_tour, custo_arestas


# ---------------------------------------------------------------------------
# Passo 1: árvore geradora mínima (Prim com vetor, O(n^2))
# ---------------------------------------------------------------------------
def prim_mst(D, raiz=0):
    """Prim na versão com vetor de chaves (sem heap).

    O grafo é completo (|E| = n(n-1)/2), então a versão com vetor, O(n^2),
    é assintoticamente melhor que a com heap binário, O(|E| log n) = O(n^2 log n).
    Retorna a lista de arestas (pai, filho) da árvore.
    """
    n = D.shape[0]
    na_arvore = np.zeros(n, dtype=bool)
    chave = np.full(n, np.inf)
    pai = np.full(n, -1, dtype=np.int64)
    chave[raiz] = 0.0
    arestas = []

    for _ in range(n):
        # vértice fora da árvore com menor chave
        u = int(np.argmin(np.where(na_arvore, np.inf, chave)))
        na_arvore[u] = True
        if pai[u] != -1:
            arestas.append((int(pai[u]), u))
        # relaxa as chaves usando a linha u de D (D[u] é view, não cópia)
        linha = D[u]
        melhora = (~na_arvore) & (linha < chave)
        chave[melhora] = linha[melhora]
        pai[melhora] = u

    return arestas


# ---------------------------------------------------------------------------
# Passo 2a: vértices de grau ímpar em T
# ---------------------------------------------------------------------------
def vertices_grau_impar(n, arestas):
    grau = np.zeros(n, dtype=np.int64)
    for u, v in arestas:
        grau[u] += 1
        grau[v] += 1
    impares = [int(v) for v in np.flatnonzero(grau % 2 == 1)]
    return impares, grau


# ---------------------------------------------------------------------------
# Passo 2b: matching perfeito de peso mínimo (único uso do networkx)
# ---------------------------------------------------------------------------
def matching_perfeito_minimo(D, impares):
    """Matching perfeito de peso mínimo no subgrafo induzido por I.

    Usa nx.min_weight_matching (NetworkX >= 3.0). Internamente, a função
    inverte os pesos, w'(u,v) = W + 1 - w(u,v), e chama o Blossom de peso
    máximo com maxcardinality=True. Num grafo completo com |I| par,
    cardinalidade máxima = matching perfeito, e entre matchings perfeitos
    (todos com |I|/2 arestas):
        max sum(W + 1 - w) = (|I|/2)(W + 1) - min sum(w).
    Antes da 3.0 a função não exigia cardinalidade máxima, por isso a versão
    é fixada e o resultado é verificado abaixo.
    """
    k = len(impares)
    if k == 0:
        return []
    assert k % 2 == 0, "Lema do aperto de mãos: |I| deve ser par"

    # pesos como tipos nativos do Python (.item()) -> inteiros exatos na TSPLIB
    G = nx.Graph()
    for i in range(k):
        u = impares[i]
        for j in range(i + 1, k):
            v = impares[j]
            G.add_edge(u, v, weight=D[u, v].item())

    M = nx.min_weight_matching(G, weight="weight")

    # auditoria: o matching precisa ser perfeito e cobrir exatamente I
    assert 2 * len(M) == k, "Matching retornado não é perfeito"
    assert {x for e in M for x in e} == set(impares), "Matching não cobre I"
    return [(int(u), int(v)) for u, v in M]

# ---------------------------------------------------------------------------
# Passo 3: circuito euleriano no multigrafo T + M (Hierholzer via DFS)
# ---------------------------------------------------------------------------
def circuito_euleriano(n, arestas, inicio=0):
    """Hierholzer iterativo (DFS com pilha explícita), O(|E|).

    Cada aresta recebe um id: uma aresta que está em T e em M aparece duas
    vezes no multigrafo, e marcar por id (e não pelo par u,v) é o que permite
    percorrer as duas cópias. Pilha explícita evita o limite de recursão.
    """
    adj = [[] for _ in range(n)]
    for eid, (u, v) in enumerate(arestas):
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    usada = [False] * len(arestas)
    ptr = [0] * n            # próxima posição a examinar em adj[u]
    pilha = [inicio]
    circuito = []

    while pilha:
        u = pilha[-1]
        while ptr[u] < len(adj[u]) and usada[adj[u][ptr[u]][1]]:
            ptr[u] += 1
        if ptr[u] == len(adj[u]):
            circuito.append(pilha.pop())   # u sem arestas livres: fecha
        else:
            v, eid = adj[u][ptr[u]]
            usada[eid] = True
            pilha.append(v)

    circuito.reverse()
    assert len(circuito) == len(arestas) + 1, "Multigrafo não é euleriano"
    return circuito


# ---------------------------------------------------------------------------
# Passo 4: atalhos (shortcutting)
# ---------------------------------------------------------------------------
def atalhar(circuito, n):
    """Mantém a primeira ocorrência de cada vértice. Pular u-w-v para u-v
    nunca aumenta o custo pela desigualdade triangular."""
    visto = [False] * n
    tour = []
    for v in circuito:
        if not visto[v]:
            visto[v] = True
            tour.append(v)
    return tour  # o retorno tour[-1] -> tour[0] fica implícito


# ---------------------------------------------------------------------------
# Algoritmo completo
# ---------------------------------------------------------------------------
def christofides(D, inicio=0, medir_memoria=False):
    """Retorna (tour, custo, stats).

    medir_memoria=True liga o tracemalloc, que deixa o código bem mais lento:
    meça tempo e memória em execuções SEPARADAS para não contaminar os tempos.
    """
    D = np.asarray(D)
    n = D.shape[0]
    if n <= 3:
        tour = list(range(n))
        return tour, (custo_tour(D, tour) if n > 1 else 0), {}

    ligou_aqui = medir_memoria and not tracemalloc.is_tracing()
    if ligou_aqui:
        tracemalloc.start()
    med = Medidor(medir_memoria)
    t_total = time.perf_counter()

    try:
        with med.etapa("mst"):
            T = prim_mst(D, raiz=inicio)

        with med.etapa("impares"):
            I, grau = vertices_grau_impar(n, T)

        with med.etapa("matching"):
            M = matching_perfeito_minimo(D, I)

        with med.etapa("euler"):
            circuito = circuito_euleriano(n, T + M, inicio=inicio)

        with med.etapa("atalho"):
            tour = atalhar(circuito, n)
    finally:
        if ligou_aqui:
            tracemalloc.stop()

    custo = custo_tour(D, tour)
    c_T, c_M = custo_arestas(D, T), custo_arestas(D, M)
    med.stats.update({
        "tempo_total": time.perf_counter() - t_total,
        "n": n,
        "n_impares": len(I),          # tamanho da entrada do Blossom
        "grau_max_mst": int(grau.max()),
        "custo_mst": c_T,             # limite inferior: c(T) <= OPT
        "custo_matching": c_M,
        "custo_euleriano": c_T + c_M,  # >= custo do tour (atalhos só reduzem)
        "custo_tour": custo,
    })
    return tour, custo, med.stats
