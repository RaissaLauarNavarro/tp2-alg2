"""
Funções utilitárias compartilhadas entre os algoritmos de TSP.
"""

import numpy as np


def custo_tour(D, tour):
    """Calcula o custo total de um ciclo hamiltoniano (tour 0-based) na matriz D."""
    t = np.asarray(tour)
    return int(D[t, np.roll(t, -1)].sum())


def custo_arestas(D, arestas):
    """Soma os pesos de uma lista de arestas (u, v) na matriz D."""
    return sum(D[u, v].item() for u, v in arestas)


def matriz_euc2d(coords):
    """Gera matriz de distâncias inteiras EUC_2D (TSPLIB95) a partir de pontos 2D."""
    c = np.asarray(coords, dtype=np.float64)
    dif = c[:, None, :] - c[None, :, :]
    return np.floor(np.sqrt((dif ** 2).sum(axis=2)) + 0.5).astype(np.int64)
