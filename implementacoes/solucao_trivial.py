"""
Inicialização trivial (Cold Start) para o TSP: 1 -> 2 -> ... -> n -> 1.
"""

import numpy as np


def solucao_trivial(D):
    """Cold start: 1 -> 2 -> ... -> n -> 1 (0-based: 0 -> 1 -> ... -> n-1 -> 0)."""
    D = np.asarray(D)
    return list(range(D.shape[0]))
