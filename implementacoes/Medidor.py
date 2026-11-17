"""
Instrumentação de tempo e pico de memória por sub-rotina.
"""

import time
import tracemalloc
from contextlib import contextmanager


class Medidor:
    def __init__(self, medir_memoria):
        self.medir_memoria = medir_memoria
        self.stats = {}

    @contextmanager
    def etapa(self, nome):
        if self.medir_memoria:
            base, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
        t0 = time.perf_counter()
        yield
        self.stats[f"tempo_{nome}"] = time.perf_counter() - t0
        if self.medir_memoria:
            _, pico = tracemalloc.get_traced_memory()
            self.stats[f"mem_pico_{nome}"] = pico - base   # bytes acima da base
