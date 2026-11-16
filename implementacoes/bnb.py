"""
Branch-and-Bound exato para o TSP simétrico.

Árvore de busca
  Cada nó é um caminho parcial que começa no vértice 0 (fixar a origem não perde
  soluções: todo ciclo passa por 0). Ramificar = estender o caminho por um
  vértice ainda não visitado. Uma folha (caminho com n vértices) vira um ciclo
  ao somar a aresta de volta a 0.

Limite inferior (Levitin, seção 12.2)
  Num ciclo, cada vértice tem exatamente duas arestas incidentes, então
      custo(ciclo) = (1/2) * soma_v [soma das duas arestas do ciclo em v]
                  >= (1/2) * soma_v [soma das duas MENORES arestas viáveis em v].
  Para um caminho parcial, as arestas já escolhidas são forçadas:
    - vértice interno do caminho: as suas duas arestas do caminho;
    - extremidade (início ou fim): a aresta do caminho + a menor outra aresta;
    - vértice fora do caminho: as duas menores arestas incidentes.
  Com pesos inteiros (TSPLIB), lb = ceil(S / 2). O lb de um filho é calculado
  a partir do lb do pai em O(1) (só mudam as parcelas de dois vértices).

Poda
  Um nó é podado quando lb >= melhor custo conhecido (limite superior, UB).
  Como os custos são inteiros, lb >= UB significa que nada abaixo dele pode
  melhorar o UB. O UB começa com o custo de `tour_inicial` (warm-start); sem
  ele, usa a solução trivial 0 -> 1 -> ... -> n-1 -> 0 (cold start).

Quebra de simetria
  Todo ciclo aparece nos dois sentidos (0,a,...,z) e (0,z,...,a). Só o sentido
  com a < z é explorado, o que corta cerca de metade da árvore sem perder o
  ótimo. O teste custa O(1) por nó.

Estratégias
  "profundidade": DFS recursiva com backtracking; os filhos são visitados em
      ordem crescente de lb. Memória O(n^2) no pior caso (lista de filhos por
      nível), independente do tamanho da árvore.
  "melhor": best-first com heap pelo lb (desempate pelo nó mais profundo).
      Cada nó guarda o seu próprio caminho, então a memória cresce com a
      fronteira da busca; `max_fila` interrompe antes de esgotar a RAM
      (medido: 3 milhões de nós na fila ocupam ~1,3 GB, com n = 24).

Entrada: matriz de distâncias D (numpy n x n, simétrica, inteira), a mesma
que o leitor_tsp.py devolve.
"""

import heapq
import time
import tracemalloc

import numpy as np

from solucao_trivial import solucao_trivial
from utils import custo_tour


class _Interrompido(Exception):
    def __init__(self, status):
        self.status = status


# ---------------------------------------------------------------------------
# Pré-processamento do limite inferior
# ---------------------------------------------------------------------------
def _preprocessar(D):
    """Para cada vértice v: menor aresta m1[v] (para o vizinho viz1[v]) e
    segunda menor m2[v]. A matriz vira lista de listas: no laço interno do
    Python, Dl[u][v] é bem mais rápido que D[u, v] em numpy."""
    n = D.shape[0]
    Dl = D.tolist()
    m1, m2, viz1 = [0] * n, [0] * n, [0] * n
    for v in range(n):
        ordem = sorted((Dl[v][u], u) for u in range(n) if u != v)
        m1[v], viz1[v] = ordem[0]
        m2[v] = ordem[1][0]
    return Dl, m1, m2, viz1


def limite_inferior_raiz(D):
    """lb do nó raiz (só o vértice 0 no caminho): ceil(soma(m1 + m2) / 2)."""
    _, m1, m2, _ = _preprocessar(np.asarray(D))
    return (sum(m1) + sum(m2) + 1) // 2


# ---------------------------------------------------------------------------
# Busca
# ---------------------------------------------------------------------------
class _Busca:
    def __init__(self, D, ub, tour_ub, limite_s, max_fila):
        self.n = D.shape[0]
        self.Dl, self.m1, self.m2, self.viz1 = _preprocessar(D)
        self.base = [a + b for a, b in zip(self.m1, self.m2)]
        self.ub = ub
        self.tour_ub = list(tour_ub)
        self.limite_s = limite_s
        self.max_fila = max_fila
        self.t0 = time.perf_counter()
        self.prazo = self.t0 + limite_s
        self.c = dict(gerados=1, expandidos=0, podados_limite=0,
                      podados_simetria=0, folhas=0, melhorias_ub=0,
                      prof_max=1, max_fila=1)
        self.tempo_ate_melhor = 0.0

    # menor aresta de v que não seja a aresta para x
    def _alt(self, v, x):
        return self.m2[v] if self.viz1[v] == x else self.m1[v]

    def _filhos(self, caminho_ultimos, k, S, custo, cont, p1, visitado):
        """Gera os filhos de um nó. caminho_ultimos = (penúltimo, último).
        Devolve [(lb, v, S2, custo2, cont2)] só dos que sobrevivem à poda.
        Folhas (k + 1 == n) são fechadas aqui mesmo e atualizam o UB."""
        n, Dl, c = self.n, self.Dl, self.c
        p, u = caminho_ultimos
        saida = []
        restantes = n - k - 1                  # não visitados depois do filho
        for v in range(1, n):
            if visitado[v]:
                continue
            c["gerados"] += 1
            w = Dl[u][v]

            # quebra de simetria: o último vértice do ciclo tem de ser > p1
            if k == 1:
                cont2 = n - 1 - v              # não visitados maiores que v
                p1v = v
            else:
                cont2 = cont - (v > p1)
                p1v = p1
            if restantes > 0 and cont2 == 0:
                c["podados_simetria"] += 1
                continue

            # lb incremental: só mudam as parcelas de u e v
            if k == 1:   # u é a origem, ainda sem aresta forçada
                S2 = S - self.base[u] + w + self._alt(u, v)                        - self.base[v] + w + self._alt(v, u)
            else:        # u deixa de ser extremidade e vira vértice interno
                S2 = S - self._alt(u, p) + w - self.base[v] + w + self._alt(v, u)
            custo2 = custo + w

            if restantes == 0:                 # folha: ciclo completo
                c["folhas"] += 1
                total = custo2 + Dl[v][0]
                if total < self.ub:
                    self._novo_ub(total, v)
                continue

            lb = (S2 + 1) // 2
            if lb >= self.ub:
                c["podados_limite"] += 1
                continue
            saida.append((lb, v, S2, custo2, cont2, p1v))
        return saida

    def _novo_ub(self, total, ultimo):
        self.ub = total
        self.tour_ub = self._caminho_atual() + [ultimo]
        self.c["melhorias_ub"] += 1
        self.tempo_ate_melhor = time.perf_counter() - self.t0

    def _checar_tempo(self):
        if self.c["expandidos"] & 1023 == 0 and time.perf_counter() > self.prazo:
            raise _Interrompido("tempo_esgotado")

    # ---------------- profundidade ----------------
    def profundidade(self):
        n = self.n
        self.visitado = [False] * n
        self.visitado[0] = True
        self.caminho = [0]
        self._caminho_atual = lambda: list(self.caminho)
        S0 = sum(self.base)
        self._dfs(S0, 0, 0, 0)

    def _dfs(self, S, custo, cont, p1):
        c, caminho, visitado = self.c, self.caminho, self.visitado
        k = len(caminho)
        c["expandidos"] += 1
        self._checar_tempo()
        if k > c["prof_max"]:
            c["prof_max"] = k
        ultimos = (caminho[-2] if k > 1 else -1, caminho[-1])
        filhos = self._filhos(ultimos, k, S, custo, cont, p1, visitado)
        filhos.sort()                          # menor lb primeiro
        for lb, v, S2, custo2, cont2, p1v in filhos:
            if lb >= self.ub:                  # UB pode ter melhorado no irmão
                c["podados_limite"] += 1
                continue
            visitado[v] = True
            caminho.append(v)
            self._dfs(S2, custo2, cont2, p1v)
            caminho.pop()
            visitado[v] = False

    # ---------------- melhor primeiro ----------------
    def melhor(self):
        n, c = self.n, self.c
        S0 = sum(self.base)
        desempate = 0
        # (lb, -profundidade, desempate, caminho, S, custo, cont, p1)
        fila = [((S0 + 1) // 2, -1, desempate, (0,), S0, 0, 0, 0)]
        self._atual = (0,)
        self._caminho_atual = lambda: list(self._atual)
        while fila:
            lb, negk, _, caminho, S, custo, cont, p1 = heapq.heappop(fila)
            if lb >= self.ub:                  # UB melhorou depois do push
                c["podados_limite"] += 1
                continue
            k = -negk
            c["expandidos"] += 1
            self._checar_tempo()
            if k > c["prof_max"]:
                c["prof_max"] = k
            visitado = [False] * n
            for v in caminho:
                visitado[v] = True
            self._atual = caminho
            ultimos = (caminho[-2] if k > 1 else -1, caminho[-1])
            for lb2, v, S2, custo2, cont2, p1v in self._filhos(
                    ultimos, k, S, custo, cont, p1, visitado):
                desempate += 1
                heapq.heappush(fila, (lb2, -(k + 1), desempate, caminho + (v,),
                                      S2, custo2, cont2, p1v))
            if len(fila) > c["max_fila"]:
                c["max_fila"] = len(fila)
                if len(fila) > self.max_fila:
                    raise _Interrompido("memoria_esgotada")


def branch_and_bound(D, tour_inicial=None, estrategia="profundidade",
                     limite_s=1800.0, max_fila=2_000_000, medir_memoria=False):
    """Resolve o TSP de forma exata.

    tour_inicial: tour usado como limite superior inicial (warm-start).
        None = solução trivial 0 -> 1 -> ... -> n-1 -> 0 (cold start).
    Retorna um dicionário com:
        status          "otimo", "tempo_esgotado" ou "memoria_esgotada"
        tour, custo     melhor solução conhecida (ótima se status == "otimo")
        ub_inicial      custo de tour_inicial
        lb_raiz         limite inferior do nó raiz
        contadores de nós, tempos e (opcional) pico de memória.
    Interrompida pelo limite de tempo, a busca preserva os contadores até o
    momento da interrupção.
    """
    D = np.asarray(D)
    n = D.shape[0]
    if tour_inicial is None:
        tour_inicial = solucao_trivial(D)
    tour_inicial = [int(v) for v in tour_inicial]
    assert sorted(tour_inicial) == list(range(n)), "tour_inicial não é permutação"
    ub0 = custo_tour(D, tour_inicial)

    if n <= 3:   # só existe um ciclo
        return dict(status="otimo", tour=tour_inicial, custo=ub0, ub_inicial=ub0,
                    lb_raiz=ub0, n=n)

    ligou_aqui = medir_memoria and not tracemalloc.is_tracing()
    if ligou_aqui:
        tracemalloc.start()
    if medir_memoria:
        base_mem, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()

    t_pre = time.perf_counter()
    busca = _Busca(D, ub0, tour_inicial, limite_s, max_fila)
    tempo_pre = time.perf_counter() - t_pre
    lb_raiz = (sum(busca.base) + 1) // 2

    status = "otimo"
    try:
        if estrategia == "profundidade":
            busca.profundidade()
        elif estrategia == "melhor":
            busca.melhor()
        else:
            raise ValueError(f"estratégia desconhecida: {estrategia}")
    except _Interrompido as e:
        status = e.status
    tempo_busca = time.perf_counter() - busca.t0

    mem_pico = None
    if medir_memoria:
        _, pico = tracemalloc.get_traced_memory()
        mem_pico = pico - base_mem
        if ligou_aqui:
            tracemalloc.stop()

    custo = custo_tour(D, busca.tour_ub)
    assert custo == busca.ub, "custo do melhor tour diverge do UB"
    return dict(
        status=status, tour=busca.tour_ub, custo=custo, n=n,
        estrategia=estrategia, ub_inicial=ub0, lb_raiz=lb_raiz,
        **busca.c,
        tempo_preprocessamento=tempo_pre,
        tempo_busca=tempo_busca,
        tempo_ate_melhor=busca.tempo_ate_melhor,
        nos_por_s=busca.c["expandidos"] / tempo_busca if tempo_busca > 0 else 0.0,
        mem_pico=mem_pico,
    )


# ---------------------------------------------------------------------------
# Verificações
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from itertools import permutations

    from utils import matriz_euc2d

    def otimo_forca_bruta(D):
        n = D.shape[0]
        return min(custo_tour(D, [0, *p]) for p in permutations(range(1, n)))

    # Exemplo do slide (ótimo = 37)
    nomes = "abcde"
    pesos = {("a", "b"): 4, ("a", "c"): 8, ("a", "d"): 9, ("a", "e"): 12,
             ("b", "c"): 6, ("b", "d"): 8, ("b", "e"): 9,
             ("c", "d"): 10, ("c", "e"): 11, ("d", "e"): 7}
    D = np.zeros((5, 5), dtype=np.int64)
    for (x, y), w in pesos.items():
        i, j = nomes.index(x), nomes.index(y)
        D[i, j] = D[j, i] = w
    for est in ("profundidade", "melhor"):
        r = branch_and_bound(D, estrategia=est)
        print(f"Slide [{est}]: H = {'-'.join(nomes[v] for v in r['tour'] + [r['tour'][0]])}"
              f", custo {r['custo']} (UB inicial {r['ub_inicial']}, lb raiz {r['lb_raiz']}),"
              f" expandidos {r['expandidos']}, podados {r['podados_limite']}")

    # Instâncias aleatórias contra força bruta, nas duas estratégias,
    # com e sem warm-start (o próprio ótimo como UB inicial)
    rng = np.random.default_rng(0)
    casos = 0
    for n in range(4, 10):
        for _ in range(25):
            D = matriz_euc2d(rng.uniform(0, 1000, size=(n, 2)))
            opt = otimo_forca_bruta(D)
            for est in ("profundidade", "melhor"):
                r = branch_and_bound(D, estrategia=est)
                assert r["status"] == "otimo" and r["custo"] == opt, (n, est)
                assert r["lb_raiz"] <= opt
                assert sorted(r["tour"]) == list(range(n))
                r2 = branch_and_bound(D, tour_inicial=r["tour"], estrategia=est)
                assert r2["custo"] == opt and r2["melhorias_ub"] == 0
                casos += 1
    print(f"\n{casos} execuções aleatórias (n=4..9): ótimo sempre igual à força bruta")

    # Limite de tempo: a busca para e devolve o melhor tour conhecido
    D = matriz_euc2d(rng.uniform(0, 1000, size=(60, 2)))
    r = branch_and_bound(D, limite_s=1.0)
    print(f"\nn=60 com limite de 1 s: status={r['status']}, "
          f"UB {r['ub_inicial']} -> {r['custo']}, expandidos {r['expandidos']}, "
          f"{r['nos_por_s']:.0f} nós/s")
