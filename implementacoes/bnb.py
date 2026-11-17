"""
Branch-and-Bound exato para o TSP simétrico com relaxação de Held-Karp (1-tree).

Árvore de busca:
  Cada nó representa um caminho parcial que começa no vértice 0.
  A ramificação estende o caminho por um vértice ainda não visitado.
  Quando todos os vértices são visitados, fecha-se o ciclo voltando ao 0.

Lower bound (Relaxação de Held-Karp):
  Atribuímos penalidades pi_v a cada vértice v, definindo pesos modificados:
    w_pi(a, b) = d(a, b) + pi_a + pi_b

  Para um caminho parcial P = (0, ..., u) com vértices não visitados U e
  W = {u} U U:
    - Os vértices internos de P já possuem grau fixo 2;
    - A 1-tree condicionada conecta 0 a P, forma uma MST penalizada em W e
      liga 0 ao vértice y em U que minimiza d(0, y) + pi_y.

  O limite inferior lagrangiano é:
    L(pi) = custo(P) + pi_u + MST_pi(W) + min_{y em U}(d(0, y) + pi_y)
            - 2 * sum_{v em W} pi_v

  Os multiplicadores pi são ajustados via ascensão de subgradiente usando
  g_v = grau_1tree(v) - 2 e herdados (warm-start) pelos nós filhos.
  Como D possui distâncias inteiras, ceil(L(pi) - 1e-9) é um LB inteiro válido.

Poda:
  Um nó é podado quando LB >= melhor custo conhecido (UB).

Quebra de simetria:
  Cada ciclo é explorado em apenas um dos sentidos, impondo que o último
  vértice seja maior que o primeiro vértice depois da origem.

Estratégias:
  "profundidade": DFS recursiva, visitando primeiro os filhos com menor LB.
  "melhor": best-first com heap ordenada pelo LB e desempate por profundidade.

Entrada:
  Matriz de distâncias D (numpy n x n), simétrica e com distâncias inteiras.
"""

from array import array
import heapq
import math
import time
import tracemalloc

import numpy as np

from solucao_trivial import solucao_trivial
from utils import custo_tour


class _Interrompido(Exception):
    def __init__(self, status):
        self.status = status


# ---------------------------------------------------------------------------
# MST penalizada e relaxação de Held-Karp
# ---------------------------------------------------------------------------

def _mst_penalizada(Dl, vertices, pi):
    """MST via Prim O(m²) sobre w_pi(u, v) = Dl[u][v] + pi[u] + pi[v].

    Calcula os pesos penalizados sob demanda para evitar alocar matrizes n x n.
    Retorna (custo_mst_penalizado, graus_na_mst).
    """
    m = len(vertices)
    graus = [0] * m

    if m <= 1:
        return 0.0, graus

    na_arvore = [False] * m
    chave = [float("inf")] * m
    pai = [-1] * m
    chave[0] = 0.0
    custo = 0.0

    for _ in range(m):
        u = -1
        menor = float("inf")
        for i in range(m):
            if not na_arvore[i] and chave[i] < menor:
                menor = chave[i]
                u = i

        na_arvore[u] = True
        custo += chave[u]

        p = pai[u]
        if p != -1:
            graus[u] += 1
            graus[p] += 1

        vu = vertices[u]
        pi_vu = pi[vu]
        linha_vu = Dl[vu]

        for i in range(m):
            if not na_arvore[i]:
                vi = vertices[i]
                w = linha_vu[vi] + pi_vu + pi[vi]
                if w < chave[i]:
                    chave[i] = w
                    pai[i] = u

    return custo, graus


def _held_karp_raiz(Dl, ub, iters=100):
    """Calcula o lower bound de Held-Karp na raiz e retorna (lb_int, pi)."""
    n = len(Dl)
    pi = [0.0] * n

    if n <= 1:
        return 0, pi

    U = list(range(1, n))
    m = len(U)
    linha0 = Dl[0]

    melhor_lb = float("-inf")
    lam = 2.0
    sem_melhora = 0

    for _ in range(iters):
        custo_mst, graus = _mst_penalizada(Dl, U, pi)

        # Duas menores arestas saindo da origem 0 sob w_pi(0, v) = d(0, v) + pi_0 + pi_v.
        m1 = m2 = float("inf")
        idx1 = idx2 = -1
        for i, v in enumerate(U):
            w = linha0[v] + pi[v]
            if w < m1:
                m2, idx2 = m1, idx1
                m1, idx1 = w, i
            elif w < m2:
                m2, idx2 = w, i

        graus[idx1] += 1
        graus[idx2] += 1

        soma_pi_U = sum(pi[v] for v in U)
        lb = custo_mst + m1 + m2 - 2.0 * soma_pi_U

        if lb > melhor_lb + 1e-9:
            melhor_lb = lb
            sem_melhora = 0
        else:
            sem_melhora += 1
            if sem_melhora == 5:
                lam *= 0.5
                sem_melhora = 0

        lb_int = max(0, int(math.ceil(melhor_lb - 1e-9)))
        if lb_int >= ub:
            return lb_int, pi

        # Subgradiente: g_v = grau(v) - 2
        norm_sq = 0
        for g_idx in range(m):
            g = graus[g_idx] - 2
            norm_sq += g * g

        if norm_sq == 0:
            # Todos os vértices têm grau 2: a 1-tree já é um ciclo hamiltoniano.
            break

        gap = max(0.0, ub - lb)
        passo = lam * gap / norm_sq
        for i, v in enumerate(U):
            g = graus[i] - 2
            if g != 0:
                pi[v] += passo * g

    return max(0, int(math.ceil(melhor_lb - 1e-9))), pi


def limite_inferior_raiz(D, ub=None, iters=100):
    """Calcula o lower bound da raiz via Held-Karp."""
    D = np.asarray(D)
    Dl = D.tolist()
    n = len(Dl)

    if n <= 1:
        return 0

    if ub is None:
        ub = custo_tour(D, solucao_trivial(D))

    lb_int, _ = _held_karp_raiz(Dl, ub, iters=iters)
    return lb_int


# ---------------------------------------------------------------------------
# Busca
# ---------------------------------------------------------------------------

class _Busca:
    def __init__(self, D, ub, tour_ub, limite_s, max_fila,
                 iters_raiz=100, iters_no=15):
        self.n = D.shape[0]
        self.Dl = D.tolist()

        self.ub = ub
        self.tour_ub = list(tour_ub)
        self.limite_s = limite_s
        self.max_fila = max_fila
        self.iters_raiz = iters_raiz
        self.iters_no = iters_no

        self.t0 = time.perf_counter()
        self.prazo = self.t0 + limite_s

        self.c = dict(
            gerados=1,
            expandidos=0,
            podados_limite=0,
            podados_simetria=0,
            folhas=0,
            melhorias_ub=0,
            prof_max=1,
            max_fila=1,
        )

        self.tempo_ate_melhor = 0.0
        self._ao_melhorar_ub = None

    def _lower_bound_raiz(self):
        """Held-Karp na raiz: retorna (lb_int, pi_raiz)."""
        return _held_karp_raiz(self.Dl, self.ub, iters=self.iters_raiz)

    def _lower_bound(self, custo, ultimo, U, pi_pai):
        """LB de Held-Karp para P=(0,...,ultimo), com warm-start em pi_pai.

        Retorna (lb_int, pi_atualizado).
        """
        Dl = self.Dl
        pi = list(pi_pai)

        # Caso base: resta apenas 1 vértice em U; conclusão única e exata.
        if len(U) == 1:
            x = U[0]
            return custo + Dl[ultimo][x] + Dl[x][0], pi

        W = [ultimo] + U
        m = len(W)
        linha0 = Dl[0]

        melhor_lb = float("-inf")
        lam = 2.0
        sem_melhora = 0

        for _ in range(self.iters_no):
            custo_mst, graus = _mst_penalizada(Dl, W, pi)

            # Extremidade `ultimo` já possui 1 aresta fixa vinda do caminho P.
            graus[0] += 1

            # Conexão mínima da origem 0 a algum vértice y em U (= W[1:]).
            melhor_y = float("inf")
            idx_y = 1
            for i in range(1, m):
                v = W[i]
                w = linha0[v] + pi[v]
                if w < melhor_y:
                    melhor_y = w
                    idx_y = i

            graus[idx_y] += 1

            soma_pi_W = sum(pi[v] for v in W)
            lb = custo + pi[ultimo] + custo_mst + melhor_y - 2.0 * soma_pi_W

            if lb > melhor_lb + 1e-9:
                melhor_lb = lb
                sem_melhora = 0
            else:
                sem_melhora += 1
                if sem_melhora == 4:
                    lam *= 0.5
                    sem_melhora = 0

            lb_int = max(0, int(math.ceil(melhor_lb - 1e-9)))
            if lb_int >= self.ub:
                return lb_int, pi

            norm_sq = 0
            for i in range(m):
                g = graus[i] - 2
                norm_sq += g * g

            if norm_sq == 0:
                break

            gap = max(0.0, self.ub - lb)
            passo = lam * gap / norm_sq
            for i in range(m):
                g = graus[i] - 2
                if g != 0:
                    pi[W[i]] += passo * g

        return max(0, int(math.ceil(melhor_lb - 1e-9))), pi

    def _filhos(self, caminho_ultimos, k, custo, cont, p1,
                visitado, pi_pai, usar_mask=False):
        """Gera os filhos sobreviventes à poda.

        Retorna tuplas (lb, v, custo2, cont2, p1v, pi_filho).
        Folhas são fechadas aqui e podem atualizar o limite superior.
        """
        n, Dl, c = self.n, self.Dl, self.c
        _, u = caminho_ultimos

        saida = []
        restantes = n - k - 1

        if usar_mask:
            nao_visitados = [
                v for v in range(1, n)
                if not (visitado & (1 << v))
            ]
        else:
            nao_visitados = [
                v for v in range(1, n)
                if not visitado[v]
            ]

        for v in nao_visitados:
            c["gerados"] += 1
            w = Dl[u][v]

            # Quebra de simetria: o último vértice do ciclo deve ser
            # maior que o primeiro vértice escolhido depois da origem.
            if k == 1:
                cont2 = n - 1 - v
                p1v = v
            else:
                cont2 = cont - (v > p1)
                p1v = p1

            if restantes > 0 and cont2 == 0:
                c["podados_simetria"] += 1
                continue

            custo2 = custo + w

            # Se v completa o caminho, fecha o ciclo voltando à origem.
            if restantes == 0:
                c["folhas"] += 1
                total = custo2 + Dl[v][0]

                if total < self.ub:
                    self._novo_ub(total, v)

                continue

            U = [x for x in nao_visitados if x != v]
            lb, pi_filho = self._lower_bound(custo2, v, U, pi_pai)

            if lb >= self.ub:
                c["podados_limite"] += 1
                continue

            saida.append((lb, v, custo2, cont2, p1v, pi_filho))

        return saida

    def _novo_ub(self, total, ultimo):
        self.ub = total
        self.tour_ub = self._caminho_atual() + [ultimo]
        self.c["melhorias_ub"] += 1
        self.tempo_ate_melhor = time.perf_counter() - self.t0

        if self._ao_melhorar_ub is not None:
            self._ao_melhorar_ub()

    def _checar_tempo(self):
        if (self.c["expandidos"] & 1023) == 0:
            if time.perf_counter() > self.prazo:
                raise _Interrompido("tempo_esgotado")

    # ---------------- profundidade ----------------

    def profundidade(self, pi_raiz):
        self.visitado = [False] * self.n
        self.visitado[0] = True

        self.caminho = [0]
        self._caminho_atual = lambda: list(self.caminho)

        self._dfs(0, 0, 0, pi_raiz)

    def _dfs(self, custo, cont, p1, pi_atual):
        c = self.c
        caminho = self.caminho
        visitado = self.visitado

        k = len(caminho)
        c["expandidos"] += 1
        self._checar_tempo()

        if k > c["prof_max"]:
            c["prof_max"] = k

        ultimos = (
            caminho[-2] if k > 1 else -1,
            caminho[-1],
        )

        filhos = self._filhos(
            ultimos, k, custo, cont, p1, visitado, pi_atual, usar_mask=False
        )
        filhos.sort(key=lambda item: item[0])  # menor LB primeiro

        for lb, v, custo2, cont2, p1v, pi_filho in filhos:
            if lb >= self.ub:
                c["podados_limite"] += 1
                continue

            visitado[v] = True
            caminho.append(v)

            self._dfs(custo2, cont2, p1v, pi_filho)

            caminho.pop()
            visitado[v] = False

    # ---------------- melhor primeiro ----------------

    def melhor(self, lb0, pi_raiz):
        n, c = self.n, self.c

        if n > 256:
            raise ValueError(
                "Estratégia 'melhor' compacta vértices em 1 byte "
                "(máximo n=256)."
            )

        arr_custo = array("Q")
        arr_cont = array("H")
        arr_p1 = array("H")
        arr_k = array("H")
        arr_mask = []
        arr_pi = []

        buf_caminhos = bytearray()
        livres = []

        def alocar_no(k, custo, cont, p1, mask, pi_no, pai_id, novo_v):
            if livres:
                sid = livres.pop()
                arr_custo[sid] = custo
                arr_cont[sid] = cont
                arr_p1[sid] = p1
                arr_k[sid] = k
                arr_mask[sid] = mask
                arr_pi[sid] = pi_no
            else:
                sid = len(arr_custo)
                arr_custo.append(custo)
                arr_cont.append(cont)
                arr_p1.append(p1)
                arr_k.append(k)
                arr_mask.append(mask)
                arr_pi.append(pi_no)
                buf_caminhos.extend(b"\x00" * n)

            base_dst = sid * n

            if pai_id >= 0:
                base_src = pai_id * n
                buf_caminhos[base_dst:base_dst + k - 1] = (
                    buf_caminhos[base_src:base_src + k - 1]
                )

            buf_caminhos[base_dst + k - 1] = novo_v
            return sid

        MASK_ID = (1 << 32) - 1

        def empacotar_chave(lb, k, sid):
            return (lb << 48) | ((65535 - k) << 32) | sid

        sid0 = alocar_no(1, 0, 0, 0, 1, pi_raiz, -1, 0)
        fila = [empacotar_chave(lb0, 1, sid0)]

        atual_sid = sid0
        atual_k = 1

        self._caminho_atual = lambda: list(
            buf_caminhos[atual_sid * n:atual_sid * n + atual_k]
        )

        def purgar_fila():
            if len(fila) < 500_000:
                return

            limite_chave = self.ub << 48
            nova_fila = []
            podados = 0

            for ch in fila:
                if ch >= limite_chave:
                    sid_morto = ch & MASK_ID
                    arr_pi[sid_morto] = None
                    livres.append(sid_morto)
                    podados += 1
                else:
                    nova_fila.append(ch)

            if podados:
                c["podados_limite"] += podados
                fila[:] = nova_fila
                heapq.heapify(fila)

        self._ao_melhorar_ub = purgar_fila

        while fila:
            ch = heapq.heappop(fila)
            lb = ch >> 48
            sid = ch & MASK_ID

            if lb >= self.ub:
                c["podados_limite"] += 1
                arr_pi[sid] = None
                livres.append(sid)
                continue

            k = arr_k[sid]
            custo = arr_custo[sid]
            cont = arr_cont[sid]
            p1 = arr_p1[sid]
            mask = arr_mask[sid]
            pi_no = arr_pi[sid]

            c["expandidos"] += 1
            self._checar_tempo()

            if k > c["prof_max"]:
                c["prof_max"] = k

            atual_sid = sid
            atual_k = k

            base = sid * n
            ultimos = (
                buf_caminhos[base + k - 2] if k > 1 else -1,
                buf_caminhos[base + k - 1],
            )

            filhos = self._filhos(
                ultimos, k, custo, cont, p1, mask, pi_no, usar_mask=True
            )

            for lb2, v, custo2, cont2, p1v, pi_filho in filhos:
                filho_sid = alocar_no(
                    k + 1,
                    custo2,
                    cont2,
                    p1v,
                    mask | (1 << v),
                    pi_filho,
                    sid,
                    v,
                )
                heapq.heappush(
                    fila,
                    empacotar_chave(lb2, k + 1, filho_sid),
                )

            arr_pi[sid] = None
            livres.append(sid)

            if len(fila) > c["max_fila"]:
                c["max_fila"] = len(fila)
                if len(fila) > self.max_fila:
                    raise _Interrompido("memoria_esgotada")


# ---------------------------------------------------------------------------
# Interface pública
# ---------------------------------------------------------------------------

def branch_and_bound(
    D,
    tour_inicial=None,
    estrategia="profundidade",
    limite_s=1800.0,
    max_fila=2_000_000,
    medir_memoria=False,
    iters_raiz=100,
    iters_no=15,
):
    """Resolve o TSP simétrico por Branch-and-Bound com Held-Karp.

    Parâmetros
    ----------
    D : array-like
        Matriz n x n de distâncias simétrica e inteira.
    tour_inicial : sequência ou None
        Tour usado como limite superior inicial. Se None, usa solucao_trivial.
    estrategia : {"profundidade", "melhor"}
        Estratégia de busca.
    limite_s : float
        Limite de tempo em segundos.
    max_fila : int
        Número máximo de nós simultaneamente na fila da estratégia "melhor".
    medir_memoria : bool
        Se True, mede o pico de memória com tracemalloc.
    iters_raiz : int
        Iterações de subgradiente de Held-Karp na raiz.
    iters_no : int
        Iterações de subgradiente de Held-Karp por nó interno (warm-start).

    Retorno
    -------
    dict
        Status, melhor tour/custo, lower bound da raiz, contadores,
        tempos e, opcionalmente, pico de memória.
    """
    D = np.asarray(D)
    n = D.shape[0]

    if D.ndim != 2 or D.shape[1] != n:
        raise ValueError("D deve ser uma matriz quadrada.")

    if n < 1:
        raise ValueError("D deve conter pelo menos um vértice.")

    if tour_inicial is None:
        tour_inicial = solucao_trivial(D)

    tour_inicial = [int(v) for v in tour_inicial]

    assert sorted(tour_inicial) == list(range(n)), (
        "tour_inicial deve ser uma permutação dos vértices."
    )

    ub0 = custo_tour(D, tour_inicial)

    if n <= 3:
        return dict(
            status="otimo",
            tour=tour_inicial,
            custo=ub0,
            ub_inicial=ub0,
            lb_raiz=ub0,
            n=n,
        )

    ligou_aqui = medir_memoria and not tracemalloc.is_tracing()

    if ligou_aqui:
        tracemalloc.start()

    if medir_memoria:
        base_mem, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()

    t_pre = time.perf_counter()
    busca = _Busca(
        D, ub0, tour_inicial, limite_s, max_fila,
        iters_raiz=iters_raiz, iters_no=iters_no,
    )
    lb_raiz, pi_raiz = busca._lower_bound_raiz()
    tempo_pre = time.perf_counter() - t_pre

    status = "otimo"

    try:
        if lb_raiz < busca.ub:
            if estrategia == "profundidade":
                busca.profundidade(pi_raiz)
            elif estrategia == "melhor":
                busca.melhor(lb_raiz, pi_raiz)
            else:
                raise ValueError(f"Estratégia desconhecida: {estrategia}")
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

    assert custo == busca.ub, "Custo do melhor tour diverge do UB."

    return dict(
        status=status,
        tour=busca.tour_ub,
        custo=custo,
        n=n,
        estrategia=estrategia,
        ub_inicial=ub0,
        lb_raiz=lb_raiz,
        **busca.c,
        tempo_preprocessamento=tempo_pre,
        tempo_busca=tempo_busca,
        tempo_ate_melhor=busca.tempo_ate_melhor,
        nos_por_s=(
            busca.c["expandidos"] / tempo_busca
            if tempo_busca > 0 else 0.0
        ),
        mem_pico=mem_pico,
    )