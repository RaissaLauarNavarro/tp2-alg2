"""
Branch-and-Bound exato para o TSP simétrico com lower bound baseado em MST.

Árvore de busca:
  Cada nó representa um caminho parcial que começa no vértice 0.
  A ramificação estende o caminho por um vértice ainda não visitado.
  Quando todos os vértices são visitados, fecha-se o ciclo voltando ao 0.

Lower bound:
  Para um caminho parcial P = (0, ..., u), seja U o conjunto dos vértices
  ainda não visitados. Qualquer conclusão do caminho precisa:
    - conectar u a U;
    - conectar todos os vértices de U;
    - conectar U ao vértice inicial 0.

  Usamos:
    LB(P) = custo(P) + MST(U)
            + min_{x em U} d(u, x)
            + min_{y em U} d(0, y)

  Na raiz, ainda não há extremidade final distinta da origem. Portanto:
    LB(raiz) = MST(V \\ {0}) + duas menores arestas distintas saindo de 0.

  O limite é válido, embora as duas conexões mínimas do caso geral possam
  escolher o mesmo vértice: isso apenas pode subestimar o custo real.

MST:
  Prim com vetor de chaves, executado diretamente sobre a lista de listas
  de distâncias. Para m vértices, custa O(m²).

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
import time
import tracemalloc

import numpy as np

from solucao_trivial import solucao_trivial
from utils import custo_tour


class _Interrompido(Exception):
    def __init__(self, status):
        self.status = status


# ---------------------------------------------------------------------------
# MST e lower bound
# ---------------------------------------------------------------------------

def mst_custo(Dl, vertices):
    """Retorna o custo da MST induzida por `vertices`, usando Prim O(m²).

    Dl é a matriz de distâncias no formato lista de listas.
    """
    m = len(vertices)

    if m <= 1:
        return 0

    na_arvore = [False] * m
    chave = [float("inf")] * m
    chave[0] = 0
    custo = 0

    for _ in range(m):
        # Vértice fora da árvore com menor chave.
        u = -1
        menor = float("inf")
        for i in range(m):
            if not na_arvore[i] and chave[i] < menor:
                menor = chave[i]
                u = i

        na_arvore[u] = True
        custo += chave[u]

        vu = vertices[u]

        # Atualiza as chaves usando as distâncias a partir de vu.
        for i in range(m):
            if not na_arvore[i]:
                vi = vertices[i]
                w = Dl[vu][vi]
                if w < chave[i]:
                    chave[i] = w

    return custo


def limite_inferior_raiz(D):
    """Calcula o lower bound da raiz."""
    Dl = np.asarray(D).tolist()
    n = len(Dl)

    if n <= 1:
        return 0

    U = list(range(1, n))
    mst = mst_custo(Dl, U)

    duas_menores = sorted(Dl[0][v] for v in U)[:2]
    return mst + sum(duas_menores)


# ---------------------------------------------------------------------------
# Busca
# ---------------------------------------------------------------------------

class _Busca:
    def __init__(self, D, ub, tour_ub, limite_s, max_fila):
        self.n = D.shape[0]
        self.Dl = D.tolist()

        self.ub = ub
        self.tour_ub = list(tour_ub)
        self.limite_s = limite_s
        self.max_fila = max_fila

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
        """MST dos vértices 1..n-1 + duas conexões distintas à origem."""
        U = list(range(1, self.n))
        mst = mst_custo(self.Dl, U)
        duas_menores = sorted(self.Dl[0][v] for v in U)[:2]
        return mst + sum(duas_menores)

    def _lower_bound(self, custo, ultimo, U):
        """LB para P=(0,...,ultimo), com U ainda não visitado."""
        Dl = self.Dl

        mst = mst_custo(Dl, U)
        conecta_fim = min(Dl[ultimo][x] for x in U)
        conecta_inicio = min(Dl[0][x] for x in U)

        return custo + mst + conecta_fim + conecta_inicio

    def _filhos(self, caminho_ultimos, k, custo, cont, p1,
                visitado, usar_mask=False):
        """Gera os filhos sobreviventes à poda.

        Retorna tuplas (lb, v, custo2, cont2, p1v).
        Folhas são fechadas aqui e podem atualizar o limite superior.
        """
        n, Dl, c = self.n, self.Dl, self.c
        p, u = caminho_ultimos

        saida = []
        restantes = n - k - 1

        # Vértices não visitados antes de adicionar o próximo vértice.
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

            # U após adicionar v ao caminho.
            U = [x for x in nao_visitados if x != v]

            lb = self._lower_bound(custo2, v, U)

            if lb >= self.ub:
                c["podados_limite"] += 1
                continue

            saida.append((lb, v, custo2, cont2, p1v))

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

    def profundidade(self):
        self.visitado = [False] * self.n
        self.visitado[0] = True

        self.caminho = [0]
        self._caminho_atual = lambda: list(self.caminho)

        self._dfs(0, 0, 0)

    def _dfs(self, custo, cont, p1):
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
            ultimos, k, custo, cont, p1, visitado, usar_mask=False
        )
        filhos.sort()  # menor LB primeiro

        for lb, v, custo2, cont2, p1v in filhos:
            # O UB pode ter melhorado durante a exploração de outro filho.
            if lb >= self.ub:
                c["podados_limite"] += 1
                continue

            visitado[v] = True
            caminho.append(v)

            self._dfs(custo2, cont2, p1v)

            caminho.pop()
            visitado[v] = False

    # ---------------- melhor primeiro ----------------

    def melhor(self):
        n, c = self.n, self.c

        if n > 256:
            raise ValueError(
                "Estratégia 'melhor' compacta vértices em 1 byte "
                "(máximo n=256)."
            )

        # Banco plano de estados indexado por slot_id.
        arr_custo = array("Q")
        arr_cont = array("H")
        arr_p1 = array("H")
        arr_k = array("H")
        arr_mask = []

        buf_caminhos = bytearray()
        livres = []

        def alocar_no(k, custo, cont, p1, mask, pai_id, novo_v):
            if livres:
                sid = livres.pop()
                arr_custo[sid] = custo
                arr_cont[sid] = cont
                arr_p1[sid] = p1
                arr_k[sid] = k
                arr_mask[sid] = mask
            else:
                sid = len(arr_custo)
                arr_custo.append(custo)
                arr_cont.append(cont)
                arr_p1.append(p1)
                arr_k.append(k)
                arr_mask.append(mask)
                buf_caminhos.extend(b"\x00" * n)

            base_dst = sid * n

            if pai_id >= 0:
                base_src = pai_id * n
                buf_caminhos[base_dst:base_dst + k - 1] = (
                    buf_caminhos[base_src:base_src + k - 1]
                )

            buf_caminhos[base_dst + k - 1] = novo_v
            return sid

        # Chave: menor LB primeiro; em empate, maior profundidade primeiro.
        MASK_ID = (1 << 32) - 1

        def empacotar_chave(lb, k, sid):
            return (lb << 48) | ((65535 - k) << 32) | sid

        lb0 = self._lower_bound_raiz()
        sid0 = alocar_no(1, 0, 0, 0, 1, -1, 0)
        fila = [empacotar_chave(lb0, 1, sid0)]

        atual_sid = sid0
        atual_k = 1

        self._caminho_atual = lambda: list(
            buf_caminhos[atual_sid * n:atual_sid * n + atual_k]
        )

        # Purga nós obsoletos da heap quando o UB melhora e a fila é grande.
        def purgar_fila():
            if len(fila) < 500_000:
                return

            limite_chave = self.ub << 48
            nova_fila = []
            podados = 0

            for ch in fila:
                if ch >= limite_chave:
                    livres.append(ch & MASK_ID)
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
                livres.append(sid)
                continue

            k = arr_k[sid]
            custo = arr_custo[sid]
            cont = arr_cont[sid]
            p1 = arr_p1[sid]
            mask = arr_mask[sid]

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
                ultimos, k, custo, cont, p1, mask, usar_mask=True
            )

            for lb2, v, custo2, cont2, p1v in filhos:
                filho_sid = alocar_no(
                    k + 1,
                    custo2,
                    cont2,
                    p1v,
                    mask | (1 << v),
                    sid,
                    v,
                )
                heapq.heappush(
                    fila,
                    empacotar_chave(lb2, k + 1, filho_sid),
                )

            # Libera o slot do nó pai depois de gerar todos os filhos.
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
):
    """Resolve o TSP simétrico por Branch-and-Bound.

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
    busca = _Busca(D, ub0, tour_inicial, limite_s, max_fila)
    lb_raiz = busca._lower_bound_raiz()
    tempo_pre = time.perf_counter() - t_pre

    status = "otimo"

    try:
        if estrategia == "profundidade":
            busca.profundidade()
        elif estrategia == "melhor":
            busca.melhor()
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
