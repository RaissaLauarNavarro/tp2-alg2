"""
Heurística de Inserção Baseada em Fecho Convexo (Convex Hull Insertion) para o TSP.

Definição do enunciado (estratégia 3 de warm-start):
  "Formar a sub-rota inicial com o fecho convexo e inserir iterativamente os
   vértices internos de forma gulosa."

Algoritmo (Stewart, 1977; Golden, Bodin, Doyle e Stewart, 1980):
  1. Calcule o fecho convexo dos pontos. Os vértices do fecho, na ordem em que
     aparecem no contorno, formam a sub-rota inicial. (Num tour euclidiano
     ótimo, os vértices do fecho aparecem exatamente nessa ordem.)
  2. Enquanto houver vértices fora da sub-rota, escolha de forma gulosa um
     vértice k e uma aresta (i, j) da sub-rota e insira k entre i e j.
     Critério padrão ("custo", inserção mais barata): o par (k, (i, j)) que
     minimiza o aumento de custo  d(i,k) + d(k,j) - d(i,j).
     Critério alternativo ("razao", o original de Stewart): para cada k, a
     aresta de menor aumento; entre os k, o de menor razão
     (d(i,k) + d(k,j)) / d(i,j).
  3. Retorne a sub-rota, que agora contém todos os vértices.

O fecho é calculado aqui pela Varredura de Graham, como na aula 05 (DCC207,
slides 4 a 8; CLRS 33.3), em O(n log n); o SciPy só é usado no teste, para
conferir o resultado. A inserção avalia todos os pares
(vértice de fora, aresta) a cada passo, vetorizado em NumPy: O(n^2) por
inserção, O(n^3) no total. Os CUSTOS vêm sempre da matriz D real da instância;
as coordenadas servem só para escolher a sub-rota inicial.

Coordenadas usadas para o fecho (função pontos_para_fecho):
  "coords"   NODE_COORD_SECTION (EUC_2D, ATT, CEIL_2D)
  "geo"      GEO: graus.minutos (DDD.MM) convertidos para graus decimais;
             fecho no plano (latitude, longitude), aproximação plana.
  "display"  DISPLAY_DATA_SECTION (bayg29, bays29, dantzig42)
  "mds"      instâncias EXPLICIT sem coordenada nenhuma (gr17, gr21, gr24,
             fri26, swiss42, ...): pontos 2D obtidos por MDS clássico
             (Torgerson) da matriz D. Determinístico. A qualidade do
             mergulho é reportada pelo stress-1 de Kruskal.
"""

import time
import tracemalloc
from functools import cmp_to_key

import numpy as np

from Medidor import Medidor
from utils import custo_tour


# ---------------------------------------------------------------------------
# Coordenadas para o fecho
# ---------------------------------------------------------------------------
def geo_para_graus(c):
    """DDD.MM -> graus decimais (truncamento dos graus, como no leitor GEO)."""
    graus = np.trunc(c)
    return graus + (c - graus) * 100.0 / 60.0


def mds_classico(D, dim=2):
    """MDS clássico (Torgerson): B = -1/2 J D^2 J, autovetores dos `dim` maiores
    autovalores. Autovalores negativos (D não euclidiana) são zerados.
    Retorna (pontos n x dim, stress-1 de Kruskal)."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    autoval, autovet = np.linalg.eigh(B)            # ordem crescente
    idx = np.argsort(autoval)[::-1][:dim]
    L = np.sqrt(np.clip(autoval[idx], 0.0, None))
    X = autovet[:, idx] * L
    dif = X[:, None, :] - X[None, :, :]
    Dhat = np.sqrt((dif ** 2).sum(axis=2))
    iu = np.triu_indices(n, k=1)
    stress = float(np.sqrt(((D[iu] - Dhat[iu]) ** 2).sum() / (D[iu] ** 2).sum()))
    return X, stress


def pontos_para_fecho(inst):
    """Devolve (pontos n x 2, origem, stress) para uma Instancia do leitor_tsp.
    stress é None quando as coordenadas vêm do próprio arquivo."""
    if inst.coords is not None:
        if inst.tipo_peso == "GEO":
            return geo_para_graus(inst.coords), "geo", None
        return inst.coords, "coords", None
    if inst.coords_display is not None:
        return inst.coords_display, "display", None
    X, stress = mds_classico(inst.D)
    return X, "mds", stress


# ---------------------------------------------------------------------------
# Passo 1: fecho convexo (Varredura de Graham, aula 05, slides 4 a 8)
# ---------------------------------------------------------------------------
def _cruz(o, a, b):
    """Produto vetorial (a - o) x (b - o).
    > 0: o -> a -> b vira à esquerda; < 0: à direita; = 0: colineares."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def fecho_convexo(pontos):
    """Varredura de Graham (Graham, 1972; CLRS 33.3). Devolve os índices dos
    vértices do fecho convexo, no sentido anti-horário, começando pela âncora p0.

    Passos, como nos slides:
      1. Âncora p0: o ponto de menor coordenada y; em caso de empate, o mais à
         esquerda. Por ser o mais baixo, é com certeza vértice do fecho.
      2. Os demais pontos são ordenados pelo ângulo polar em relação a p0, no
         sentido anti-horário. Entre pontos com o mesmo ângulo, só o MAIS
         DISTANTE de p0 é mantido: os mais próximos estão sobre o segmento
         p0 -> mais distante, então não podem ser vértices do fecho.
      3. Pilha com p0, p1, p2. Para cada ponto seguinte pi: enquanto o caminho
         (penúltimo da pilha) -> (topo) -> pi NÃO virar à esquerda, o topo é
         vértice interno e sai da pilha. Depois, pi entra na pilha.
      4. A pilha contém os vértices do fecho.

    A ordenação usa o produto vetorial em vez de ângulos (atan2), o que é exato
    para coordenadas inteiras e não depende de ponto flutuante. Como todos os
    pontos estão acima de p0 (ou na mesma altura, à direita), os ângulos ficam
    em [0, pi) e o produto vetorial define uma ordem total.
    Custo: O(n log n) da ordenação + O(n) da varredura (cada ponto entra e sai
    da pilha no máximo uma vez).

    Suposições dos slides e como os dados reais são tratados:
      - "não há 3 pontos colineares": instâncias reais têm colineares. Os de
        mesmo ângulo em relação a p0 são resolvidos no passo 2; nos outros, o
        teste "não vira à esquerda" (cruz <= 0) desempilha o ponto do meio.
        Assim, pontos colineares sobre uma aresta do fecho NÃO entram nele;
        eles são inseridos depois, como os internos.
      - Pontos com coordenadas repetidas contam uma vez só (fica o de menor
        índice).
      - "P contém ao menos 3 pontos": se sobrarem menos de 3 pontos distintos,
        ou se todos forem colineares, devolve os 1 ou 2 pontos extremos."""
    P = [(float(x), float(y)) for x, y in np.asarray(pontos)[:, :2]]

    # pontos distintos (fica o de menor índice)
    vistos, idx = set(), []
    for i in range(len(P)):
        if P[i] not in vistos:
            vistos.add(P[i])
            idx.append(i)
    if len(idx) <= 2:
        return idx

    # 1. âncora: menor y, mais à esquerda no empate
    p0 = min(idx, key=lambda i: (P[i][1], P[i][0]))
    resto = [i for i in idx if i != p0]

    # 2. ordenação pelo ângulo polar em relação a p0 (anti-horário); no mesmo
    #    ângulo, do mais próximo para o mais distante
    def compara(i, j):
        c = _cruz(P[p0], P[i], P[j])
        if c > 0:            # j está à esquerda de p0 -> i: ângulo de i é menor
            return -1
        if c < 0:
            return 1
        return -1 if _dist2(P[p0], P[i]) < _dist2(P[p0], P[j]) else 1

    resto.sort(key=cmp_to_key(compara))

    # no mesmo ângulo, mantém só o mais distante (o último do grupo)
    ordenados = []
    for k, i in enumerate(resto):
        if k + 1 < len(resto) and _cruz(P[p0], P[i], P[resto[k + 1]]) == 0:
            continue
        ordenados.append(i)

    if len(ordenados) < 2:     # todos os pontos colineares com p0
        return [p0] + ordenados

    # 3. varredura com pilha
    pilha = [p0, ordenados[0], ordenados[1]]
    for i in ordenados[2:]:
        while len(pilha) >= 2 and _cruz(P[pilha[-2]], P[pilha[-1]], P[i]) <= 0:
            pilha.pop()
        pilha.append(i)

    # 4. a pilha é o fecho, em sentido anti-horário a partir de p0
    return pilha


# ---------------------------------------------------------------------------
# Passo 2: inserção gulosa dos vértices internos
# ---------------------------------------------------------------------------
def inserir_gulosamente(D, subrota, criterio="custo"):
    """Insere em `subrota` todos os vértices que faltam, um por vez.

    Em cada passo monta a matriz de aumentos  inc[k, e] = d(i,k) + d(k,j) - d(i,j)
    para todo vértice de fora k e toda aresta e = (i, j) da sub-rota atual.
    Desempate: menor k, depois a aresta que aparece primeiro na sub-rota
    (argmin devolve o primeiro mínimo). Retorna (tour, ordem_de_insercao)."""
    D = np.asarray(D)
    n = D.shape[0]
    tour = list(subrota)
    dentro = np.zeros(n, dtype=bool)
    dentro[tour] = True
    ordem = []

    while len(tour) < n:
        t = np.asarray(tour)
        t2 = np.roll(t, -1)                       # aresta e liga t[e] -> t2[e]
        U = np.flatnonzero(~dentro)               # vértices de fora, crescente
        d_ik = D[np.ix_(U, t)]
        d_kj = D[np.ix_(U, t2)]
        d_ij = D[t, t2]
        inc = d_ik + d_kj - d_ij[None, :]

        if criterio == "custo":
            a, e = divmod(int(np.argmin(inc)), len(t))
        elif criterio == "razao":
            e_k = np.argmin(inc, axis=1)          # melhor aresta de cada k
            linhas = np.arange(len(U))
            num = d_ik[linhas, e_k] + d_kj[linhas, e_k]
            den = d_ij[e_k].astype(np.float64)
            razao = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.inf)
            a = int(np.argmin(razao))
            e = int(e_k[a])
        else:
            raise ValueError(f"critério desconhecido: {criterio}")

        k = int(U[a])
        tour.insert(e + 1, k)                     # entre t[e] e t2[e]
        dentro[k] = True
        ordem.append(k)
    return tour, ordem


# ---------------------------------------------------------------------------
# Algoritmo completo (mesma interface do christofides / twice-around-the-tree)
# ---------------------------------------------------------------------------
def fecho_convexo_insercao(D, pontos, criterio="custo", medir_memoria=False):
    """Retorna (tour, custo, stats).

    D: matriz de distâncias real (os custos vêm daqui).
    pontos: coordenadas n x 2 usadas só para o fecho (ver pontos_para_fecho).
    medir_memoria=True liga o tracemalloc; meça tempo e memória em execuções
    SEPARADAS."""
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
        with med.etapa("fecho"):
            H = fecho_convexo(pontos)
        with med.etapa("insercao"):
            tour, _ = inserir_gulosamente(D, H, criterio)
    finally:
        if ligou_aqui:
            tracemalloc.stop()

    custo = custo_tour(D, tour)
    med.stats.update({
        "tempo_total": time.perf_counter() - t_total,
        "n": n,
        "n_fecho": len(H),
        "proporcao_fecho": len(H) / n,   # também é feature da análise de sensibilidade
        "custo_subrota_fecho": custo_tour(D, H) if len(H) > 1 else 0,
        "custo_tour": custo,
    })
    return tour, custo, med.stats


def tour_fecho_convexo(inst, criterio="custo"):
    """Atalho para o warm-start do B&B: recebe a Instancia do leitor_tsp."""
    pts, _, _ = pontos_para_fecho(inst)
    tour, _ = inserir_gulosamente(inst.D, fecho_convexo(pts), criterio)
    return tour


# ---------------------------------------------------------------------------
# Verificações
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from itertools import permutations

    from scipy.spatial import ConvexHull

    from utils import matriz_euc2d

    rng = np.random.default_rng(0)

    # (a) fecho igual ao do SciPy (Qhull), em pontos gerais e com colineares/repetidos
    for _ in range(300):
        m = int(rng.integers(4, 60))
        if rng.random() < 0.5:
            P = rng.uniform(0, 1000, size=(m, 2))
        else:
            P = rng.integers(0, 6, size=(m, 2)).astype(float)   # grade: colineares e repetidos
        try:
            ref = ConvexHull(P)
        except Exception:
            continue                                            # degenerado para o Qhull
        meu = fecho_convexo(P)
        assert {tuple(P[i]) for i in meu} == {tuple(P[i]) for i in ref.vertices}
        area = 0.5 * sum(_cruz(P[meu[0]], P[meu[i]], P[meu[i + 1]])
                         for i in range(1, len(meu) - 1))
        assert abs(area - ref.volume) < 1e-6 and area > 0       # anti-horário
    print("Fecho convexo: igual ao do SciPy em 300 conjuntos (incl. colineares e repetidos)")

    # casos fora das suposições do slide (colineares, repetidos, menos de 3 pontos)
    assert fecho_convexo(np.array([[0, 0], [1, 1], [2, 2], [3, 3]])) == [0, 3]
    assert fecho_convexo(np.array([[5, 5], [5, 5], [5, 5]])) == [0]
    quadrado = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1], [1, 1]])
    assert fecho_convexo(quadrado) == [0, 2, 4, 6]   # pontos sobre as arestas ficam de fora
    print("Casos degenerados (todos colineares, todos iguais, pontos sobre as arestas): ok")

    # (b) inserção vetorizada igual a uma versão de referência em Python puro
    def insercao_referencia(D, sub):
        tour, fora = list(sub), sorted(set(range(len(D))) - set(sub))
        while fora:
            best = None
            for k in fora:
                for e in range(len(tour)):
                    i, j = tour[e], tour[(e + 1) % len(tour)]
                    inc = D[i][k] + D[k][j] - D[i][j]
                    if best is None or inc < best[0]:
                        best = (inc, k, e)
            _, k, e = best
            tour.insert(e + 1, k)
            fora.remove(k)
        return tour

    def otimo_forca_bruta(D):
        n = D.shape[0]
        return min(custo_tour(D, [0, *p]) for p in permutations(range(1, n)))

    pior = {"custo": 1.0, "razao": 1.0}
    for n in range(4, 10):
        for _ in range(30):
            P = rng.uniform(0, 1000, size=(n, 2))
            D = matriz_euc2d(P)
            tour, custo, st = fecho_convexo_insercao(D, P)
            assert tour == insercao_referencia(D.tolist(), fecho_convexo(P))
            assert sorted(tour) == list(range(n))
            opt = otimo_forca_bruta(D)
            pior["custo"] = max(pior["custo"], custo / opt)
            t2, c2, _ = fecho_convexo_insercao(D, P, criterio="razao")
            assert sorted(t2) == list(range(n))
            pior["razao"] = max(pior["razao"], c2 / opt)
    print(f"180 instâncias aleatórias (n=4..9): inserção igual à referência; pior razão "
          f"custo/ótimo = {pior['custo']:.4f} (custo), {pior['razao']:.4f} (razão)")

    # (c) MDS clássico recupera pontos euclidianos (stress ~ 0)
    P = rng.uniform(0, 1000, size=(30, 2))
    dif = P[:, None] - P[None, :]
    _, s = mds_classico(np.sqrt((dif ** 2).sum(axis=2)))
    print(f"MDS clássico em distâncias euclidianas exatas: stress = {s:.2e}")

    # (d) instâncias da pasta data/
    from leitor_tsp import ler_instancia, listar_instancias
    pasta = sys.argv[1] if len(sys.argv) > 1 else "../data"
    try:
        caminhos = listar_instancias(pasta)
    except FileNotFoundError:
        caminhos = []
    if caminhos:
        print(f"\n{'instância':<11}{'n':>4} {'coords':<8}{'stress':>7}{'|fecho|':>8}"
              f"{'ótimo':>9}{'custo':>9}{'gap %':>8}{'razão':>9}{'gap %':>8}{'tempo':>10}")
    for c in caminhos:
        try:
            inst = ler_instancia(c)
        except (ValueError, KeyError):
            continue
        pts, origem, stress = pontos_para_fecho(inst)
        tour, custo, st = fecho_convexo_insercao(inst.D, pts)
        _, c_raz, _ = fecho_convexo_insercao(inst.D, pts, criterio="razao")
        opt = inst.otimo
        g = lambda x: f"{100 * (x - opt) / opt:.2f}" if opt else "NA"
        print(f"{inst.nome:<11}{inst.n:>4} {origem:<8}"
              f"{(f'{stress:.3f}' if stress is not None else '-'):>7}{st['n_fecho']:>8}"
              f"{str(opt):>9}{custo:>9}{g(custo):>8}{c_raz:>9}{g(c_raz):>8}"
              f"{st['tempo_total'] * 1e3:>8.2f}ms")