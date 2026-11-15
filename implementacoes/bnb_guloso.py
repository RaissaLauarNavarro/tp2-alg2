"""
Heurística do Vizinho Mais Próximo (Nearest Neighbor), uma busca gulosa para o TSP.

Algoritmo (Levitin, seção 12.3):
  1. Comece num vértice arbitrário (aqui, o vértice 1 da TSPLIB = índice 0).
  2. Repita até visitar todos: vá do vértice atual para o vértice NÃO
     visitado mais próximo dele.
  3. Volte ao vértice inicial.

É guloso porque cada passo escolhe a menor aresta disponível a partir do
vértice atual, sem nunca rever uma escolha. Custo O(n^2): n passos, cada um
varrendo uma linha de D.

Sem garantia de aproximação: para qualquer constante r existem instâncias,
mesmo euclidianas, em que custo(NN) / ótimo > r (Rosenkrantz, Stearns e Lewis,
1977, mostram razão Θ(log n) no caso métrico). Por isso é usada aqui como
limite superior inicial (warm-start) do Branch-and-Bound, e não como algoritmo
aproximativo com garantia.

Desempate: entre vizinhos à mesma distância, o de menor índice. Isso torna
o resultado determinístico e reproduzível.

Entrada: matriz de distâncias D (numpy n x n), a mesma que o leitor_tsp.py
devolve. Nenhuma biblioteca de grafos é usada.
"""

import time
import tracemalloc

import numpy as np

from christofides import custo_tour, _Medidor


# ---------------------------------------------------------------------------
# Busca gulosa
# ---------------------------------------------------------------------------
def tour_vizinho_mais_proximo(D, inicio=0):
    """Devolve o tour (lista de vértices, 0-based) construído a partir de `inicio`.

    `chave` guarda a linha do vértice atual com +inf nos já visitados; o argmin
    devolve o primeiro índice mínimo, ou seja, o desempate pelo menor índice."""
    D = np.asarray(D)
    n = D.shape[0]
    visitado = np.zeros(n, dtype=bool)
    tour = [inicio]
    visitado[inicio] = True
    atual = inicio
    for _ in range(n - 1):
        chave = np.where(visitado, np.inf, D[atual])
        prox = int(np.argmin(chave))
        tour.append(prox)
        visitado[prox] = True
        atual = prox
    return tour


def melhor_inicio(D):
    """Variante multi-start: roda o NN a partir de cada um dos n vértices e fica
    com o melhor tour. Custo O(n^3). Útil para medir o quanto a escolha
    arbitrária do vértice inicial pesa no resultado."""
    D = np.asarray(D)
    melhor, melhor_custo, custos = None, None, []
    for s in range(D.shape[0]):
        t = tour_vizinho_mais_proximo(D, s)
        c = custo_tour(D, t)
        custos.append(c)
        if melhor_custo is None or c < melhor_custo:
            melhor, melhor_custo = t, c
    return melhor, melhor_custo, custos


# ---------------------------------------------------------------------------
# Versão instrumentada (mesmo formato do christofides / twice-around-the-tree)
# ---------------------------------------------------------------------------
def vizinho_mais_proximo(D, inicio=0, medir_memoria=False):
    """Retorna (tour, custo, stats). medir_memoria=True liga o tracemalloc;
    meça tempo e memória em execuções SEPARADAS."""
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
        with med.etapa("guloso"):
            tour = tour_vizinho_mais_proximo(D, inicio)
    finally:
        if ligou_aqui:
            tracemalloc.stop()

    custo = custo_tour(D, tour)
    t = np.asarray(tour)
    arestas = D[t, np.roll(t, -1)]
    med.stats.update({
        "tempo_total": time.perf_counter() - t_total,
        "n": n,
        "custo_tour": custo,
        # a última aresta (volta ao início) não é escolhida pelo guloso; em
        # geral é a mais longa do tour e explica boa parte do gap
        "aresta_retorno": arestas[-1].item(),
        "maior_aresta": int(arestas.max()),
    })
    return tour, custo, med.stats


# ---------------------------------------------------------------------------
# Verificações
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from itertools import permutations

    from christofides import matriz_euc2d

    # Exemplo do slide (mesmo grafo dos outros algoritmos), a..e -> 0..4
    nomes = "abcde"
    pesos = {("a", "b"): 4, ("a", "c"): 8, ("a", "d"): 9, ("a", "e"): 12,
             ("b", "c"): 6, ("b", "d"): 8, ("b", "e"): 9,
             ("c", "d"): 10, ("c", "e"): 11, ("d", "e"): 7}
    D = np.zeros((5, 5), dtype=np.int64)
    for (x, y), w in pesos.items():
        i, j = nomes.index(x), nomes.index(y)
        D[i, j] = D[j, i] = w
    tour, custo, st = vizinho_mais_proximo(D)
    print("Exemplo do slide (ótimo 37)")
    print("  H  =", "-".join(nomes[v] for v in tour + [tour[0]]), "custo", custo,
          "| aresta de retorno", st["aresta_retorno"])

    # Conferência contra uma implementação independente, em Python puro
    def nn_referencia(D, s):
        n = len(D)
        tour, livres = [s], set(range(n)) - {s}
        while livres:
            u = tour[-1]
            v = min(livres, key=lambda x: (D[u][x], x))
            tour.append(v)
            livres.remove(v)
        return tour

    def otimo_forca_bruta(D):
        n = D.shape[0]
        return min(custo_tour(D, [0, *p]) for p in permutations(range(1, n)))

    rng = np.random.default_rng(0)
    pior = 1.0
    for n in range(4, 10):
        for _ in range(30):
            D = matriz_euc2d(rng.uniform(0, 100, size=(n, 2)))  # muitos empates
            s = int(rng.integers(n))
            tour, custo, _ = vizinho_mais_proximo(D, inicio=s)
            assert tour == nn_referencia(D.tolist(), s)
            assert sorted(tour) == list(range(n)) and tour[0] == s
            pior = max(pior, custo / otimo_forca_bruta(D))
    print(f"\n180 instâncias aleatórias (n=4..9, início sorteado): igual à referência; "
          f"pior razão custo/ótimo = {pior:.4f}")

    # Instâncias da pasta data/ (se existir): início no vértice 1 e multi-start
    from leitor_tsp import ler_instancia, listar_instancias
    pasta = sys.argv[1] if len(sys.argv) > 1 else "../data"
    try:
        caminhos = listar_instancias(pasta)
    except FileNotFoundError:
        caminhos = []
    if caminhos:
        print(f"\n{'instância':<11}{'n':>4}{'ótimo':>9}{'NN(1)':>9}{'gap %':>8}"
              f"{'melhor início':>15}{'gap %':>8}{'pior início':>13}{'tempo':>10}")
    for c in caminhos:
        try:
            inst = ler_instancia(c)
        except (ValueError, KeyError):
            continue
        tour, custo, st = vizinho_mais_proximo(inst.D)
        _, c_melhor, custos = melhor_inicio(inst.D)
        opt = inst.otimo
        g = lambda x: f"{100 * (x - opt) / opt:.2f}" if opt else "NA"
        print(f"{inst.nome:<11}{inst.n:>4}{str(opt):>9}{custo:>9}{g(custo):>8}"
              f"{c_melhor:>15}{g(c_melhor):>8}{max(custos):>13}"
              f"{st['tempo_total'] * 1e3:>8.3f}ms")