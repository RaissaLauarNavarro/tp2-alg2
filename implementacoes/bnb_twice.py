"""
Roda o Branch-and-Bound em todas as instâncias TSPLIB da pasta data/ e salva as
métricas num CSV (parte da "base de dados com as métricas coletadas").

Uso (a partir da pasta implementacoes/; a pasta data/ fica em ../data):
    python rodar_teste_bnb.py                                # cold start, DFS, 30 min
    python rodar_teste_bnb.py --estrategia melhor            # best-first
    python rodar_teste_bnb.py --nmax 40 --limite 1800
    python rodar_teste_bnb.py --continuar                    # retoma um CSV parcial
    python rodar_teste_bnb.py outra_pasta/ --saida outro.csv

Inicialização do limite superior (--inicial):
    trivial       1 -> 2 -> ... -> n -> 1 (cold start)
    vizinho       Vizinho Mais Próximo a partir do vértice 1 (busca gulosa)
    fecho         inserção mais barata a partir do fecho convexo
    twice         Twice-around-the-tree
    christofides  Christofides
Para adicionar outra estratégia, registre no dicionário INICIALIZACOES.
O tempo da heurística é medido (tempo_inicializacao_s) e descontado do limite.

Cada instância pode levar até --limite segundos (padrão 1800 = 30 min). Quem
estoura o limite fica com status "tempo_esgotado", custo_otimo e tempo_total_s
marcados como NA, mas mantém os contadores de nós até a interrupção e o melhor
custo encontrado. O CSV é gravado linha a linha: se o processo cair no meio,
rode de novo com --continuar e as instâncias já feitas são puladas.

--memoria liga o tracemalloc durante a busca. Ele deixa o B&B bem mais lento
(e portanto distorce tempos e contagens de nós sob limite de tempo); use numa
rodada separada.
"""

import argparse
import csv
import json
import os
import sys
import time

from bnb_trivial import branch_and_bound, solucao_trivial
from christofides import christofides
from bnb_fechoc import tour_fecho_convexo
from leitor_tsp import ler_instancia, listar_instancias, custo_tour
from twice_around import twice_around_the_tree
from bnb_guloso import tour_vizinho_mais_proximo

# nome -> função Instancia -> tour. Para adicionar um warm-start, registre aqui.
# Recebe a instância inteira (e não só D) porque o fecho convexo usa coordenadas.
INICIALIZACOES = {
    "trivial":      lambda inst: solucao_trivial(inst.D),                 # 1. cold start
    "vizinho":      lambda inst: tour_vizinho_mais_proximo(inst.D, 0),    # 2. busca gulosa
    "fecho":        lambda inst: tour_fecho_convexo(inst),                # 3. fecho convexo
    "twice":        lambda inst: twice_around_the_tree(inst.D)[0],        # 4. 2-aproximado
    "christofides": lambda inst: christofides(inst.D)[0],                 # 5. 1,5-aproximado
}

CAMPOS = [
    "instancia", "n", "tipo_peso", "inicial", "estrategia", "limite_s",
    "status", "otimo_publicado", "custo_otimo", "melhor_custo",
    "confere_otimo_publicado", "gap_melhor_pct",
    "ub_inicial", "gap_ub_inicial_pct", "lb_raiz", "gap_lb_raiz_pct",
    "nos_gerados", "nos_expandidos", "nos_podados_limite", "nos_podados_simetria",
    "folhas", "melhorias_ub", "prof_max", "max_fila",
    "tempo_inicializacao_s", "tempo_preprocessamento_s", "tempo_busca_s",
    "tempo_total_s", "tempo_decorrido_s", "tempo_ate_melhor_s", "nos_por_s",
    "mem_pico_bytes", "trajetoria_ub",
]


def pct(a, b):
    return round(100 * (a - b) / b, 4)


def avaliar(inst, inicial, estrategia, limite_s, max_fila, medir_memoria):
    t0 = time.perf_counter()
    tour0 = INICIALIZACOES[inicial](inst)
    t_ini = time.perf_counter() - t0

    # o tempo da heurística conta dentro do limite de 30 min da execução
    r = branch_and_bound(inst.D, tour_inicial=tour0, estrategia=estrategia,
                         limite_s=max(limite_s - t_ini, 0.0), max_fila=max_fila,
                         medir_memoria=medir_memoria)

    assert sorted(r["tour"]) == list(range(inst.n)), f"{inst.nome}: tour inválido"
    assert r["custo"] == custo_tour(inst.D, r["tour"]), f"{inst.nome}: custo divergente"

    resolvido = r["status"] == "otimo"
    opt = inst.otimo
    decorrido = t_ini + r.get("tempo_preprocessamento", 0) + r.get("tempo_busca", 0)
    return {
        "instancia": inst.nome,
        "n": inst.n,
        "tipo_peso": inst.tipo_peso,
        "inicial": inicial,
        "estrategia": estrategia,
        "limite_s": limite_s,
        "status": r["status"],
        "otimo_publicado": opt if opt is not None else "NA",
        "custo_otimo": r["custo"] if resolvido else "NA",
        "melhor_custo": r["custo"],
        "confere_otimo_publicado": (r["custo"] == opt) if (resolvido and opt) else "NA",
        "gap_melhor_pct": pct(r["custo"], opt) if opt else "NA",
        "ub_inicial": r["ub_inicial"],
        "gap_ub_inicial_pct": pct(r["ub_inicial"], opt) if opt else "NA",
        "lb_raiz": r["lb_raiz"],
        # quanto o lb da raiz fica abaixo do ótimo: (ótimo - lb) / ótimo
        "gap_lb_raiz_pct": round(100 * (opt - r["lb_raiz"]) / opt, 4) if opt else "NA",
        "nos_gerados": r.get("gerados", 0),
        "nos_expandidos": r.get("expandidos", 0),
        "nos_podados_limite": r.get("podados_limite", 0),
        "nos_podados_simetria": r.get("podados_simetria", 0),
        "folhas": r.get("folhas", 0),
        "melhorias_ub": r.get("melhorias_ub", 0),
        "prof_max": r.get("prof_max", 0),
        "max_fila": r.get("max_fila", 0),
        "tempo_inicializacao_s": t_ini,
        "tempo_preprocessamento_s": r.get("tempo_preprocessamento", 0),
        "tempo_busca_s": r.get("tempo_busca", 0),
        "tempo_total_s": decorrido if resolvido else "NA",
        "tempo_decorrido_s": decorrido,
        "tempo_ate_melhor_s": r.get("tempo_ate_melhor", 0),
        "nos_por_s": round(r.get("nos_por_s", 0), 1),
        "mem_pico_bytes": r["mem_pico"] if r.get("mem_pico") is not None else "NA",
        # JSON: [[nós expandidos, segundos, custo], ...] a cada melhoria do UB
        "trajetoria_ub": json.dumps(r.get("trajetoria_ub", [])),
    }


def main():
    ap = argparse.ArgumentParser(description="Branch-and-Bound nas instâncias TSPLIB")
    ap.add_argument("pasta", nargs="?", default="../data",
                    help="pasta com os .tsp/.tsp.gz (padrão: ../data)")
    ap.add_argument("--inicial", choices=sorted(INICIALIZACOES), default="trivial")
    ap.add_argument("--estrategia", choices=["profundidade", "melhor"],
                    default="profundidade")
    ap.add_argument("--limite", type=float, default=1800.0,
                    help="segundos por instância (padrão: 1800 = 30 min)")
    ap.add_argument("--max-fila", type=int, default=2_000_000,
                    help="só best-first: nós na fila antes de abortar (~0,9 GB)")
    ap.add_argument("--nmin", type=int, default=0)
    ap.add_argument("--nmax", type=int, default=10**9)
    ap.add_argument("--so-euc2d", action="store_true")
    ap.add_argument("--memoria", action="store_true")
    ap.add_argument("--saida", default=None,
                    help="padrão: resultados_bnb_<inicial>_<estrategia>[_memoria].csv")
    ap.add_argument("--continuar", action="store_true",
                    help="acrescenta ao CSV existente, pulando instâncias já feitas")
    args = ap.parse_args()

    sufixo = "_memoria" if args.memoria else ""
    saida = args.saida or f"resultados_bnb_{args.inicial}_{args.estrategia}{sufixo}.csv"
    if not os.path.isdir(args.pasta):
        sys.exit(f"Pasta '{args.pasta}' não encontrada. Rode a partir de "
                 f"implementacoes/ ou passe o caminho: python rodar_teste_bnb.py caminho/")

    caminhos = listar_instancias(args.pasta, args.nmin, args.nmax)
    if not caminhos:
        sys.exit(f"Nenhum .tsp ou .tsp.gz em '{args.pasta}' com {args.nmin} <= n <= {args.nmax}.")

    feitas = set()
    if args.continuar and os.path.exists(saida):
        with open(saida, encoding="utf-8") as f:
            feitas = {r["instancia"] for r in csv.DictReader(f)}
        modo = "a"
    else:
        modo = "w"

    print(f"inicial={args.inicial}  estratégia={args.estrategia}  "
          f"limite={args.limite:.0f}s  -> {saida}")
    print(f"{'instância':<11}{'n':>4}  {'status':<17}{'custo':>9}{'ótimo':>9}"
          f"{'UB0':>9}{'expandidos':>13}{'podados':>13}{'tempo':>10}")

    puladas = []
    with open(saida, modo, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if modo == "w":
            w.writeheader()
        for caminho in caminhos:
            arq = os.path.basename(caminho)
            try:
                inst = ler_instancia(caminho)
            except (ValueError, KeyError) as e:
                puladas.append((arq, str(e)))
                continue
            if args.so_euc2d and inst.tipo_peso != "EUC_2D":
                continue
            if inst.nome in feitas:
                print(f"{inst.nome:<11}{inst.n:>4}  (já está no CSV)")
                continue

            r = avaliar(inst, args.inicial, args.estrategia, args.limite,
                        args.max_fila, args.memoria)
            w.writerow(r)
            f.flush()
            print(f"{r['instancia']:<11}{r['n']:>4}  {r['status']:<17}{r['melhor_custo']:>9}"
                  f"{str(r['otimo_publicado']):>9}{r['ub_inicial']:>9}"
                  f"{r['nos_expandidos']:>13}{r['nos_podados_limite']:>13}"
                  f"{r['tempo_decorrido_s']:>9.1f}s", flush=True)

    for arq, motivo in puladas:
        print(f"PULADA {arq}: {motivo}")
    print(f"\nResultados em {saida}")


if __name__ == "__main__":
    main()