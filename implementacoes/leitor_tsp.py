"""
Leitor de instâncias TSPLIB (problemas TSP simétricos).

Tipos de peso suportados (EDGE_WEIGHT_TYPE), seguindo as fórmulas do documento
oficial da TSPLIB95:
  EUC_2D   d = nint(sqrt(dx^2 + dy^2))
  CEIL_2D  d = ceil(sqrt(dx^2 + dy^2))
  ATT      distância pseudo-euclidiana (att48, att532)
  GEO      distância geodésica em km (ulysses16, ulysses22, burma14)
  EXPLICIT matriz dada no arquivo (gr24, fri26, bays29, swiss42, ...), em
           qualquer EDGE_WEIGHT_FORMAT (FULL_MATRIX, UPPER_ROW, LOWER_DIAG_ROW...)

Aceita arquivos .tsp e .tsp.gz (o site oficial distribui compactado).
Sempre devolve a matriz de distâncias D (numpy int64, n x n), que é a
entrada que o christofides.py (e os outros algoritmos) recebem.

Rodar este arquivo diretamente valida o leitor (pasta padrão: data/):
    python leitor_tsp.py
    python leitor_tsp.py outra_pasta/
Para cada X.opt.tour presente na pasta, calcula o custo do tour ótimo com a
matriz lida de X.tsp e confere com o ótimo publicado. Se bater, o parser e a
função de distância daquele tipo estão corretos.
"""

import gzip
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from utils import custo_tour


# Valores ótimos publicados na TSPLIB (todas as instâncias TSP simétricas)
OTIMOS = {
    "a280": 2579, "ali535": 202339, "att48": 10628, "att532": 27686,
    "bayg29": 1610, "bays29": 2020, "berlin52": 7542, "bier127": 118282,
    "brazil58": 25395, "brd14051": 469385, "brg180": 1950, "burma14": 3323,
    "ch130": 6110, "ch150": 6528, "d198": 15780, "d493": 35002, "d657": 48912,
    "d1291": 50801, "d1655": 62128, "d2103": 80450, "d15112": 1573084,
    "d18512": 645238, "dantzig42": 699, "dsj1000": 18660188, "eil51": 426,
    "eil76": 538, "eil101": 629, "fl417": 11861, "fl1400": 20127,
    "fl1577": 22249, "fl3795": 28772, "fnl4461": 182566, "fri26": 937,
    "gil262": 2378, "gr17": 2085, "gr21": 2707, "gr24": 1272, "gr48": 5046,
    "gr96": 55209, "gr120": 6942, "gr137": 69853, "gr202": 40160,
    "gr229": 134602, "gr431": 171414, "gr666": 294358, "hk48": 11461,
    "kroA100": 21282, "kroB100": 22141, "kroC100": 20749, "kroD100": 21294,
    "kroE100": 22068, "kroA150": 26524, "kroB150": 26130, "kroA200": 29368,
    "kroB200": 29437, "lin105": 14379, "lin318": 42029, "linhp318": 41345,
    "nrw1379": 56638, "p654": 34643, "pa561": 2763, "pcb442": 50778,
    "pcb1173": 56892, "pcb3038": 137694, "pla7397": 23260728,
    "pla33810": 66048945, "pla85900": 142382641, "pr76": 108159,
    "pr107": 44303, "pr124": 59030, "pr136": 96772, "pr144": 58537,
    "pr152": 73682, "pr226": 80369, "pr264": 49135, "pr299": 48191,
    "pr439": 107217, "pr1002": 259045, "pr2392": 378032, "rat99": 1211,
    "rat195": 2323, "rat575": 6773, "rat783": 8806, "rd100": 7910,
    "rd400": 15281, "rl1304": 252948, "rl1323": 270199, "rl1889": 316536,
    "rl5915": 565530, "rl5934": 556045, "rl11849": 923288, "si175": 21407,
    "si535": 48450, "si1032": 92650, "st70": 675, "swiss42": 1273,
    "ts225": 126643, "tsp225": 3916, "u159": 42080, "u574": 36905,
    "u724": 41910, "u1060": 224094, "u1432": 152970, "u1817": 57201,
    "u2152": 64253, "u2319": 234256, "ulysses16": 6859, "ulysses22": 7013,
    "usa13509": 19982859, "vm1084": 239297, "vm1748": 336556,
}

_SECOES = {
    "NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION", "DISPLAY_DATA_SECTION",
    "TOUR_SECTION", "FIXED_EDGES_SECTION", "DEPOT_SECTION", "DEMAND_SECTION",
}


@dataclass
class Instancia:
    nome: str
    n: int
    tipo_peso: str                 # EUC_2D, GEO, ATT, EXPLICIT, ...
    formato_peso: str | None       # só para EXPLICIT
    D: np.ndarray                  # matriz de distâncias n x n (int64)
    coords: np.ndarray | None      # NODE_COORD_SECTION, se houver
    coords_display: np.ndarray | None  # DISPLAY_DATA_SECTION, se houver
    comentario: str = ""
    cabecalho: dict = field(default_factory=dict)

    @property
    def pontos(self):
        """Coordenadas para features geométricas (fecho convexo, bounding box).
        Instâncias EXPLICIT só têm pontos se trouxerem DISPLAY_DATA_SECTION;
        caso contrário devolve None e só as features baseadas em D se aplicam."""
        return self.coords if self.coords is not None else self.coords_display

    @property
    def otimo(self):
        return OTIMOS.get(self.nome)


# ---------------------------------------------------------------------------
# Leitura bruta do arquivo
# ---------------------------------------------------------------------------
def _abrir(caminho):
    if caminho.endswith(".gz"):
        return gzip.open(caminho, "rt", encoding="latin-1")
    return open(caminho, "r", encoding="latin-1")


def _ler_blocos(caminho):
    """Separa o arquivo em cabeçalho (dict) e seções (dict de listas de tokens)."""
    cabecalho, secoes = {}, {}
    atual = None
    with _abrir(caminho) as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            if linha == "EOF":
                break
            primeiro = linha.split()[0].rstrip(":")
            if primeiro in _SECOES:
                atual = primeiro
                secoes[atual] = linha.split()[1:]   # dados na mesma linha (raro)
            elif linha[0].isalpha() and ":" in linha:
                chave, valor = linha.split(":", 1)
                cabecalho[chave.strip().upper()] = valor.strip()
                atual = None
            elif atual is not None:
                secoes[atual].extend(linha.split())
            else:
                raise ValueError(f"{caminho}: linha fora de seção: {linha!r}")
    return cabecalho, secoes


def _ler_coords(tokens, n, caminho, secao):
    vals = [float(t) for t in tokens]
    if len(vals) < 3 * n:
        raise ValueError(f"{caminho}: {secao} tem {len(vals)//3} nós, esperado {n}")
    arr = np.array(vals[: 3 * n]).reshape(n, 3)
    ids = arr[:, 0].astype(int)
    if not np.array_equal(np.sort(ids), np.arange(1, n + 1)):
        raise ValueError(f"{caminho}: ids de nós em {secao} não são 1..{n}")
    coords = np.empty((n, 2))
    coords[ids - 1] = arr[:, 1:]          # respeita o id, não a ordem das linhas
    return coords


# ---------------------------------------------------------------------------
# Funções de distância (vetorizadas, fórmulas do TSPLIB95)
# ---------------------------------------------------------------------------
def _nint(x):
    return np.floor(x + 0.5)


def dist_euc_2d(c):
    dif = c[:, None, :] - c[None, :, :]
    return _nint(np.sqrt((dif ** 2).sum(axis=2))).astype(np.int64)


def dist_ceil_2d(c):
    dif = c[:, None, :] - c[None, :, :]
    return np.ceil(np.sqrt((dif ** 2).sum(axis=2))).astype(np.int64)


def dist_att(c):
    dif = c[:, None, :] - c[None, :, :]
    r = np.sqrt((dif ** 2).sum(axis=2) / 10.0)
    t = _nint(r)
    return np.where(t < r, t + 1, t).astype(np.int64)


def dist_geo(c):
    """Coordenadas no formato DDD.MM (graus.minutos), x = latitude, y = longitude.
    Os graus são obtidos por truncamento, como no código C de referência:
    38.55 significa 38 graus e 55 minutos. Arredondar (nint) daria 39 graus e
    -45 minutos e os ótimos publicados deixam de bater."""
    PI, RRR = 3.141592, 6378.388          # constantes exatamente como na TSPLIB
    graus = np.trunc(c)
    minutos = c - graus
    rad = PI * (graus + 5.0 * minutos / 3.0) / 180.0
    lat, lon = rad[:, 0], rad[:, 1]
    q1 = np.cos(lon[:, None] - lon[None, :])
    q2 = np.cos(lat[:, None] - lat[None, :])
    q3 = np.cos(lat[:, None] + lat[None, :])
    arg = np.clip(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3), -1.0, 1.0)
    D = np.trunc(RRR * np.arccos(arg) + 1.0).astype(np.int64)
    np.fill_diagonal(D, 0)                # a fórmula daria 1 na diagonal
    return D


_FUNCOES = {"EUC_2D": dist_euc_2d, "CEIL_2D": dist_ceil_2d,
            "ATT": dist_att, "GEO": dist_geo}


def _indices_explicit(formato, n):
    """Sequência de posições (i, j) na ordem em que os pesos aparecem.
    Para matriz simétrica, *_COL é a transposta de *_ROW do triângulo oposto."""
    if formato == "FULL_MATRIX":
        return [(i, j) for i in range(n) for j in range(n)]
    if formato in ("UPPER_ROW", "LOWER_COL"):
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if formato in ("LOWER_ROW", "UPPER_COL"):
        return [(i, j) for i in range(n) for j in range(i)]
    if formato in ("UPPER_DIAG_ROW", "LOWER_DIAG_COL"):
        return [(i, j) for i in range(n) for j in range(i, n)]
    if formato in ("LOWER_DIAG_ROW", "UPPER_DIAG_COL"):
        return [(i, j) for i in range(n) for j in range(i + 1)]
    raise ValueError(f"EDGE_WEIGHT_FORMAT não suportado: {formato}")


def _matriz_explicit(tokens, formato, n, caminho):
    pos = _indices_explicit(formato, n)
    if len(tokens) != len(pos):
        raise ValueError(f"{caminho}: {len(tokens)} pesos, esperado {len(pos)} "
                         f"para {formato} com n={n}")
    D = np.zeros((n, n), dtype=np.int64)
    for (i, j), t in zip(pos, tokens):
        D[i, j] = int(float(t))
        if formato != "FULL_MATRIX":
            D[j, i] = D[i, j]
    if not np.array_equal(D, D.T):
        raise ValueError(f"{caminho}: matriz não é simétrica")
    return D


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def ler_instancia(caminho):
    cab, sec = _ler_blocos(caminho)

    tipo_problema = cab.get("TYPE", "TSP").split()[0]
    if tipo_problema != "TSP":
        raise ValueError(f"{caminho}: TYPE={tipo_problema}; só TSP simétrico é suportado")

    n = int(cab["DIMENSION"])
    tipo = cab["EDGE_WEIGHT_TYPE"].split()[0]
    formato = None
    if tipo == "EXPLICIT":                # FORMAT só importa para EXPLICIT
        formato = cab["EDGE_WEIGHT_FORMAT"].split()[0]

    coords = None
    if "NODE_COORD_SECTION" in sec:
        coords = _ler_coords(sec["NODE_COORD_SECTION"], n, caminho, "NODE_COORD_SECTION")
    coords_display = None
    if "DISPLAY_DATA_SECTION" in sec:
        coords_display = _ler_coords(sec["DISPLAY_DATA_SECTION"], n, caminho,
                                     "DISPLAY_DATA_SECTION")

    if tipo == "EXPLICIT":
        D = _matriz_explicit(sec["EDGE_WEIGHT_SECTION"], formato, n, caminho)
    elif tipo in _FUNCOES:
        if coords is None:
            raise ValueError(f"{caminho}: {tipo} sem NODE_COORD_SECTION")
        D = _FUNCOES[tipo](coords)
    else:
        raise ValueError(f"{caminho}: EDGE_WEIGHT_TYPE {tipo} não suportado")

    nome = cab.get("NAME", os.path.basename(caminho)).split()[0]
    nome = nome.removesuffix(".tsp")      # ulysses16 tem "NAME: ulysses16.tsp"
    return Instancia(nome=nome, n=n, tipo_peso=tipo, formato_peso=formato, D=D,
                     coords=coords, coords_display=coords_display,
                     comentario=cab.get("COMMENT", ""), cabecalho=cab)


def ler_tour(caminho):
    """Lê um arquivo .opt.tour e devolve o tour com índices 0-based."""
    _, sec = _ler_blocos(caminho)
    tour = []
    for t in sec["TOUR_SECTION"]:
        v = int(t)
        if v == -1:
            break
        tour.append(v - 1)
    return tour


def violacoes_desigualdade_triangular(D):
    """Quantos pares (i, j), i < j, têm um atalho k com D[i,k] + D[k,j] < D[i,j].

    A garantia de 1.5 do Christofides (e de 2 do twice-around-the-tree) depende
    da desigualdade triangular. Distâncias rodoviárias (EXPLICIT) e o
    arredondamento nint podem violá-la. Retorna (n_pares, maior_violacao).
    Processa uma linha por vez: tempo O(n^3), mas memória só O(n^2)."""
    n = D.shape[0]
    pares, maior = 0, 0
    for i in range(n):
        via_k = (D[i][:, None] + D).min(axis=0)   # min_k D[i,k] + D[k,j]
        falta = D[i] - via_k
        pares += int((falta[i + 1:] > 0).sum())
        maior = max(maior, int(falta.max()))
    return pares, maior


def listar_instancias(pasta, n_min=0, n_max=10**9):
    """Caminhos dos .tsp/.tsp.gz da pasta com n_min <= DIMENSION <= n_max,
    ordenados por tamanho. Lê só o cabeçalho de cada arquivo."""
    achados = []
    for arq in os.listdir(pasta):
        if not (arq.endswith(".tsp") or arq.endswith(".tsp.gz")):
            continue
        caminho = os.path.join(pasta, arq)
        with _abrir(caminho) as f:
            for linha in f:
                if linha.upper().startswith("DIMENSION"):
                    try:
                        n = int(linha.split(":", 1)[1])
                    except ValueError:
                        break
                    if n_min <= n <= n_max:
                        achados.append((n, arq, caminho))
                    break
    return [c for _, _, c in sorted(achados)]


# ---------------------------------------------------------------------------
# Validação do leitor contra os tours ótimos publicados
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pasta = sys.argv[1] if len(sys.argv) > 1 else "data"
    if not os.path.isdir(pasta):
        sys.exit(f"Pasta '{pasta}' não encontrada. Rode a partir da raiz do "
                 f"projeto ou passe o caminho: python leitor_tsp.py caminho/da/pasta")
    print(f"{'instância':<11}{'n':>4}  {'tipo':<22}{'ótimo':>8}{'opt.tour':>10}  "
          f"{'ok':<4}{'viol. Δ (pares/máx)':>21}{'pontos':>8}")
    falhas = 0
    for caminho in listar_instancias(pasta):
        try:
            inst = ler_instancia(caminho)
        except (ValueError, KeyError) as e:
            print(f"{os.path.basename(caminho):<15} PULADA: {e}")
            continue
        tipo = inst.tipo_peso + (f"/{inst.formato_peso}" if inst.formato_peso else "")
        base = caminho.removesuffix(".gz").removesuffix(".tsp")
        tour_arq = next((base + s for s in (".opt.tour", ".opt.tour.gz")
                         if os.path.exists(base + s)), None)
        c_opt, ok = "-", "-"
        if tour_arq:
            tour = ler_tour(tour_arq)
            assert sorted(tour) == list(range(inst.n))
            c_opt = custo_tour(inst.D, tour)
            ok = "sim" if c_opt == inst.otimo else "NÃO"
            falhas += ok == "NÃO"
        nviol, vmax = violacoes_desigualdade_triangular(inst.D)
        print(f"{inst.nome:<11}{inst.n:>4}  {tipo:<22}{str(inst.otimo):>8}"
              f"{str(c_opt):>10}  {ok:<4}{f'{nviol}/{vmax}':>21}"
              f"{'sim' if inst.pontos is not None else 'não':>8}")
    print("\nTodas as conferências bateram." if falhas == 0 else f"\n{falhas} FALHA(S).")
