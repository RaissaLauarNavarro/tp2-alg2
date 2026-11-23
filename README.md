# TP2 – Caixeiro Viajante (TSP)

Jean Berly (2023034773) e Raissa Lauar Navarro (2023034790)

Estudo experimental de algoritmos exatos e aproximativos para o Problema do Caixeiro Viajante simétrico, usando instâncias da TSPLIB com N entre 14 e 70 vértices.

Algoritmos implementados:

- **Branch-and-Bound** com duas estratégias de busca (*best-first* e *depth-first*) e cinco *warm-starts* para o limite superior inicial: solução trivial, vizinho mais próximo, fecho convexo (Graham), Twice-around-the-tree e Christofides.
- **Twice-around-the-tree** (2-aproximativo quando vale a desigualdade triangular).
- **Christofides** (1,5-aproximativo quando vale a desigualdade triangular).

## Estrutura do repositório

```
.
├── data/                        # Instâncias TSPLIB originais
├── data_respeita_desig_triang/  # Instâncias que satisfazem a desigualdade triangular
├── implementacoes/              # Código-fonte dos algoritmos
├── testes/                      # Scripts de execução e coleta de métricas
├── resultados/                  # Métricas coletadas (CSV)
├── documentacao/                # Relatório: PDF, código-fonte LaTeX, links prompts LLMs
├── historicos_llms              # Link dos prompts LLMS
└── README.md
```

### `data/`
Instâncias da TSPLIB usadas nos experimentos (`.tsp`) e, quando disponíveis, os tours ótimos publicados (`.opt.tour`). Cada instância é lida com a métrica do próprio arquivo (`EDGE_WEIGHT_TYPE`: `GEO`, `ATT`, `EUC_2D` ou `EXPLICIT`), sem converter tudo para `EUC_2D`.

| Faixa | Instâncias |
|---|---|
| Principal (N ≤ 40) | burma14, ulysses16, gr17, gr21, ulysses22, gr24, fri26, bayg29, bays29, dantzig42.tsp|
| Limite (N ∈ [45, 75]) | att48, eil51, st70 |


### `implementacoes/`
Código-fonte dos algoritmos:

- Branch-and-Bound (best-first e depth-first) e os limites inferiores;
- heurísticas de *warm-start* (trivial, vizinho mais próximo, fecho convexo por Graham);
- Twice-around-the-tree (MST por Prim + caminhada pré-ordem com atalhos);
- Christofides (MST + emparelhamento perfeito de custo mínimo + circuito euleriano + atalhos);
- leitura das instâncias e cálculo da matriz de distâncias.


### `testes/`
Scripts que executam os algoritmos sobre as instâncias e registram as métricas: tempo de execução por subrotina, pico de memória, custo da solução, gap em relação ao ótimo e, no Branch-and-Bound, nós expandidos e podados. Execuções do Branch-and-Bound têm limite de 30 minutos; ao estourar o tempo, o custo é registrado como `NA` e as estatísticas até a interrupção são mantidas.

### `resultados/`
Base de dados com as métricas coletadas, em CSV. Para cada algoritmo há um arquivo de tempo/qualidade e outro de memória (`*_memoria.csv`):

### `documentacao/`
Relatório no formato de artigo científico (modelo SBC):

- `relatorio.pdf` – versão compilada;
- `latex/` – código-fonte LaTeX e figuras;
- `historicos_llms` - links dos prompts de LLMS.

## Como executar

Cada arquivo apresenta no cabeçalho como ele deve ser rodado e suas funcionalidades.