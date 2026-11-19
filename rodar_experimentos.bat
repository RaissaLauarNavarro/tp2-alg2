@echo off
echo === FASE 1: COM MEMORIA (tracemalloc) ===
py .\testes\rodar_bnb.py --estrategia melhor --inicial trivial --memoria
py .\testes\rodar_bnb.py --estrategia melhor --inicial vizinho --memoria
py .\testes\rodar_bnb.py --estrategia melhor --inicial fecho --memoria
py .\testes\rodar_bnb.py --estrategia profundidade --inicial trivial --memoria
py .\testes\rodar_bnb.py --estrategia profundidade --inicial vizinho --memoria
py .\testes\rodar_bnb.py --estrategia profundidade --inicial fecho --memoria
py .\testes\rodar_bnb.py --estrategia profundidade --inicial twice --memoria
py .\testes\rodar_bnb.py --estrategia profundidade --inicial christofides --memoria

echo === FASE 2: SEM MEMORIA (tempo e nos reais) ===
py .\testes\rodar_bnb.py --estrategia melhor --inicial trivial
py .\testes\rodar_bnb.py --estrategia melhor --inicial vizinho
py .\testes\rodar_bnb.py --estrategia melhor --inicial fecho
py .\testes\rodar_bnb.py --estrategia profundidade --inicial trivial
py .\testes\rodar_bnb.py --estrategia profundidade --inicial vizinho
py .\testes\rodar_bnb.py --estrategia profundidade --inicial fecho
py .\testes\rodar_bnb.py --estrategia profundidade --inicial twice
py .\testes\rodar_bnb.py --estrategia profundidade --inicial christofides
echo === CONCLUIDO ===