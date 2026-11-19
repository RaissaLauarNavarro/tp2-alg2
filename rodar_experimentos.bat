@echo off
echo === FASE 2: SEM MEMORIA (tempo e nos reais) ===
py .\testes\rodar_bnb.py --estrategia melhor --inicial christofides
py .\testes\rodar_bnb.py --estrategia melhor --inicial twice
py .\testes\rodar_bnb.py --estrategia melhor --inicial trivial
py .\testes\rodar_bnb.py --estrategia melhor --inicial vizinho
py .\testes\rodar_bnb.py --estrategia melhor --inicial fecho
py .\testes\rodar_bnb.py --estrategia profundidade --inicial trivial
py .\testes\rodar_bnb.py --estrategia profundidade --inicial vizinho
py .\testes\rodar_bnb.py --estrategia profundidade --inicial fecho
py .\testes\rodar_bnb.py --estrategia profundidade --inicial twice
py .\testes\rodar_bnb.py --estrategia profundidade --inicial christofides
echo === CONCLUIDO ===