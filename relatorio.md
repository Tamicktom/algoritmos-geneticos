# Relatório: Algoritmos Genéticos

## Introdução

Os algoritmos genéticos são métodos de busca e otimização baseados nos princípios de seleção natural e evolução. Muito utilizados na ciência da computação e investigação operacional para resolver problemas comuns complexos que os métodos clássicos não são eficientes. Este relatório tem como objetivo demonstrar os resultados da implementação de tais algoritmos na resolução de um problema.

Todas as implementações foram feitas em Python, utilizando as bibliotecas comuns como Numpy e Matplotlib. Para cada problema, diferentes experimentos com parâmetros foram feitos: crossover, mutação, população e assim por diante.

## Problema 1 - Reconhecimento de padrões.

### Descrição do problema

O algoritmo deve reconhecer o número 0, representado pela bitstring: [1 1 1 1 0 1 1 0 1 1 1 1].

### Experimento e resultados

Foi realizado três conjuntos de experimentos:

1. **Comparação de diferentes taxas de crossover e mutação:** Foram testadas combinações de taxas
de crossover `(0.6, 0.7, 0.8, 0.9)` e taxas de mutação `(0.01, 0.05, 0.1, 0.2)`.

2. **Comparação de operadores genéticos:** Foram realizados experimentos utilizando apenas crossover,
apenas mutação, e ambos os operadores.

3. **Execução de um único experimento detalhado:** Foi executado um experimento com a melhor
configuração encontrada, registrando a evolução do fitness ao longo das gerações.