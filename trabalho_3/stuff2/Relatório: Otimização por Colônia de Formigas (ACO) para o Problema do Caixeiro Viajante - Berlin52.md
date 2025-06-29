# Relatório: Otimização por Colônia de Formigas (ACO) para o Problema do Caixeiro Viajante - Berlin52

**Autor:** Manus AI  
**Data:** 28 de junho de 2025  
**Problema:** Traveling Salesman Problem (TSP) - Berlin52  

---

## Resumo Executivo

Este relatório apresenta a implementação e análise de resultados do algoritmo de Otimização por Colônia de Formigas (Ant Colony Optimization - ACO) aplicado ao problema clássico do Caixeiro Viajante utilizando o conjunto de dados Berlin52. O algoritmo foi desenvolvido em Python e executado com configurações otimizadas para encontrar o menor caminho que conecta todas as 52 cidades de Berlim, retornando ao ponto de origem.

Os resultados obtidos demonstram a eficácia do ACO na resolução de problemas de otimização combinatória complexos. O algoritmo convergiu para uma solução com distância total de **7659.25 unidades**, representando uma melhoria de **42.68%** em relação à solução inicial aleatória. A execução foi concluída em aproximadamente 29 segundos, utilizando 50 formigas artificiais ao longo de 500 iterações.




## 1. Introdução

O Problema do Caixeiro Viajante (Traveling Salesman Problem - TSP) é um dos problemas de otimização combinatória mais estudados na ciência da computação e pesquisa operacional. Classificado como NP-difícil, o TSP consiste em encontrar o menor caminho que visita todas as cidades de um conjunto exatamente uma vez e retorna à cidade de origem. A complexidade computacional cresce exponencialmente com o número de cidades, tornando impraticável a busca exaustiva para instâncias com mais de algumas dezenas de cidades.

O conjunto de dados Berlin52 representa um benchmark clássico na literatura de TSP, contendo as coordenadas de 52 localizações em Berlim, Alemanha. Este problema foi originalmente proposto por Grötschel e tem sido amplamente utilizado para avaliar a performance de algoritmos de otimização. Com 52 cidades, o espaço de busca contém aproximadamente 1.55 × 10^67 possíveis soluções, demonstrando a necessidade de algoritmos heurísticos eficientes.

A Otimização por Colônia de Formigas (Ant Colony Optimization - ACO) é uma meta-heurística inspirada no comportamento de formigas reais na busca por alimento. Desenvolvida por Marco Dorigo em 1992, esta técnica baseia-se na capacidade das formigas de encontrar caminhos ótimos através da deposição e seguimento de rastros de feromônios. No contexto computacional, formigas artificiais constroem soluções de forma probabilística, utilizando informações heurísticas locais e feromônios globais acumulados por soluções anteriores.

## 2. Fundamentação Teórica

### 2.1 Algoritmo de Otimização por Colônia de Formigas

O ACO opera através de um processo iterativo onde múltiplas formigas artificiais constroem soluções para o problema de otimização. Cada formiga toma decisões probabilísticas baseadas em dois componentes principais: a informação heurística local (visibilidade) e a informação de feromônio global (experiência coletiva).

A probabilidade de uma formiga k escolher a cidade j a partir da cidade i é dada pela fórmula:

```
P(i,j) = [τ(i,j)^α × η(i,j)^β] / Σ[τ(i,l)^α × η(i,l)^β]
```

Onde:
- τ(i,j) representa a concentração de feromônio na aresta entre as cidades i e j
- η(i,j) é a informação heurística, tipicamente o inverso da distância entre i e j
- α controla a influência relativa do feromônio
- β controla a influência relativa da informação heurística
- A somatória é calculada sobre todas as cidades l ainda não visitadas

### 2.2 Atualização de Feromônios

O mecanismo de atualização de feromônios é fundamental para o sucesso do algoritmo. Após todas as formigas completarem suas soluções, os feromônios são atualizados em duas etapas:

**Evaporação:** Simula a evaporação natural dos feromônios, evitando convergência prematura:
```
τ(i,j) = (1 - ρ) × τ(i,j)
```

**Deposição:** Formigas depositam feromônios proporcionalmente à qualidade de suas soluções:
```
τ(i,j) = τ(i,j) + Σ(Q / L_k)
```

Onde ρ é a taxa de evaporação, Q é uma constante de deposição, e L_k é o comprimento da solução da formiga k.

### 2.3 Características do Berlin52

O conjunto Berlin52 apresenta características específicas que influenciam a aplicação do ACO:

- **Distribuição espacial:** As 52 cidades estão distribuídas de forma irregular no plano cartesiano, com coordenadas variando de (25, 5) a (1740, 1175)
- **Distâncias euclidianas:** O problema utiliza distâncias euclidianas bidimensionais, calculadas pela fórmula padrão
- **Solução ótima conhecida:** A literatura reporta que a solução ótima para Berlin52 tem comprimento aproximado de 7542 unidades
- **Densidade de conexões:** Todas as cidades estão conectadas a todas as outras, caracterizando um grafo completo


## 3. Metodologia

### 3.1 Implementação do Algoritmo

A implementação do ACO foi desenvolvida em Python 3.11, utilizando as bibliotecas NumPy para operações matriciais eficientes e Matplotlib para visualização dos resultados. A arquitetura do código foi estruturada em uma classe principal `ACO_TSP` que encapsula todos os componentes do algoritmo.

**Estruturas de Dados Principais:**
- Matriz de distâncias: Array bidimensional 52×52 contendo distâncias euclidianas entre todas as cidades
- Matriz de feromônios: Array bidimensional 52×52 inicializado com valores pequenos (0.1)
- Matriz heurística: Inverso das distâncias, fornecendo informação local sobre atratividade das conexões

**Processo de Construção de Soluções:**
Cada formiga inicia sua jornada em uma cidade aleatória e constrói uma solução completa seguindo o processo:

1. **Seleção probabilística:** A próxima cidade é escolhida usando a regra de transição probabilística do ACO
2. **Atualização da lista tabu:** Cidades visitadas são removidas das opções disponíveis
3. **Cálculo de custos:** A distância total é computada ao final do percurso

**Mecanismo de Seleção por Roleta:**
A implementação utiliza o método de seleção por roleta para escolha probabilística das próximas cidades. Este método garante que cidades com maior probabilidade tenham maior chance de seleção, mantendo ainda a possibilidade de exploração de alternativas menos prováveis.

### 3.2 Configuração de Parâmetros

A configuração dos parâmetros do ACO foi baseada em valores amplamente utilizados na literatura e ajustados empiricamente para o problema Berlin52:

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| Número de formigas (m) | 50 | Quantidade de formigas artificiais por iteração |
| Número de iterações | 500 | Total de ciclos de construção e atualização |
| Alpha (α) | 1.0 | Peso da influência do feromônio |
| Beta (β) | 2.0 | Peso da informação heurística |
| Rho (ρ) | 0.1 | Taxa de evaporação do feromônio |
| Q | 100.0 | Constante de deposição de feromônio |

**Justificativa dos Parâmetros:**

O valor de α = 1.0 proporciona influência moderada do feromônio, permitindo que a experiência coletiva guie a busca sem dominar completamente as decisões. O β = 2.0 enfatiza a importância da informação heurística local, favorecendo conexões mais curtas. A taxa de evaporação ρ = 0.1 é conservadora, mantendo informações históricas por períodos mais longos e evitando convergência prematura.

O número de formigas (50) foi escolhido para balancear exploração do espaço de busca com eficiência computacional. Valores muito baixos podem resultar em exploração insuficiente, enquanto valores muito altos aumentam o custo computacional sem benefícios proporcionais.

### 3.3 Critérios de Parada e Avaliação

O algoritmo utiliza um critério de parada fixo baseado no número de iterações (500). Este valor foi determinado através de análise empírica, observando que a convergência típica ocorre entre 200-300 iterações para este problema específico.

**Métricas de Avaliação:**
- Distância da melhor solução encontrada
- Histórico de convergência ao longo das iterações
- Tempo de execução total
- Percentual de melhoria em relação à solução inicial
- Qualidade da solução em relação ao ótimo conhecido

### 3.4 Estrutura de Dados e Otimizações

A implementação incorpora várias otimizações para melhorar a eficiência computacional:

**Pré-computação de Distâncias:** Todas as distâncias euclidianas são calculadas uma única vez no início e armazenadas em matriz, evitando recálculos repetitivos durante a execução.

**Operações Vetorizadas:** Utilização de operações NumPy para atualização de feromônios e cálculos probabilísticos, aproveitando otimizações de baixo nível.

**Gestão de Memória:** Estruturas de dados são reutilizadas entre iterações, minimizando alocações desnecessárias de memória.

**Validação de Soluções:** Verificação automática da validade das soluções construídas, garantindo que todas as cidades sejam visitadas exatamente uma vez.


## 4. Resultados e Análise

### 4.1 Resultados Quantitativos

A execução do algoritmo ACO no problema Berlin52 produziu os seguintes resultados principais:

| Métrica | Valor |
|---------|-------|
| **Melhor distância encontrada** | 7659.25 unidades |
| **Distância inicial (primeira iteração)** | 13361.65 unidades |
| **Melhoria percentual** | 42.68% |
| **Tempo de execução** | 29.38 segundos |
| **Iteração da melhor solução** | 250 |
| **Gap em relação ao ótimo conhecido** | ~1.55% |

**Sequência do Melhor Caminho Encontrado:**
O caminho ótimo identificado pelo ACO conecta as cidades na seguinte ordem: 23 → 20 → 50 → 16 → 46 → 44 → 34 → 35 → 36 → 39 → 40 → 37 → 38 → 48 → 24 → 5 → 15 → 6 → 4 → 25 → 12 → 28 → 27 → 26 → 47 → 14 → 13 → 52 → 11 → 51 → 33 → 43 → 10 → 9 → 8 → 41 → 19 → 45 → 32 → 49 → 1 → 22 → 31 → 18 → 3 → 17 → 21 → 42 → 7 → 2 → 30 → 29 → 23.

### 4.2 Análise de Convergência

![Gráfico de Convergência](https://private-us-east-1.manuscdn.com/sessionFile/6Y2grDk6dc2RH9ZA1J0p0P/sandbox/bxFCugKXNJKryc68s4a5a9-images_1751140144010_na1fn_L2hvbWUvdWJ1bnR1L2NvbnZlcmdlbmNpYV9hY28.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvNlkyZ3JEazZkYzJSSDlaQTFKMHAwUC9zYW5kYm94L2J4RkN1Z0tYTkpLcnljNjhzNGE1YTktaW1hZ2VzXzE3NTExNDAxNDQwMTBfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyTnZiblpsY21kbGJtTnBZVjloWTI4LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=cOPG7z9ZqLpySzXg2JH-kDSIfW6T1ZqHzLHfPNLf~J7SHuW~JnmJaZ4Mgv7GnLNoa4awC9qAtX8m9S1Nxfzxgi0IVI5SnBWcVKt1b3dscfQNUzXojzdHTiBBA6-gn0XusYL1HNuRNC7ntiKGECFrcMbU9ZCIrdJ9toqKudbePT08EiCI-8tFIDCkRWgbhSkihoyOzpmBnA9Rd~n9rq6dUsl4fCPcDmuWUlVndzHWTv0gloMlkUDDsTdeBrDKvvBdKZEjsjwlFWwK2V~sCHT844sFWQ066s6v9xWLVSeV7M4gNP9Iq0lm0Ve3hljEYnFwgkH93~96hB~7x-0aMs1JaA__)

O gráfico de convergência revela características importantes do comportamento do algoritmo ACO:

**Fase de Exploração Inicial (Iterações 1-80):** O algoritmo demonstra rápida melhoria nas primeiras iterações, com a distância total diminuindo drasticamente de aproximadamente 13.400 para cerca de 8.200 unidades. Esta fase caracteriza-se pela exploração ativa do espaço de busca, com as formigas descobrindo regiões promissoras.

**Fase de Intensificação (Iterações 80-250):** Observa-se uma convergência mais gradual, com melhorias incrementais que levam à descoberta da melhor solução na iteração 250. Durante este período, o algoritmo refina soluções existentes através do acúmulo de feromônios em arestas de alta qualidade.

**Fase de Estabilização (Iterações 250-500):** Após a iteração 250, o algoritmo mantém a melhor solução encontrada, indicando convergência efetiva. A ausência de melhorias adicionais sugere que o algoritmo explorou adequadamente o espaço de busca local.

**Taxa de Convergência:** A melhoria de 42.68% demonstra a eficácia do ACO em escapar de soluções iniciais de baixa qualidade. A convergência em aproximadamente 250 iterações (50% do total) indica eficiência computacional adequada.

### 4.3 Análise Espacial da Solução

![Visualização da Solução](https://private-us-east-1.manuscdn.com/sessionFile/6Y2grDk6dc2RH9ZA1J0p0P/sandbox/bxFCugKXNJKryc68s4a5a9-images_1751140144010_na1fn_L2hvbWUvdWJ1bnR1L3NvbHVjYW9fYWNv.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvNlkyZ3JEazZkYzJSSDlaQTFKMHAwUC9zYW5kYm94L2J4RkN1Z0tYTkpLcnljNjhzNGE1YTktaW1hZ2VzXzE3NTExNDAxNDQwMTBfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzTnZiSFZqWVc5ZllXTnYucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=OFqRXAKX3ULZHC96m2SUcMsmR56K6rq5jzwX8qy-uT73ytnMoKK63CruE8thET6pwLXWu~of912sQw3yALEzZZ1UzksFNAWKtlt3dweO~CmLIKjZ6DJCWgcwdsiUh64s9UKJmzutIYEuHbUNEOrNs-3zFUf1t7gzwcIklYncQUanttGsZ8dEtt1oufVQ4zcJHraRFqpKNaZ-TV2aRuUvtTnNiRgBXcH-Cg2IZam--AE2M6T~w39CTAPXNEviZnfS1-LE5SFpO5z2j~pVkge-vrKX-naIiDGuUiSZf9CTi13-fGhjSXlI~sAmTXdBjDbL7OzvMly88Tvdo84Y6wWTNA__)

A visualização espacial da solução encontrada revela padrões interessantes na estrutura do caminho ótimo:

**Distribuição Geográfica:** O caminho demonstra uma estratégia eficiente de visitação que minimiza cruzamentos desnecessários. A solução apresenta uma estrutura aproximadamente circular, visitando clusters de cidades próximas antes de se mover para regiões distantes.

**Análise de Clusters:** O algoritmo identificou naturalmente agrupamentos de cidades e os visitou de forma sequencial. Observa-se que cidades geograficamente próximas tendem a ser visitadas em sequência, reduzindo a distância total percorrida.

**Evitação de Cruzamentos:** A solução apresenta poucos cruzamentos de arestas, característica desejável em soluções de TSP de alta qualidade. Cruzamentos desnecessários geralmente indicam oportunidades de melhoria através de operações de 2-opt.

**Pontos Extremos:** O algoritmo tratou adequadamente cidades isoladas ou em posições extremas do mapa, integrando-as ao caminho de forma eficiente sem criar desvios excessivos.

### 4.4 Comparação com Benchmarks

A solução encontrada (7659.25) apresenta qualidade competitiva quando comparada com resultados reportados na literatura:

- **Ótimo conhecido:** ~7542 unidades
- **Gap de otimalidade:** 1.55%
- **Qualidade relativa:** 98.45% do ótimo

Este resultado posiciona a implementação ACO desenvolvida entre as soluções de alta qualidade para o Berlin52, demonstrando a eficácia da configuração de parâmetros e implementação utilizada.

### 4.5 Análise de Performance Computacional

**Eficiência Temporal:** O tempo de execução de 29.38 segundos para 500 iterações com 50 formigas demonstra eficiência computacional adequada. Considerando que cada iteração processa 50 soluções completas, o algoritmo avalia aproximadamente 25.000 soluções candidatas.

**Escalabilidade:** A complexidade temporal O(n²) por formiga por iteração é aceitável para problemas de tamanho médio como o Berlin52. Para instâncias maiores, otimizações adicionais como listas de candidatos ou paralelização poderiam ser implementadas.

**Utilização de Recursos:** O algoritmo demonstra uso eficiente de memória, mantendo estruturas de dados compactas e reutilizando alocações entre iterações.


## 5. Discussão

### 5.1 Eficácia do Algoritmo ACO

Os resultados obtidos confirmam a eficácia do algoritmo ACO para resolver o problema do Caixeiro Viajante em instâncias de tamanho médio. A melhoria de 42.68% em relação à solução inicial e o gap de apenas 1.55% em relação ao ótimo conhecido demonstram que a implementação conseguiu equilibrar adequadamente exploração e explotação do espaço de busca.

A convergência observada na iteração 250 sugere que o algoritmo encontrou uma região promissora do espaço de busca e concentrou seus esforços na refinação local. Este comportamento é característico de meta-heurísticas bem ajustadas, onde a fase inicial de exploração é seguida por intensificação em regiões de alta qualidade.

### 5.2 Influência dos Parâmetros

A configuração de parâmetros utilizada mostrou-se efetiva para o problema Berlin52. O valor β = 2.0 (influência heurística) superior ao α = 1.0 (influência do feromônio) favoreceu a utilização de informação local, importante nas fases iniciais quando os rastros de feromônio ainda não estão bem estabelecidos.

A taxa de evaporação ρ = 0.1 permitiu manutenção adequada da memória histórica sem causar estagnação prematura. Valores mais altos de ρ poderiam resultar em perda excessiva de informação, enquanto valores menores poderiam levar à convergência prematura.

### 5.3 Características da Solução Encontrada

A análise espacial da solução revela que o ACO desenvolveu uma estratégia inteligente de visitação. A tendência de agrupar cidades próximas e minimizar cruzamentos indica que o algoritmo capturou princípios fundamentais de otimização geométrica em problemas TSP.

A estrutura aproximadamente circular da solução é típica de soluções de alta qualidade em TSP euclidiano, onde a minimização da distância total naturalmente leva a configurações que evitam desvios desnecessários.

### 5.4 Comparação com Outras Meta-heurísticas

Embora este estudo tenha focado exclusivamente no ACO, é importante contextualizar os resultados em relação a outras abordagens:

- **Algoritmos Genéticos:** Tipicamente produzem resultados similares para Berlin52, com gaps de 1-3% em relação ao ótimo
- **Simulated Annealing:** Pode alcançar soluções ligeiramente melhores com tempo de execução comparável
- **Busca Tabu:** Frequentemente encontra soluções de qualidade similar com convergência mais rápida

O ACO demonstra competitividade adequada neste contexto, especialmente considerando sua capacidade de paralelização natural e robustez em diferentes tipos de problemas.

## 6. Limitações e Trabalhos Futuros

### 6.1 Limitações da Implementação Atual

**Configuração de Parâmetros:** A configuração utilizada foi baseada em valores padrão da literatura e ajustes empíricos limitados. Uma otimização sistemática dos parâmetros através de técnicas como grid search ou algoritmos evolutivos poderia melhorar os resultados.

**Critério de Parada:** O uso de número fixo de iterações pode ser ineficiente. Critérios adaptativos baseados na estagnação da melhor solução ou convergência da população poderiam reduzir o tempo computacional.

**Estruturas de Vizinhança:** A implementação atual não incorpora operadores de busca local como 2-opt ou 3-opt, que poderiam refinar as soluções encontradas pelas formigas.

**Diversidade da População:** Não foram implementados mecanismos explícitos para manutenção da diversidade, como reinicialização de feromônios ou múltiplas colônias.

### 6.2 Propostas de Melhoria

**ACO Híbrido:** Integração com algoritmos de busca local para refinamento das soluções construídas pelas formigas. A aplicação de 2-opt nas melhores soluções de cada iteração poderia reduzir significativamente o gap de otimalidade.

**Paralelização:** Implementação de processamento paralelo para construção simultânea de soluções por múltiplas formigas, aproveitando arquiteturas multi-core modernas.

**Adaptação Dinâmica:** Desenvolvimento de mecanismos para ajuste automático de parâmetros durante a execução, baseado no progresso da busca e características da instância.

**Múltiplas Colônias:** Implementação de múltiplas colônias independentes com comunicação periódica, aumentando a diversidade e robustez do algoritmo.

### 6.3 Extensões para Problemas Relacionados

A implementação desenvolvida pode ser adaptada para variantes do TSP e problemas relacionados:

- **TSP Assimétrico:** Modificação para lidar com matrizes de distância não simétricas
- **Multiple TSP:** Extensão para múltiplos vendedores
- **Vehicle Routing Problem:** Adaptação para problemas de roteamento com restrições de capacidade
- **TSP com Janelas de Tempo:** Incorporação de restrições temporais

## 7. Conclusões

Este estudo demonstrou a aplicação bem-sucedida do algoritmo de Otimização por Colônia de Formigas ao problema clássico do Caixeiro Viajante utilizando o conjunto de dados Berlin52. Os resultados obtidos confirmam a eficácia desta meta-heurística para problemas de otimização combinatória de complexidade média.

### 7.1 Principais Contribuições

**Implementação Robusta:** Desenvolvimento de uma implementação completa e eficiente do ACO em Python, com estruturas de dados otimizadas e interface clara para análise de resultados.

**Análise Detalhada:** Condução de análise abrangente dos resultados, incluindo convergência temporal, qualidade da solução e características espaciais do caminho encontrado.

**Documentação Completa:** Produção de documentação técnica detalhada que pode servir como referência para implementações futuras e estudos comparativos.

### 7.2 Resultados Principais

- **Qualidade da Solução:** Gap de apenas 1.55% em relação ao ótimo conhecido, demonstrando alta qualidade da solução encontrada
- **Eficiência Computacional:** Convergência em tempo razoável (29 segundos) com recursos computacionais modestos
- **Melhoria Significativa:** Redução de 42.68% na distância total em relação à solução inicial aleatória

### 7.3 Implicações Práticas

Os resultados obtidos têm implicações importantes para aplicações práticas do ACO:

**Problemas de Roteamento:** A qualidade das soluções e eficiência computacional tornam o ACO viável para problemas reais de roteamento e logística.

**Otimização Industrial:** A robustez do algoritmo e facilidade de implementação facilitam sua aplicação em contextos industriais onde soluções de boa qualidade são mais importantes que otimalidade global.

**Pesquisa e Desenvolvimento:** A implementação serve como base sólida para desenvolvimento de variantes mais sofisticadas e estudos comparativos com outras meta-heurísticas.

### 7.4 Considerações Finais

O algoritmo ACO demonstrou ser uma ferramenta valiosa para resolução do TSP, combinando simplicidade conceitual com eficácia prática. A inspiração biológica do algoritmo não apenas facilita sua compreensão e implementação, mas também fornece insights sobre mecanismos de otimização coletiva que podem ser aplicados em diversos contextos.

A qualidade dos resultados obtidos para o Berlin52, combinada com a flexibilidade e extensibilidade da implementação, confirma o ACO como uma escolha sólida para problemas de otimização combinatória onde soluções de alta qualidade são requeridas em tempo computacional razoável.

Este trabalho contribui para o corpo de conhecimento sobre aplicações práticas do ACO e fornece uma base sólida para pesquisas futuras em otimização bio-inspirada e problemas de roteamento complexos.

---

## Anexos

### Anexo A: Configuração Completa dos Parâmetros

```python
# Configuração utilizada na execução
aco_config = {
    'n_ants': 50,
    'n_iterations': 500,
    'alpha': 1.0,
    'beta': 2.0,
    'rho': 0.1,
    'Q': 100.0,
    'initial_pheromone': 0.1
}
```

### Anexo B: Especificações do Ambiente de Execução

- **Sistema Operacional:** Ubuntu 22.04 LTS
- **Python:** Versão 3.11.0rc1
- **Bibliotecas:** NumPy 1.24+, Matplotlib 3.6+
- **Hardware:** Ambiente virtualizado com recursos padrão
- **Tempo de Execução:** 29.38 segundos

### Anexo C: Coordenadas das Cidades Berlin52

As coordenadas completas das 52 cidades estão disponíveis no arquivo original berlin52.tsp, seguindo o formato padrão TSPLIB com distâncias euclidianas bidimensionais.

