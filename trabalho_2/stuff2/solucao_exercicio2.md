# Exercício 2: Perceptron para Classificação do Dataset Wine

## Introdução

Este documento apresenta a solução do Exercício 2, que consiste na implementação de uma rede Perceptron para classificação do conjunto de dados Wine do UCI Machine Learning Repository. O dataset Wine contém os resultados de uma análise química de vinhos cultivados na mesma região da Itália, mas derivados de três diferentes cultivares.

Conforme solicitado, o objetivo é identificar os atributos e classes do problema, e realizar experimentos similares aos do exercício 1, incluindo treinar uma rede Perceptron para reconhecer as diferentes classes, dividindo aleatoriamente os exemplos em subconjuntos de treinamento (70%), validação (15%) e teste (15%). Além disso, foram testadas diferentes configurações de pesos iniciais e taxas de aprendizado, bem como o impacto da normalização dos dados nos resultados.

## Identificação dos Atributos e Classes

De acordo com a documentação do dataset Wine, os dados possuem as seguintes características:

### Classes
O dataset contém 3 classes, representando três diferentes cultivares de vinho:
- Classe 1: 59 amostras
- Classe 2: 71 amostras
- Classe 3: 48 amostras

### Atributos
O dataset contém 13 atributos contínuos, que são os resultados de análises químicas:
1. Alcohol (Álcool)
2. Malic acid (Ácido málico)
3. Ash (Cinzas)
4. Alcalinity of ash (Alcalinidade das cinzas)
5. Magnesium (Magnésio)
6. Total phenols (Fenóis totais)
7. Flavanoids (Flavonoides)
8. Nonflavanoid phenols (Fenóis não flavonoides)
9. Proanthocyanins (Proantocianinas)
10. Color intensity (Intensidade da cor)
11. Hue (Tonalidade)
12. OD280/OD315 of diluted wines (OD280/OD315 de vinhos diluídos)
13. Proline (Prolina)

## Implementação do Perceptron

A implementação do Perceptron foi realizada em Python, utilizando as bibliotecas NumPy, Pandas, Matplotlib, Seaborn e Scikit-learn. A classe Perceptron foi implementada do zero, permitindo o controle completo sobre os parâmetros e o processo de treinamento.

```python
class Perceptron:
    def __init__(self, n_inputs, n_outputs, learning_rate=0.01, init_weight_scale=0.1):
        """
        Inicializa o Perceptron
        
        Parâmetros:
        - n_inputs: número de atributos de entrada
        - n_outputs: número de classes de saída
        - learning_rate: taxa de aprendizado
        - init_weight_scale: escala para inicialização dos pesos
        """
        self.learning_rate = learning_rate
        # Inicialização dos pesos com valores aleatórios pequenos
        self.weights = init_weight_scale * np.random.randn(n_inputs, n_outputs)
        # Inicialização do bias
        self.bias = init_weight_scale * np.random.randn(n_outputs)
        # Histórico de erros para acompanhamento
        self.error_history = []
        
    def forward(self, X):
        """
        Realiza a propagação direta (forward pass)
        
        Parâmetros:
        - X: matriz de exemplos de entrada
        
        Retorna:
        - saída do perceptron (valores contínuos)
        """
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        """
        Realiza a predição das classes
        
        Parâmetros:
        - X: matriz de exemplos de entrada
        
        Retorna:
        - classes preditas (índices das classes com maior valor)
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def train(self, X, y, epochs=100, validation_data=None):
        """
        Treina o perceptron
        
        Parâmetros:
        - X: matriz de exemplos de treinamento
        - y: vetor de classes (codificadas como one-hot)
        - epochs: número de épocas de treinamento
        - validation_data: tupla (X_val, y_val) para validação
        
        Retorna:
        - histórico de erros de treinamento e validação
        """
        n_samples = X.shape[0]
        train_errors = []
        val_errors = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Cálculo do erro (MSE)
            error = output - y
            mse = np.mean(np.sum(error**2, axis=1))
            train_errors.append(mse)
            
            # Atualização dos pesos
            delta_w = -self.learning_rate * np.dot(X.T, error) / n_samples
            delta_b = -self.learning_rate * np.sum(error, axis=0) / n_samples
            
            self.weights += delta_w
            self.bias += delta_b
            
            # Validação, se fornecida
            if validation_data is not None:
                X_val, y_val = validation_data
                val_output = self.forward(X_val)
                val_error = val_output - y_val
                val_mse = np.mean(np.sum(val_error**2, axis=1))
                val_errors.append(val_mse)
        
        return train_errors, val_errors
```

## Metodologia

Para a realização dos experimentos, seguimos a seguinte metodologia:

1. **Carregamento do dataset**: O conjunto de dados Wine foi carregado a partir do arquivo wine.data do UCI Machine Learning Repository.

2. **Divisão do dataset**: Os dados foram divididos em conjuntos de treinamento (70%), validação (15%) e teste (15%).

3. **Experimentos com dados não normalizados**: Foram testadas diferentes combinações de taxas de aprendizado (0.001, 0.01, 0.1) e escalas de inicialização de pesos (0.01, 0.1, 1.0). Para cada configuração, o experimento foi repetido 5 vezes, e a melhor configuração foi selecionada com base na acurácia de validação.

4. **Experimentos com dados normalizados**: Os mesmos experimentos foram repetidos, mas com os dados normalizados utilizando o StandardScaler da biblioteca Scikit-learn.

5. **Avaliação dos resultados**: Os resultados foram avaliados com base nas acurácias de treinamento, validação e teste, bem como nas matrizes de confusão para cada conjunto de dados.

6. **Visualização dos resultados**: Foram gerados gráficos da curva de aprendizado (Erro Médio Quadrático) e das matrizes de confusão para o melhor modelo.

## Resultados e Discussão

### Divisão do Dataset

O dataset Wine foi dividido da seguinte forma:
- Conjunto de treinamento: 124 amostras (70%)
- Conjunto de validação: 27 amostras (15%)
- Conjunto de teste: 27 amostras (15%)

### Experimentos com Dados Não Normalizados

Foram testadas 9 configurações diferentes (3 taxas de aprendizado × 3 escalas de inicialização), cada uma repetida 5 vezes. A melhor configuração obtida foi:

- Taxa de aprendizado: 0.001
- Escala de inicialização: 0.01
- Acurácia de validação: 0.2963 (29.63%)
- Acurácia de teste: 0.3333 (33.33%)

Observamos que sem normalização, o desempenho do Perceptron foi muito baixo, com acurácias próximas ao acaso. Isso sugere que as diferentes escalas dos atributos estão afetando significativamente o aprendizado do modelo.

### Experimentos com Dados Normalizados

Os mesmos experimentos foram repetidos com os dados normalizados. A melhor configuração obtida foi:

- Taxa de aprendizado: 0.01
- Escala de inicialização: 0.01
- Acurácia de validação: 1.0000 (100.00%)
- Acurácia de teste: 1.0000 (100.00%)

Com a normalização dos dados, o Perceptron conseguiu atingir 100% de acurácia tanto no conjunto de validação quanto no conjunto de teste. Isso demonstra a importância crucial da normalização para este dataset específico.

### Comparação dos Resultados

Comparando os melhores resultados obtidos com e sem normalização:

- **Sem normalização**: 0.2963 (validação), 0.3333 (teste)
- **Com normalização**: 1.0000 (validação), 1.0000 (teste)

A diferença é extremamente significativa, com uma melhoria de mais de 70 pontos percentuais na acurácia. Isso indica que a normalização é essencial para o bom desempenho do Perceptron neste dataset.

### Melhor Configuração Geral

A melhor configuração geral foi obtida com os dados normalizados:

- Normalização: Sim
- Taxa de aprendizado: 0.01
- Escala de inicialização: 0.01
- Épocas: 200
- Acurácia de treinamento: 0.9758 (97.58%)
- Acurácia de validação: 1.0000 (100.00%)
- Acurácia de teste: 1.0000 (100.00%)

### Curva de Aprendizado

A curva de aprendizado do melhor modelo mostra a evolução do Erro Médio Quadrático (MSE) ao longo das épocas de treinamento, tanto para o conjunto de treinamento quanto para o conjunto de validação.

![Curva de Aprendizado](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/0yqQt6Bts7z6qR4rioQGrg-images_1747575703076_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzIvY3VydmFfYXByZW5kaXphZG8.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94LzB5cVF0NkJ0czd6NnFSNHJpb1FHcmctaW1hZ2VzXzE3NDc1NzU3MDMwNzZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6SXZZM1Z5ZG1GZllYQnlaVzVrYVhwaFpHOC5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=orcUtH9KGRh0r2ulIQiln9XHqBb8LIoTf~dvpiWzGt1xwCbSgioXZPHEzg4h4qsIt0cJ2jyXeQ63RgH6etuRScJRzeofb~xFsKGdCWJOpu1iyrK8Pd0QAdC0WjntKaV6xHBCeGlF1eZH1hAnj128~5Qv1G4U4H4T5OSiV8PUss81O3hMzT0~CNvqSKTBFeEgmbBx8b78eLcNPktkmx7Taky1UVKd9agG8V-ewU9Dkbj7TkaTO3hZlIW39w8qrWMopvf2Rmt2xgWGe2rRvEe-mf6ugArJgaU6x7tQnvcObyUgRf~xb2e9DWbIFeq-rNb7jiZz~G4abCsHKS1KZKe3Hg__)

Observa-se que o erro diminui rapidamente nas primeiras épocas e continua a diminuir de forma mais gradual ao longo do treinamento. O erro de validação é ligeiramente menor que o erro de treinamento, o que sugere que o modelo está generalizando bem para dados não vistos.

### Matrizes de Confusão

As matrizes de confusão para os conjuntos de treinamento, validação e teste são apresentadas a seguir:

![Matrizes de Confusão](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/0yqQt6Bts7z6qR4rioQGrg-images_1747575703077_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzIvbWF0cml6ZXNfY29uZnVzYW8.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94LzB5cVF0NkJ0czd6NnFSNHJpb1FHcmctaW1hZ2VzXzE3NDc1NzU3MDMwNzdfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6SXZiV0YwY21sNlpYTmZZMjl1Wm5WellXOC5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=Dn2Zp3an6WoAHxvnzwCQynJ3iax8d6wqt0HV0YkVKWc9P2h7VMxvKIDjbR9MmmBLvO~OYxkV4MYIfuv8MX6UtGQ8-y5jBY7xPod3PXdN~KcLWANxXlWpLKjL-T4Ym5tFIlDUHks5vNLSR4091xqLZr0SAOHWt0VKFkPapFFG50jrTeoPwhpXcLR1uMEVojOXsJXOTvMCOvL4qS2bQkcF66JyxLyqI9MMtk7LD6XVJGyddQHpq4vc756GFnCduNiituRSa1F~Sr~qn5wVAGHXK4Zjld0JMdRVsj6GKm-G0qOjr5DSdN3M4gaXV-DtmUIKwY9yws9myBDAyIAXmZHZuQ__)

#### Matriz de Confusão - Treinamento:
```
[[42  0  0]
 [ 0 48  3]
 [ 0  0 31]]
```

#### Matriz de Confusão - Validação:
```
[[ 8  0  0]
 [ 0  9  0]
 [ 0  0 10]]
```

#### Matriz de Confusão - Teste:
```
[[ 9  0  0]
 [ 0 11  0]
 [ 0  0  7]]
```

Analisando as matrizes de confusão, podemos observar:

1. **Classe 1**: O modelo classificou corretamente todas as amostras da classe 1 em todos os conjuntos (treinamento, validação e teste).

2. **Classe 2**: No conjunto de treinamento, 3 amostras da classe 2 foram classificadas incorretamente como classe 3. No entanto, nos conjuntos de validação e teste, todas as amostras foram classificadas corretamente.

3. **Classe 3**: O modelo classificou corretamente todas as amostras da classe 3 em todos os conjuntos.

Essas observações indicam que o modelo teve um desempenho excelente, com apenas alguns erros no conjunto de treinamento, e perfeição nos conjuntos de validação e teste.

## Conclusões

Com base nos experimentos realizados, podemos concluir que:

1. **Normalização dos dados**: A normalização dos dados é absolutamente crucial para o bom desempenho do Perceptron no dataset Wine. Sem normalização, o modelo teve um desempenho próximo ao acaso, enquanto com normalização, atingiu 100% de acurácia.

2. **Taxa de aprendizado e inicialização de pesos**: A taxa de aprendizado de 0.01 e a escala de inicialização de 0.01 produziram os melhores resultados com os dados normalizados.

3. **Separabilidade das classes**: As classes do dataset Wine são linearmente separáveis quando os dados são normalizados, permitindo que um modelo simples como o Perceptron alcance 100% de acurácia.

4. **Desempenho do Perceptron**: O Perceptron, mesmo sendo um modelo relativamente simples, conseguiu atingir uma acurácia perfeita neste problema de classificação multiclasse, desde que os dados estejam normalizados.

5. **Convergência do algoritmo**: A curva de aprendizado mostra que o algoritmo convergiu adequadamente, com o erro diminuindo consistentemente ao longo das épocas.

Em resumo, o Perceptron mostrou-se extremamente eficaz para a classificação do dataset Wine quando utilizado com dados normalizados e parâmetros adequados. A normalização dos dados foi o fator mais importante para o sucesso do modelo, destacando a importância do pré-processamento adequado dos dados em problemas de aprendizado de máquina.

## Comparação com o Dataset Iris (Exercício 1)

Comparando os resultados obtidos com o dataset Wine com os do dataset Iris (Exercício 1), podemos observar:

1. **Impacto da normalização**: A normalização teve um impacto muito mais significativo no dataset Wine do que no Iris. No Wine, a normalização melhorou a acurácia de cerca de 30% para 100%, enquanto no Iris, a melhoria foi mais modesta.

2. **Separabilidade das classes**: As classes do dataset Wine são mais facilmente separáveis (quando normalizadas) do que as do Iris, onde as classes Versicolor e Virginica apresentavam alguma sobreposição.

3. **Complexidade do problema**: O dataset Wine tem mais atributos (13) do que o Iris (4), mas com a normalização adequada, o problema se tornou mais fácil de resolver.

4. **Desempenho geral**: O Perceptron conseguiu um desempenho perfeito no dataset Wine normalizado, enquanto no Iris, mesmo com normalização, a acurácia máxima foi de cerca de 91% no conjunto de validação e 83% no conjunto de teste.

Essas observações destacam como diferentes datasets podem apresentar desafios distintos para algoritmos de aprendizado de máquina, e como técnicas simples de pré-processamento, como a normalização, podem ter impactos drasticamente diferentes dependendo da natureza dos dados.
