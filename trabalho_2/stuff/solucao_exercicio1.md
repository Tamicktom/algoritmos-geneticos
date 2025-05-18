# Exercício 1: Perceptron para Classificação do Dataset Iris

## Introdução

Este documento apresenta a solução do Exercício 1, que consiste na implementação de uma rede Perceptron para classificação do conjunto de dados Iris. O dataset Iris contém 150 amostras de flores, sendo 50 de cada uma das três espécies: Iris setosa, Iris virginica e Iris versicolor. Para cada amostra, foram medidos quatro atributos: comprimento e largura da sépala e da pétala.

Conforme solicitado, o objetivo é treinar uma rede Perceptron para reconhecer as três diferentes classes, dividindo aleatoriamente os exemplos em subconjuntos de treinamento (70%), validação (15%) e teste (15%). Além disso, foram testadas diferentes configurações de pesos iniciais e taxas de aprendizado, bem como o impacto da normalização dos dados nos resultados.

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

1. **Carregamento e divisão do dataset**: O conjunto de dados Iris foi carregado e dividido em conjuntos de treinamento (70%), validação (15%) e teste (15%).

2. **Experimentos com dados não normalizados**: Foram testadas diferentes combinações de taxas de aprendizado (0.001, 0.01, 0.1) e escalas de inicialização de pesos (0.01, 0.1, 1.0). Para cada configuração, o experimento foi repetido 5 vezes, e a melhor configuração foi selecionada com base na acurácia de validação.

3. **Experimentos com dados normalizados**: Os mesmos experimentos foram repetidos, mas com os dados normalizados utilizando o StandardScaler da biblioteca Scikit-learn.

4. **Avaliação dos resultados**: Os resultados foram avaliados com base nas acurácias de treinamento, validação e teste, bem como nas matrizes de confusão para cada conjunto de dados.

5. **Visualização dos resultados**: Foram gerados gráficos da curva de aprendizado (Erro Médio Quadrático) e das matrizes de confusão para o melhor modelo.

## Resultados e Discussão

### Divisão do Dataset

O dataset Iris foi dividido da seguinte forma:
- Conjunto de treinamento: 104 amostras (70%)
- Conjunto de validação: 23 amostras (15%)
- Conjunto de teste: 23 amostras (15%)

### Experimentos com Dados Não Normalizados

Foram testadas 9 configurações diferentes (3 taxas de aprendizado × 3 escalas de inicialização), cada uma repetida 5 vezes. A melhor configuração obtida foi:

- Taxa de aprendizado: 0.01
- Escala de inicialização: 1.0
- Acurácia de validação: 0.8696 (86.96%)
- Acurácia de teste: 0.8261 (82.61%)

### Experimentos com Dados Normalizados

Os mesmos experimentos foram repetidos com os dados normalizados. A melhor configuração obtida foi:

- Taxa de aprendizado: 0.01
- Escala de inicialização: 0.1
- Acurácia de validação: 0.9130 (91.30%)
- Acurácia de teste: 0.8261 (82.61%)

### Comparação dos Resultados

Comparando os melhores resultados obtidos com e sem normalização:

- **Sem normalização**: 0.8696 (validação), 0.8261 (teste)
- **Com normalização**: 0.9130 (validação), 0.8261 (teste)

Observa-se que a normalização dos dados melhorou a acurácia de validação em aproximadamente 4.34 pontos percentuais, enquanto a acurácia de teste permaneceu a mesma. Isso sugere que a normalização ajudou o modelo a generalizar melhor para o conjunto de validação, mas não teve impacto significativo no conjunto de teste.

### Melhor Configuração Geral

A melhor configuração geral foi obtida com os dados normalizados:

- Normalização: Sim
- Taxa de aprendizado: 0.01
- Escala de inicialização: 0.1
- Épocas: 200
- Acurácia de treinamento: 0.8077 (80.77%)
- Acurácia de validação: 0.9130 (91.30%)
- Acurácia de teste: 0.8261 (82.61%)

### Curva de Aprendizado

A curva de aprendizado do melhor modelo mostra a evolução do Erro Médio Quadrático (MSE) ao longo das épocas de treinamento, tanto para o conjunto de treinamento quanto para o conjunto de validação.

![Curva de Aprendizado](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/lVVIKyKzZrKlDGIXu1rTki-images_1747515907815_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzEvY3VydmFfYXByZW5kaXphZG8.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2xWVklLeUt6WnJLbERHSVh1MXJUa2ktaW1hZ2VzXzE3NDc1MTU5MDc4MTVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6RXZZM1Z5ZG1GZllYQnlaVzVrYVhwaFpHOC5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=iGJ1hYwIEY6jiO4ovrMqoEXnbIaulAQDW4XWud~lz~-~tobPCc8XHBq8aJSlqFALnCbhqbWZ63rp3-zDJu-59pIGd6T4H94M~H-i7bjexXzw-OewMXRLZPq~KoPhTs3LoRUfU3q-N-O0eabvs25jj0kQFq1hrik-qQODoJmIG1Gj2OUFpEKgfkjRM-l7fROKBRulLSxthFcZHteCrJSkrfshm7YMDiFFBM3vfm54UiEk6AL0SAyePvw3B-qnwmWj4aKvsnjgCvRA3E-mYetkU2A7XRrsEkvLP-ohRCbr-2GcczpSO2yXWlxQql0L~cprlSuRr1ooyIaL8cHL2W-ylQ__)

Observa-se que o erro diminui consistentemente ao longo das épocas, indicando que o modelo está aprendendo corretamente. Além disso, o erro de validação é ligeiramente menor que o erro de treinamento, o que sugere que o modelo está generalizando bem para dados não vistos.

### Matrizes de Confusão

As matrizes de confusão para os conjuntos de treinamento, validação e teste são apresentadas a seguir:

![Matrizes de Confusão](https://private-us-east-1.manuscdn.com/sessionFile/2jSsdmejQ7pam6GX81Nf9s/sandbox/lVVIKyKzZrKlDGIXu1rTki-images_1747515907815_na1fn_L2hvbWUvdWJ1bnR1L2V4ZXJjaWNpbzEvbWF0cml6ZXNfY29uZnVzYW8.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvMmpTc2RtZWpRN3BhbTZHWDgxTmY5cy9zYW5kYm94L2xWVklLeUt6WnJLbERHSVh1MXJUa2ktaW1hZ2VzXzE3NDc1MTU5MDc4MTVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyVjRaWEpqYVdOcGJ6RXZiV0YwY21sNlpYTmZZMjl1Wm5WellXOC5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=XgoT5mgX1NTCraqv22dOHsztqxuhlpv-kNdTS~-DKLz3ZLB~fyavPvA~FtwoqYPgf-u6NYh~IGhNMJSBhS0NQ7OKWaa91TikGFUBPXhg83nG6lL~a9I6qYsypVE1jk7~WK3OHW1wQndn8arW-A2pQFCHkK5BTDUGnIa81Ee2LoYxrdO8mSjEkjrW1TGyNlcj61irhsQjnMvEiSw~eVOw4N2lsVwJX6P512z2yfy~xf4cgXCKpH0qcw80nBBTmIXKd-C7CYRakKHtPmcgJfzS3hBbz1uT-M33OplZItDvR01-JVDd2KAKNGNpdLqWy-kaSkpuKYsArPgqEue~zcA16Q__)

#### Matriz de Confusão - Treinamento:
```
[[34  0  0]
 [ 0 16 18]
 [ 0  2 34]]
```

#### Matriz de Confusão - Validação:
```
[[8 0 0]
 [0 5 2]
 [0 0 8]]
```

#### Matriz de Confusão - Teste:
```
[[8 0 0]
 [0 5 4]
 [0 0 6]]
```

Analisando as matrizes de confusão, podemos observar:

1. **Classe Setosa (0)**: O modelo classificou corretamente todas as amostras da classe Setosa em todos os conjuntos (treinamento, validação e teste). Isso indica que a classe Setosa é facilmente separável das outras classes.

2. **Classe Versicolor (1)**: O modelo teve dificuldade em distinguir entre Versicolor e Virginica. No conjunto de treinamento, 18 amostras de Versicolor foram classificadas incorretamente como Virginica. No conjunto de validação, 2 amostras foram classificadas incorretamente, e no conjunto de teste, 4 amostras.

3. **Classe Virginica (2)**: O modelo classificou corretamente a maioria das amostras de Virginica, com apenas 2 amostras classificadas incorretamente como Versicolor no conjunto de treinamento, e nenhuma no conjunto de validação e teste.

Essas observações sugerem que as classes Versicolor e Virginica são mais difíceis de separar, o que é um resultado esperado, considerando que essas duas espécies são mais semelhantes entre si do que em relação à espécie Setosa.

## Conclusões

Com base nos experimentos realizados, podemos concluir que:

1. **Normalização dos dados**: A normalização dos dados melhorou a acurácia de validação, indicando que é uma prática recomendada para este tipo de problema.

2. **Taxa de aprendizado e inicialização de pesos**: A taxa de aprendizado de 0.01 e a escala de inicialização de 0.1 produziram os melhores resultados com os dados normalizados. Taxas de aprendizado muito altas (0.1) ou muito baixas (0.001) resultaram em desempenho inferior.

3. **Separabilidade das classes**: A classe Setosa é facilmente separável das outras classes, enquanto as classes Versicolor e Virginica são mais difíceis de distinguir, o que é consistente com as características biológicas dessas espécies.

4. **Desempenho do Perceptron**: O Perceptron, mesmo sendo um modelo relativamente simples, conseguiu atingir uma acurácia de teste de 82.61%, o que é um resultado satisfatório para este problema de classificação multiclasse.

5. **Convergência do algoritmo**: A curva de aprendizado mostra que o algoritmo convergiu adequadamente, com o erro diminuindo consistentemente ao longo das épocas.

Em resumo, o Perceptron mostrou-se eficaz para a classificação do dataset Iris, especialmente quando utilizado com dados normalizados e parâmetros adequados. No entanto, a dificuldade em separar completamente as classes Versicolor e Virginica sugere que modelos mais complexos, como redes neurais com camadas ocultas, poderiam potencialmente alcançar resultados ainda melhores.

## Código Completo

O código completo da implementação está disponível no arquivo `perceptron_iris.py`. Abaixo está o código principal que executa os experimentos:

```python
def main():
    # Carregar o dataset Iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Dividir o dataset em treino, validação e teste (70%/15%/15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42  # 0.1765 * 0.85 = 0.15
    )
    
    print(f"Tamanho do conjunto de treinamento: {X_train.shape[0]}")
    print(f"Tamanho do conjunto de validação: {X_val.shape[0]}")
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]}")
    
    # Experimentos com dados não normalizados
    print("\n=== Experimentos com dados não normalizados ===")
    
    # Testar diferentes taxas de aprendizado e inicializações de pesos
    learning_rates = [0.001, 0.01, 0.1]
    init_scales = [0.01, 0.1, 1.0]
    epochs = 200
    
    results_no_norm = []
    
    for lr in learning_rates:
        for init_scale in init_scales:
            print(f"\nTaxa de aprendizado: {lr}, Escala de inicialização: {init_scale}")
            
            # Executar o experimento 5 vezes para cada configuração
            avg_val_acc = 0
            best_result = None
            best_val_acc = 0
            
            for i in range(5):
                result = run_experiment(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    learning_rate=lr, init_weight_scale=init_scale, epochs=epochs
                )
                
                avg_val_acc += result['val_acc']
                
                if result['val_acc'] > best_val_acc:
                    best_val_acc = result['val_acc']
                    best_result = result
            
            avg_val_acc /= 5
            print(f"Acurácia média de validação: {avg_val_acc:.4f}")
            print(f"Melhor acurácia de validação: {best_val_acc:.4f}")
            print(f"Acurácia de teste correspondente: {best_result['test_acc']:.4f}")
            
            results_no_norm.append({
                'lr': lr,
                'init_scale': init_scale,
                'avg_val_acc': avg_val_acc,
                'best_val_acc': best_val_acc,
                'test_acc': best_result['test_acc'],
                'result': best_result
            })
    
    # Encontrar a melhor configuração sem normalização
    best_no_norm = max(results_no_norm, key=lambda x: x['best_val_acc'])
    print("\nMelhor configuração sem normalização:")
    print(f"Taxa de aprendizado: {best_no_norm['lr']}")
    print(f"Escala de inicialização: {best_no_norm['init_scale']}")
    print(f"Acurácia de validação: {best_no_norm['best_val_acc']:.4f}")
    print(f"Acurácia de teste: {best_no_norm['test_acc']:.4f}")
    
    # Experimentos com dados normalizados
    print("\n=== Experimentos com dados normalizados ===")
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    results_norm = []
    
    for lr in learning_rates:
        for init_scale in init_scales:
            print(f"\nTaxa de aprendizado: {lr}, Escala de inicialização: {init_scale}")
            
            # Executar o experimento 5 vezes para cada configuração
            avg_val_acc = 0
            best_result = None
            best_val_acc = 0
            
            for i in range(5):
                result = run_experiment(
                    X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test,
                    learning_rate=lr, init_weight_scale=init_scale, epochs=epochs
                )
                
                avg_val_acc += result['val_acc']
                
                if result['val_acc'] > best_val_acc:
                    best_val_acc = result['val_acc']
                    best_result = result
            
            avg_val_acc /= 5
            print(f"Acurácia média de validação: {avg_val_acc:.4f}")
            print(f"Melhor acurácia de validação: {best_val_acc:.4f}")
            print(f"Acurácia de teste correspondente: {best_result['test_acc']:.4f}")
            
            results_norm.append({
                'lr': lr,
                'init_scale': init_scale,
                'avg_val_acc': avg_val_acc,
                'best_val_acc': best_val_acc,
                'test_acc': best_result['test_acc'],
                'result': best_result
            })
    
    # Encontrar a melhor configuração com normalização
    best_norm = max(results_norm, key=lambda x: x['best_val_acc'])
    print("\nMelhor configuração com normalização:")
    print(f"Taxa de aprendizado: {best_norm['lr']}")
    print(f"Escala de inicialização: {best_norm['init_scale']}")
    print(f"Acurácia de validação: {best_norm['best_val_acc']:.4f}")
    print(f"Acurácia de teste: {best_norm['test_acc']:.4f}")
    
    # Comparar os melhores resultados
    print("\n=== Comparação dos melhores resultados ===")
    print(f"Sem normalização: {best_no_norm['best_val_acc']:.4f} (validação), {best_no_norm['test_acc']:.4f} (teste)")
    print(f"Com normalização: {best_norm['best_val_acc']:.4f} (validação), {best_norm['test_acc']:.4f} (teste)")
    
    # Selecionar o melhor resultado geral
    best_overall = best_norm if best_norm['best_val_acc'] > best_no_norm['best_val_acc'] else best_no_norm
    best_result = best_overall['result']
    
    # Plotar curvas de erro para o melhor modelo
    plt.figure(figsize=(10, 6))
    plt.plot(best_result['train_errors'], label='Erro de Treinamento')
    plt.plot(best_result['val_errors'], label='Erro de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Erro Médio Quadrático')
    plt.title('Curva de Aprendizado - Melhor Modelo')
    plt.legend()
    plt.grid(True)
    plt.savefig('curva_aprendizado.png', dpi=300, bbox_inches='tight')
    
    # Plotar matrizes de confusão
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(best_result['train_cm'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusão - Treinamento')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    
    plt.subplot(1, 3, 2)
    sns.heatmap(best_result['val_cm'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusão - Validação')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    
    plt.subplot(1, 3, 3)
    sns.heatmap(best_result['test_cm'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusão - Teste')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    
    plt.tight_layout()
    plt.savefig('matrizes_confusao.png', dpi=300, bbox_inches='tight')
```
