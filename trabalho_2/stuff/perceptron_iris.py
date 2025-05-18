"""
Implementação do Perceptron para classificação do dataset Iris
Exercício 1 - Computação Inspirada pela Natureza
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, mean_squared_error
import seaborn as sns

# Configuração para exibir caracteres especiais em português
plt.rcParams['font.family'] = 'DejaVu Sans'

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

# Função para converter rótulos para codificação one-hot
def to_one_hot(y, n_classes):
    """
    Converte rótulos para codificação one-hot
    
    Parâmetros:
    - y: vetor de rótulos de classe
    - n_classes: número total de classes
    
    Retorna:
    - matriz de codificação one-hot
    """
    one_hot = np.zeros((len(y), n_classes))
    for i, label in enumerate(y):
        one_hot[i, label] = 1
    return one_hot

# Função para executar experimentos com diferentes configurações
def run_experiment(X_train, y_train, X_val, y_val, X_test, y_test, 
                  learning_rate=0.01, init_weight_scale=0.1, epochs=100):
    """
    Executa um experimento com o Perceptron
    
    Parâmetros:
    - X_train, y_train: dados de treinamento
    - X_val, y_val: dados de validação
    - X_test, y_test: dados de teste
    - learning_rate: taxa de aprendizado
    - init_weight_scale: escala para inicialização dos pesos
    - epochs: número de épocas
    
    Retorna:
    - resultados do experimento (acurácias, histórico de erros, etc.)
    """
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    # Converter rótulos para one-hot
    y_train_one_hot = to_one_hot(y_train, n_classes)
    y_val_one_hot = to_one_hot(y_val, n_classes)
    y_test_one_hot = to_one_hot(y_test, n_classes)
    
    # Inicializar e treinar o perceptron
    perceptron = Perceptron(n_features, n_classes, learning_rate, init_weight_scale)
    train_errors, val_errors = perceptron.train(
        X_train, y_train_one_hot, epochs, validation_data=(X_val, y_val_one_hot)
    )
    
    # Avaliar o modelo
    y_train_pred = perceptron.predict(X_train)
    y_val_pred = perceptron.predict(X_val)
    y_test_pred = perceptron.predict(X_test)
    
    # Calcular acurácias
    train_acc = np.mean(y_train_pred == y_train)
    val_acc = np.mean(y_val_pred == y_val)
    test_acc = np.mean(y_test_pred == y_test)
    
    # Calcular matrizes de confusão
    train_cm = confusion_matrix(y_train, y_train_pred)
    val_cm = confusion_matrix(y_val, y_val_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    return {
        'perceptron': perceptron,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_errors': train_errors,
        'val_errors': val_errors,
        'train_cm': train_cm,
        'val_cm': val_cm,
        'test_cm': test_cm
    }

# Função principal para executar todos os experimentos
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
    
    # Salvar os resultados em um arquivo de texto
    with open('resultados.txt', 'w') as f:
        f.write("=== Resultados do Experimento ===\n\n")
        
        f.write("Melhor configuração:\n")
        f.write(f"Normalização: {'Sim' if best_overall == best_norm else 'Não'}\n")
        f.write(f"Taxa de aprendizado: {best_overall['lr']}\n")
        f.write(f"Escala de inicialização: {best_overall['init_scale']}\n")
        f.write(f"Épocas: {epochs}\n\n")
        
        f.write("Acurácias:\n")
        f.write(f"Treinamento: {best_result['train_acc']:.4f}\n")
        f.write(f"Validação: {best_result['val_acc']:.4f}\n")
        f.write(f"Teste: {best_result['test_acc']:.4f}\n\n")
        
        f.write("Matrizes de Confusão:\n")
        f.write("Treinamento:\n")
        f.write(str(best_result['train_cm']) + "\n\n")
        f.write("Validação:\n")
        f.write(str(best_result['val_cm']) + "\n\n")
        f.write("Teste:\n")
        f.write(str(best_result['test_cm']) + "\n\n")

if __name__ == "__main__":
    main()
