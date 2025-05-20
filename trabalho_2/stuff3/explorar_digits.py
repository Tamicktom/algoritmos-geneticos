"""
Exploração e visualização do dataset Digits do scikit-learn
Exercício 3 - Computação Inspirada pela Natureza
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Carregar o dataset Digits
print("Carregando o dataset Digits do scikit-learn...")
digits = load_digits()

# Informações básicas sobre o dataset
print(f"Formato dos dados: {digits.data.shape}")
print(f"Formato dos alvos: {digits.target.shape}")
print(f"Classes: {np.unique(digits.target)}")

# Visualizar alguns exemplos
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f'Dígito: {digits.target[i]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('exemplos_digits.png')

# Dividir o dataset em treino, validação e teste (70%/15%/15%)
X = digits.data
y = digits.target

# Primeiro, separamos 15% para teste
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Em seguida, separamos 15% do total (que é ~17.65% do restante) para validação
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42
)

print(f"Tamanho do conjunto de treinamento: {X_train.shape[0]}")
print(f"Tamanho do conjunto de validação: {X_val.shape[0]}")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]}")

# Verificar a distribuição das classes em cada conjunto
def print_class_distribution(y, name):
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nDistribuição de classes no conjunto {name}:")
    for cls, count in zip(unique, counts):
        print(f"  Classe {cls}: {count} exemplos")

print_class_distribution(y_train, "de treinamento")
print_class_distribution(y_val, "de validação")
print_class_distribution(y_test, "de teste")

# Salvar informações sobre o dataset
with open('info_digits.txt', 'w') as f:
    f.write("=== Informações sobre o Dataset Digits ===\n\n")
    f.write(f"Número total de amostras: {len(digits.data)}\n")
    f.write(f"Dimensão de cada imagem: {digits.images[0].shape}\n")
    f.write(f"Dimensão dos dados de entrada: {digits.data.shape[1]}\n")
    f.write(f"Número de classes: {len(np.unique(digits.target))}\n\n")
    
    f.write("Divisão do dataset:\n")
    f.write(f"  Treinamento: {X_train.shape[0]} amostras\n")
    f.write(f"  Validação: {X_val.shape[0]} amostras\n")
    f.write(f"  Teste: {X_test.shape[0]} amostras\n\n")
    
    f.write("Distribuição de classes:\n")
    unique, counts = np.unique(digits.target, return_counts=True)
    for cls, count in zip(unique, counts):
        f.write(f"  Classe {cls}: {count} exemplos\n")

print("\nExploração concluída. Informações salvas em 'info_digits.txt' e exemplos visualizados em 'exemplos_digits.png'.")
