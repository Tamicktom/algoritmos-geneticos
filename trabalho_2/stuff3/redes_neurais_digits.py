"""
Implementação e comparação de diferentes arquiteturas de redes neurais para o dataset Digits
Exercício 3 - Computação Inspirada pela Natureza
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import os

# Configurar o TensorFlow para usar apenas a CPU se necessário
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configurar o TensorFlow para usar memória de GPU de forma dinâmica
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Configuração para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

# Carregar o dataset Digits
print("Carregando o dataset Digits do scikit-learn...")
digits = load_digits()

# Preparar os dados
X = digits.data
y = digits.target

# Dividir o dataset em treino, validação e teste (70%/15%/15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42
)

print(f"Tamanho do conjunto de treinamento: {X_train.shape[0]}")
print(f"Tamanho do conjunto de validação: {X_val.shape[0]}")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]}")

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Converter os rótulos para one-hot encoding
y_train_onehot = to_categorical(y_train, 10)
y_val_onehot = to_categorical(y_val, 10)
y_test_onehot = to_categorical(y_test, 10)

# Preparar dados para CNN (reshape para formato de imagem)
X_train_cnn = X_train_scaled.reshape(-1, 8, 8, 1)
X_val_cnn = X_val_scaled.reshape(-1, 8, 8, 1)
X_test_cnn = X_test_scaled.reshape(-1, 8, 8, 1)

# Definir callbacks para treinamento
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Função para criar e treinar modelos
def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100, model_name="model"):
    # Criar diretório para salvar modelos se não existir
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Callback para salvar o melhor modelo
    model_checkpoint = ModelCheckpoint(
        f'models/{model_name}.h5',
        monitor='val_loss',
        save_best_only=True
    )
    
    # Registrar o tempo de início
    start_time = time.time()
    
    # Treinar o modelo
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Calcular o tempo de treinamento
    training_time = time.time() - start_time
    
    return model, history, training_time

# Função para avaliar modelos
def evaluate_model(model, X_test, y_test, y_test_onehot, model_name="model"):
    # Avaliar o modelo
    loss, accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
    
    # Fazer previsões
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calcular a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    # Gerar relatório de classificação
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Plotar a matriz de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10),
                yticklabels=range(10))
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.savefig(f'matriz_confusao_{model_name}.png', dpi=300, bbox_inches='tight')
    
    return loss, accuracy, report

# Função para plotar histórico de treinamento
def plot_training_history(history, model_name="model"):
    plt.figure(figsize=(12, 5))
    
    # Plotar acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treinamento')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title(f'Acurácia - {model_name}')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    
    # Plotar perda
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treinamento')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title(f'Perda - {model_name}')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'historico_{model_name}.png', dpi=300, bbox_inches='tight')

# Lista para armazenar resultados
results = []

# 1. MLP Simples (1 camada oculta)
print("\n=== Treinando MLP Simples (1 camada oculta) ===")
mlp_simple = Sequential([
    Dense(100, activation='relu', input_shape=(64,)),
    Dense(10, activation='softmax')
])

mlp_simple.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

mlp_simple, mlp_simple_history, mlp_simple_time = train_model(
    mlp_simple, X_train_scaled, y_train_onehot, X_val_scaled, y_val_onehot,
    model_name="mlp_simple"
)

mlp_simple_loss, mlp_simple_acc, mlp_simple_report = evaluate_model(
    mlp_simple, X_test_scaled, y_test, y_test_onehot, model_name="mlp_simple"
)

plot_training_history(mlp_simple_history, model_name="mlp_simple")

results.append({
    'model_name': 'MLP Simples',
    'architecture': '1 camada oculta (100 neurônios)',
    'params': mlp_simple.count_params(),
    'accuracy': mlp_simple_acc,
    'training_time': mlp_simple_time,
    'epochs': len(mlp_simple_history.history['loss'])
})

# 2. MLP Profunda (3 camadas ocultas)
print("\n=== Treinando MLP Profunda (3 camadas ocultas) ===")
mlp_deep = Sequential([
    Dense(128, activation='relu', input_shape=(64,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

mlp_deep.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

mlp_deep, mlp_deep_history, mlp_deep_time = train_model(
    mlp_deep, X_train_scaled, y_train_onehot, X_val_scaled, y_val_onehot,
    model_name="mlp_deep"
)

mlp_deep_loss, mlp_deep_acc, mlp_deep_report = evaluate_model(
    mlp_deep, X_test_scaled, y_test, y_test_onehot, model_name="mlp_deep"
)

plot_training_history(mlp_deep_history, model_name="mlp_deep")

results.append({
    'model_name': 'MLP Profunda',
    'architecture': '3 camadas ocultas (128-64-32 neurônios)',
    'params': mlp_deep.count_params(),
    'accuracy': mlp_deep_acc,
    'training_time': mlp_deep_time,
    'epochs': len(mlp_deep_history.history['loss'])
})

# 3. CNN Simples
print("\n=== Treinando CNN Simples ===")
cnn_simple = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(8, 8, 1), padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

cnn_simple.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_simple, cnn_simple_history, cnn_simple_time = train_model(
    cnn_simple, X_train_cnn, y_train_onehot, X_val_cnn, y_val_onehot,
    model_name="cnn_simple"
)

cnn_simple_loss, cnn_simple_acc, cnn_simple_report = evaluate_model(
    cnn_simple, X_test_cnn, y_test, y_test_onehot, model_name="cnn_simple"
)

plot_training_history(cnn_simple_history, model_name="cnn_simple")

results.append({
    'model_name': 'CNN Simples',
    'architecture': '1 camada convolucional (32 filtros)',
    'params': cnn_simple.count_params(),
    'accuracy': cnn_simple_acc,
    'training_time': cnn_simple_time,
    'epochs': len(cnn_simple_history.history['loss'])
})

# 4. CNN Profunda
print("\n=== Treinando CNN Profunda ===")
cnn_deep = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(8, 8, 1), padding='same'),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

cnn_deep.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_deep, cnn_deep_history, cnn_deep_time = train_model(
    cnn_deep, X_train_cnn, y_train_onehot, X_val_cnn, y_val_onehot,
    model_name="cnn_deep"
)

cnn_deep_loss, cnn_deep_acc, cnn_deep_report = evaluate_model(
    cnn_deep, X_test_cnn, y_test, y_test_onehot, model_name="cnn_deep"
)

plot_training_history(cnn_deep_history, model_name="cnn_deep")

results.append({
    'model_name': 'CNN Profunda',
    'architecture': '2 camadas convolucionais (32-64 filtros)',
    'params': cnn_deep.count_params(),
    'accuracy': cnn_deep_acc,
    'training_time': cnn_deep_time,
    'epochs': len(cnn_deep_history.history['loss'])
})

# 5. MLP com diferentes otimizadores (SGD)
print("\n=== Treinando MLP com SGD ===")
mlp_sgd = Sequential([
    Dense(100, activation='relu', input_shape=(64,)),
    Dense(10, activation='softmax')
])

mlp_sgd.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

mlp_sgd, mlp_sgd_history, mlp_sgd_time = train_model(
    mlp_sgd, X_train_scaled, y_train_onehot, X_val_scaled, y_val_onehot,
    model_name="mlp_sgd"
)

mlp_sgd_loss, mlp_sgd_acc, mlp_sgd_report = evaluate_model(
    mlp_sgd, X_test_scaled, y_test, y_test_onehot, model_name="mlp_sgd"
)

plot_training_history(mlp_sgd_history, model_name="mlp_sgd")

results.append({
    'model_name': 'MLP com SGD',
    'architecture': '1 camada oculta (100 neurônios)',
    'params': mlp_sgd.count_params(),
    'accuracy': mlp_sgd_acc,
    'training_time': mlp_sgd_time,
    'epochs': len(mlp_sgd_history.history['loss'])
})

# Criar tabela comparativa
results_df = pd.DataFrame(results)
print("\n=== Comparação dos Modelos ===")
print(results_df[['model_name', 'architecture', 'params', 'accuracy', 'training_time', 'epochs']])

# Salvar resultados em arquivo
results_df.to_csv('comparacao_modelos.csv', index=False)

# Plotar gráfico comparativo de acurácia
plt.figure(figsize=(12, 6))
plt.bar(results_df['model_name'], results_df['accuracy'], color='skyblue')
plt.title('Comparação de Acurácia entre Modelos')
plt.xlabel('Modelo')
plt.ylabel('Acurácia no Conjunto de Teste')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('comparacao_acuracia.png', dpi=300, bbox_inches='tight')

# Plotar gráfico comparativo de tempo de treinamento
plt.figure(figsize=(12, 6))
plt.bar(results_df['model_name'], results_df['training_time'], color='salmon')
plt.title('Comparação de Tempo de Treinamento entre Modelos')
plt.xlabel('Modelo')
plt.ylabel('Tempo de Treinamento (segundos)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('comparacao_tempo.png', dpi=300, bbox_inches='tight')

# Plotar gráfico comparativo de número de parâmetros
plt.figure(figsize=(12, 6))
plt.bar(results_df['model_name'], results_df['params'], color='lightgreen')
plt.title('Comparação de Número de Parâmetros entre Modelos')
plt.xlabel('Modelo')
plt.ylabel('Número de Parâmetros')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('comparacao_parametros.png', dpi=300, bbox_inches='tight')

# Salvar resumo dos resultados em arquivo de texto
with open('resultados_redes_neurais.txt', 'w') as f:
    f.write("=== Resultados dos Experimentos com Redes Neurais ===\n\n")
    
    f.write("Dataset: Digits (scikit-learn)\n")
    f.write(f"Número total de amostras: {len(digits.data)}\n")
    f.write(f"Dimensão dos dados de entrada: {digits.data.shape[1]}\n")
    f.write(f"Número de classes: {len(np.unique(digits.target))}\n\n")
    
    f.write("Divisão do dataset:\n")
    f.write(f"  Treinamento: {X_train.shape[0]} amostras\n")
    f.write(f"  Validação: {X_val.shape[0]} amostras\n")
    f.write(f"  Teste: {X_test.shape[0]} amostras\n\n")
    
    f.write("Comparação dos Modelos:\n")
    for i, row in results_df.iterrows():
        f.write(f"\n{row['model_name']} ({row['architecture']}):\n")
        f.write(f"  Número de parâmetros: {row['params']}\n")
        f.write(f"  Acurácia no teste: {row['accuracy']:.4f}\n")
        f.write(f"  Tempo de treinamento: {row['training_time']:.2f} segundos\n")
        f.write(f"  Épocas de treinamento: {row['epochs']}\n")
    
    f.write("\n\nMelhor modelo: ")
    best_model_idx = results_df['accuracy'].idxmax()
    best_model = results_df.iloc[best_model_idx]
    f.write(f"{best_model['model_name']} com acurácia de {best_model['accuracy']:.4f}\n")

print("\nExperimentos concluídos. Resultados salvos em 'resultados_redes_neurais.txt' e gráficos comparativos gerados.")
