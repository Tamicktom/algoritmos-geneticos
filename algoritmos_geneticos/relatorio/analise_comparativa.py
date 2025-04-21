"""
Script para gerar gráficos e análises comparativas dos resultados obtidos nos experimentos.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

# Adiciona o diretório pai ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa os módulos dos algoritmos implementados
from reconhecimento_padroes.pattern_recognition import PatternRecognitionGA
from maximizacao_funcao.function_maximization import FunctionMaximizationGA, g, HillClimbing, SimulatedAnnealing
from minimizacao_rosenbrock.rosenbrock_minimization import RosenbrockGA, rosenbrock

def create_report_directory() -> str:
    """
    Cria o diretório para armazenar o relatório e os gráficos.
    
    Returns:
        Caminho para o diretório do relatório
    """
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graficos')
    os.makedirs(report_dir, exist_ok=True)
    return report_dir

def plot_pattern_recognition_results(save_dir: str) -> None:
    """
    Gera gráficos comparativos para o problema de reconhecimento de padrões.
    
    Args:
        save_dir: Diretório para salvar os gráficos
    """
    # Dados dos experimentos (obtidos dos testes anteriores)
    crossover_rates = [0.6, 0.7, 0.8, 0.9]
    mutation_rates = [0.01, 0.05, 0.1, 0.2]
    
    # Matriz de resultados (gerações médias)
    generations_matrix = np.array([
        [6.80, 5.57, 6.20, 9.43],  # CR = 0.6
        [4.87, 6.17, 5.57, 8.37],  # CR = 0.7
        [5.10, 5.07, 5.63, 10.70], # CR = 0.8
        [5.80, 5.33, 5.63, 11.53]  # CR = 0.9
    ])
    
    # Resultados dos operadores
    operators = ['Ambos', 'Apenas Crossover', 'Apenas Mutação']
    operator_generations = [6.90, 5.03, 6.60]
    operator_std = [3.38, 3.43, 4.22]
    
    # Plota o gráfico de calor para taxas de crossover e mutação
    plt.figure(figsize=(12, 8))
    plt.imshow(generations_matrix, interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Número médio de gerações')
    
    # Configura os eixos
    plt.xticks(range(len(mutation_rates)), [str(mr) for mr in mutation_rates])
    plt.yticks(range(len(crossover_rates)), [str(cr) for cr in crossover_rates])
    plt.xlabel('Taxa de Mutação')
    plt.ylabel('Taxa de Crossover')
    plt.title('Número Médio de Gerações para Convergência - Reconhecimento de Padrões')
    
    # Adiciona os valores na matriz
    for i in range(len(crossover_rates)):
        for j in range(len(mutation_rates)):
            plt.text(j, i, f"{generations_matrix[i, j]:.1f}", 
                     ha="center", va="center", color="white" if generations_matrix[i, j] > np.mean(generations_matrix) else "black")
    
    plt.savefig(os.path.join(save_dir, 'reconhecimento_parametros.png'))
    
    # Plota o gráfico de barras para comparação de operadores
    plt.figure(figsize=(10, 6))
    
    plt.bar(operators, operator_generations, yerr=operator_std, capsize=10)
    plt.ylabel('Número Médio de Gerações')
    plt.title('Comparação entre Operadores Genéticos - Reconhecimento de Padrões')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adiciona os valores nas barras
    for i, v in enumerate(operator_generations):
        plt.text(i, v + operator_std[i] + 0.5, f"{v:.1f}", ha='center')
    
    plt.savefig(os.path.join(save_dir, 'reconhecimento_operadores.png'))
    
    # Plota um exemplo de evolução do fitness
    # Cria um algoritmo genético para gerar dados de exemplo
    ga = PatternRecognitionGA(population_size=100, crossover_rate=0.8, mutation_rate=0.1)
    ga.initialize_population()
    
    # Executa algumas gerações
    for _ in range(10):
        ga.evolve()
    
    # Plota a evolução do fitness
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(ga.best_fitness_history)), ga.best_fitness_history, 'b-', label='Melhor Fitness')
    plt.plot(range(len(ga.avg_fitness_history)), ga.avg_fitness_history, 'r-', label='Fitness Médio')
    plt.title('Evolução do Fitness - Reconhecimento de Padrões')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'reconhecimento_evolucao.png'))

def plot_function_maximization_results(save_dir: str) -> None:
    """
    Gera gráficos comparativos para o problema de maximização de função.
    
    Args:
        save_dir: Diretório para salvar os gráficos
    """
    # Plota a função g(x)
    x_values = np.linspace(0, 1, 1000)
    y_values = [g(x) for x in x_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values)
    plt.title('Função g(x) = 2^(-2((x-0,1)/0,9)²) (sin(5πx))^6')
    plt.xlabel('x')
    plt.ylabel('g(x)')
    plt.grid(True)
    
    # Marca o ponto máximo (x=0.1)
    plt.plot(0.1, 1.0, 'ro', markersize=8)
    plt.annotate('Máximo Global (0.1, 1.0)', xy=(0.1, 1.0), xytext=(0.2, 0.8),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.savefig(os.path.join(save_dir, 'maximizacao_funcao.png'))
    
    # Dados dos experimentos (obtidos dos testes anteriores)
    methods = ["roulette", "tournament", "sus"]
    method_generations = [1000.00, 1000.00, 1000.00]  # Todos atingiram o máximo de gerações
    method_values = [1.000000, 1.000000, 1.000000]    # Todos encontraram o valor ótimo
    
    algorithms = ["Algoritmo Genético", "Subida da Colina", "Recozimento Simulado"]
    algorithm_values = [1.000000, 0.752050, 0.841810]
    algorithm_std_values = [0.000000, 0.230197, 0.165586]
    algorithm_times = [1.9595, 0.0043, 0.0022]
    algorithm_iterations = [1000.00, 206.90, 449.00]
    
    # Plota o gráfico de barras para comparação de métodos de seleção
    plt.figure(figsize=(12, 6))
    
    # Gráfico de barras para valores
    plt.subplot(1, 2, 2)
    plt.bar(methods, method_values)
    plt.ylabel('Valor Médio de g(x)')
    plt.title('Qualidade da Solução por Método de Seleção')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0.9, 1.01)  # Ajusta a escala para mostrar pequenas diferenças
    
    # Adiciona os valores nas barras
    for i, v in enumerate(method_values):
        plt.text(i, v - 0.02, f"{v:.4f}", ha='center')
    
    plt.savefig(os.path.join(save_dir, 'maximizacao_metodos_selecao.png'))
    
    # Plota o gráfico de barras para comparação de algoritmos
    plt.figure(figsize=(15, 5))
    
    # Gráfico de barras para valores
    plt.subplot(1, 3, 1)
    plt.bar(algorithms, algorithm_values, yerr=algorithm_std_values, capsize=10)
    plt.ylabel('Valor Médio de g(x)')
    plt.title('Qualidade da Solução por Algoritmo')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    # Gráfico de barras para tempos
    plt.subplot(1, 3, 2)
    plt.bar(algorithms, algorithm_times)
    plt.ylabel('Tempo Médio (s)')
    plt.title('Tempo de Execução por Algoritmo')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    # Gráfico de barras para iterações
    plt.subplot(1, 3, 3)
    plt.bar(algorithms, algorithm_iterations)
    plt.ylabel('Número Médio de Iterações')
    plt.title('Iterações até Convergência por Algoritmo')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'maximizacao_comparacao_algoritmos.png'))
    
    # Plota um exemplo de evolução do fitness
    # Cria um algoritmo genético para gerar dados de exemplo
    ga = FunctionMaximizationGA(population_size=100, chromosome_length=20, crossover_rate=0.8, mutation_rate=0.1)
    ga.initialize_population()
    
    # Executa algumas gerações
    for _ in range(20):
        ga.evolve()
    
    # Plota a evolução do fitness
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(ga.best_fitness_history)), ga.best_fitness_history, 'b-', label='Melhor Fitness')
    plt.plot(range(len(ga.avg_fitness_history)), ga.avg_fitness_history, 'r-', label='Fitness Médio')
    plt.title('Evolução do Fitness - Maximização de Função')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'maximizacao_evolucao.png'))

def plot_rosenbrock_results(save_dir: str) -> None:
    """
    Gera gráficos comparativos para o problema de minimização da função de Rosenbrock.
    
    Args:
        save_dir: Diretório para salvar os gráficos
    """
    # Plota a função de Rosenbrock em 2D (contorno)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Calcula o valor da função para cada ponto
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = rosenbrock(X[j, i], Y[j, i])
    
    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contorno da Função de Rosenbrock')
    plt.grid(True)
    
    # Marca o ponto mínimo global (1, 1)
    plt.plot(1, 1, 'ro', markersize=8)
    plt.annotate('Mínimo Global (1, 1)', xy=(1, 1), xytext=(1.2, 1.2),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.savefig(os.path.join(save_dir, 'rosenbrock_funcao.png'))
    
    # Dados dos experimentos (obtidos dos testes anteriores)
    # Comparação com e sem elitismo
    elitism_labels = ['Com Elitismo', 'Sem Elitismo']
    elitism_fitness = [0.999976, 0.999465]
    elitism_x = [1.002152, 1.005844]
    elitism_y = [1.004379, 1.011765]
    
    # Diferentes tamanhos de população
    population_sizes = [20, 50, 100, 200]
    population_fitness = [0.892605, 0.949882, 0.999973, 0.999991]
    population_x = [0.805707, 0.896071, 1.001755, 0.998489]
    population_y = [0.819257, 0.892317, 1.003616, 0.997055]
    population_times = [0.9752, 3.0590, 6.9266, 18.3566]
    
    # Plota o gráfico de barras para comparação com e sem elitismo
    plt.figure(figsize=(15, 5))
    
    # Configurações para os gráficos
    x_pos = np.arange(len(elitism_labels))
    width = 0.35
    
    # Gráfico para o valor da função
    plt.subplot(1, 3, 1)
    function_values = [1.0 / f - 1.0 for f in elitism_fitness]
    plt.bar(x_pos, function_values, width)
    plt.ylabel('Valor da Função')
    plt.title('Qualidade da Solução')
    plt.xticks(x_pos, elitism_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Gráfico para o erro em x
    plt.subplot(1, 3, 2)
    x_errors = [abs(x - 1.0) for x in elitism_x]
    plt.bar(x_pos, x_errors, width)
    plt.ylabel('Erro Absoluto em x')
    plt.title('Precisão em x')
    plt.xticks(x_pos, elitism_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Gráfico para o erro em y
    plt.subplot(1, 3, 3)
    y_errors = [abs(y - 1.0) for y in elitism_y]
    plt.bar(x_pos, y_errors, width)
    plt.ylabel('Erro Absoluto em y')
    plt.title('Precisão em y')
    plt.xticks(x_pos, elitism_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'rosenbrock_elitismo.png'))
    
    # Plota os gráficos para diferentes tamanhos de população
    plt.figure(figsize=(15, 10))
    
    # Gráfico para o valor da função
    plt.subplot(2, 2, 1)
    function_values = [1.0 / f - 1.0 for f in population_fitness]
    plt.plot(population_sizes, function_values, marker='o', linestyle='-')
    plt.xlabel('Tamanho da População')
    plt.ylabel('Valor da Função')
    plt.title('Qualidade da Solução')
    plt.grid(True)
    
    # Gráfico para o valor de x
    plt.subplot(2, 2, 2)
    plt.plot(population_sizes, population_x, marker='o', linestyle='-')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Valor Ótimo (x=1)')
    plt.xlabel('Tamanho da População')
    plt.ylabel('Valor Médio de x')
    plt.title('Convergência para x Ótimo')
    plt.legend()
    plt.grid(True)
    
    # Gráfico para o valor de y
    plt.subplot(2, 2, 3)
    plt.plot(population_sizes, population_y, marker='o', linestyle='-')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Valor Ótimo (y=1)')
    plt.xlabel('Tamanho da População')
    plt.ylabel('Valor Médio de y')
    plt.title('Convergência para y Ótimo')
    plt.legend()
    plt.grid(True)
    
    # Gráfico para o tempo de execução
    plt.subplot(2, 2, 4)
    plt.plot(population_sizes, population_times, marker='o', linestyle='-')
    plt.xlabel('Tamanho da População')
    plt.ylabel('Tempo Médio (s)')
    plt.title('Tempo de Execução')
    plt.grid(True)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'rosenbrock_populacao.png'))
    
    # Plota um exemplo de evolução do fitness
    # Cria um algoritmo genético para gerar dados de exemplo
    ga = RosenbrockGA(population_size=100, chromosome_length=16, crossover_rate=0.8, mutation_rate=0.1)
    ga.initialize_population()
    
    # Executa algumas gerações
    for _ in range(20):
        ga.evolve()
    
    # Plota a evolução do fitness
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(ga.best_fitness_history)), ga.best_fitness_history, 'b-', label='Melhor Fitness')
    plt.plot(range(len(ga.avg_fitness_history)), ga.avg_fitness_history, 'r-', label='Fitness Médio')
    plt.title('Evolução do Fitness - Minimização da Função de Rosenbrock')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'rosenbrock_evolucao.png'))

def create_summary_table(save_dir: str) -> None:
    """
    Cria uma tabela resumo com os principais resultados dos experimentos.
    
    Args:
        save_dir: Diretório para salvar a tabela
    """
    # Dados para a tabela de reconhecimento de padrões
    pattern_data = {
        'Configuração': [
            'CR=0.6, MR=0.01', 'CR=0.7, MR=0.01', 'CR=0.8, MR=0.01', 'CR=0.9, MR=0.01',
            'Apenas Crossover', 'Apenas Mutação', 'Ambos'
        ],
        'Gerações Médias': [6.80, 4.87, 5.10, 5.80, 5.03, 6.60, 6.90],
        'Desvio Padrão': [1.87, 2.28, 2.37, 2.88, 3.43, 4.22, 3.38],
        'Tempo Médio (s)': [0.0131, 0.0097, 0.0101, 0.0112, 0.0103, 0.0122, 0.0131]
    }
    
    # Dados para a tabela de maximização de função
    maximization_data = {
        'Algoritmo': ['Algoritmo Genético', 'Subida da Colina', 'Recozimento Simulado'],
        'Valor g(x)': [1.000000, 0.752050, 0.841810],
        'Desvio Padrão': [0.000000, 0.230197, 0.165586],
        'Iterações': [1000.00, 206.90, 449.00],
        'Tempo (s)': [1.9595, 0.0043, 0.0022]
    }
    
    # Dados para a tabela de minimização da função de Rosenbrock
    rosenbrock_data = {
        'Configuração': ['Com Elitismo', 'Sem Elitismo', 'Pop=20', 'Pop=50', 'Pop=100', 'Pop=200'],
        'Fitness': [0.999976, 0.999465, 0.892605, 0.949882, 0.999973, 0.999991],
        'Valor x': [1.002152, 1.005844, 0.805707, 0.896071, 1.001755, 0.998489],
        'Valor y': [1.004379, 1.011765, 0.819257, 0.892317, 1.003616, 0.997055],
        'Tempo (s)': [6.7188, 6.9528, 0.9752, 3.0590, 6.9266, 18.3566]
    }
    
    # Cria os DataFrames
    pattern_df = pd.DataFrame(pattern_data)
    maximization_df = pd.DataFrame(maximization_data)
    rosenbrock_df = pd.DataFrame(rosenbrock_data)
    
    # Salva as tabelas em formato CSV
    pattern_df.to_csv(os.path.join(save_dir, 'reconhecimento_resultados.csv'), index=False)
    maximization_df.to_csv(os.path.join(save_dir, 'maximizacao_resultados.csv'), index=False)
    rosenbrock_df.to_csv(os.path.join(save_dir, 'rosenbrock_resultados.csv'), index=False)
    
    # Cria uma tabela HTML para visualização
    html_content = """
    <html>
    <head>
        <title>Resultados dos Experimentos com Algoritmos Genéticos</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Resultados dos Experimentos com Algoritmos Genéticos</h1>
        
        <h2>1. Reconhecimento de Padrões</h2>
        {0}
        
        <h2>2. Maximização de Função</h2>
        {1}
        
        <h2>3. Minimização da Função de Rosenbrock</h2>
        {2}
    </body>
    </html>
    """.format(
        pattern_df.to_html(index=False),
        maximization_df.to_html(index=False),
        rosenbrock_df.to_html(index=False)
    )
    
    # Salva a tabela HTML
    with open(os.path.join(save_dir, 'resultados_tabela.html'), 'w') as f:
        f.write(html_content)

def create_comparative_analysis(save_dir: str) -> None:
    """
    Cria uma análise comparativa dos três problemas.
    
    Args:
        save_dir: Diretório para salvar a análise
    """
    # Dados para a análise comparativa
    problems = ['Reconhecimento de Padrões', 'Maximização de Função', 'Minimização de Rosenbrock']
    
    # Melhor configuração para cada problema
    best_configs = ['CR=0.7, MR=0.01', 'Roleta/Torneio/SUS', 'Pop=200, Com Elitismo']
    
    # Gerações médias para convergência
    generations = [4.87, 1000.00, 1000.00]
    
    # Tempo médio de execução (s)
    times = [0.0097, 1.9595, 18.3566]
    
    # Qualidade da solução (normalizada para [0, 1])
    quality = [1.0, 1.0, 0.999991]
    
    # Plota o gráfico comparativo
    plt.figure(figsize=(15, 5))
    
    # Configurações para os gráficos
    x_pos = np.arange(len(problems))
    width = 0.35
    
    # Gráfico para o tempo de execução
    plt.subplot(1, 3, 1)
    plt.bar(x_pos, times, width)
    plt.ylabel('Tempo Médio (s)')
    plt.title('Tempo de Execução por Problema')
    plt.xticks(x_pos, problems, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Gráfico para a qualidade da solução
    plt.subplot(1, 3, 2)
    plt.bar(x_pos, quality, width)
    plt.ylabel('Qualidade da Solução')
    plt.title('Precisão por Problema')
    plt.xticks(x_pos, problems, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0.9, 1.01)  # Ajusta a escala para mostrar pequenas diferenças
    
    # Gráfico para as gerações
    plt.subplot(1, 3, 3)
    plt.bar(x_pos, generations, width)
    plt.ylabel('Gerações Médias')
    plt.title('Gerações até Convergência por Problema')
    plt.xticks(x_pos, problems, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'analise_comparativa.png'))
    
    # Cria uma tabela com as melhores configurações
    best_data = {
        'Problema': problems,
        'Melhor Configuração': best_configs,
        'Gerações Médias': generations,
        'Tempo Médio (s)': times,
        'Qualidade da Solução': quality
    }
    
    best_df = pd.DataFrame(best_data)
    
    # Salva a tabela em formato CSV
    best_df.to_csv(os.path.join(save_dir, 'melhores_configuracoes.csv'), index=False)
    
    # Cria uma tabela HTML para visualização
    html_content = """
    <html>
    <head>
        <title>Análise Comparativa dos Problemas</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Análise Comparativa dos Problemas</h1>
        {0}
    </body>
    </html>
    """.format(best_df.to_html(index=False))
    
    # Salva a tabela HTML
    with open(os.path.join(save_dir, 'analise_comparativa.html'), 'w') as f:
        f.write(html_content)

def main():
    """
    Função principal para gerar todos os gráficos e análises.
    """
    # Cria o diretório para o relatório
    report_dir = create_report_directory()
    
    # Gera os gráficos para cada problema
    plot_pattern_recognition_results(report_dir)
    plot_function_maximization_results(report_dir)
    plot_rosenbrock_results(report_dir)
    
    # Cria a tabela resumo
    create_summary_table(report_dir)
    
    # Cria a análise comparativa
    create_comparative_analysis(report_dir)
    
    print(f"Análise comparativa gerada com sucesso no diretório: {report_dir}")

if __name__ == "__main__":
    main()
