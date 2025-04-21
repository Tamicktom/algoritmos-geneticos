"""
Implementação de algoritmo genético para reconhecimento de padrões.
Objetivo: Reconhecer o número 0 representado pela bitstring [1 1 1 1 0 1 1 0 1 1 1 1].
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# Adiciona o diretório pai ao path para importar o módulo utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.genetic_algorithm import GeneticAlgorithm, Individual

# Padrão alvo a ser reconhecido (número 0)
TARGET_PATTERN = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]

def calculate_fitness(chromosome: List[int]) -> float:
    """
    Calcula o fitness de um cromossomo para o problema de reconhecimento de padrões.
    O fitness é baseado na similaridade com o padrão alvo.
    
    Args:
        chromosome: Lista de genes que compõem o cromossomo
        
    Returns:
        Valor de fitness (quanto maior, melhor)
    """
    # Calcula a similaridade como o número de genes que correspondem ao padrão alvo
    matches = sum(1 for i in range(len(chromosome)) if chromosome[i] == TARGET_PATTERN[i])
    
    # Normaliza o fitness para o intervalo [0, 1]
    return matches / len(TARGET_PATTERN)

class PatternRecognitionGA(GeneticAlgorithm):
    """
    Algoritmo genético específico para reconhecimento de padrões.
    """
    def __init__(
        self,
        population_size: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism: bool = True,
        elitism_size: int = 2,
        max_generations: int = 1000,
        crossover_only: bool = False,
        mutation_only: bool = False
    ):
        """
        Inicializa o algoritmo genético para reconhecimento de padrões.
        
        Args:
            population_size: Tamanho da população
            crossover_rate: Taxa de crossover
            mutation_rate: Taxa de mutação
            elitism: Se True, os melhores indivíduos são preservados entre gerações
            elitism_size: Número de indivíduos preservados pelo elitismo
            max_generations: Número máximo de gerações
            crossover_only: Se True, apenas o operador de crossover é utilizado
            mutation_only: Se True, apenas o operador de mutação é utilizado
        """
        # Verifica se as configurações são válidas
        if crossover_only and mutation_only:
            raise ValueError("Não é possível usar apenas crossover e apenas mutação simultaneamente")
        
        # Ajusta as taxas de acordo com as configurações
        if crossover_only:
            mutation_rate = 0.0
        if mutation_only:
            crossover_rate = 0.0
        
        # Inicializa a classe pai
        super().__init__(
            population_size=population_size,
            chromosome_length=len(TARGET_PATTERN),
            fitness_function=calculate_fitness,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism=elitism,
            elitism_size=elitism_size,
            max_generations=max_generations,
            target_fitness=1.0  # Fitness perfeito quando todos os genes correspondem
        )
        
        self.crossover_only = crossover_only
        self.mutation_only = mutation_only
        self.execution_time = 0

    def run_experiment(self) -> Tuple[Individual, int, float]:
        """
        Executa o experimento e mede o tempo de execução.
        
        Returns:
            Tupla contendo o melhor indivíduo, número de gerações e tempo de execução
        """
        start_time = time.time()
        best_individual, generations, _, _ = self.run()
        self.execution_time = time.time() - start_time
        
        return best_individual, generations, self.execution_time

def run_multiple_experiments(
    num_experiments: int = 30,
    population_size: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    elitism: bool = True,
    crossover_only: bool = False,
    mutation_only: bool = False
) -> Tuple[float, float, float, float]:
    """
    Executa múltiplos experimentos e calcula estatísticas.
    
    Args:
        num_experiments: Número de experimentos a serem executados
        population_size: Tamanho da população
        crossover_rate: Taxa de crossover
        mutation_rate: Taxa de mutação
        elitism: Se True, os melhores indivíduos são preservados entre gerações
        crossover_only: Se True, apenas o operador de crossover é utilizado
        mutation_only: Se True, apenas o operador de mutação é utilizado
        
    Returns:
        Tupla contendo média e desvio padrão do número de gerações e do tempo de execução
    """
    generations_list = []
    time_list = []
    
    for _ in range(num_experiments):
        ga = PatternRecognitionGA(
            population_size=population_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism=elitism,
            crossover_only=crossover_only,
            mutation_only=mutation_only
        )
        
        _, generations, execution_time = ga.run_experiment()
        generations_list.append(generations)
        time_list.append(execution_time)
    
    # Calcula estatísticas
    avg_generations = np.mean(generations_list)
    std_generations = np.std(generations_list)
    avg_time = np.mean(time_list)
    std_time = np.std(time_list)
    
    return avg_generations, std_generations, avg_time, std_time

def compare_parameters(
    population_size: int = 100,
    crossover_rates: List[float] = [0.6, 0.7, 0.8, 0.9],
    mutation_rates: List[float] = [0.01, 0.05, 0.1, 0.2],
    num_experiments: int = 30,
    save_dir: Optional[str] = None
) -> None:
    """
    Compara diferentes taxas de crossover e mutação.
    
    Args:
        population_size: Tamanho da população
        crossover_rates: Lista de taxas de crossover a serem testadas
        mutation_rates: Lista de taxas de mutação a serem testadas
        num_experiments: Número de experimentos para cada combinação
        save_dir: Diretório para salvar os gráficos (opcional)
    """
    results = []
    
    # Testa todas as combinações de taxas
    for cr in crossover_rates:
        for mr in mutation_rates:
            avg_gen, std_gen, avg_time, std_time = run_multiple_experiments(
                num_experiments=num_experiments,
                population_size=population_size,
                crossover_rate=cr,
                mutation_rate=mr
            )
            
            results.append({
                'crossover_rate': cr,
                'mutation_rate': mr,
                'avg_generations': avg_gen,
                'std_generations': std_gen,
                'avg_time': avg_time,
                'std_time': std_time
            })
            
            print(f"Crossover: {cr}, Mutação: {mr}, Gerações: {avg_gen:.2f} ± {std_gen:.2f}, Tempo: {avg_time:.4f} ± {std_time:.4f}s")
    
    # Plota os resultados
    plt.figure(figsize=(12, 8))
    
    # Organiza os dados para o gráfico
    cr_values = sorted(list(set(r['crossover_rate'] for r in results)))
    mr_values = sorted(list(set(r['mutation_rate'] for r in results)))
    
    # Cria uma matriz para armazenar os valores médios de gerações
    generations_matrix = np.zeros((len(cr_values), len(mr_values)))
    
    for i, cr in enumerate(cr_values):
        for j, mr in enumerate(mr_values):
            for r in results:
                if r['crossover_rate'] == cr and r['mutation_rate'] == mr:
                    generations_matrix[i, j] = r['avg_generations']
    
    # Plota o gráfico de calor
    plt.imshow(generations_matrix, interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Número médio de gerações')
    
    # Configura os eixos
    plt.xticks(range(len(mr_values)), [str(mr) for mr in mr_values])
    plt.yticks(range(len(cr_values)), [str(cr) for cr in cr_values])
    plt.xlabel('Taxa de Mutação')
    plt.ylabel('Taxa de Crossover')
    plt.title('Número Médio de Gerações para Convergência')
    
    # Adiciona os valores na matriz
    for i in range(len(cr_values)):
        for j in range(len(mr_values)):
            plt.text(j, i, f"{generations_matrix[i, j]:.1f}", 
                     ha="center", va="center", color="white" if generations_matrix[i, j] > np.mean(generations_matrix) else "black")
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'comparacao_parametros.png'))
    
    plt.show()

def compare_operators(
    population_size: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    num_experiments: int = 30,
    save_dir: Optional[str] = None
) -> None:
    """
    Compara o uso de apenas crossover, apenas mutação, e ambos.
    
    Args:
        population_size: Tamanho da população
        crossover_rate: Taxa de crossover
        mutation_rate: Taxa de mutação
        num_experiments: Número de experimentos para cada configuração
        save_dir: Diretório para salvar os gráficos (opcional)
    """
    # Executa experimentos com ambos os operadores
    avg_gen_both, std_gen_both, avg_time_both, std_time_both = run_multiple_experiments(
        num_experiments=num_experiments,
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate
    )
    
    # Executa experimentos apenas com crossover
    avg_gen_cross, std_gen_cross, avg_time_cross, std_time_cross = run_multiple_experiments(
        num_experiments=num_experiments,
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        crossover_only=True
    )
    
    # Executa experimentos apenas com mutação
    avg_gen_mut, std_gen_mut, avg_time_mut, std_time_mut = run_multiple_experiments(
        num_experiments=num_experiments,
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_only=True
    )
    
    # Imprime os resultados
    print(f"Ambos: Gerações: {avg_gen_both:.2f} ± {std_gen_both:.2f}, Tempo: {avg_time_both:.4f} ± {std_time_both:.4f}s")
    print(f"Apenas Crossover: Gerações: {avg_gen_cross:.2f} ± {std_gen_cross:.2f}, Tempo: {avg_time_cross:.4f} ± {std_time_cross:.4f}s")
    print(f"Apenas Mutação: Gerações: {avg_gen_mut:.2f} ± {std_gen_mut:.2f}, Tempo: {avg_time_mut:.4f} ± {std_time_mut:.4f}s")
    
    # Plota os resultados
    plt.figure(figsize=(10, 6))
    
    operators = ['Ambos', 'Apenas Crossover', 'Apenas Mutação']
    generations = [avg_gen_both, avg_gen_cross, avg_gen_mut]
    errors = [std_gen_both, std_gen_cross, std_gen_mut]
    
    plt.bar(operators, generations, yerr=errors, capsize=10)
    plt.ylabel('Número Médio de Gerações')
    plt.title('Comparação entre Operadores Genéticos')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adiciona os valores nas barras
    for i, v in enumerate(generations):
        plt.text(i, v + errors[i] + 5, f"{v:.1f}", ha='center')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'comparacao_operadores.png'))
    
    plt.show()

def run_single_experiment(
    population_size: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    elitism: bool = True,
    save_dir: Optional[str] = None
) -> None:
    """
    Executa um único experimento e plota a evolução do fitness.
    
    Args:
        population_size: Tamanho da população
        crossover_rate: Taxa de crossover
        mutation_rate: Taxa de mutação
        elitism: Se True, os melhores indivíduos são preservados entre gerações
        save_dir: Diretório para salvar os gráficos (opcional)
    """
    # Cria e executa o algoritmo genético
    ga = PatternRecognitionGA(
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism=elitism
    )
    
    best_individual, generations, execution_time = ga.run_experiment()
    
    # Imprime os resultados
    print(f"Melhor indivíduo: {best_individual}")
    print(f"Gerações: {generations}")
    print(f"Tempo de execução: {execution_time:.4f}s")
    
    # Plota a evolução do fitness
    title = f"Evolução do Fitness (Pop={population_size}, CR={crossover_rate}, MR={mutation_rate})"
    ga.plot_fitness_history(title=title, save_path=os.path.join(save_dir, 'evolucao_fitness.png') if save_dir else None)

if __name__ == "__main__":
    # Cria diretório para salvar os resultados
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')
    os.makedirs(results_dir, exist_ok=True)
    
    # Executa um único experimento para visualizar a evolução do fitness
    run_single_experiment(save_dir=results_dir)
    
    # Compara diferentes taxas de crossover e mutação
    compare_parameters(save_dir=results_dir)
    
    # Compara o uso de apenas crossover, apenas mutação, e ambos
    compare_operators(save_dir=results_dir)
