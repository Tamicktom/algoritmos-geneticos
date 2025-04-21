"""
Implementação de algoritmo genético para maximização de função.
Objetivo: Maximizar a função g(x) = 2^(-2((x-0,1)/0,9)²) (sin(5πx))^6 no intervalo [0, 1].
"""
import sys
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable

# Adiciona o diretório pai ao path para importar o módulo utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.genetic_algorithm import GeneticAlgorithm, Individual

def g(x: float) -> float:
    """
    Função a ser maximizada: g(x) = 2^(-2((x-0,1)/0,9)²) (sin(5πx))^6
    
    Args:
        x: Valor no intervalo [0, 1]
        
    Returns:
        Valor da função g(x)
    """
    exponent = -2 * ((x - 0.1) / 0.9) ** 2
    sine_term = math.sin(5 * math.pi * x) ** 6
    return (2 ** exponent) * sine_term

def binary_to_decimal(binary: List[int], min_val: float, max_val: float) -> float:
    """
    Converte uma representação binária para um valor decimal no intervalo [min_val, max_val].
    
    Args:
        binary: Lista de bits (0 ou 1)
        min_val: Valor mínimo do intervalo
        max_val: Valor máximo do intervalo
        
    Returns:
        Valor decimal correspondente
    """
    # Converte a lista de bits para um número inteiro
    decimal = 0
    for bit in binary:
        decimal = decimal * 2 + bit
    
    # Normaliza para o intervalo [min_val, max_val]
    max_binary = 2 ** len(binary) - 1
    return min_val + (max_val - min_val) * decimal / max_binary

def calculate_fitness(chromosome: List[int]) -> float:
    """
    Calcula o fitness de um cromossomo para o problema de maximização da função g(x).
    
    Args:
        chromosome: Lista de genes que compõem o cromossomo
        
    Returns:
        Valor de fitness (quanto maior, melhor)
    """
    # Converte o cromossomo binário para um valor decimal no intervalo [0, 1]
    x = binary_to_decimal(chromosome, 0.0, 1.0)
    
    # Calcula o valor da função g(x)
    return g(x)

class FunctionMaximizationGA(GeneticAlgorithm):
    """
    Algoritmo genético específico para maximização da função g(x).
    """
    def __init__(
        self,
        population_size: int = 100,
        chromosome_length: int = 20,  # Precisão de pelo menos 3 casas decimais
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism: bool = True,
        elitism_size: int = 2,
        max_generations: int = 1000,
        selection_method: str = "roulette"  # "roulette", "tournament", "sus"
    ):
        """
        Inicializa o algoritmo genético para maximização da função g(x).
        
        Args:
            population_size: Tamanho da população
            chromosome_length: Comprimento do cromossomo (número de bits)
            crossover_rate: Taxa de crossover
            mutation_rate: Taxa de mutação
            elitism: Se True, os melhores indivíduos são preservados entre gerações
            elitism_size: Número de indivíduos preservados pelo elitismo
            max_generations: Número máximo de gerações
            selection_method: Método de seleção ("roulette", "tournament", "sus")
        """
        super().__init__(
            population_size=population_size,
            chromosome_length=chromosome_length,
            fitness_function=calculate_fitness,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism=elitism,
            elitism_size=elitism_size,
            max_generations=max_generations
        )
        
        self.selection_method = selection_method
        self.execution_time = 0
        self.best_x = None
        self.best_g_x = None
    
    def evolve(self) -> None:
        """
        Executa uma geração do algoritmo genético com o método de seleção especificado.
        """
        new_population = []
        
        # Aplica elitismo se configurado
        if self.elitism:
            # Ordena a população pelo fitness (decrescente)
            sorted_population = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
            # Adiciona os melhores indivíduos à nova população
            for i in range(min(self.elitism_size, len(sorted_population))):
                new_population.append(sorted_population[i].copy())
        
        # Preenche o resto da população com novos indivíduos
        while len(new_population) < self.population_size:
            # Seleciona dois pais usando o método especificado
            if self.selection_method == "tournament":
                parent1 = self.selection_tournament()
                parent2 = self.selection_tournament()
            elif self.selection_method == "sus":
                # Para SUS, selecionamos todos os pais de uma vez
                if len(new_population) == (self.elitism_size if self.elitism else 0):
                    # Número de pais necessários para completar a população
                    num_parents_needed = self.population_size - len(new_population)
                    # Arredonda para cima para um número par
                    num_parents = (num_parents_needed + 1) // 2 * 2
                    parents = self.selection_sus(num_parents)
                    
                    # Realiza crossover e mutação para cada par de pais
                    for i in range(0, len(parents), 2):
                        if i + 1 < len(parents):
                            child1, child2 = self.crossover_single_point(parents[i], parents[i+1])
                            self.mutate(child1)
                            self.mutate(child2)
                            new_population.append(child1)
                            if len(new_population) < self.population_size:
                                new_population.append(child2)
                    
                    # Pula o resto do loop
                    continue
                else:
                    # Não deveria chegar aqui, mas por segurança
                    parent1 = self.selection_roulette()
                    parent2 = self.selection_roulette()
            else:  # "roulette" (padrão)
                parent1 = self.selection_roulette()
                parent2 = self.selection_roulette()
            
            # Realiza crossover
            child1, child2 = self.crossover_single_point(parent1, parent2)
            
            # Aplica mutação
            self.mutate(child1)
            self.mutate(child2)
            
            # Adiciona os filhos à nova população
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # Substitui a população antiga pela nova
        self.population = new_population
        
        # Avalia a nova população
        self.evaluate_population()
        
        # Incrementa o contador de gerações
        self.generation += 1
    
    def run_experiment(self) -> Tuple[Individual, float, float, int, float]:
        """
        Executa o experimento e mede o tempo de execução.
        
        Returns:
            Tupla contendo o melhor indivíduo, melhor valor de x, melhor valor de g(x),
            número de gerações e tempo de execução
        """
        start_time = time.time()
        best_individual, generations, _, _ = self.run()
        self.execution_time = time.time() - start_time
        
        # Calcula o valor de x correspondente ao melhor indivíduo
        self.best_x = binary_to_decimal(best_individual.chromosome, 0.0, 1.0)
        self.best_g_x = g(self.best_x)
        
        return best_individual, self.best_x, self.best_g_x, generations, self.execution_time

class HillClimbing:
    """
    Implementação do algoritmo de Subida da Colina para comparação.
    """
    def __init__(
        self,
        objective_function: Callable[[float], float],
        min_val: float,
        max_val: float,
        step_size: float = 0.01,
        max_iterations: int = 1000,
        max_stagnation: int = 100
    ):
        """
        Inicializa o algoritmo de Subida da Colina.
        
        Args:
            objective_function: Função a ser maximizada
            min_val: Valor mínimo do intervalo de busca
            max_val: Valor máximo do intervalo de busca
            step_size: Tamanho do passo para explorar a vizinhança
            max_iterations: Número máximo de iterações
            max_stagnation: Número máximo de iterações sem melhoria
        """
        self.objective_function = objective_function
        self.min_val = min_val
        self.max_val = max_val
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.max_stagnation = max_stagnation
        
        self.best_x = None
        self.best_value = float('-inf')
        self.iterations = 0
        self.execution_time = 0
        self.fitness_history = []
    
    def run(self) -> Tuple[float, float, int, float]:
        """
        Executa o algoritmo de Subida da Colina.
        
        Returns:
            Tupla contendo o melhor valor de x, melhor valor da função,
            número de iterações e tempo de execução
        """
        start_time = time.time()
        
        # Inicializa com um ponto aleatório no intervalo
        current_x = np.random.uniform(self.min_val, self.max_val)
        current_value = self.objective_function(current_x)
        
        self.best_x = current_x
        self.best_value = current_value
        self.fitness_history.append(current_value)
        
        stagnation_count = 0
        
        for i in range(self.max_iterations):
            self.iterations = i + 1
            
            # Gera um vizinho aleatório dentro do intervalo
            step = np.random.uniform(-self.step_size, self.step_size)
            neighbor_x = current_x + step
            
            # Garante que o vizinho está dentro do intervalo
            neighbor_x = max(self.min_val, min(self.max_val, neighbor_x))
            
            # Avalia o vizinho
            neighbor_value = self.objective_function(neighbor_x)
            
            # Se o vizinho for melhor, move-se para ele
            if neighbor_value > current_value:
                current_x = neighbor_x
                current_value = neighbor_value
                stagnation_count = 0
                
                # Atualiza o melhor encontrado
                if current_value > self.best_value:
                    self.best_x = current_x
                    self.best_value = current_value
            else:
                stagnation_count += 1
            
            self.fitness_history.append(self.best_value)
            
            # Critério de parada: estagnação
            if stagnation_count >= self.max_stagnation:
                break
        
        self.execution_time = time.time() - start_time
        
        return self.best_x, self.best_value, self.iterations, self.execution_time

class SimulatedAnnealing:
    """
    Implementação do algoritmo de Recozimento Simulado para comparação.
    """
    def __init__(
        self,
        objective_function: Callable[[float], float],
        min_val: float,
        max_val: float,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.95,
        step_size: float = 0.1,
        max_iterations: int = 1000
    ):
        """
        Inicializa o algoritmo de Recozimento Simulado.
        
        Args:
            objective_function: Função a ser maximizada
            min_val: Valor mínimo do intervalo de busca
            max_val: Valor máximo do intervalo de busca
            initial_temperature: Temperatura inicial
            cooling_rate: Taxa de resfriamento
            step_size: Tamanho do passo para explorar a vizinhança
            max_iterations: Número máximo de iterações
        """
        self.objective_function = objective_function
        self.min_val = min_val
        self.max_val = max_val
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.step_size = step_size
        self.max_iterations = max_iterations
        
        self.best_x = None
        self.best_value = float('-inf')
        self.iterations = 0
        self.execution_time = 0
        self.fitness_history = []
    
    def run(self) -> Tuple[float, float, int, float]:
        """
        Executa o algoritmo de Recozimento Simulado.
        
        Returns:
            Tupla contendo o melhor valor de x, melhor valor da função,
            número de iterações e tempo de execução
        """
        start_time = time.time()
        
        # Inicializa com um ponto aleatório no intervalo
        current_x = np.random.uniform(self.min_val, self.max_val)
        current_value = self.objective_function(current_x)
        
        self.best_x = current_x
        self.best_value = current_value
        self.fitness_history.append(current_value)
        
        temperature = self.initial_temperature
        
        for i in range(self.max_iterations):
            self.iterations = i + 1
            
            # Gera um vizinho aleatório dentro do intervalo
            step = np.random.uniform(-self.step_size, self.step_size)
            neighbor_x = current_x + step
            
            # Garante que o vizinho está dentro do intervalo
            neighbor_x = max(self.min_val, min(self.max_val, neighbor_x))
            
            # Avalia o vizinho
            neighbor_value = self.objective_function(neighbor_x)
            
            # Calcula a diferença de energia (negativa para maximização)
            delta_e = neighbor_value - current_value
            
            # Decide se aceita o novo estado
            if delta_e > 0:  # Sempre aceita se for melhor
                current_x = neighbor_x
                current_value = neighbor_value
                
                # Atualiza o melhor encontrado
                if current_value > self.best_value:
                    self.best_x = current_x
                    self.best_value = current_value
            else:
                # Aceita com uma probabilidade que depende da temperatura
                probability = math.exp(delta_e / temperature)
                if np.random.random() < probability:
                    current_x = neighbor_x
                    current_value = neighbor_value
            
            self.fitness_history.append(self.best_value)
            
            # Reduz a temperatura
            temperature *= self.cooling_rate
            
            # Critério de parada: temperatura muito baixa
            if temperature < 1e-10:
                break
        
        self.execution_time = time.time() - start_time
        
        return self.best_x, self.best_value, self.iterations, self.execution_time

def plot_function(save_dir: Optional[str] = None) -> None:
    """
    Plota a função g(x) no intervalo [0, 1].
    
    Args:
        save_dir: Diretório para salvar o gráfico (opcional)
    """
    x_values = np.linspace(0, 1, 1000)
    y_values = [g(x) for x in x_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values)
    plt.title('Função g(x) = 2^(-2((x-0,1)/0,9)²) (sin(5πx))^6')
    plt.xlabel('x')
    plt.ylabel('g(x)')
    plt.grid(True)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'funcao_g_x.png'))
    
    plt.show()

def compare_selection_methods(
    population_size: int = 100,
    chromosome_length: int = 20,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    num_experiments: int = 10,
    save_dir: Optional[str] = None
) -> None:
    """
    Compara diferentes métodos de seleção.
    
    Args:
        population_size: Tamanho da população
        chromosome_length: Comprimento do cromossomo
        crossover_rate: Taxa de crossover
        mutation_rate: Taxa de mutação
        num_experiments: Número de experimentos para cada método
        save_dir: Diretório para salvar os gráficos (opcional)
    """
    methods = ["roulette", "tournament", "sus"]
    results = {method: {"generations": [], "times": [], "values": []} for method in methods}
    
    for method in methods:
        print(f"Testando método de seleção: {method}")
        
        for _ in range(num_experiments):
            ga = FunctionMaximizationGA(
                population_size=population_size,
                chromosome_length=chromosome_length,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                selection_method=method
            )
            
            _, best_x, best_g_x, generations, execution_time = ga.run_experiment()
            
            results[method]["generations"].append(generations)
            results[method]["times"].append(execution_time)
            results[method]["values"].append(best_g_x)
        
        avg_gen = np.mean(results[method]["generations"])
        std_gen = np.std(results[method]["generations"])
        avg_time = np.mean(results[method]["times"])
        std_time = np.std(results[method]["times"])
        avg_value = np.mean(results[method]["values"])
        std_value = np.std(results[method]["values"])
        
        print(f"  Gerações: {avg_gen:.2f} ± {std_gen:.2f}")
        print(f"  Tempo: {avg_time:.4f} ± {std_time:.4f}s")
        print(f"  Valor g(x): {avg_value:.6f} ± {std_value:.6f}")
    
    # Plota os resultados
    plt.figure(figsize=(12, 6))
    
    # Dados para o gráfico
    method_names = list(results.keys())
    avg_generations = [np.mean(results[method]["generations"]) for method in method_names]
    std_generations = [np.std(results[method]["generations"]) for method in method_names]
    avg_values = [np.mean(results[method]["values"]) for method in method_names]
    
    # Gráfico de barras para gerações
    plt.subplot(1, 2, 1)
    plt.bar(method_names, avg_generations, yerr=std_generations, capsize=10)
    plt.ylabel('Número Médio de Gerações')
    plt.title('Gerações até Convergência')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Gráfico de barras para valores
    plt.subplot(1, 2, 2)
    plt.bar(method_names, avg_values)
    plt.ylabel('Valor Médio de g(x)')
    plt.title('Qualidade da Solução')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'comparacao_metodos_selecao.png'))
    
    plt.show()

def compare_algorithms(
    population_size: int = 100,
    chromosome_length: int = 20,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    num_experiments: int = 10,
    save_dir: Optional[str] = None
) -> None:
    """
    Compara o algoritmo genético com Subida da Colina e Recozimento Simulado.
    
    Args:
        population_size: Tamanho da população para o algoritmo genético
        chromosome_length: Comprimento do cromossomo para o algoritmo genético
        crossover_rate: Taxa de crossover para o algoritmo genético
        mutation_rate: Taxa de mutação para o algoritmo genético
        num_experiments: Número de experimentos para cada algoritmo
        save_dir: Diretório para salvar os gráficos (opcional)
    """
    # Resultados para cada algoritmo
    ga_results = {"values": [], "times": [], "iterations": []}
    hc_results = {"values": [], "times": [], "iterations": []}
    sa_results = {"values": [], "times": [], "iterations": []}
    
    # Executa experimentos com o algoritmo genético
    print("Testando Algoritmo Genético")
    for _ in range(num_experiments):
        ga = FunctionMaximizationGA(
            population_size=population_size,
            chromosome_length=chromosome_length,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate
        )
        
        _, _, best_g_x, generations, execution_time = ga.run_experiment()
        
        ga_results["values"].append(best_g_x)
        ga_results["times"].append(execution_time)
        ga_results["iterations"].append(generations)
    
    # Executa experimentos com Subida da Colina
    print("Testando Subida da Colina")
    for _ in range(num_experiments):
        hc = HillClimbing(
            objective_function=g,
            min_val=0.0,
            max_val=1.0
        )
        
        _, best_value, iterations, execution_time = hc.run()
        
        hc_results["values"].append(best_value)
        hc_results["times"].append(execution_time)
        hc_results["iterations"].append(iterations)
    
    # Executa experimentos com Recozimento Simulado
    print("Testando Recozimento Simulado")
    for _ in range(num_experiments):
        sa = SimulatedAnnealing(
            objective_function=g,
            min_val=0.0,
            max_val=1.0
        )
        
        _, best_value, iterations, execution_time = sa.run()
        
        sa_results["values"].append(best_value)
        sa_results["times"].append(execution_time)
        sa_results["iterations"].append(iterations)
    
    # Calcula estatísticas
    algorithms = ["Algoritmo Genético", "Subida da Colina", "Recozimento Simulado"]
    all_results = [ga_results, hc_results, sa_results]
    
    for i, alg in enumerate(algorithms):
        avg_value = np.mean(all_results[i]["values"])
        std_value = np.std(all_results[i]["values"])
        avg_time = np.mean(all_results[i]["times"])
        std_time = np.std(all_results[i]["times"])
        avg_iter = np.mean(all_results[i]["iterations"])
        std_iter = np.std(all_results[i]["iterations"])
        
        print(f"{alg}:")
        print(f"  Valor g(x): {avg_value:.6f} ± {std_value:.6f}")
        print(f"  Tempo: {avg_time:.4f} ± {std_time:.4f}s")
        print(f"  Iterações: {avg_iter:.2f} ± {std_iter:.2f}")
    
    # Plota os resultados
    plt.figure(figsize=(15, 5))
    
    # Dados para o gráfico
    avg_values = [np.mean(res["values"]) for res in all_results]
    std_values = [np.std(res["values"]) for res in all_results]
    avg_times = [np.mean(res["times"]) for res in all_results]
    std_times = [np.std(res["times"]) for res in all_results]
    avg_iters = [np.mean(res["iterations"]) for res in all_results]
    std_iters = [np.std(res["iterations"]) for res in all_results]
    
    # Gráfico de barras para valores
    plt.subplot(1, 3, 1)
    plt.bar(algorithms, avg_values, yerr=std_values, capsize=10)
    plt.ylabel('Valor Médio de g(x)')
    plt.title('Qualidade da Solução')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    # Gráfico de barras para tempos
    plt.subplot(1, 3, 2)
    plt.bar(algorithms, avg_times, yerr=std_times, capsize=10)
    plt.ylabel('Tempo Médio (s)')
    plt.title('Tempo de Execução')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    # Gráfico de barras para iterações
    plt.subplot(1, 3, 3)
    plt.bar(algorithms, avg_iters, yerr=std_iters, capsize=10)
    plt.ylabel('Número Médio de Iterações')
    plt.title('Iterações até Convergência')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'comparacao_algoritmos.png'))
    
    plt.show()

def run_single_experiment(
    population_size: int = 100,
    chromosome_length: int = 20,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    selection_method: str = "roulette",
    save_dir: Optional[str] = None
) -> None:
    """
    Executa um único experimento com o algoritmo genético e plota a evolução do fitness.
    
    Args:
        population_size: Tamanho da população
        chromosome_length: Comprimento do cromossomo
        crossover_rate: Taxa de crossover
        mutation_rate: Taxa de mutação
        selection_method: Método de seleção
        save_dir: Diretório para salvar os gráficos (opcional)
    """
    # Cria e executa o algoritmo genético
    ga = FunctionMaximizationGA(
        population_size=population_size,
        chromosome_length=chromosome_length,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        selection_method=selection_method
    )
    
    _, best_x, best_g_x, generations, execution_time = ga.run_experiment()
    
    # Imprime os resultados
    print(f"Melhor valor de x: {best_x:.6f}")
    print(f"Melhor valor de g(x): {best_g_x:.6f}")
    print(f"Gerações: {generations}")
    print(f"Tempo de execução: {execution_time:.4f}s")
    
    # Plota a evolução do fitness
    title = f"Evolução do Fitness (Pop={population_size}, CR={crossover_rate}, MR={mutation_rate}, {selection_method})"
    ga.plot_fitness_history(title=title, save_path=os.path.join(save_dir, 'evolucao_fitness.png') if save_dir else None)

if __name__ == "__main__":
    # Cria diretório para salvar os resultados
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')
    os.makedirs(results_dir, exist_ok=True)
    
    # Plota a função g(x)
    plot_function(save_dir=results_dir)
    
    # Executa um único experimento para visualizar a evolução do fitness
    run_single_experiment(save_dir=results_dir)
    
    # Compara diferentes métodos de seleção
    compare_selection_methods(save_dir=results_dir)
    
    # Compara o algoritmo genético com Subida da Colina e Recozimento Simulado
    compare_algorithms(save_dir=results_dir)
