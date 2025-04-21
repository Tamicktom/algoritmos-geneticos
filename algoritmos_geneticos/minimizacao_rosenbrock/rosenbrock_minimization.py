"""
Implementação de algoritmo genético para minimização da função de Rosenbrock.
Objetivo: Minimizar a função f(x,y) = (1-x)² + 100(y-x²)² no intervalo [-10, 10].
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
from utils.genetic_algorithm import Individual

def rosenbrock(x: float, y: float) -> float:
    """
    Função de Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    
    Args:
        x: Valor no intervalo [-10, 10]
        y: Valor no intervalo [-10, 10]
        
    Returns:
        Valor da função de Rosenbrock
    """
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

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

class RosenbrockIndividual(Individual):
    """
    Classe que representa um indivíduo para o problema de minimização da função de Rosenbrock.
    """
    def __init__(self, chromosome: List[int], chromosome_length: int, fitness: float = None):
        """
        Inicializa um indivíduo com um cromossomo e valor de fitness opcional.
        
        Args:
            chromosome: Lista de genes que compõem o cromossomo
            chromosome_length: Comprimento do cromossomo para cada variável (x e y)
            fitness: Valor de aptidão do indivíduo (calculado posteriormente se None)
        """
        super().__init__(chromosome, fitness)
        self.chromosome_length = chromosome_length
        
    def get_x_y(self, min_val: float = -10.0, max_val: float = 10.0) -> Tuple[float, float]:
        """
        Obtém os valores de x e y a partir do cromossomo.
        
        Args:
            min_val: Valor mínimo do intervalo
            max_val: Valor máximo do intervalo
            
        Returns:
            Tupla contendo os valores de x e y
        """
        # Divide o cromossomo em duas partes para x e y
        x_binary = self.chromosome[:self.chromosome_length]
        y_binary = self.chromosome[self.chromosome_length:]
        
        # Converte para valores decimais
        x = binary_to_decimal(x_binary, min_val, max_val)
        y = binary_to_decimal(y_binary, min_val, max_val)
        
        return x, y
    
    def __str__(self) -> str:
        """Representação em string do indivíduo."""
        x, y = self.get_x_y()
        return f"x: {x:.6f}, y: {y:.6f}, Fitness: {self.fitness}"

class RosenbrockGA:
    """
    Algoritmo genético para minimização da função de Rosenbrock.
    """
    def __init__(
        self,
        population_size: int = 100,
        chromosome_length: int = 16,  # Para cada variável (x e y)
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism: bool = True,
        elitism_size: int = 2,
        max_generations: int = 1000,
        selection_method: str = "tournament"  # "roulette", "tournament", "sus"
    ):
        """
        Inicializa o algoritmo genético para minimização da função de Rosenbrock.
        
        Args:
            population_size: Tamanho da população
            chromosome_length: Comprimento do cromossomo para cada variável (x e y)
            crossover_rate: Taxa de crossover
            mutation_rate: Taxa de mutação
            elitism: Se True, os melhores indivíduos são preservados entre gerações
            elitism_size: Número de indivíduos preservados pelo elitismo
            max_generations: Número máximo de gerações
            selection_method: Método de seleção ("roulette", "tournament", "sus")
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.total_length = chromosome_length * 2  # x e y
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.elitism_size = elitism_size
        self.max_generations = max_generations
        self.selection_method = selection_method
        
        self.population = []
        self.best_individual = None
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.execution_time = 0
        
    def calculate_fitness(self, individual: RosenbrockIndividual) -> float:
        """
        Calcula o fitness de um indivíduo para o problema de minimização da função de Rosenbrock.
        
        Args:
            individual: Indivíduo a ser avaliado
            
        Returns:
            Valor de fitness (quanto maior, melhor)
        """
        x, y = individual.get_x_y()
        
        # Calcula o valor da função de Rosenbrock (queremos minimizar)
        rosenbrock_value = rosenbrock(x, y)
        
        # Como queremos maximizar o fitness, invertemos o valor
        # Adicionamos 1 para evitar divisão por zero e limitamos para evitar valores muito grandes
        return 1.0 / (1.0 + rosenbrock_value)
    
    def initialize_population(self) -> None:
        """
        Inicializa a população com indivíduos aleatórios.
        """
        self.population = []
        for _ in range(self.population_size):
            # Cria um cromossomo aleatório com genes 0 ou 1
            chromosome = [np.random.randint(0, 2) for _ in range(self.total_length)]
            individual = RosenbrockIndividual(chromosome, self.chromosome_length)
            self.population.append(individual)
        
        # Avalia o fitness de cada indivíduo na população inicial
        self.evaluate_population()
    
    def evaluate_population(self) -> None:
        """
        Avalia o fitness de cada indivíduo na população.
        """
        for individual in self.population:
            if individual.fitness is None:
                individual.fitness = self.calculate_fitness(individual)
        
        # Atualiza o melhor indivíduo
        current_best = max(self.population, key=lambda ind: ind.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = RosenbrockIndividual(current_best.chromosome.copy(), self.chromosome_length, current_best.fitness)
        
        # Registra estatísticas
        fitnesses = [ind.fitness for ind in self.population]
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
    
    def selection_roulette(self) -> RosenbrockIndividual:
        """
        Seleciona um indivíduo usando o método da roleta.
        
        Returns:
            O indivíduo selecionado
        """
        # Calcula a soma total de fitness
        total_fitness = sum(ind.fitness for ind in self.population)
        
        # Se todos os fitness forem zero, seleciona aleatoriamente
        if total_fitness == 0:
            return np.random.choice(self.population)
        
        # Seleciona um ponto aleatório na roleta
        selection_point = np.random.uniform(0, total_fitness)
        
        # Encontra o indivíduo correspondente ao ponto selecionado
        current_sum = 0
        for individual in self.population:
            current_sum += individual.fitness
            if current_sum >= selection_point:
                return individual
        
        # Caso de segurança (não deve ocorrer)
        return self.population[-1]
    
    def selection_tournament(self, tournament_size: int = 3) -> RosenbrockIndividual:
        """
        Seleciona um indivíduo usando o método de torneio.
        
        Args:
            tournament_size: Número de indivíduos que participam do torneio
            
        Returns:
            O indivíduo vencedor do torneio
        """
        # Seleciona aleatoriamente indivíduos para o torneio
        tournament = np.random.choice(self.population, min(tournament_size, len(self.population)), replace=False)
        
        # Retorna o indivíduo com maior fitness
        return max(tournament, key=lambda ind: ind.fitness)
    
    def selection_sus(self, num_parents: int) -> List[RosenbrockIndividual]:
        """
        Seleciona indivíduos usando Amostragem Universal Estocástica.
        
        Args:
            num_parents: Número de indivíduos a serem selecionados
            
        Returns:
            Lista de indivíduos selecionados
        """
        # Calcula a soma total de fitness
        total_fitness = sum(ind.fitness for ind in self.population)
        
        # Se todos os fitness forem zero, seleciona aleatoriamente
        if total_fitness == 0:
            return np.random.choice(self.population, num_parents, replace=True).tolist()
        
        # Calcula o passo entre pontos de seleção
        step = total_fitness / num_parents
        
        # Seleciona um ponto inicial aleatório
        start = np.random.uniform(0, step)
        
        # Pontos de seleção
        points = [start + i * step for i in range(num_parents)]
        
        # Seleciona os indivíduos
        selected = []
        for point in points:
            current_sum = 0
            for individual in self.population:
                current_sum += individual.fitness
                if current_sum >= point:
                    selected.append(individual)
                    break
            
            # Caso de segurança (não deve ocorrer)
            if len(selected) < len(points):
                selected.append(self.population[-1])
        
        return selected
    
    def crossover_single_point(self, parent1: RosenbrockIndividual, parent2: RosenbrockIndividual) -> Tuple[RosenbrockIndividual, RosenbrockIndividual]:
        """
        Realiza crossover de ponto único entre dois pais.
        
        Args:
            parent1: Primeiro pai
            parent2: Segundo pai
            
        Returns:
            Tupla com os dois filhos gerados
        """
        # Verifica se o crossover ocorrerá
        if np.random.random() > self.crossover_rate:
            return (
                RosenbrockIndividual(parent1.chromosome.copy(), self.chromosome_length),
                RosenbrockIndividual(parent2.chromosome.copy(), self.chromosome_length)
            )
        
        # Seleciona um ponto de corte aleatório
        crossover_point = np.random.randint(1, self.total_length - 1)
        
        # Cria os cromossomos dos filhos
        child1_chromosome = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
        child2_chromosome = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]
        
        # Cria os indivíduos filhos
        child1 = RosenbrockIndividual(child1_chromosome, self.chromosome_length)
        child2 = RosenbrockIndividual(child2_chromosome, self.chromosome_length)
        
        return child1, child2
    
    def crossover_two_point(self, parent1: RosenbrockIndividual, parent2: RosenbrockIndividual) -> Tuple[RosenbrockIndividual, RosenbrockIndividual]:
        """
        Realiza crossover de dois pontos entre dois pais.
        
        Args:
            parent1: Primeiro pai
            parent2: Segundo pai
            
        Returns:
            Tupla com os dois filhos gerados
        """
        # Verifica se o crossover ocorrerá
        if np.random.random() > self.crossover_rate:
            return (
                RosenbrockIndividual(parent1.chromosome.copy(), self.chromosome_length),
                RosenbrockIndividual(parent2.chromosome.copy(), self.chromosome_length)
            )
        
        # Seleciona dois pontos de corte aleatórios
        point1 = np.random.randint(1, self.total_length - 2)
        point2 = np.random.randint(point1 + 1, self.total_length - 1)
        
        # Cria os cromossomos dos filhos
        child1_chromosome = parent1.chromosome[:point1] + parent2.chromosome[point1:point2] + parent1.chromosome[point2:]
        child2_chromosome = parent2.chromosome[:point1] + parent1.chromosome[point1:point2] + parent2.chromosome[point2:]
        
        # Cria os indivíduos filhos
        child1 = RosenbrockIndividual(child1_chromosome, self.chromosome_length)
        child2 = RosenbrockIndividual(child2_chromosome, self.chromosome_length)
        
        return child1, child2
    
    def mutate(self, individual: RosenbrockIndividual) -> None:
        """
        Aplica mutação a um indivíduo.
        
        Args:
            individual: Indivíduo a ser mutado
        """
        for i in range(len(individual.chromosome)):
            # Para cada gene, verifica se ocorrerá mutação
            if np.random.random() < self.mutation_rate:
                # Inverte o valor do gene (0 -> 1 ou 1 -> 0)
                individual.chromosome[i] = 1 - individual.chromosome[i]
        
        # Reseta o fitness para que seja recalculado
        individual.fitness = None
    
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
                new_population.append(RosenbrockIndividual(sorted_population[i].chromosome.copy(), self.chromosome_length, sorted_population[i].fitness))
        
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
    
    def run(self) -> Tuple[RosenbrockIndividual, int, List[float], List[float]]:
        """
        Executa o algoritmo genético até atingir o critério de parada.
        
        Returns:
            Tupla contendo o melhor indivíduo, número de gerações,
            histórico do melhor fitness e histórico do fitness médio
        """
        # Inicializa a população
        self.initialize_population()
        
        # Loop principal
        for _ in range(self.max_generations):
            # Executa uma geração
            self.evolve()
        
        return (
            self.best_individual,
            self.generation,
            self.best_fitness_history,
            self.avg_fitness_history
        )
    
    def run_experiment(self) -> Tuple[RosenbrockIndividual, float, float, int, float]:
        """
        Executa o experimento e mede o tempo de execução.
        
        Returns:
            Tupla contendo o melhor indivíduo, melhor valor de x, melhor valor de y,
            número de gerações e tempo de execução
        """
        start_time = time.time()
        best_individual, generations, _, _ = self.run()
        self.execution_time = time.time() - start_time
        
        # Calcula os valores de x e y correspondentes ao melhor indivíduo
        x, y = best_individual.get_x_y()
        
        return best_individual, x, y, generations, self.execution_time
    
    def plot_fitness_history(self, title: str = "Evolução do Fitness", save_path: Optional[str] = None) -> None:
        """
        Plota o histórico de fitness.
        
        Args:
            title: Título do gráfico
            save_path: Caminho para salvar o gráfico (opcional)
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.best_fitness_history)), self.best_fitness_history, 'b-', label='Melhor Fitness')
        plt.plot(range(len(self.avg_fitness_history)), self.avg_fitness_history, 'r-', label='Fitness Médio')
        plt.title(title)
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

def plot_rosenbrock_function(save_dir: Optional[str] = None) -> None:
    """
    Plota a função de Rosenbrock em 3D.
    
    Args:
        save_dir: Diretório para salvar o gráfico (opcional)
    """
    # Cria uma grade de pontos
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Calcula o valor da função para cada ponto
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = rosenbrock(X[j, i], Y[j, i])
    
    # Plota a superfície 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Adiciona uma barra de cores
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Configura os eixos
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title('Função de Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²')
    
    # Ajusta a visualização
    ax.view_init(elev=30, azim=45)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'funcao_rosenbrock_3d.png'))
    
    plt.show()
    
    # Plota o contorno 2D
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
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'funcao_rosenbrock_contorno.png'))
    
    plt.show()

def test_population_sizes(
    population_sizes: List[int] = [20, 50, 100, 200, 500],
    chromosome_length: int = 16,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    with_elitism: bool = True,
    num_experiments: int = 10,
    save_dir: Optional[str] = None
) -> None:
    """
    Testa diferentes tamanhos de população.
    
    Args:
        population_sizes: Lista de tamanhos de população a serem testados
        chromosome_length: Comprimento do cromossomo para cada variável (x e y)
        crossover_rate: Taxa de crossover
        mutation_rate: Taxa de mutação
        with_elitism: Se True, testa com elitismo; se False, testa sem elitismo
        num_experiments: Número de experimentos para cada tamanho
        save_dir: Diretório para salvar os gráficos (opcional)
    """
    results = {size: {"fitness": [], "x": [], "y": [], "generations": [], "times": []} for size in population_sizes}
    
    for size in population_sizes:
        print(f"Testando tamanho de população: {size}")
        
        for _ in range(num_experiments):
            ga = RosenbrockGA(
                population_size=size,
                chromosome_length=chromosome_length,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                elitism=with_elitism
            )
            
            best_individual, x, y, generations, execution_time = ga.run_experiment()
            
            results[size]["fitness"].append(best_individual.fitness)
            results[size]["x"].append(x)
            results[size]["y"].append(y)
            results[size]["generations"].append(generations)
            results[size]["times"].append(execution_time)
        
        # Calcula estatísticas
        avg_fitness = np.mean(results[size]["fitness"])
        std_fitness = np.std(results[size]["fitness"])
        avg_x = np.mean(results[size]["x"])
        std_x = np.std(results[size]["x"])
        avg_y = np.mean(results[size]["y"])
        std_y = np.std(results[size]["y"])
        avg_time = np.mean(results[size]["times"])
        std_time = np.std(results[size]["times"])
        
        print(f"  Fitness: {avg_fitness:.6f} ± {std_fitness:.6f}")
        print(f"  x: {avg_x:.6f} ± {std_x:.6f}")
        print(f"  y: {avg_y:.6f} ± {std_y:.6f}")
        print(f"  Tempo: {avg_time:.4f} ± {std_time:.4f}s")
    
    # Plota os resultados
    plt.figure(figsize=(15, 10))
    
    # Dados para o gráfico
    sizes = list(results.keys())
    avg_fitness = [1.0 / np.mean(results[size]["fitness"]) - 1.0 for size in sizes]  # Converte de volta para o valor da função
    std_fitness = [np.std(results[size]["fitness"]) for size in sizes]
    avg_x = [np.mean(results[size]["x"]) for size in sizes]
    std_x = [np.std(results[size]["x"]) for size in sizes]
    avg_y = [np.mean(results[size]["y"]) for size in sizes]
    std_y = [np.std(results[size]["y"]) for size in sizes]
    avg_times = [np.mean(results[size]["times"]) for size in sizes]
    
    # Gráfico para o valor da função
    plt.subplot(2, 2, 1)
    plt.errorbar(sizes, avg_fitness, yerr=std_fitness, marker='o', linestyle='-')
    plt.xlabel('Tamanho da População')
    plt.ylabel('Valor Médio da Função')
    plt.title('Qualidade da Solução')
    plt.grid(True)
    
    # Gráfico para o valor de x
    plt.subplot(2, 2, 2)
    plt.errorbar(sizes, avg_x, yerr=std_x, marker='o', linestyle='-')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Valor Ótimo (x=1)')
    plt.xlabel('Tamanho da População')
    plt.ylabel('Valor Médio de x')
    plt.title('Convergência para x Ótimo')
    plt.legend()
    plt.grid(True)
    
    # Gráfico para o valor de y
    plt.subplot(2, 2, 3)
    plt.errorbar(sizes, avg_y, yerr=std_y, marker='o', linestyle='-')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Valor Ótimo (y=1)')
    plt.xlabel('Tamanho da População')
    plt.ylabel('Valor Médio de y')
    plt.title('Convergência para y Ótimo')
    plt.legend()
    plt.grid(True)
    
    # Gráfico para o tempo de execução
    plt.subplot(2, 2, 4)
    plt.plot(sizes, avg_times, marker='o', linestyle='-')
    plt.xlabel('Tamanho da População')
    plt.ylabel('Tempo Médio (s)')
    plt.title('Tempo de Execução')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'comparacao_tamanhos_populacao_{"com" if with_elitism else "sem"}_elitismo.png'))
    
    plt.show()

def compare_elitism(
    population_size: int = 100,
    chromosome_length: int = 16,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    num_experiments: int = 10,
    save_dir: Optional[str] = None
) -> None:
    """
    Compara o desempenho com e sem elitismo.
    
    Args:
        population_size: Tamanho da população
        chromosome_length: Comprimento do cromossomo para cada variável (x e y)
        crossover_rate: Taxa de crossover
        mutation_rate: Taxa de mutação
        num_experiments: Número de experimentos para cada configuração
        save_dir: Diretório para salvar os gráficos (opcional)
    """
    # Resultados para cada configuração
    with_elitism_results = {"fitness": [], "x": [], "y": [], "generations": [], "times": [], "history": []}
    without_elitism_results = {"fitness": [], "x": [], "y": [], "generations": [], "times": [], "history": []}
    
    # Executa experimentos com elitismo
    print("Testando com elitismo")
    for _ in range(num_experiments):
        ga = RosenbrockGA(
            population_size=population_size,
            chromosome_length=chromosome_length,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism=True
        )
        
        best_individual, x, y, generations, execution_time = ga.run_experiment()
        
        with_elitism_results["fitness"].append(best_individual.fitness)
        with_elitism_results["x"].append(x)
        with_elitism_results["y"].append(y)
        with_elitism_results["generations"].append(generations)
        with_elitism_results["times"].append(execution_time)
        with_elitism_results["history"].append(ga.best_fitness_history)
    
    # Executa experimentos sem elitismo
    print("Testando sem elitismo")
    for _ in range(num_experiments):
        ga = RosenbrockGA(
            population_size=population_size,
            chromosome_length=chromosome_length,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism=False
        )
        
        best_individual, x, y, generations, execution_time = ga.run_experiment()
        
        without_elitism_results["fitness"].append(best_individual.fitness)
        without_elitism_results["x"].append(x)
        without_elitism_results["y"].append(y)
        without_elitism_results["generations"].append(generations)
        without_elitism_results["times"].append(execution_time)
        without_elitism_results["history"].append(ga.best_fitness_history)
    
    # Calcula estatísticas
    with_elitism_avg_fitness = np.mean(with_elitism_results["fitness"])
    with_elitism_std_fitness = np.std(with_elitism_results["fitness"])
    with_elitism_avg_x = np.mean(with_elitism_results["x"])
    with_elitism_std_x = np.std(with_elitism_results["x"])
    with_elitism_avg_y = np.mean(with_elitism_results["y"])
    with_elitism_std_y = np.std(with_elitism_results["y"])
    with_elitism_avg_time = np.mean(with_elitism_results["times"])
    with_elitism_std_time = np.std(with_elitism_results["times"])
    
    without_elitism_avg_fitness = np.mean(without_elitism_results["fitness"])
    without_elitism_std_fitness = np.std(without_elitism_results["fitness"])
    without_elitism_avg_x = np.mean(without_elitism_results["x"])
    without_elitism_std_x = np.std(without_elitism_results["x"])
    without_elitism_avg_y = np.mean(without_elitism_results["y"])
    without_elitism_std_y = np.std(without_elitism_results["y"])
    without_elitism_avg_time = np.mean(without_elitism_results["times"])
    without_elitism_std_time = np.std(without_elitism_results["times"])
    
    print("Com Elitismo:")
    print(f"  Fitness: {with_elitism_avg_fitness:.6f} ± {with_elitism_std_fitness:.6f}")
    print(f"  x: {with_elitism_avg_x:.6f} ± {with_elitism_std_x:.6f}")
    print(f"  y: {with_elitism_avg_y:.6f} ± {with_elitism_std_y:.6f}")
    print(f"  Tempo: {with_elitism_avg_time:.4f} ± {with_elitism_std_time:.4f}s")
    
    print("Sem Elitismo:")
    print(f"  Fitness: {without_elitism_avg_fitness:.6f} ± {without_elitism_std_fitness:.6f}")
    print(f"  x: {without_elitism_avg_x:.6f} ± {without_elitism_std_x:.6f}")
    print(f"  y: {without_elitism_avg_y:.6f} ± {without_elitism_std_y:.6f}")
    print(f"  Tempo: {without_elitism_avg_time:.4f} ± {without_elitism_std_time:.4f}s")
    
    # Plota a evolução do fitness médio
    plt.figure(figsize=(12, 6))
    
    # Calcula a média do histórico de fitness para cada geração
    with_elitism_avg_history = np.mean(with_elitism_results["history"], axis=0)
    without_elitism_avg_history = np.mean(without_elitism_results["history"], axis=0)
    
    plt.plot(range(len(with_elitism_avg_history)), with_elitism_avg_history, 'b-', label='Com Elitismo')
    plt.plot(range(len(without_elitism_avg_history)), without_elitism_avg_history, 'r-', label='Sem Elitismo')
    plt.xlabel('Geração')
    plt.ylabel('Fitness Médio')
    plt.title('Evolução do Fitness com e sem Elitismo')
    plt.legend()
    plt.grid(True)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'comparacao_elitismo.png'))
    
    plt.show()
    
    # Plota os resultados em barras
    plt.figure(figsize=(15, 5))
    
    # Configurações para os gráficos
    labels = ['Com Elitismo', 'Sem Elitismo']
    x_pos = np.arange(len(labels))
    width = 0.35
    
    # Gráfico para o valor da função
    plt.subplot(1, 3, 1)
    function_values = [1.0 / with_elitism_avg_fitness - 1.0, 1.0 / without_elitism_avg_fitness - 1.0]
    function_errors = [with_elitism_std_fitness, without_elitism_std_fitness]
    plt.bar(x_pos, function_values, width, yerr=function_errors, capsize=10)
    plt.ylabel('Valor da Função')
    plt.title('Qualidade da Solução')
    plt.xticks(x_pos, labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Gráfico para o erro em x
    plt.subplot(1, 3, 2)
    x_errors = [abs(with_elitism_avg_x - 1.0), abs(without_elitism_avg_x - 1.0)]
    plt.bar(x_pos, x_errors, width, capsize=10)
    plt.ylabel('Erro Absoluto em x')
    plt.title('Precisão em x')
    plt.xticks(x_pos, labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Gráfico para o erro em y
    plt.subplot(1, 3, 3)
    y_errors = [abs(with_elitism_avg_y - 1.0), abs(without_elitism_avg_y - 1.0)]
    plt.bar(x_pos, y_errors, width, capsize=10)
    plt.ylabel('Erro Absoluto em y')
    plt.title('Precisão em y')
    plt.xticks(x_pos, labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'comparacao_elitismo_barras.png'))
    
    plt.show()

def run_single_experiment(
    population_size: int = 100,
    chromosome_length: int = 16,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    elitism: bool = True,
    save_dir: Optional[str] = None
) -> None:
    """
    Executa um único experimento com o algoritmo genético e plota a evolução do fitness.
    
    Args:
        population_size: Tamanho da população
        chromosome_length: Comprimento do cromossomo para cada variável (x e y)
        crossover_rate: Taxa de crossover
        mutation_rate: Taxa de mutação
        elitism: Se True, usa elitismo
        save_dir: Diretório para salvar os gráficos (opcional)
    """
    # Cria e executa o algoritmo genético
    ga = RosenbrockGA(
        population_size=population_size,
        chromosome_length=chromosome_length,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism=elitism
    )
    
    best_individual, x, y, generations, execution_time = ga.run_experiment()
    
    # Calcula o valor da função de Rosenbrock
    rosenbrock_value = rosenbrock(x, y)
    
    # Imprime os resultados
    print(f"Melhor indivíduo: {best_individual}")
    print(f"Valor de x: {x:.6f}")
    print(f"Valor de y: {y:.6f}")
    print(f"Valor da função de Rosenbrock: {rosenbrock_value:.6f}")
    print(f"Gerações: {generations}")
    print(f"Tempo de execução: {execution_time:.4f}s")
    
    # Plota a evolução do fitness
    title = f"Evolução do Fitness (Pop={population_size}, CR={crossover_rate}, MR={mutation_rate}, {'Com' if elitism else 'Sem'} Elitismo)"
    ga.plot_fitness_history(title=title, save_path=os.path.join(save_dir, 'evolucao_fitness.png') if save_dir else None)

if __name__ == "__main__":
    # Cria diretório para salvar os resultados
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultados')
    os.makedirs(results_dir, exist_ok=True)
    
    # Plota a função de Rosenbrock
    plot_rosenbrock_function(save_dir=results_dir)
    
    # Executa um único experimento para visualizar a evolução do fitness
    run_single_experiment(save_dir=results_dir)
    
    # Compara o desempenho com e sem elitismo
    compare_elitism(save_dir=results_dir)
    
    # Testa diferentes tamanhos de população com elitismo
    test_population_sizes(with_elitism=True, save_dir=results_dir)
    
    # Testa diferentes tamanhos de população sem elitismo
    test_population_sizes(with_elitism=False, save_dir=results_dir)
