"""
Módulo base para algoritmos genéticos.
Contém classes e funções genéricas que serão utilizadas pelos algoritmos específicos.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple, Any, Optional

class Individual:
    """
    Classe que representa um indivíduo na população do algoritmo genético.
    """
    def __init__(self, chromosome: List[int], fitness: float = None):
        """
        Inicializa um indivíduo com um cromossomo e valor de fitness opcional.
        
        Args:
            chromosome: Lista de genes que compõem o cromossomo
            fitness: Valor de aptidão do indivíduo (calculado posteriormente se None)
        """
        self.chromosome = chromosome
        self.fitness = fitness
    
    def __str__(self) -> str:
        """Representação em string do indivíduo."""
        return f"Cromossomo: {''.join(map(str, self.chromosome))}, Fitness: {self.fitness}"
    
    def copy(self) -> 'Individual':
        """Cria uma cópia do indivíduo."""
        return Individual(self.chromosome.copy(), self.fitness)

class GeneticAlgorithm:
    """
    Classe base para implementação de algoritmos genéticos.
    """
    def __init__(
        self,
        population_size: int,
        chromosome_length: int,
        fitness_function: Callable[[List[int]], float],
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism: bool = False,
        elitism_size: int = 1,
        max_generations: int = 1000,
        target_fitness: Optional[float] = None
    ):
        """
        Inicializa o algoritmo genético com os parâmetros especificados.
        
        Args:
            population_size: Tamanho da população
            chromosome_length: Comprimento do cromossomo de cada indivíduo
            fitness_function: Função que calcula o fitness de um cromossomo
            crossover_rate: Taxa de crossover (probabilidade de ocorrer cruzamento)
            mutation_rate: Taxa de mutação (probabilidade de um gene sofrer mutação)
            elitism: Se True, os melhores indivíduos são preservados entre gerações
            elitism_size: Número de indivíduos preservados pelo elitismo
            max_generations: Número máximo de gerações
            target_fitness: Valor de fitness alvo para critério de parada
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.fitness_function = fitness_function
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.elitism_size = elitism_size
        self.max_generations = max_generations
        self.target_fitness = target_fitness
        
        self.population = []
        self.best_individual = None
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def initialize_population(self) -> None:
        """
        Inicializa a população com indivíduos aleatórios.
        """
        self.population = []
        for _ in range(self.population_size):
            # Cria um cromossomo aleatório com genes 0 ou 1
            chromosome = [random.randint(0, 1) for _ in range(self.chromosome_length)]
            individual = Individual(chromosome)
            self.population.append(individual)
        
        # Avalia o fitness de cada indivíduo na população inicial
        self.evaluate_population()
    
    def evaluate_population(self) -> None:
        """
        Avalia o fitness de cada indivíduo na população.
        """
        for individual in self.population:
            if individual.fitness is None:
                individual.fitness = self.fitness_function(individual.chromosome)
        
        # Atualiza o melhor indivíduo
        current_best = max(self.population, key=lambda ind: ind.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.copy()
        
        # Registra estatísticas
        fitnesses = [ind.fitness for ind in self.population]
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
    
    def selection_roulette(self) -> Individual:
        """
        Seleciona um indivíduo usando o método da roleta.
        
        Returns:
            O indivíduo selecionado
        """
        # Calcula a soma total de fitness
        total_fitness = sum(ind.fitness for ind in self.population)
        
        # Se todos os fitness forem zero, seleciona aleatoriamente
        if total_fitness == 0:
            return random.choice(self.population)
        
        # Seleciona um ponto aleatório na roleta
        selection_point = random.uniform(0, total_fitness)
        
        # Encontra o indivíduo correspondente ao ponto selecionado
        current_sum = 0
        for individual in self.population:
            current_sum += individual.fitness
            if current_sum >= selection_point:
                return individual
        
        # Caso de segurança (não deve ocorrer)
        return self.population[-1]
    
    def selection_tournament(self, tournament_size: int = 3) -> Individual:
        """
        Seleciona um indivíduo usando o método de torneio.
        
        Args:
            tournament_size: Número de indivíduos que participam do torneio
            
        Returns:
            O indivíduo vencedor do torneio
        """
        # Seleciona aleatoriamente indivíduos para o torneio
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        
        # Retorna o indivíduo com maior fitness
        return max(tournament, key=lambda ind: ind.fitness)
    
    def selection_sus(self, num_parents: int) -> List[Individual]:
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
            return random.choices(self.population, k=num_parents)
        
        # Calcula o passo entre pontos de seleção
        step = total_fitness / num_parents
        
        # Seleciona um ponto inicial aleatório
        start = random.uniform(0, step)
        
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
    
    def crossover_single_point(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Realiza crossover de ponto único entre dois pais.
        
        Args:
            parent1: Primeiro pai
            parent2: Segundo pai
            
        Returns:
            Tupla com os dois filhos gerados
        """
        # Verifica se o crossover ocorrerá
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Seleciona um ponto de corte aleatório
        crossover_point = random.randint(1, self.chromosome_length - 1)
        
        # Cria os cromossomos dos filhos
        child1_chromosome = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
        child2_chromosome = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]
        
        # Cria os indivíduos filhos
        child1 = Individual(child1_chromosome)
        child2 = Individual(child2_chromosome)
        
        return child1, child2
    
    def crossover_two_point(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Realiza crossover de dois pontos entre dois pais.
        
        Args:
            parent1: Primeiro pai
            parent2: Segundo pai
            
        Returns:
            Tupla com os dois filhos gerados
        """
        # Verifica se o crossover ocorrerá
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Seleciona dois pontos de corte aleatórios
        point1 = random.randint(1, self.chromosome_length - 2)
        point2 = random.randint(point1 + 1, self.chromosome_length - 1)
        
        # Cria os cromossomos dos filhos
        child1_chromosome = parent1.chromosome[:point1] + parent2.chromosome[point1:point2] + parent1.chromosome[point2:]
        child2_chromosome = parent2.chromosome[:point1] + parent1.chromosome[point1:point2] + parent2.chromosome[point2:]
        
        # Cria os indivíduos filhos
        child1 = Individual(child1_chromosome)
        child2 = Individual(child2_chromosome)
        
        return child1, child2
    
    def crossover_uniform(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Realiza crossover uniforme entre dois pais.
        
        Args:
            parent1: Primeiro pai
            parent2: Segundo pai
            
        Returns:
            Tupla com os dois filhos gerados
        """
        # Verifica se o crossover ocorrerá
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Cria os cromossomos dos filhos
        child1_chromosome = []
        child2_chromosome = []
        
        # Para cada gene, decide de qual pai ele virá
        for i in range(self.chromosome_length):
            if random.random() < 0.5:
                child1_chromosome.append(parent1.chromosome[i])
                child2_chromosome.append(parent2.chromosome[i])
            else:
                child1_chromosome.append(parent2.chromosome[i])
                child2_chromosome.append(parent1.chromosome[i])
        
        # Cria os indivíduos filhos
        child1 = Individual(child1_chromosome)
        child2 = Individual(child2_chromosome)
        
        return child1, child2
    
    def mutate(self, individual: Individual) -> None:
        """
        Aplica mutação a um indivíduo.
        
        Args:
            individual: Indivíduo a ser mutado
        """
        for i in range(len(individual.chromosome)):
            # Para cada gene, verifica se ocorrerá mutação
            if random.random() < self.mutation_rate:
                # Inverte o valor do gene (0 -> 1 ou 1 -> 0)
                individual.chromosome[i] = 1 - individual.chromosome[i]
        
        # Reseta o fitness para que seja recalculado
        individual.fitness = None
    
    def evolve(self) -> None:
        """
        Executa uma geração do algoritmo genético.
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
            # Seleciona dois pais
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
    
    def run(self) -> Tuple[Individual, int, List[float], List[float]]:
        """
        Executa o algoritmo genético até atingir o critério de parada.
        
        Returns:
            Tupla contendo o melhor indivíduo, número de gerações,
            histórico do melhor fitness e histórico do fitness médio
        """
        # Inicializa a população
        self.initialize_population()
        
        # Loop principal
        while self.generation < self.max_generations:
            # Executa uma geração
            self.evolve()
            
            # Verifica se atingiu o fitness alvo
            if self.target_fitness is not None and self.best_individual.fitness >= self.target_fitness:
                break
        
        return (
            self.best_individual,
            self.generation,
            self.best_fitness_history,
            self.avg_fitness_history
        )
    
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
