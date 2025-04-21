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
     
(Content truncated due to size limit. Use line ranges to read in chunks)