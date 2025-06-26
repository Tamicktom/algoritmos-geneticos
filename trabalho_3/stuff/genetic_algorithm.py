import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

class GeneticAlgorithm:
    def __init__(self, population_size=50, max_generations=100, 
                 crossover_rate=0.8, mutation_rate=0.1, elite_size=2):
        """
        Inicializa o Algoritmo Genético
        
        Parâmetros:
        - population_size: tamanho da população
        - max_generations: número máximo de gerações
        - crossover_rate: taxa de cruzamento
        - mutation_rate: taxa de mutação
        - elite_size: número de indivíduos elite preservados
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # Limites do espaço de busca
        self.bounds = [(-5, 5), (-5, 5)]
        self.dimensions = len(self.bounds)
        
        # Histórico para análise
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.global_best_individual = None
        self.global_best_fitness = float('inf')
        
    def rosenbrock_function(self, x, y):
        """
        Função de Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
        """
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def fitness_function(self, individual):
        """
        Função de fitness (objetivo a ser minimizada)
        """
        x, y = individual
        return self.rosenbrock_function(x, y)
    
    def create_individual(self):
        """
        Cria um indivíduo aleatório
        """
        return [random.uniform(self.bounds[i][0], self.bounds[i][1]) 
                for i in range(self.dimensions)]
    
    def create_population(self):
        """
        Cria a população inicial
        """
        return [self.create_individual() for _ in range(self.population_size)]
    
    def evaluate_population(self, population):
        """
        Avalia toda a população e retorna lista de fitness
        """
        fitness_values = []
        for individual in population:
            fitness = self.fitness_function(individual)
            fitness_values.append(fitness)
            
            # Atualiza melhor global
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_individual = individual.copy()
        
        return fitness_values
    
    def tournament_selection(self, population, fitness_values, tournament_size=3):
        """
        Seleção por torneio
        """
        selected = []
        
        for _ in range(self.population_size - self.elite_size):
            # Seleciona indivíduos aleatórios para o torneio
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_values[i] for i in tournament_indices]
            
            # Seleciona o melhor do torneio (menor fitness)
            winner_index = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_index].copy())
        
        return selected
    
    def arithmetic_crossover(self, parent1, parent2):
        """
        Cruzamento aritmético
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        alpha = random.random()
        child1 = [alpha * parent1[i] + (1 - alpha) * parent2[i] 
                  for i in range(self.dimensions)]
        child2 = [(1 - alpha) * parent1[i] + alpha * parent2[i] 
                  for i in range(self.dimensions)]
        
        return child1, child2
    
    def gaussian_mutation(self, individual):
        """
        Mutação gaussiana
        """
        mutated = individual.copy()
        
        for i in range(self.dimensions):
            if random.random() < self.mutation_rate:
                # Mutação gaussiana com desvio padrão adaptativo
                sigma = (self.bounds[i][1] - self.bounds[i][0]) * 0.1
                mutation = random.gauss(0, sigma)
                mutated[i] += mutation
                
                # Aplica os limites
                mutated[i] = max(self.bounds[i][0], 
                               min(self.bounds[i][1], mutated[i]))
        
        return mutated
    
    def get_elite(self, population, fitness_values):
        """
        Seleciona os melhores indivíduos (elite)
        """
        # Ordena por fitness (menor é melhor)
        sorted_indices = np.argsort(fitness_values)
        elite = [population[i].copy() for i in sorted_indices[:self.elite_size]]
        return elite
    
    def optimize(self):
        """
        Executa o algoritmo genético
        """
        # Inicializa população
        population = self.create_population()
        
        print(f"AG iniciado com população de {self.population_size} indivíduos")
        print(f"Configurações: crossover={self.crossover_rate}, "
              f"mutação={self.mutation_rate}, elite={self.elite_size}")
        
        for generation in range(self.max_generations):
            # Avalia população
            fitness_values = self.evaluate_population(population)
            
            # Armazena histórico
            self.best_fitness_history.append(min(fitness_values))
            self.mean_fitness_history.append(np.mean(fitness_values))
            
            # Mostra progresso a cada 10 gerações
            if (generation + 1) % 10 == 0:
                print(f"Geração {generation + 1}: Melhor = {min(fitness_values):.6f}, "
                      f"Média = {np.mean(fitness_values):.6f}")
            
            # Seleciona elite
            elite = self.get_elite(population, fitness_values)
            
            # Seleção para reprodução
            selected = self.tournament_selection(population, fitness_values)
            
            # Cruzamento e mutação
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                # Seleciona dois pais
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                # Cruzamento
                child1, child2 = self.arithmetic_crossover(parent1, parent2)
                
                # Mutação
                child1 = self.gaussian_mutation(child1)
                child2 = self.gaussian_mutation(child2)
                
                new_population.extend([child1, child2])
            
            # Mantém o tamanho da população
            population = new_population[:self.population_size]
        
        # Avaliação final
        final_fitness = self.evaluate_population(population)
        
        print(f"\nOtimização concluída!")
        print(f"Melhor solução encontrada: x = {self.global_best_individual[0]:.6f}, "
              f"y = {self.global_best_individual[1]:.6f}")
        print(f"Valor da função: {self.global_best_fitness:.6f}")
        
        return {
            'best_individual': self.global_best_individual,
            'best_fitness': self.global_best_fitness,
            'best_history': self.best_fitness_history,
            'mean_history': self.mean_fitness_history
        }

def compare_algorithms():
    """
    Compara PSO e AG executando múltiplas vezes
    """
    num_runs = 10
    
    print("=== COMPARAÇÃO PSO vs AG ===")
    print(f"Executando {num_runs} execuções de cada algoritmo...\n")
    
    # Resultados PSO
    pso_results = []
    print("Executando PSO...")
    
    from pso_implementation import PSO
    
    for run in range(num_runs):
        pso = PSO(num_particles=30, max_iterations=100)
        result = pso.optimize()
        pso_results.append(result['best_fitness'])
        print(f"PSO Run {run+1}: {result['best_fitness']:.6f}")
    
    # Resultados AG
    ag_results = []
    print("\nExecutando AG...")
    
    for run in range(num_runs):
        ga = GeneticAlgorithm(population_size=50, max_generations=100)
        result = ga.optimize()
        ag_results.append(result['best_fitness'])
        print(f"AG Run {run+1}: {result['best_fitness']:.6f}")
    
    # Estatísticas
    print("\n=== ESTATÍSTICAS ===")
    print(f"PSO - Média: {np.mean(pso_results):.6f}, "
          f"Desvio: {np.std(pso_results):.6f}, "
          f"Melhor: {np.min(pso_results):.6f}")
    print(f"AG  - Média: {np.mean(ag_results):.6f}, "
          f"Desvio: {np.std(ag_results):.6f}, "
          f"Melhor: {np.min(ag_results):.6f}")
    
    return {
        'pso_results': pso_results,
        'ag_results': ag_results,
        'pso_stats': {
            'mean': np.mean(pso_results),
            'std': np.std(pso_results),
            'best': np.min(pso_results)
        },
        'ag_stats': {
            'mean': np.mean(ag_results),
            'std': np.std(ag_results),
            'best': np.min(ag_results)
        }
    }

if __name__ == "__main__":
    # Executa uma instância do AG
    ga = GeneticAlgorithm(population_size=50, max_generations=100)
    ga_result = ga.optimize()
    
    # Salva resultado para comparação posterior
    np.save('/home/ubuntu/ga_result.npy', ga_result)
    
    print("\nResultado do AG salvo em ga_result.npy")

