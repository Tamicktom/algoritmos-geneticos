import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random

class PSO:
    def __init__(self, num_particles=30, max_iterations=100, w=0.729, c1=1.49445, c2=1.49445):
        """
        Inicializa o algoritmo PSO
        
        Parâmetros:
        - num_particles: número de partículas no enxame
        - max_iterations: número máximo de iterações
        - w: peso de inércia
        - c1: coeficiente de aceleração cognitiva
        - c2: coeficiente de aceleração social
        """
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Limites do espaço de busca
        self.bounds = [(-5, 5), (-5, 5)]
        self.dimensions = len(self.bounds)
        
        # Histórico para análise
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
    def rosenbrock_function(self, x, y):
        """
        Função de Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
        """
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def fitness_function(self, position):
        """
        Função de fitness (objetivo a ser minimizada)
        """
        x, y = position
        return self.rosenbrock_function(x, y)
    
    def initialize_particles(self):
        """
        Inicializa as partículas com posições e velocidades aleatórias
        """
        particles = []
        
        for i in range(self.num_particles):
            # Posição inicial aleatória dentro dos limites
            position = np.array([
                random.uniform(self.bounds[0][0], self.bounds[0][1]),
                random.uniform(self.bounds[1][0], self.bounds[1][1])
            ])
            
            # Velocidade inicial aleatória (pequena)
            velocity = np.array([
                random.uniform(-1, 1),
                random.uniform(-1, 1)
            ])
            
            # Melhor posição pessoal inicial é a posição atual
            best_position = position.copy()
            best_fitness = self.fitness_function(position)
            
            particle = {
                'position': position,
                'velocity': velocity,
                'best_position': best_position,
                'best_fitness': best_fitness
            }
            
            particles.append(particle)
            
            # Atualiza o melhor global se necessário
            if best_fitness < self.global_best_fitness:
                self.global_best_fitness = best_fitness
                self.global_best_position = best_position.copy()
        
        return particles
    
    def update_velocity(self, particle):
        """
        Atualiza a velocidade da partícula usando a equação do PSO
        """
        r1 = random.random()
        r2 = random.random()
        
        cognitive_component = self.c1 * r1 * (particle['best_position'] - particle['position'])
        social_component = self.c2 * r2 * (self.global_best_position - particle['position'])
        
        particle['velocity'] = (self.w * particle['velocity'] + 
                               cognitive_component + 
                               social_component)
        
        # Limita a velocidade para evitar explosão
        max_velocity = 2.0
        particle['velocity'] = np.clip(particle['velocity'], -max_velocity, max_velocity)
    
    def update_position(self, particle):
        """
        Atualiza a posição da partícula
        """
        particle['position'] += particle['velocity']
        
        # Aplica os limites do espaço de busca
        for i in range(self.dimensions):
            if particle['position'][i] < self.bounds[i][0]:
                particle['position'][i] = self.bounds[i][0]
                particle['velocity'][i] = 0  # Para a velocidade na direção da parede
            elif particle['position'][i] > self.bounds[i][1]:
                particle['position'][i] = self.bounds[i][1]
                particle['velocity'][i] = 0
    
    def optimize(self):
        """
        Executa o algoritmo PSO
        """
        particles = self.initialize_particles()
        
        print(f"PSO iniciado com {self.num_particles} partículas")
        print(f"Configurações: w={self.w}, c1={self.c1}, c2={self.c2}")
        print(f"Melhor inicial: {self.global_best_fitness:.6f}")
        
        for iteration in range(self.max_iterations):
            fitness_values = []
            
            for particle in particles:
                # Calcula fitness atual
                current_fitness = self.fitness_function(particle['position'])
                fitness_values.append(current_fitness)
                
                # Atualiza melhor pessoal
                if current_fitness < particle['best_fitness']:
                    particle['best_fitness'] = current_fitness
                    particle['best_position'] = particle['position'].copy()
                
                # Atualiza melhor global
                if current_fitness < self.global_best_fitness:
                    self.global_best_fitness = current_fitness
                    self.global_best_position = particle['position'].copy()
                
                # Atualiza velocidade e posição
                self.update_velocity(particle)
                self.update_position(particle)
            
            # Armazena histórico
            self.best_fitness_history.append(self.global_best_fitness)
            self.mean_fitness_history.append(np.mean(fitness_values))
            
            # Mostra progresso a cada 10 iterações
            if (iteration + 1) % 10 == 0:
                print(f"Iteração {iteration + 1}: Melhor = {self.global_best_fitness:.6f}, "
                      f"Média = {np.mean(fitness_values):.6f}")
        
        print(f"\nOtimização concluída!")
        print(f"Melhor solução encontrada: x = {self.global_best_position[0]:.6f}, "
              f"y = {self.global_best_position[1]:.6f}")
        print(f"Valor da função: {self.global_best_fitness:.6f}")
        
        return {
            'best_position': self.global_best_position,
            'best_fitness': self.global_best_fitness,
            'best_history': self.best_fitness_history,
            'mean_history': self.mean_fitness_history
        }

def plot_convergence(pso_results, title="Convergência do PSO"):
    """
    Plota a convergência do algoritmo
    """
    plt.figure(figsize=(12, 5))
    
    # Gráfico de convergência
    plt.subplot(1, 2, 1)
    iterations = range(1, len(pso_results['best_history']) + 1)
    plt.plot(iterations, pso_results['best_history'], 'b-', label='Melhor valor', linewidth=2)
    plt.plot(iterations, pso_results['mean_history'], 'r--', label='Valor médio', linewidth=2)
    plt.xlabel('Iteração')
    plt.ylabel('Valor da função')
    plt.title('Convergência do PSO')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Gráfico da função de Rosenbrock
    plt.subplot(1, 2, 2)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    
    # Usa escala logarítmica para melhor visualização
    Z_log = np.log10(Z + 1)
    
    contour = plt.contour(X, Y, Z_log, levels=20)
    plt.colorbar(contour, label='log10(f(x,y) + 1)')
    
    # Marca a solução ótima teórica (1, 1)
    plt.plot(1, 1, 'r*', markersize=15, label='Ótimo teórico (1,1)')
    
    # Marca a melhor solução encontrada
    best_x, best_y = pso_results['best_position']
    plt.plot(best_x, best_y, 'bo', markersize=10, label=f'PSO ({best_x:.3f},{best_y:.3f})')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Função de Rosenbrock')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

if __name__ == "__main__":
    # Executa o PSO
    pso = PSO(num_particles=30, max_iterations=100)
    results = pso.optimize()
    
    # Plota os resultados
    plt = plot_convergence(results)
    plt.savefig('/home/ubuntu/pso_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()

