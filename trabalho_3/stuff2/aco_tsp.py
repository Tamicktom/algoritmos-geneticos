import numpy as np
import matplotlib.pyplot as plt
import random
import math
from typing import List, Tuple
import time

class ACO_TSP:
    def __init__(self, cities: List[Tuple[float, float]], 
                 n_ants: int = 50, 
                 n_iterations: int = 500,
                 alpha: float = 1.0,  # influência do feromônio
                 beta: float = 2.0,   # influência da heurística
                 rho: float = 0.1,    # taxa de evaporação
                 Q: float = 100.0):   # constante de deposição de feromônio
        
        self.cities = cities
        self.n_cities = len(cities)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        # Matriz de distâncias
        self.distances = self._calculate_distances()
        
        # Matriz de feromônios (inicializada com valor pequeno)
        self.pheromones = np.ones((self.n_cities, self.n_cities)) * 0.1
        
        # Heurística (inverso da distância)
        self.heuristic = 1.0 / (self.distances + 1e-10)
        
        # Para armazenar resultados
        self.best_distance = float('inf')
        self.best_path = []
        self.distance_history = []
        
    def _calculate_distances(self) -> np.ndarray:
        """Calcula matriz de distâncias euclidianas entre todas as cidades"""
        distances = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    x1, y1 = self.cities[i]
                    x2, y2 = self.cities[j]
                    distances[i][j] = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distances
    
    def _calculate_path_distance(self, path: List[int]) -> float:
        """Calcula a distância total de um caminho"""
        total_distance = 0
        for i in range(len(path)):
            current_city = path[i]
            next_city = path[(i + 1) % len(path)]
            total_distance += self.distances[current_city][next_city]
        return total_distance
    
    def _select_next_city(self, current_city: int, unvisited: List[int]) -> int:
        """Seleciona a próxima cidade baseada na probabilidade ACO"""
        if not unvisited:
            return None
            
        probabilities = []
        for city in unvisited:
            pheromone = self.pheromones[current_city][city] ** self.alpha
            heuristic = self.heuristic[current_city][city] ** self.beta
            probabilities.append(pheromone * heuristic)
        
        # Normalizar probabilidades
        total_prob = sum(probabilities)
        if total_prob == 0:
            return random.choice(unvisited)
        
        probabilities = [p / total_prob for p in probabilities]
        
        # Seleção por roleta
        r = random.random()
        cumulative_prob = 0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return unvisited[i]
        
        return unvisited[-1]
    
    def _construct_ant_solution(self) -> List[int]:
        """Constrói uma solução para uma formiga"""
        start_city = random.randint(0, self.n_cities - 1)
        path = [start_city]
        unvisited = list(range(self.n_cities))
        unvisited.remove(start_city)
        
        current_city = start_city
        while unvisited:
            next_city = self._select_next_city(current_city, unvisited)
            path.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        return path
    
    def _update_pheromones(self, ant_paths: List[List[int]], ant_distances: List[float]):
        """Atualiza os feromônios baseado nas soluções das formigas"""
        # Evaporação
        self.pheromones *= (1 - self.rho)
        
        # Deposição
        for path, distance in zip(ant_paths, ant_distances):
            pheromone_deposit = self.Q / distance
            for i in range(len(path)):
                current_city = path[i]
                next_city = path[(i + 1) % len(path)]
                self.pheromones[current_city][next_city] += pheromone_deposit
                self.pheromones[next_city][current_city] += pheromone_deposit
    
    def solve(self) -> Tuple[List[int], float, List[float]]:
        """Executa o algoritmo ACO"""
        print(f"Iniciando ACO com {self.n_ants} formigas por {self.n_iterations} iterações...")
        print(f"Parâmetros: α={self.alpha}, β={self.beta}, ρ={self.rho}, Q={self.Q}")
        
        start_time = time.time()
        
        for iteration in range(self.n_iterations):
            # Construir soluções para todas as formigas
            ant_paths = []
            ant_distances = []
            
            for ant in range(self.n_ants):
                path = self._construct_ant_solution()
                distance = self._calculate_path_distance(path)
                ant_paths.append(path)
                ant_distances.append(distance)
                
                # Atualizar melhor solução
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path.copy()
            
            # Atualizar feromônios
            self._update_pheromones(ant_paths, ant_distances)
            
            # Armazenar histórico
            self.distance_history.append(self.best_distance)
            
            # Log do progresso
            if (iteration + 1) % 50 == 0:
                print(f"Iteração {iteration + 1}: Melhor distância = {self.best_distance:.2f}")
        
        end_time = time.time()
        print(f"\nACO concluído em {end_time - start_time:.2f} segundos")
        print(f"Melhor distância encontrada: {self.best_distance:.2f}")
        
        return self.best_path, self.best_distance, self.distance_history

def read_tsp_file(filename: str) -> List[Tuple[float, float]]:
    """Lê arquivo TSP e retorna lista de coordenadas das cidades"""
    cities = []
    with open(filename, 'r') as file:
        reading_coords = False
        for line in file:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading_coords = True
                continue
            elif line == "EOF":
                break
            elif reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    city_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    cities.append((x, y))
    return cities

def plot_convergence(distance_history: List[float], save_path: str = None):
    """Plota gráfico de convergência do algoritmo"""
    plt.figure(figsize=(12, 6))
    plt.plot(distance_history, 'b-', linewidth=2)
    plt.title('Convergência do ACO - Distância Total ao Longo das Iterações', fontsize=14)
    plt.xlabel('Iteração', fontsize=12)
    plt.ylabel('Distância Total', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_solution(cities: List[Tuple[float, float]], path: List[int], 
                 distance: float, save_path: str = None):
    """Plota visualização da solução encontrada"""
    plt.figure(figsize=(12, 10))
    
    # Plotar cidades
    x_coords = [city[0] for city in cities]
    y_coords = [city[1] for city in cities]
    plt.scatter(x_coords, y_coords, c='red', s=50, zorder=3)
    
    # Numerar cidades
    for i, (x, y) in enumerate(cities):
        plt.annotate(str(i+1), (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    # Plotar caminho
    path_x = [cities[i][0] for i in path] + [cities[path[0]][0]]
    path_y = [cities[i][1] for i in path] + [cities[path[0]][1]]
    plt.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, zorder=2)
    
    # Destacar cidade inicial
    start_city = path[0]
    plt.scatter(cities[start_city][0], cities[start_city][1], 
               c='green', s=100, marker='s', zorder=4, label='Cidade Inicial')
    
    plt.title(f'Solução ACO para Berlin52 TSP\nDistância Total: {distance:.2f}', fontsize=14)
    plt.xlabel('Coordenada X', fontsize=12)
    plt.ylabel('Coordenada Y', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Ler dados do arquivo TSP
    cities = read_tsp_file('/home/ubuntu/upload/berlin52.tsp')
    print(f"Carregadas {len(cities)} cidades do arquivo berlin52.tsp")
    
    # Configurar e executar ACO
    aco = ACO_TSP(cities, 
                  n_ants=50, 
                  n_iterations=500,
                  alpha=1.0,
                  beta=2.0,
                  rho=0.1,
                  Q=100.0)
    
    # Resolver problema
    best_path, best_distance, distance_history = aco.solve()
    
    # Gerar gráficos
    plot_convergence(distance_history, '/home/ubuntu/convergencia_aco.png')
    plot_solution(cities, best_path, best_distance, '/home/ubuntu/solucao_aco.png')
    
    # Salvar resultados
    with open('/home/ubuntu/resultados_aco.txt', 'w') as f:
        f.write(f"Resultados do ACO para Berlin52 TSP\n")
        f.write(f"=====================================\n\n")
        f.write(f"Configurações utilizadas:\n")
        f.write(f"- Número de formigas: {aco.n_ants}\n")
        f.write(f"- Número de iterações: {aco.n_iterations}\n")
        f.write(f"- Alpha (influência feromônio): {aco.alpha}\n")
        f.write(f"- Beta (influência heurística): {aco.beta}\n")
        f.write(f"- Rho (taxa evaporação): {aco.rho}\n")
        f.write(f"- Q (constante deposição): {aco.Q}\n\n")
        f.write(f"Resultados:\n")
        f.write(f"- Melhor distância encontrada: {best_distance:.2f}\n")
        f.write(f"- Melhor caminho: {[i+1 for i in best_path]}\n")
        f.write(f"- Distância inicial: {distance_history[0]:.2f}\n")
        f.write(f"- Melhoria obtida: {((distance_history[0] - best_distance) / distance_history[0] * 100):.2f}%\n")
    
    print(f"\nResultados salvos em:")
    print(f"- Gráfico de convergência: /home/ubuntu/convergencia_aco.png")
    print(f"- Visualização da solução: /home/ubuntu/solucao_aco.png")
    print(f"- Arquivo de resultados: /home/ubuntu/resultados_aco.txt")

