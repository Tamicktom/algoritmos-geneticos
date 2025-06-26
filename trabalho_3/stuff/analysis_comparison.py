import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from pso_implementation import PSO
from genetic_algorithm import GeneticAlgorithm

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def run_multiple_experiments(num_runs=20):
    """
    Executa múltiplas execuções de PSO e AG para análise estatística
    """
    print(f"Executando {num_runs} experimentos de cada algoritmo...")
    
    pso_results = []
    ag_results = []
    pso_convergence = []
    ag_convergence = []
    
    # Executa PSO
    print("Executando PSO...")
    for run in range(num_runs):
        pso = PSO(num_particles=30, max_iterations=100, w=0.729, c1=1.49445, c2=1.49445)
        result = pso.optimize()
        pso_results.append(result['best_fitness'])
        pso_convergence.append(result['best_history'])
        if (run + 1) % 5 == 0:
            print(f"  PSO: {run + 1}/{num_runs} concluído")
    
    # Executa AG
    print("Executando AG...")
    for run in range(num_runs):
        ga = GeneticAlgorithm(population_size=50, max_generations=100, 
                             crossover_rate=0.8, mutation_rate=0.1, elite_size=2)
        result = ga.optimize()
        ag_results.append(result['best_fitness'])
        ag_convergence.append(result['best_history'])
        if (run + 1) % 5 == 0:
            print(f"  AG: {run + 1}/{num_runs} concluído")
    
    return {
        'pso_results': pso_results,
        'ag_results': ag_results,
        'pso_convergence': pso_convergence,
        'ag_convergence': ag_convergence
    }

def statistical_analysis(pso_results, ag_results):
    """
    Realiza análise estatística dos resultados
    """
    print("\n=== ANÁLISE ESTATÍSTICA ===")
    
    # Estatísticas descritivas
    pso_stats = {
        'mean': np.mean(pso_results),
        'std': np.std(pso_results),
        'min': np.min(pso_results),
        'max': np.max(pso_results),
        'median': np.median(pso_results),
        'q25': np.percentile(pso_results, 25),
        'q75': np.percentile(pso_results, 75)
    }
    
    ag_stats = {
        'mean': np.mean(ag_results),
        'std': np.std(ag_results),
        'min': np.min(ag_results),
        'max': np.max(ag_results),
        'median': np.median(ag_results),
        'q25': np.percentile(ag_results, 25),
        'q75': np.percentile(ag_results, 75)
    }
    
    print("PSO:")
    print(f"  Média: {pso_stats['mean']:.6f}")
    print(f"  Desvio padrão: {pso_stats['std']:.6f}")
    print(f"  Mediana: {pso_stats['median']:.6f}")
    print(f"  Min: {pso_stats['min']:.6f}")
    print(f"  Max: {pso_stats['max']:.6f}")
    
    print("AG:")
    print(f"  Média: {ag_stats['mean']:.6f}")
    print(f"  Desvio padrão: {ag_stats['std']:.6f}")
    print(f"  Mediana: {ag_stats['median']:.6f}")
    print(f"  Min: {ag_stats['min']:.6f}")
    print(f"  Max: {ag_stats['max']:.6f}")
    
    # Teste de significância (Mann-Whitney U)
    statistic, p_value = stats.mannwhitneyu(pso_results, ag_results, alternative='two-sided')
    print(f"\nTeste Mann-Whitney U:")
    print(f"  Estatística: {statistic:.2f}")
    print(f"  p-valor: {p_value:.6f}")
    print(f"  Significativo (α=0.05): {'Sim' if p_value < 0.05 else 'Não'}")
    
    # Taxa de sucesso (valores < 0.01)
    pso_success_rate = np.sum(np.array(pso_results) < 0.01) / len(pso_results) * 100
    ag_success_rate = np.sum(np.array(ag_results) < 0.01) / len(ag_results) * 100
    
    print(f"\nTaxa de sucesso (f < 0.01):")
    print(f"  PSO: {pso_success_rate:.1f}%")
    print(f"  AG: {ag_success_rate:.1f}%")
    
    return pso_stats, ag_stats, p_value

def create_comprehensive_plots(data, pso_stats, ag_stats):
    """
    Cria visualizações abrangentes dos resultados
    """
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Boxplot comparativo
    plt.subplot(3, 4, 1)
    box_data = [data['pso_results'], data['ag_results']]
    bp = plt.boxplot(box_data, labels=['PSO', 'AG'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    plt.ylabel('Valor da função')
    plt.title('Distribuição dos Resultados')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 2. Histograma PSO
    plt.subplot(3, 4, 2)
    plt.hist(data['pso_results'], bins=15, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(pso_stats['mean'], color='red', linestyle='--', label=f'Média: {pso_stats["mean"]:.4f}')
    plt.xlabel('Valor da função')
    plt.ylabel('Frequência')
    plt.title('Distribuição PSO')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Histograma AG
    plt.subplot(3, 4, 3)
    plt.hist(data['ag_results'], bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(ag_stats['mean'], color='red', linestyle='--', label=f'Média: {ag_stats["mean"]:.4f}')
    plt.xlabel('Valor da função')
    plt.ylabel('Frequência')
    plt.title('Distribuição AG')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Convergência média
    plt.subplot(3, 4, 4)
    pso_mean_conv = np.mean(data['pso_convergence'], axis=0)
    ag_mean_conv = np.mean(data['ag_convergence'], axis=0)
    
    iterations = range(1, len(pso_mean_conv) + 1)
    plt.plot(iterations, pso_mean_conv, 'b-', label='PSO', linewidth=2)
    plt.plot(iterations, ag_mean_conv, 'r-', label='AG', linewidth=2)
    plt.xlabel('Iteração/Geração')
    plt.ylabel('Melhor valor (log)')
    plt.title('Convergência Média')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Convergência individual PSO
    plt.subplot(3, 4, 5)
    for i, conv in enumerate(data['pso_convergence'][:5]):
        plt.plot(conv, alpha=0.6, linewidth=1)
    plt.plot(pso_mean_conv, 'k-', linewidth=3, label='Média')
    plt.xlabel('Iteração')
    plt.ylabel('Melhor valor (log)')
    plt.title('Convergência PSO (5 execuções)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Convergência individual AG
    plt.subplot(3, 4, 6)
    for i, conv in enumerate(data['ag_convergence'][:5]):
        plt.plot(conv, alpha=0.6, linewidth=1)
    plt.plot(ag_mean_conv, 'k-', linewidth=3, label='Média')
    plt.xlabel('Geração')
    plt.ylabel('Melhor valor (log)')
    plt.title('Convergência AG (5 execuções)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Função de Rosenbrock 3D
    ax = plt.subplot(3, 4, 7, projection='3d')
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-1, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.scatter([1], [1], [0], color='red', s=100, label='Ótimo global')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title('Função de Rosenbrock')
    
    # 8. Contorno da função
    plt.subplot(3, 4, 8)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    
    levels = np.logspace(0, 3, 20)
    contour = plt.contour(X, Y, Z, levels=levels)
    plt.colorbar(contour, label='f(x,y)')
    plt.plot(1, 1, 'r*', markersize=15, label='Ótimo global (1,1)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Curvas de Nível')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Comparação de velocidade de convergência
    plt.subplot(3, 4, 9)
    # Calcula quantas iterações para atingir diferentes thresholds
    thresholds = [1, 0.1, 0.01, 0.001]
    pso_conv_speed = []
    ag_conv_speed = []
    
    for threshold in thresholds:
        pso_iters = []
        ag_iters = []
        
        for conv in data['pso_convergence']:
            iter_to_threshold = next((i for i, val in enumerate(conv) if val < threshold), len(conv))
            pso_iters.append(iter_to_threshold)
        
        for conv in data['ag_convergence']:
            iter_to_threshold = next((i for i, val in enumerate(conv) if val < threshold), len(conv))
            ag_iters.append(iter_to_threshold)
        
        pso_conv_speed.append(np.mean(pso_iters))
        ag_conv_speed.append(np.mean(ag_iters))
    
    x_pos = np.arange(len(thresholds))
    width = 0.35
    
    plt.bar(x_pos - width/2, pso_conv_speed, width, label='PSO', color='lightblue')
    plt.bar(x_pos + width/2, ag_conv_speed, width, label='AG', color='lightcoral')
    plt.xlabel('Threshold')
    plt.ylabel('Iterações médias')
    plt.title('Velocidade de Convergência')
    plt.xticks(x_pos, [f'{t}' for t in thresholds])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Robustez (desvio padrão vs média)
    plt.subplot(3, 4, 10)
    plt.scatter(pso_stats['mean'], pso_stats['std'], s=200, color='blue', 
               label='PSO', alpha=0.7, edgecolors='black')
    plt.scatter(ag_stats['mean'], ag_stats['std'], s=200, color='red', 
               label='AG', alpha=0.7, edgecolors='black')
    plt.xlabel('Média')
    plt.ylabel('Desvio Padrão')
    plt.title('Robustez (Menor = Melhor)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    # 11. Taxa de sucesso por threshold
    plt.subplot(3, 4, 11)
    thresholds_success = [1, 0.1, 0.01, 0.001, 0.0001]
    pso_success_rates = []
    ag_success_rates = []
    
    for threshold in thresholds_success:
        pso_rate = np.sum(np.array(data['pso_results']) < threshold) / len(data['pso_results']) * 100
        ag_rate = np.sum(np.array(data['ag_results']) < threshold) / len(data['ag_results']) * 100
        pso_success_rates.append(pso_rate)
        ag_success_rates.append(ag_rate)
    
    x_pos = np.arange(len(thresholds_success))
    width = 0.35
    
    plt.bar(x_pos - width/2, pso_success_rates, width, label='PSO', color='lightblue')
    plt.bar(x_pos + width/2, ag_success_rates, width, label='AG', color='lightcoral')
    plt.xlabel('Threshold')
    plt.ylabel('Taxa de Sucesso (%)')
    plt.title('Taxa de Sucesso por Threshold')
    plt.xticks(x_pos, [f'{t}' for t in thresholds_success])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 12. Evolução da diversidade (apenas para AG)
    plt.subplot(3, 4, 12)
    # Simula diversidade baseada na convergência
    diversity_pso = []
    diversity_ag = []
    
    for conv in data['pso_convergence'][:5]:
        # Diversidade inversamente proporcional à convergência
        div = [1.0 / (1.0 + val) for val in conv]
        diversity_pso.append(div)
    
    for conv in data['ag_convergence'][:5]:
        div = [1.0 / (1.0 + val) for val in conv]
        diversity_ag.append(div)
    
    mean_div_pso = np.mean(diversity_pso, axis=0)
    mean_div_ag = np.mean(diversity_ag, axis=0)
    
    iterations = range(1, len(mean_div_pso) + 1)
    plt.plot(iterations, mean_div_pso, 'b-', label='PSO', linewidth=2)
    plt.plot(iterations, mean_div_ag, 'r-', label='AG', linewidth=2)
    plt.xlabel('Iteração/Geração')
    plt.ylabel('Diversidade (estimada)')
    plt.title('Evolução da Diversidade')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """
    Função principal que executa toda a análise
    """
    print("=== ANÁLISE COMPARATIVA PSO vs AG ===")
    print("Função de Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²")
    print("Domínio: [-5, 5] × [-5, 5]")
    print("Ótimo global: (1, 1) com f(1,1) = 0")
    
    # Executa experimentos
    data = run_multiple_experiments(num_runs=20)
    
    # Análise estatística
    pso_stats, ag_stats, p_value = statistical_analysis(data['pso_results'], data['ag_results'])
    
    # Cria visualizações
    print("\nGerando visualizações...")
    fig = create_comprehensive_plots(data, pso_stats, ag_stats)
    
    # Salva gráfico
    plt.savefig('/home/ubuntu/analise_comparativa.png', dpi=300, bbox_inches='tight')
    print("Gráfico salvo como 'analise_comparativa.png'")
    
    # Salva dados para o relatório
    results_summary = {
        'pso_stats': pso_stats,
        'ag_stats': ag_stats,
        'p_value': p_value,
        'num_runs': len(data['pso_results']),
        'pso_results': data['pso_results'],
        'ag_results': data['ag_results']
    }
    
    np.save('/home/ubuntu/results_summary.npy', results_summary)
    print("Dados salvos como 'results_summary.npy'")
    
    plt.show()
    
    return results_summary

if __name__ == "__main__":
    results = main()

