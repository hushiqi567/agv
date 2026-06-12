"""实验结果图表生成"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_exp1_rl_vs_astar(csv_path, output_dir="logs/metrics"):
    import csv
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    rows = []
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    ratios = [float(r['ratio']) for r in rows if float(r['ratio']) > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(range(len(ratios)), ratios, color='steelblue', alpha=0.7)
    ax1.axhline(y=1.0, color='red', linestyle='--', label='A* baseline (=1.0)')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('RL Path Length / A* Path Length')
    ax1.set_title('Experiment 1: RL vs A* Path Length Ratio')
    ax1.legend()

    ax2.hist(ratios, bins=15, edgecolor='black', alpha=0.7)
    ax2.axvline(x=1.0, color='red', linestyle='--')
    ax2.set_xlabel('Path Length Ratio')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Path Length Ratios')

    plt.tight_layout()
    path = os.path.join(output_dir, "exp1_rl_vs_astar.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Saved: {path}")

def plot_exp2_ablation(csv_path, output_dir="logs/metrics"):
    import csv
    if not os.path.exists(csv_path):
        return
    rows = []
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    configs = [r['config'] for r in rows]
    tasks = [int(r['tasks_completed']) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    ax.bar(configs, tasks, color=colors[:len(configs)], edgecolor='black')
    ax.set_ylabel('Tasks Completed')
    ax.set_title('Experiment 2: Ablation Study')
    for i, v in enumerate(tasks):
        ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, "exp2_ablation.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)

def plot_exp3_scalability(csv_path, output_dir="logs/metrics"):
    import csv
    if not os.path.exists(csv_path):
        return
    rows = []
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    agvs = [int(r['num_agvs']) for r in rows]
    tasks = [int(r['tasks_completed']) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(agvs, tasks, 'o-', linewidth=2, markersize=10, color='#2c3e50')
    ax.set_xlabel('Number of AGVs')
    ax.set_ylabel('Tasks Completed')
    ax.set_title('Experiment 3: Multi-AGV Scalability')
    ax.set_xticks(agvs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "exp3_scalability.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)

def plot_exp4_comparison(csv_path, output_dir="logs/metrics"):
    import csv
    if not os.path.exists(csv_path):
        return
    rows = []
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    policies = [r['policy'] for r in rows]
    tasks = [int(r['tasks_completed']) for r in rows]
    deadlocks = [int(r['deadlocks']) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#27ae60', '#2980b9', '#e74c3c']
    ax1.bar(policies, tasks, color=colors[:len(policies)], edgecolor='black')
    ax1.set_ylabel('Tasks Completed')
    ax1.set_title('Task Completion by Method')

    ax2.bar(policies, deadlocks, color=colors[:len(policies)], edgecolor='black')
    ax2.set_ylabel('Deadlock Count')
    ax2.set_title('Deadlocks by Method')

    plt.tight_layout()
    path = os.path.join(output_dir, "exp4_comparison.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)

if __name__ == "__main__":
    d = "logs/metrics"
    plot_exp1_rl_vs_astar(os.path.join(d, "exp1_rl_vs_astar.csv"), d)
    plot_exp2_ablation(os.path.join(d, "exp2_ablation.csv"), d)
    plot_exp3_scalability(os.path.join(d, "exp3_scalability.csv"), d)
    plot_exp4_comparison(os.path.join(d, "exp4_comparison.csv"), d)
    print("All plots generated.")
