"""指标采集器 — 挂载 MessageBus 记录并导出 CSV/图表"""
import os
import csv
import logging
from typing import List, Dict, Optional
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

class MetricsCollector:
    """
    指标采集器。

    采集仿真指标:
      - 每步每个 AGV 的位置和状态
      - 任务完成时间和路径长度
      - 碰撞次数
      - 死锁触发次数和恢复耗时
      - 训练过程的损失曲线、奖励曲线、探索率变化
    """

    def __init__(self, export_dir: str = "logs/metrics", export_interval: int = 100):
        self.export_dir = export_dir
        self.export_interval = export_interval
        os.makedirs(export_dir, exist_ok=True)
        self.logger = logging.getLogger("AGVProject.Metrics")

        self.step_records: List[Dict] = []
        self.task_records: List[Dict] = []
        self.collision_count = 0
        self.deadlock_events: List[Dict] = []
        self.training_losses: List[float] = []
        self.training_rewards: List[float] = []
        self.exploration_rates: List[float] = []

        self._step = 0

    def record_step(self, agv_states: Dict[int, dict], tasks_completed: int, step: int):
        self._step = step
        for agv_id, state in agv_states.items():
            self.step_records.append({
                'step': step,
                'agv_id': agv_id,
                'x': state.get('position', (0, 0))[0],
                'y': state.get('position', (0, 0))[1],
                'status': str(state.get('status', 'unknown')),
                'is_loaded': state.get('is_loaded', False),
                'battery': state.get('battery', 100.0),
            })

    def record_task_complete(self, task_id: int, agv_id: int, path_length: int,
                             elapsed_steps: int, step: int):
        self.task_records.append({
            'task_id': task_id,
            'agv_id': agv_id,
            'path_length': path_length,
            'elapsed_steps': elapsed_steps,
            'completed_at_step': step,
        })

    def record_collision(self, agv_id: int, step: int):
        self.collision_count += 1

    def record_deadlock(self, cycle: List[int], recovered_agv: int, step: int):
        self.deadlock_events.append({
            'step': step,
            'cycle': str(cycle),
            'recovered_agv': recovered_agv,
        })

    def record_training(self, loss: Optional[float], reward: Optional[float],
                        epsilon: Optional[float]):
        if loss is not None:
            self.training_losses.append(loss)
        if reward is not None:
            self.training_rewards.append(reward)
        if epsilon is not None:
            self.exploration_rates.append(epsilon)

    def export_csv(self, tag: str = ""):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{tag}_" if tag else ""

        def write_csv(filename, rows):
            if not rows:
                return
            path = os.path.join(self.export_dir, f"{prefix}{filename}")
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            self.logger.info(f"Exported: {path}")

        write_csv(f"{ts}_steps.csv", self.step_records)
        write_csv(f"{ts}_tasks.csv", self.task_records)
        write_csv(f"{ts}_deadlocks.csv", self.deadlock_events)

        if self.training_losses:
            path = os.path.join(self.export_dir, f"{prefix}{ts}_training.csv")
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['step', 'loss', 'reward', 'epsilon'])
                n = max(len(self.training_losses),
                        len(self.training_rewards),
                        len(self.exploration_rates))
                for i in range(n):
                    w.writerow([
                        i,
                        self.training_losses[i] if i < len(self.training_losses) else '',
                        self.training_rewards[i] if i < len(self.training_rewards) else '',
                        self.exploration_rates[i] if i < len(self.exploration_rates) else '',
                    ])
            self.logger.info(f"Exported: {path}")

    def export_charts(self, tag: str = ""):
        if not HAS_MPL:
            self.logger.warning("matplotlib not installed, skipping chart export")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{tag}_" if tag else ""

        if self.task_records:
            fig, ax = plt.subplots(figsize=(10, 6))
            times = [t['elapsed_steps'] for t in self.task_records]
            ax.hist(times, bins=20, edgecolor='black')
            ax.set_xlabel('Elapsed Steps')
            ax.set_ylabel('Count')
            ax.set_title('Task Completion Time Distribution')
            path = os.path.join(self.export_dir, f"{prefix}{ts}_task_times.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)

        if self.training_losses:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].plot(self.training_losses, alpha=0.7, linewidth=0.5)
            axes[0].set_title('Training Loss')
            axes[0].set_xlabel('Training Step')
            if len(self.training_rewards) > 0:
                window = min(100, len(self.training_rewards))
                smoothed = [
                    sum(self.training_rewards[max(0, i-window):i+1]) /
                    min(i+1, window)
                    for i in range(len(self.training_rewards))
                ]
                axes[1].plot(smoothed, linewidth=1)
            axes[1].set_title('Smoothed Reward')
            axes[1].set_xlabel('Episode')
            if self.exploration_rates:
                axes[2].plot(self.exploration_rates, linewidth=1)
            axes[2].set_title('Exploration Rate (Epsilon)')
            axes[2].set_xlabel('Step')
            plt.tight_layout()
            path = os.path.join(self.export_dir, f"{prefix}{ts}_training.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)

        if self.deadlock_events:
            fig, ax = plt.subplots(figsize=(12, 4))
            steps = [d['step'] for d in self.deadlock_events]
            ax.eventplot([steps], lineoffsets=0, linelengths=0.8, colors='red')
            ax.set_xlabel('Step')
            ax.set_title('Deadlock Events Timeline')
            path = os.path.join(self.export_dir, f"{prefix}{ts}_deadlocks.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)

    def get_summary(self) -> dict:
        return {
            'total_steps_recorded': self._step,
            'total_tasks_completed': len(self.task_records),
            'total_collisions': self.collision_count,
            'total_deadlocks': len(self.deadlock_events),
            'total_training_steps': len(self.training_losses),
        }

    def reset(self):
        self.step_records.clear()
        self.task_records.clear()
        self.deadlock_events.clear()
        self.collision_count = 0
        self._step = 0
