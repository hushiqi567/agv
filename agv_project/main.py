"""
============================================
AGV仓库仿真系统 - 主入口
============================================
本文件整合所有模块，启动完整的AGV仓库仿真。

仿真流程：
1. 初始化所有模块（环境、调度、路径规划、控制器、渲染器）
2. 主循环：环境更新 → 任务生成 → 任务分配 → AGV移动 → 渲染
3. 用户交互：ESC退出、R重置、SPACE暂停/继续

使用方式：
    python main.py
"""

import sys
import os
import logging
from typing import Optional

# 将项目根目录添加到系统路径
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from interface.communication import MessageBus, MessageType, Message, BaseModule
from interface.config import get_config
from interface.data_types import TaskStatus
from env.warehouse_env import WarehouseEnv, MAP_WIDTH, MAP_HEIGHT
from scheduler.od_flow import ODFlowManager
from scheduler.task_allocator import TaskAllocator
from path_planning.mapf_planner import MAPFPlanner
from path_planning.rl_collision_avoidance import RLCollisionAvoidance
from path_planning.agv_controller import AGVController


class Simulation:
    """
    仿真主控制器
    
    整合所有模块，管理仿真循环。
    
    Usage:
        >>> sim = Simulation()
        >>> sim.run()
    """
    
    def __init__(self, render: bool = True):
        """
        初始化仿真
        
        Args:
            render: 是否启用可视化渲染
        """
        self.logger = logging.getLogger("AGVProject.Main")
        self.render_enabled = render
        
        # 从全局配置读取参数
        config = get_config()
        self.max_steps = config.simulation.max_steps
        self.render_fps = config.simulation.render_fps
        
        # 仿真状态
        self.current_step = 0
        self.is_running = False
        self.is_paused = False
        
        # 初始化所有模块
        self._init_modules()
        
        self.logger.info("仿真系统初始化完成")
    
    def _init_modules(self):
        """初始化所有仿真模块"""
        self.logger.info("正在初始化仿真模块...")
        
        # 1. 创建环境
        self.env = WarehouseEnv()
        self.env.reset()
        
        # 2. 创建调度模块
        self.task_allocator = TaskAllocator(self.env.loading_zones, self.env.unloading_zones)
        
        # 3. 创建路径规划模块
        self.mapf_planner = MAPFPlanner(self.env.grid, self.env.width, self.env.height)
        self.rl_avoidance = RLCollisionAvoidance()
        
        # 4. 创建AGV控制器
        self.controller = AGVController(self.env, self.mapf_planner, self.rl_avoidance)
        self.controller.set_task_allocator(self.task_allocator)
        self.controller.reset()
        
        # 5. 设置控制器模式
        self.controller.use_rl_primary = True

        # 6. 创建渲染器（懒加载pygame）
        if self.render_enabled:
            from env.renderer import WarehouseRenderer
            self.renderer = WarehouseRenderer(self.env, fps=self.render_fps)
            self.renderer.set_agv_controller(self.controller)
        
        # 6. 设置模块间引用
        self.task_allocator.set_controller(self.controller)
        
        self.logger.info("所有模块初始化完成")
    
    def run(self):
        """
        运行仿真主循环
        
        流程：
        1. 发布仿真开始消息
        2. 主循环：环境更新 → 任务生成 → 任务分配 → AGV移动 → 渲染
        3. 处理用户输入
        4. 发布仿真结束消息
        """
        self.is_running = True
        
        # 发布仿真开始消息
        self._publish(MessageType.MAIN_SIMULATION_START, {
            "max_steps": self.max_steps
        })
        
        self.logger.info("=" * 50)
        self.logger.info("仿真开始")
        self.logger.info(f"地图: {MAP_WIDTH}x{MAP_HEIGHT}")
        self.logger.info(f"最大步数: {self.max_steps}")
        self.logger.info("=" * 50)
        
        # 主循环
        while self.is_running and self.current_step < self.max_steps:
            # 处理事件
            if self.render_enabled:
                if not self._handle_events():
                    break
            
            if not self.is_paused:
                # 执行一步仿真
                self._step()
            
            # 渲染
            if self.render_enabled:
                self.renderer.render()
        
        # 仿真结束
        self._shutdown()
    
    def _step(self):
        """
        执行一步仿真
        
        顺序：
        1. 环境更新（障碍物移动）
        2. 任务生成（泊松分布）
        3. 任务分配（最近距离）
        4. AGV移动（MAPF+RL）
        """
        self.current_step += 1
        
        # 1. 环境更新
        self.env.step()
        
        # 更新MAPF规划器的网格
        self.mapf_planner.update_grid(self.env.grid)
        
        # 2. 任务生成和分配（由TaskAllocator内部处理）
        self.task_allocator.step()
        
        # 3. AGV移动
        self.controller.step()
        
        # 4. 更新渲染器的出货口状态（红色格子 = loading_zones）
        # 检查每个出货口是否有任务正在等待AGV来取货
        if self.render_enabled:
            unloading_status = {}
            # 获取所有待分配和活跃任务中涉及的取货点
            active_pickup_positions = set()
            for task in self.task_allocator.od_flow.task_pool.values():
                if task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED, 
                                   TaskStatus.MOVING_TO_PICKUP, TaskStatus.LOADING]:
                    active_pickup_positions.add(task.pickup_pos)
            
            for pos in self.env.loading_zones:
                # 如果该出货口有待取货的任务，则显示"有货"
                has_goods = pos in active_pickup_positions
                unloading_status[pos] = has_goods
            self.renderer.set_unloading_zone_status(unloading_status)
        
        # 每步打印前10步的状态
        if self.current_step <= 10 or self.current_step % 100 == 0:
            stats = self.controller.get_statistics()
            self.logger.info(
                f"步数 {self.current_step}: "
                f"已完成 {stats['tasks_completed']} 个任务, "
                f"AGV: 空闲{stats['agvs']['idle']} "
                f"移动中{stats['agvs']['moving']}"
            )
    
    def _handle_events(self) -> bool:
        """处理用户输入事件"""
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                
                elif event.key == pygame.K_r:
                    self._reset()
                
                elif event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                    self.logger.info(f"{'已暂停' if self.is_paused else '继续运行'}")
        
        return True
    
    def _reset(self):
        """重置仿真"""
        self.current_step = 0
        
        # 重置所有模块
        self.env.reset()
        self.task_allocator.reset()
        self.controller.reset()
        self.mapf_planner.update_grid(self.env.grid)
        
        self.is_paused = False
        self.logger.info("仿真已重置")
    
    def _shutdown(self):
        """关闭仿真"""
        self.is_running = False
        
        # 发布仿真停止消息
        self._publish(MessageType.MAIN_SIMULATION_STOP, {
            "step": self.current_step,
            "reason": "completed" if self.current_step >= self.max_steps else "user_stopped"
        })
        
        # 关闭渲染器
        if self.render_enabled:
            self.renderer.close()
        
        # 打印统计信息
        stats = self.controller.get_statistics()
        self.logger.info("=" * 50)
        self.logger.info("仿真结束")
        self.logger.info(f"总步数: {self.current_step}")
        self.logger.info(f"完成任务: {stats['tasks_completed']}")
        self.logger.info(f"AGV总移动步数: {stats['steps_taken']}")
        self.logger.info("=" * 50)

        # 导出指标
        if hasattr(self, 'controller') and hasattr(self.controller, 'metrics'):
            self.controller.metrics.export_csv()
            self.controller.metrics.export_charts()
            summary = self.controller.metrics.get_summary()
            self.logger.info(f"Metrics summary: {summary}")
    
    def _publish(self, msg_type: MessageType, data: dict):
        """发布消息的便捷方法"""
        bus = MessageBus()
        bus.publish(Message(
            msg_type=msg_type,
            sender="main",
            data=data
        ))


def main():
    """主函数"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="AGV仓库仿真系统")
    parser.add_argument("--no-render", action="store_true", help="无界面模式")
    parser.add_argument("--steps", type=int, default=None, help="仿真步数")
    parser.add_argument("--train", action="store_true", help="启用RL训练模式")
    parser.add_argument("--train-episodes", type=int, default=10, help="训练回合数")
    parser.add_argument("--load-model", type=str, default=None, help="加载预训练模型")
    parser.add_argument("--save-model", type=str, default=None, help="保存训练好的模型")
    parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--cbs-primary", action="store_true",
                        help="使用CBS主导路径规划 (旧模式)")
    parser.add_argument("--curriculum", action="store_true",
                        help="启用课程学习训练")
    parser.add_argument("--export-metrics", action="store_true",
                        help="导出指标CSV和图表")
    parser.add_argument("--experiment", type=str, default=None,
                        choices=["1", "2", "3", "4"],
                        help="运行指定实验 (1-4)")
    parser.add_argument("--ppo", action="store_true", help="使用PPO算法替代DQN")
    parser.add_argument("--marl", action="store_true", help="启用MARL多智能体训练模式")
    args = parser.parse_args()
    
    # 从全局配置读取日志级别
    config = get_config()
    log_level = getattr(logging, config.simulation.log_level.upper(), logging.INFO)
    
    # 设置日志
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("=" * 60)
    print("AGV仓库仿真系统")
    print("=" * 60)
    print(f"地图尺寸: {MAP_WIDTH}x{MAP_HEIGHT}")
    print(f"AGV数量: {config.agv.num_agvs}")
    print(f"障碍物数量: 10")
    print(f"最大步数: {args.steps or config.simulation.max_steps}")
    print(f"RL训练模式: {'启用' if args.train else '禁用'}")
    print("=" * 60)
    print("操作说明:")
    print("  ESC - 退出")
    print("  R   - 重置仿真")
    print("  SPACE - 暂停/继续")
    print("=" * 60)
    
    if args.train:
        # ===== RL训练模式 =====
        print("\n启动RL训练模式...")
        print(f"训练回合数: {args.train_episodes}")
        print(f"每回合最大步数: {args.steps or config.simulation.max_steps}")
        print("=" * 60)

        # 创建仿真（无界面模式）
        sim = Simulation(render=not args.no_render)

        # 启用RL训练
        sim.rl_avoidance.set_training(True)

        # 加载预训练模型
        if args.load_model:
            sim.rl_avoidance.load_model(args.load_model)
            print(f"已加载模型: {args.load_model}")

        # 训练循环
        for episode in range(1, args.train_episodes + 1):
            print(f"\n{'='*50}")
            print(f"训练回合 {episode}/{args.train_episodes}")
            print(f"{'='*50}")

            # 重置仿真
            sim._reset()

            # 运行一个回合
            sim.max_steps = args.steps or config.simulation.max_steps
            sim.run()

            # 回合结束
            sim.rl_avoidance.end_episode()

            # 打印训练统计
            stats = sim.rl_avoidance.get_training_stats()
            print(f"  探索率: {stats['epsilon']:.3f}")
            print(f"  经验池: {stats['memory_size']}")
            print(f"  平均损失: {stats['avg_loss_100']:.4f}")
            print(f"  总奖励: {stats['total_rewards']:.1f}")

        # 保存模型
        if args.save_model:
            sim.rl_avoidance.save_model(args.save_model)
            print(f"\n模型已保存到: {args.save_model}")

        # 导出指标
        if args.export_metrics:
            sim.controller.metrics.export_csv()
            sim.controller.metrics.export_charts()
            print("指标已导出")

        print("\n训练完成！")

    elif args.experiment:
        print(f"\n运行实验 {args.experiment}...")
        if args.experiment == "1":
            from experiments.run_experiment_1_single import run_experiment_1
            run_experiment_1(num_trials=20)
        elif args.experiment == "2":
            from experiments.run_experiment_2_ablation import run_experiment_2
            run_experiment_2(steps=args.steps or 200)
        elif args.experiment == "3":
            from experiments.run_experiment_3_scalability import run_experiment_3
            run_experiment_3(steps=args.steps or 300)
        elif args.experiment == "4":
            from experiments.run_experiment_4_comparison import run_experiment_4
            run_experiment_4(steps=args.steps or 200)

    elif args.marl:
        # ===== MARL 多智能体训练模式 =====
        print("\n启动MARL多智能体训练模式...")
        from path_planning.rl.dqn_agent import DQNAgent
        from path_planning.rl.marl_trainer import MARLTrainer

        sim = Simulation(render=False)
        agent = DQNAgent(grid_size=15)
        if args.load_model:
            agent.load_model(args.load_model)
        marl = MARLTrainer(agent, num_agvs=config.agv.num_agvs)

        for ep in range(1, args.train_episodes + 1):
            result = marl.train_episode(sim.env, sim.task_allocator, max_steps=args.steps or 500)
            print(f"  Episode {ep}: tasks={result['tasks_completed']} "
                  f"collisions={result['collisions']} reward={result['avg_reward']:.2f}")

        if args.save_model:
            agent.save_model(args.save_model)
        print("\nMARL训练完成！")

    else:
        # ===== 普通仿真模式 =====
        # 创建并运行仿真
        sim = Simulation(render=not args.no_render)

        # 覆盖最大步数
        if args.steps:
            sim.max_steps = args.steps

        # CBS主导模式
        if args.cbs_primary:
            sim.controller.use_rl_primary = False
            print("使用 CBS 主导路径规划模式")

        sim.run()
    
    print("\n仿真结束，感谢使用！")


if __name__ == "__main__":
    main()
