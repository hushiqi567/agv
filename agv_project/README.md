# 基于强化学习的多AGV协同路径规划系统

## 项目概述

本项目实现了一个基于**强化学习**的**多AGV（自动导引车）**协同路径规划仿真系统。系统包含四个核心模块，通过**发布-订阅消息模式**实现模块间松耦合通信。

## 项目结构

```
agv_project/
│
├── main.py                          # 主入口 - 仿真控制器
│
├── interface/                       # 接口定义模块
│   ├── __init__.py                  # 模块初始化，统一导出
│   ├── data_types.py                # 数据结构定义（Task, AGVState, CellType等）
│   ├── config.py                    # 全局配置管理（单例模式）
│   └── communication.py             # 模块间通信（发布-订阅模式）
│
├── env/                             # 仓库环境模块
│   ├── __init__.py
│   ├── warehouse_env.py             # 50×50仓库环境核心逻辑
│   └── renderer.py                  # Pygame可视化渲染器
│
├── scheduler/                       # 调度方案模块
│   ├── __init__.py
│   ├── task_allocator.py            # 任务分配算法（最近距离优先）
│   └── od_flow.py                   # OD流程管理（任务生命周期）
│
├── path_planning/                   # 路径规划模块
│   ├── __init__.py
│   ├── mapf_planner.py              # MAPF全局规划（CBS算法）
│   ├── rl_collision_avoidance.py    # RL实时避撞（DQN算法）
│   └── agv_controller.py            # AGV控制器（整合MAPF+RL）
│
└── logs/                            # 日志文件目录
```

## 模块间通信

系统采用**发布-订阅模式**实现模块间松耦合通信：

```
[环境模块] ──发布──→ [消息总线] ──通知──→ [调度模块]
    ↑                                      │
    │                                      ↓
    └────────────── [消息总线] ←──── [路径规划模块]
```

### 消息类型

| 消息类型 | 发送方 | 说明 |
|---------|--------|------|
| ENV_STATE_UPDATE | 环境 | 环境状态更新 |
| ENV_TASK_COMPLETED | 环境 | 任务完成通知 |
| ENV_COLLISION_DETECTED | 环境 | 碰撞检测通知 |
| SCHEDULER_TASK_GENERATED | 调度 | 新任务生成 |
| SCHEDULER_TASK_ASSIGNED | 调度 | 任务分配结果 |
| PLANNER_PATH_UPDATED | 路径规划 | 路径更新 |
| PLANNER_CONFLICT_RESOLVED | 路径规划 | 冲突解决 |
| MAIN_SIMULATION_START/STOP | 主控制器 | 仿真控制 |

## 快速开始

### 1. 安装依赖

```bash
pip install numpy gymnasium matplotlib pygame torch
```

### 2. 运行仿真

```bash
# 默认配置运行
python main.py

# 无界面模式，500步
python main.py --no-render --steps 500

# 查看帮助
python main.py --help
```

### 3. 配置文件示例 (config.json)

```json
{
    "map": {
        "width": 50,
        "height": 50,
        "cell_size": 50
    },
    "agv": {
        "num_agvs": 8,
        "max_speed": 1.0,
        "battery_capacity": 100.0
    },
    "simulation": {
        "max_steps": 1000,
        "task_generation_interval": 10,
        "render_mode": "human"
    },
    "rl": {
        "algorithm": "DQN",
        "learning_rate": 0.001,
        "gamma": 0.99
    },
    "reward": {
        "task_complete": 100.0,
        "collision_penalty": -50.0,
        "step_penalty": -1.0
    }
}
```

## 仿真流程

```
时间步 t:
  1. 环境模块更新障碍物位置
  2. 环境发布 ENV_STATE_UPDATE
  3. 调度模块生成新任务（泊松分布）
  4. 调度模块分配任务给空闲AGV（最近距离优先）
  5. 调度发布 SCHEDULER_TASK_ASSIGNED
  6. AGV控制器使用MAPF规划全局路径
  7. AGV控制器使用RL处理动态避撞
  8. AGV移动，检查任务完成
  9. 渲染器更新显示
时间步 t+1:
  ...
```

## 核心算法

### MAPF (Multi-Agent Path Finding)
- **算法**: Conflict-Based Search (CBS)
- **单AGV寻路**: A*算法（含时空约束）
- **冲突检测**: 顶点冲突 + 边冲突

### RL (Reinforcement Learning)
- **算法**: Deep Q-Network (DQN)
- **状态表示**: 5×5局部网格（3通道）
- **动作空间**: 上、下、左、右、等待

### 任务调度
- **任务生成**: 泊松分布 (λ=0.5)
- **任务分配**: 最近距离优先
- **出货口管理**: 占用/释放机制

## 许可证

本项目仅供学习研究使用。
