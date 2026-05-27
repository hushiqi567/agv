# 基于强化学习的多AGV协同路径规划系统

> 第十八届大学生科研训练计划（SRTP）项目

## 项目概述

本项目实现了一个 **50×50 无人仓储网格**中的多AGV（自动导引车）路径规划仿真系统。系统采用 **CBS（Conflict-Based Search）全局规划 + DQN（Deep Q-Network）局部避撞** 的混合架构，通过发布-订阅消息总线实现模块间松耦合通信，支持可视化仿真与强化学习离线训练。

### 核心特性

- **仓库环境**：50×50网格，20个进货口、12个出货口、10个随机移动障碍物
- **MAPF全局规划**：基于CBS算法的多AGV无冲突路径规划，底层使用带时空约束的A\*搜索
- **RL实时避撞**：基于DQN的5×5局部网格感知与动态避撞，支持ε-贪心探索与经验回放训练
- **任务调度**：泊松分布生成任务，最近距离优先分配，OD流程全生命周期管理
- **可视化渲染**：Pygame实现，显示网格地图、障碍物、AGV实时状态与统计信息
- **训练模式**：支持命令行启动RL训练，模型持久化与加载

## 项目结构

```
agv_project/
├── README.md
├── main.py                              # 仿真主入口
├── rl_model.pth / rl_model_v2.pth       # 预训练RL模型权重
│
├── interface/                           # 接口定义层
│   ├── data_types.py                    # 数据结构（Task, AGVState, CellType等）
│   ├── config.py                        # 全局配置管理（单例模式，支持JSON读写）
│   └── communication.py                 # 发布-订阅消息总线
│
├── env/                                 # 仓库环境层
│   ├── warehouse_env.py                 # 50×50网格环境核心逻辑
│   └── renderer.py                      # Pygame可视化渲染器
│
├── scheduler/                           # 任务调度层
│   ├── task_allocator.py                # 任务分配器（泊松生成 + 最近距离分配）
│   └── od_flow.py                       # OD流程管理器（任务生命周期 + 出货口占用）
│
├── path_planning/                       # 路径规划层
│   ├── mapf_planner.py                  # CBS多AGV全局路径规划
│   ├── rl_collision_avoidance.py        # DQN实时避撞控制器
│   └── agv_controller.py                # AGV控制器（整合MAPF + RL）
│
├── generate_report.py                   # 报告生成脚本
├── logs/                                # 日志输出目录
└── doc/                                 # SRTP申报文档
```

## 系统架构

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
│  scheduler  │────▶│  agv_controller  │────▶│     env      │
│  任务调度    │     │    AGV控制器      │     │   仓库环境    │
└─────────────┘     └────────┬────────┘     └──────┬───────┘
                             │                      │
                      ┌──────▼───────┐              │
                      │  mapf_planner │              │
                      │  CBS全局规划  │              │
                      └──────┬───────┘              │
                             │                      │
                      ┌──────▼───────────┐          │
                      │ rl_avoidance     │◀─────────┘
                      │  DQN实时避撞     │
                      └──────────────────┘

         ◀───────────  MessageBus (发布-订阅) ───────────▶
```

### 仿真流程（每时间步）

1. **环境更新**：所有障碍物随机移动一格
2. **任务生成**：按 Poisson(λ) 分布生成新货物，随机分配取货点与空闲出货口
3. **任务分配**：为每个待分配任务选择距离取货点最近的空闲AGV
4. **路径规划**：CBS算法为所有活跃AGV规划无冲突全局路径
5. **AGV移动**：沿MAPF路径前进；遇动态障碍物时切换到DQN局部避撞
6. **状态渲染**：更新Pygame窗口显示

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch（RL训练与推理）
- Pygame（可视化渲染）
- NumPy、Matplotlib

### 安装依赖

```bash
pip install numpy matplotlib pygame torch
```

### 运行仿真

```bash
# 默认配置运行（可视化模式）
cd agv_project
python main.py

# 无界面模式，运行500步
python main.py --no-render --steps 500

# RL训练模式
python main.py --train --train-episodes 50 --save-model my_model.pth

# 加载预训练模型进行推理
python main.py --load-model rl_model_v2.pth

# 查看完整命令行参数
python main.py --help
```

### 操作说明

| 按键 | 功能 |
|------|------|
| ESC | 退出仿真 |
| R | 重置仿真 |
| SPACE | 暂停/继续 |

## 核心算法

### CBS（Conflict-Based Search）

- 为每个AGV独立运行A\*规划初始路径
- 检测路径间的顶点冲突和边冲突
- 通过约束树搜索逐步消解冲突
- 返回所有AGV的无冲突路径集合

### DQN（Deep Q-Network）

- **网络结构**：2层卷积 + 2层全连接
- **状态空间**：5×5局部网格 × 3通道（静态障碍物、动态实体、目标方向）
- **动作空间**：上、下、左、右、等待（5个离散动作）
- **奖励函数**：到达目标 +100，完成任务 +150，碰撞 -50，接近/远离目标 ±1，步数惩罚 -0.1
- **训练机制**：经验回放（容量100K）、目标网络（每100步同步）、ε-贪心探索（1.0 → 0.05）

### 任务调度

- **任务生成**：每步按 Poisson(λ=0.2) 随机生成新货物
- **取货点**：从20个进货口中等概率随机选择
- **送货点**：从12个空闲出货口中等概率随机选择
- **分配策略**：最近曼哈顿距离优先，占用出货口直至任务完成释放

## 配置说明

系统使用单例模式的 `ConfigManager` 管理全局配置，支持运行时修改与JSON文件持久化。

```python
from interface.config import get_config

config = get_config()
config.update("simulation", "max_steps", 2000)
config.update("rl", "learning_rate", 0.0005)
config.save("my_config.json")
```

### 主要配置项

| 配置节 | 关键参数 | 默认值 | 说明 |
|--------|---------|--------|------|
| map | width/height | 50/50 | 地图尺寸 |
| agv | num_agvs | 8 | AGV数量 |
| simulation | max_steps | 1000 | 最大仿真步数 |
| simulation | poisson_lambda | 0.2 | 任务生成速率 |
| rl | learning_rate | 0.001 | DQN学习率 |
| rl | gamma | 0.99 | 折扣因子 |
| reward | task_complete | 100.0 | 任务完成奖励 |
| reward | collision_penalty | -50.0 | 碰撞惩罚 |

## 模块通信

系统定义了完整的发布-订阅消息总线（`MessageBus`），支持以下消息类型：

| 消息类型 | 发送方 | 用途 |
|---------|--------|------|
| `ENV_STATE_UPDATE` | 环境模块 | 网格状态与AGV位置同步 |
| `ENV_TASK_COMPLETED` | AGV控制器 | 任务完成通知 |
| `SCHEDULER_TASK_GENERATED` | 调度模块 | 新任务生成通知 |
| `SCHEDULER_TASK_ASSIGNED` | 调度模块 | 任务分配结果下发给AGV |
| `PLANNER_PATH_UPDATED` | 路径规划 | 路径更新通知 |

## 许可证

本项目仅供学习研究使用。
