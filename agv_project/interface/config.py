"""
============================================
全局配置管理模块
============================================
本模块负责管理整个项目的全局配置，包括：
1. 仓库地图配置（尺寸、布局）
2. AGV参数配置（数量、速度、电量等）
3. 仿真运行配置（步数、渲染模式等）
4. 强化学习训练配置（学习率、折扣因子等）
5. 日志配置

使用单例模式确保全局配置唯一性。
"""

import json
import os
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


# ==========================================
# 仓库地图配置
# ==========================================

# 默认仓库地图 (10x10)
# 0=空地, 1=障碍物/货架, 2=装货区, 3=卸货区, 4=充电站
DEFAULT_WAREHOUSE_MAP = [
    [2, 0, 0, 1, 0, 0, 1, 0, 0, 3],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [3, 0, 0, 1, 0, 0, 1, 0, 0, 4],
]


@dataclass
class MapConfig:
    """
    地图配置类
    
    定义仓库的物理布局，包括尺寸、障碍物、功能区位置等。
    
    Attributes:
        width: 地图宽度（列数），默认10
        height: 地图高度（行数），默认10
        grid: 二维网格地图，每个元素为整数表示单元格类型
        cell_size: 每个单元格的像素大小（用于可视化），默认50
    """
    width: int = 10
    height: int = 10
    grid: list = field(default_factory=lambda: DEFAULT_WAREHOUSE_MAP)
    cell_size: int = 50


@dataclass
class AGVConfig:
    """
    AGV参数配置类
    
    定义AGV的物理和行为参数。
    
    Attributes:
        num_agvs: AGV数量，默认4
        max_speed: 最大移动速度（格/时间步），默认1.0
        battery_capacity: 电池容量，默认100.0
        battery_consumption_per_step: 每步耗电量，默认0.5
        charge_rate: 充电速率（每时间步），默认5.0
        load_capacity: 最大载重，默认10.0
        sensor_range: 传感器探测范围（格），默认3
    """
    num_agvs: int = 4
    max_speed: float = 1.0
    battery_capacity: float = 100.0
    battery_consumption_per_step: float = 0.5
    charge_rate: float = 5.0
    load_capacity: float = 10.0
    sensor_range: int = 3


@dataclass
class SimulationConfig:
    """
    仿真运行配置类
    
    控制仿真环境的运行参数。
    
    Attributes:
        max_steps: 最大仿真步数，默认1000
        task_generation_interval: 任务生成间隔（步数），默认10
        max_concurrent_tasks: 最大并发任务数，默认20
        render_mode: 渲染模式，可选 "human", "rgb_array", None
        render_fps: 渲染帧率，默认10
        seed: 随机种子，默认42
        log_level: 日志级别，默认 "INFO"
        save_log: 是否保存日志到文件，默认True
        log_dir: 日志保存目录，默认 "logs"
    """
    max_steps: int = 1000
    task_generation_interval: int = 10
    max_concurrent_tasks: int = 20
    poisson_lambda: float = 0.2  # 泊松分布参数，控制货物到达频率（平均每 1/λ 步生成1个货物）
    render_mode: Optional[str] = "human"
    render_fps: int = 10
    seed: int = 42
    log_level: str = "INFO"
    save_log: bool = True
    log_dir: str = "logs"


@dataclass
class RLConfig:
    """
    强化学习训练配置类
    
    定义强化学习算法的超参数。
    
    Attributes:
        algorithm: RL算法，默认 "DQN"（可选: "DQN", "PPO", "A2C"）
        learning_rate: 学习率，默认 0.001
        gamma: 折扣因子，默认 0.99
        epsilon_start: 探索率初始值，默认 1.0
        epsilon_end: 探索率最终值，默认 0.01
        epsilon_decay: 探索率衰减率，默认 0.995
        batch_size: 训练批次大小，默认 64
        memory_size: 经验回放缓冲区大小，默认 100000
        target_update_interval: 目标网络更新间隔，默认 100
        hidden_dim: 隐藏层维度，默认 256
        num_episodes: 训练回合数，默认 1000
        max_steps_per_episode: 每回合最大步数，默认 200
    """
    algorithm: str = "DQN"
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    memory_size: int = 100000
    target_update_interval: int = 100
    hidden_dim: int = 256
    num_episodes: int = 1000
    max_steps_per_episode: int = 200


@dataclass
class RewardConfig:
    """
    奖励函数配置类
    
    定义强化学习中的奖励值。
    
    Attributes:
        task_complete: 完成任务奖励，默认 100.0
        collision_penalty: 碰撞惩罚，默认 -50.0
        step_penalty: 每步惩罚（鼓励最短路径），默认 -1.0
        load_success: 装货成功奖励，默认 10.0
        unload_success: 卸货成功奖励，默认 10.0
        battery_low_penalty: 低电量惩罚，默认 -20.0
        idle_penalty: 空闲惩罚（鼓励工作），默认 -0.5
        conflict_wait_penalty: 冲突等待惩罚，默认 -2.0
    """
    task_complete: float = 100.0
    collision_penalty: float = -50.0
    step_penalty: float = -1.0
    load_success: float = 10.0
    unload_success: float = 10.0
    battery_low_penalty: float = -20.0
    idle_penalty: float = -0.5
    conflict_wait_penalty: float = -2.0


# ==========================================
# 全局配置管理器（单例模式）
# ==========================================

class ConfigManager:
    """
    全局配置管理器
    
    使用单例模式，确保整个项目使用同一份配置。
    支持从JSON文件加载配置和保存配置到JSON文件。
    
    Usage:
        >>> config = ConfigManager()
        >>> config.simulation.max_steps = 2000
        >>> config.save("config.json")
        >>> config.load("config.json")
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """单例模式：确保只有一个ConfigManager实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化配置（只执行一次）"""
        if not ConfigManager._initialized:
            self.map = MapConfig()
            self.agv = AGVConfig()
            self.simulation = SimulationConfig()
            self.rl = RLConfig()
            self.reward = RewardConfig()
            self._setup_logging()
            ConfigManager._initialized = True
    
    def _setup_logging(self):
        """
        设置日志系统
        
        根据配置初始化日志记录器，支持控制台输出和文件输出。
        """
        log_level = getattr(logging, self.simulation.log_level.upper(), logging.INFO)
        
        # 创建日志记录器
        self.logger = logging.getLogger("AGVProject")
        self.logger.setLevel(log_level)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # 文件处理器（可选）
        if self.simulation.save_log:
            log_dir = self.simulation.log_dir
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"simulation_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_format = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"日志文件已创建: {log_file}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将所有配置转换为字典
        
        Returns:
            包含所有配置的字典
        
        Example:
            >>> config = ConfigManager()
            >>> config_dict = config.to_dict()
            >>> print(config_dict['simulation']['max_steps'])
            1000
        """
        return {
            "map": asdict(self.map),
            "agv": asdict(self.agv),
            "simulation": asdict(self.simulation),
            "rl": asdict(self.rl),
            "reward": asdict(self.reward),
        }
    
    def save(self, filepath: str = "config.json"):
        """
        保存配置到JSON文件
        
        Args:
            filepath: 保存路径，默认 "config.json"
        
        Example:
            >>> config = ConfigManager()
            >>> config.save("my_config.json")
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
        self.logger.info(f"配置已保存到: {filepath}")
    
    def load(self, filepath: str = "config.json"):
        """
        从JSON文件加载配置
        
        Args:
            filepath: 配置文件路径，默认 "config.json"
        
        Example:
            >>> config = ConfigManager()
            >>> config.load("my_config.json")
        """
        if not os.path.exists(filepath):
            self.logger.warning(f"配置文件不存在: {filepath}，使用默认配置")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 更新各配置项
        if "map" in data:
            self.map = MapConfig(**data["map"])
        if "agv" in data:
            self.agv = AGVConfig(**data["agv"])
        if "simulation" in data:
            self.simulation = SimulationConfig(**data["simulation"])
        if "rl" in data:
            self.rl = RLConfig(**data["rl"])
        if "reward" in data:
            self.reward = RewardConfig(**data["reward"])
        
        self.logger.info(f"配置已从 {filepath} 加载")
    
    def update(self, section: str, key: str, value: Any):
        """
        更新单个配置项
        
        Args:
            section: 配置节名称（"map", "agv", "simulation", "rl", "reward"）
            key: 配置键名
            value: 新的配置值
        
        Example:
            >>> config = ConfigManager()
            >>> config.update("simulation", "max_steps", 2000)
            >>> config.update("rl", "learning_rate", 0.0001)
        """
        if hasattr(self, section):
            section_obj = getattr(self, section)
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
                self.logger.info(f"配置已更新: {section}.{key} = {value}")
            else:
                self.logger.error(f"配置节 '{section}' 中不存在键 '{key}'")
        else:
            self.logger.error(f"不存在配置节 '{section}'")
    
    def print_config(self):
        """
        打印当前所有配置（用于调试）
        
        Example:
            >>> config = ConfigManager()
            >>> config.print_config()
        """
        print("=" * 50)
        print("当前配置:")
        print("=" * 50)
        for section, config_dict in self.to_dict().items():
            print(f"\n[{section}]")
            for key, value in config_dict.items():
                print(f"  {key}: {value}")
        print("=" * 50)


# ==========================================
# 便捷函数：获取全局配置实例
# ==========================================

def get_config() -> ConfigManager:
    """
    获取全局配置管理器实例
    
    这是获取配置的推荐方式，确保整个项目使用同一份配置。
    
    Returns:
        ConfigManager 单例实例
    
    Example:
        >>> config = get_config()
        >>> print(config.simulation.max_steps)
        1000
    """
    return ConfigManager()


# ==========================================
# 测试代码
# ==========================================

if __name__ == "__main__":
    """运行本文件测试配置功能"""
    # 获取配置
    config = get_config()
    
    # 打印当前配置
    config.print_config()
    
    # 修改配置
    config.update("simulation", "max_steps", 2000)
    config.update("rl", "learning_rate", 0.0005)
    
    # 保存配置到文件
    config.save("test_config.json")
    
    # 从文件加载配置
    config.load("test_config.json")
    
    print("\n配置测试完成！")
