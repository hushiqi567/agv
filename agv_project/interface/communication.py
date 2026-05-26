"""
============================================
模块间通信接口模块
============================================
本模块定义了各模块之间的通信协议和接口规范。
采用"发布-订阅"模式，实现模块间的松耦合通信。

通信流程：
1. 环境模块(env) 发布：地图状态、AGV位置、任务状态
2. 调度模块(scheduler) 订阅：AGV状态、任务池 → 发布：任务分配结果
3. 路径规划模块(planner) 订阅：任务分配、地图状态 → 发布：规划路径
4. 主控制器(main) 协调所有模块的数据流

每个模块通过 MessageBus 发送和接收消息，不直接调用其他模块。
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict
import logging
from datetime import datetime


# ==========================================
# 消息类型定义
# ==========================================

class MessageType(Enum):
    """
    消息类型枚举
    
    定义了系统中所有可能的通信消息类型。
    命名规则: [发送方]_[内容]_[动作]
    """
    # ===== 环境模块消息 (env → *) =====
    ENV_STATE_UPDATE = "env_state_update"           # 环境状态更新（地图、AGV位置等）
    ENV_AGV_POSITION_UPDATE = "env_agv_position_update"  # AGV位置更新
    ENV_TASK_COMPLETED = "env_task_completed"       # 任务完成通知
    ENV_COLLISION_DETECTED = "env_collision_detected"    # 碰撞检测通知
    ENV_STEP_FINISHED = "env_step_finished"         # 单步执行完成
    
    # ===== 调度模块消息 (scheduler → *) =====
    SCHEDULER_TASK_GENERATED = "scheduler_task_generated"    # 新任务生成
    SCHEDULER_TASK_ASSIGNED = "scheduler_task_assigned"      # 任务分配结果
    SCHEDULER_TASK_CANCELLED = "scheduler_task_cancelled"    # 任务取消
    SCHEDULER_OD_FLOW_UPDATED = "scheduler_od_flow_updated"  # OD流程更新
    
    # ===== 路径规划模块消息 (planner → *) =====
    PLANNER_PATH_UPDATED = "planner_path_updated"           # 路径更新
    PLANNER_PATH_BLOCKED = "planner_path_blocked"           # 路径阻塞
    PLANNER_CONFLICT_RESOLVED = "planner_conflict_resolved" # 冲突解决
    PLANNER_REPLAN_REQUESTED = "planner_replan_requested"   # 请求重新规划
    
    # ===== 主控制器消息 (main → *) =====
    MAIN_SIMULATION_START = "main_simulation_start"         # 仿真开始
    MAIN_SIMULATION_PAUSE = "main_simulation_pause"         # 仿真暂停
    MAIN_SIMULATION_RESUME = "main_simulation_resume"       # 仿真恢复
    MAIN_SIMULATION_STOP = "main_simulation_stop"           # 仿真停止
    MAIN_SIMULATION_RESET = "main_simulation_reset"         # 仿真重置
    MAIN_CONFIG_UPDATED = "main_config_updated"             # 配置更新


# ==========================================
# 消息数据结构
# ==========================================

@dataclass
class Message:
    """
    消息数据结构
    
    所有模块间通信都使用此消息格式。
    
    Attributes:
        msg_type: 消息类型（MessageType枚举）
        sender: 发送方模块名称（如 "env", "scheduler", "planner", "main"）
        timestamp: 消息发送时间戳
        data: 消息负载数据（字典格式）
        priority: 消息优先级（0-10，数字越大优先级越高）
        msg_id: 消息唯一标识符（自动生成）
    """
    msg_type: MessageType
    sender: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: int = 5
    msg_id: str = field(default_factory=lambda: f"msg_{datetime.now().timestamp()}")


# ==========================================
# 消息总线（核心通信组件）
# ==========================================

class MessageBus:
    """
    消息总线（单例模式）
    
    采用发布-订阅模式，实现模块间的松耦合通信。
    任何模块都可以发布消息或订阅感兴趣的消息类型。
    
    Usage:
        >>> bus = MessageBus()
        >>> 
        >>> # 订阅消息
        >>> def on_task_assigned(msg):
        ...     print(f"收到任务分配: {msg.data}")
        >>> bus.subscribe(MessageType.SCHEDULER_TASK_ASSIGNED, on_task_assigned)
        >>> 
        >>> # 发布消息
        >>> bus.publish(Message(
        ...     msg_type=MessageType.SCHEDULER_TASK_ASSIGNED,
        ...     sender="scheduler",
        ...     data={"task_id": 1, "agv_id": 0}
        ... ))
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """单例模式：确保只有一个MessageBus实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化消息总线（只执行一次）"""
        if not MessageBus._initialized:
            # 订阅者字典: {MessageType: [callback1, callback2, ...]}
            self._subscribers: Dict[MessageType, List[Callable]] = defaultdict(list)
            
            # 消息历史记录（用于调试和回放）
            self._message_history: List[Message] = []
            self._max_history: int = 1000
            
            # 日志记录器
            self.logger = logging.getLogger("AGVProject.MessageBus")
            
            MessageBus._initialized = True
    
    def subscribe(self, msg_type: MessageType, callback: Callable):
        """
        订阅指定类型的消息
        
        Args:
            msg_type: 要订阅的消息类型
            callback: 收到消息时的回调函数，接收Message对象作为参数
        
        Example:
            >>> def handle_path_update(msg):
            ...     print(f"路径更新: {msg.data}")
            >>> bus.subscribe(MessageType.PLANNER_PATH_UPDATED, handle_path_update)
        """
        if callback not in self._subscribers[msg_type]:
            self._subscribers[msg_type].append(callback)
            self.logger.debug(f"新订阅: {msg_type.value}")
    
    def unsubscribe(self, msg_type: MessageType, callback: Callable):
        """
        取消订阅指定类型的消息
        
        Args:
            msg_type: 要取消订阅的消息类型
            callback: 之前注册的回调函数
        
        Example:
            >>> bus.unsubscribe(MessageType.PLANNER_PATH_UPDATED, handle_path_update)
        """
        if callback in self._subscribers[msg_type]:
            self._subscribers[msg_type].remove(callback)
            self.logger.debug(f"取消订阅: {msg_type.value}")
    
    def publish(self, message: Message):
        """
        发布消息到所有订阅者
        
        Args:
            message: 要发布的消息对象
        
        消息会按优先级排序后发送给所有订阅者。
        高优先级的消息会先被处理。
        
        Example:
            >>> bus.publish(Message(
            ...     msg_type=MessageType.ENV_STATE_UPDATE,
            ...     sender="env",
            ...     data={"step": 10, "agv_positions": {...}}
            ... ))
        """
        # 记录消息历史
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history.pop(0)
        
        # 获取该消息类型的所有订阅者
        subscribers = self._subscribers.get(message.msg_type, [])
        
        if not subscribers:
            self.logger.debug(f"消息 {message.msg_type.value} 无订阅者")
            return
        
        # 按优先级排序（高优先级先处理）
        # 这里简单起见，直接按注册顺序调用
        self.logger.debug(
            f"发布消息: {message.msg_type.value} "
            f"(来自 {message.sender}, 优先级 {message.priority})"
        )
        
        for callback in subscribers:
            try:
                callback(message)
            except Exception as e:
                self.logger.error(
                    f"处理消息 {message.msg_type.value} 时出错: {e}"
                )
    
    def get_history(self, 
                    msg_type: Optional[MessageType] = None,
                    sender: Optional[str] = None,
                    limit: int = 10) -> List[Message]:
        """
        获取消息历史记录
        
        Args:
            msg_type: 按消息类型筛选（可选）
            sender: 按发送方筛选（可选）
            limit: 返回的最大消息数量，默认10
        
        Returns:
            符合条件的消息列表（按时间倒序）
        
        Example:
            >>> # 获取最近5条路径更新消息
            >>> history = bus.get_history(
            ...     msg_type=MessageType.PLANNER_PATH_UPDATED,
            ...     limit=5
            ... )
        """
        result = self._message_history.copy()
        
        if msg_type:
            result = [m for m in result if m.msg_type == msg_type]
        if sender:
            result = [m for m in result if m.sender == sender]
        
        # 按时间倒序排列（最新的在前）
        result.reverse()
        return result[:limit]
    
    def clear_history(self):
        """清空消息历史记录"""
        self._message_history.clear()
        self.logger.info("消息历史已清空")
    
    @property
    def subscriber_count(self) -> int:
        """获取当前订阅总数"""
        return sum(len(subs) for subs in self._subscribers.values())
    
    def print_subscribers(self):
        """打印所有订阅者信息（用于调试）"""
        print("=" * 50)
        print("当前订阅者列表:")
        print("=" * 50)
        for msg_type, callbacks in self._subscribers.items():
            if callbacks:
                print(f"\n[{msg_type.value}] ({len(callbacks)} 个订阅者)")
                for i, cb in enumerate(callbacks):
                    print(f"  {i+1}. {cb.__name__}")
        print("=" * 50)


# ==========================================
# 模块基类（所有模块的父类）
# ==========================================

class BaseModule:
    """
    模块基类
    
    所有功能模块（环境、调度、路径规划）都应继承此类。
    提供了与消息总线交互的标准接口。
    
    Attributes:
        module_name: 模块名称
        bus: 消息总线实例
        logger: 日志记录器
    """
    
    def __init__(self, module_name: str):
        """
        初始化模块
        
        Args:
            module_name: 模块名称（如 "env", "scheduler", "planner"）
        """
        self.module_name = module_name
        self.bus = MessageBus()
        self.logger = logging.getLogger(f"AGVProject.{module_name}")
        self._setup_subscriptions()
        
        self.logger.info(f"模块 {module_name} 初始化完成")
    
    def _setup_subscriptions(self):
        """
        设置消息订阅
        
        子类应重写此方法，注册需要订阅的消息类型。
        
        Example:
            >>> class EnvModule(BaseModule):
            ...     def _setup_subscriptions(self):
            ...         self.subscribe(MessageType.MAIN_SIMULATION_START, self.on_start)
            ...         self.subscribe(MessageType.MAIN_SIMULATION_STOP, self.on_stop)
        """
        pass
    
    def subscribe(self, msg_type: MessageType, callback: Callable):
        """
        订阅消息的便捷方法
        
        Args:
            msg_type: 消息类型
            callback: 回调函数
        """
        self.bus.subscribe(msg_type, callback)
    
    def publish(self, msg_type: MessageType, data: Dict[str, Any], 
                priority: int = 5):
        """
        发布消息的便捷方法
        
        Args:
            msg_type: 消息类型
            data: 消息数据
            priority: 优先级（0-10）
        """
        message = Message(
            msg_type=msg_type,
            sender=self.module_name,
            data=data,
            priority=priority
        )
        self.bus.publish(message)
    
    def on_start(self, message: Message):
        """
        仿真开始时的回调
        
        Args:
            message: 启动消息
        """
        self.logger.info(f"仿真开始: {message.data}")
    
    def on_stop(self, message: Message):
        """
        仿真停止时的回调
        
        Args:
            message: 停止消息
        """
        self.logger.info(f"仿真停止: {message.data}")
    
    def on_reset(self, message: Message):
        """
        仿真重置时的回调
        
        Args:
            message: 重置消息
        """
        self.logger.info(f"仿真重置: {message.data}")


# ==========================================
# 便捷函数：获取消息总线实例
# ==========================================

def get_message_bus() -> MessageBus:
    """
    获取全局消息总线实例
    
    这是获取消息总线的推荐方式，确保整个项目使用同一个总线。
    
    Returns:
        MessageBus 单例实例
    
    Example:
        >>> bus = get_message_bus()
        >>> bus.subscribe(MessageType.ENV_STATE_UPDATE, my_callback)
    """
    return MessageBus()


# ==========================================
# 测试代码
# ==========================================

if __name__ == "__main__":
    """运行本文件测试通信功能"""
    import time
    
    # 设置日志
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 50)
    print("通信模块测试")
    print("=" * 50)
    
    # 获取消息总线
    bus = get_message_bus()
    
    # 定义测试回调函数
    def on_state_update(msg):
        print(f"[订阅者1] 收到状态更新: {msg.data}")
    
    def on_task_assigned(msg):
        print(f"[订阅者2] 收到任务分配: {msg.data}")
    
    # 订阅消息
    bus.subscribe(MessageType.ENV_STATE_UPDATE, on_state_update)
    bus.subscribe(MessageType.SCHEDULER_TASK_ASSIGNED, on_task_assigned)
    
    # 发布测试消息
    print("\n发布测试消息...")
    bus.publish(Message(
        msg_type=MessageType.ENV_STATE_UPDATE,
        sender="env",
        data={"step": 1, "agv_count": 4}
    ))
    
    bus.publish(Message(
        msg_type=MessageType.SCHEDULER_TASK_ASSIGNED,
        sender="scheduler",
        data={"task_id": 101, "agv_id": 2}
    ))
    
    # 查看订阅者
    print()
    bus.print_subscribers()
    
    # 查看消息历史
    print("\n最近消息历史:")
    for msg in bus.get_history(limit=5):
        print(f"  [{msg.timestamp}] {msg.msg_type.value} (来自 {msg.sender})")
    
    print("\n通信模块测试完成！")
