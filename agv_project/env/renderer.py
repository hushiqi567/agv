"""
============================================
仓库环境 - Pygame 可视化渲染器
============================================
本模块使用 Pygame 实现 50×50 仓库网格的可视化，
展示进货口（蓝色）、出货口（红色）、障碍物（黑色）、AGV（彩色）的实时状态。

使用方式：
    python renderer.py  # 独立运行测试
"""

import sys
import os
import pygame
import logging
from typing import List, Tuple, Optional, Dict

# 将项目根目录添加到系统路径（独立运行时使用）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from env.warehouse_env import (
    WarehouseEnv, CELL_EMPTY, CELL_OBSTACLE, CELL_LOADING, CELL_UNLOADING,
    CELL_CHARGING,
    COLOR_EMPTY, COLOR_LOADING, COLOR_UNLOADING, COLOR_OBSTACLE,
    COLOR_CHARGING, COLOR_GRID_LINE, COLOR_BACKGROUND,
    MAP_WIDTH, MAP_HEIGHT
)
from interface.data_types import AGVStatus


# ==========================================
# 渲染配置
# ==========================================

# 窗口尺寸（像素）
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700

# 网格渲染模式
# 50x50 网格，每个格子大小根据窗口自适应
CELL_SIZE = min(
    (WINDOW_WIDTH - 100) // MAP_WIDTH,   # 左右留边距
    (WINDOW_HEIGHT - 180) // MAP_HEIGHT   # 上下留边距（底部留信息栏）
)

# 网格偏移（居中显示）
GRID_OFFSET_X = (WINDOW_WIDTH - CELL_SIZE * MAP_WIDTH) // 2
GRID_OFFSET_Y = (WINDOW_HEIGHT - 150 - CELL_SIZE * MAP_HEIGHT) // 2

# 帧率
FPS = 30

# 信息栏高度
INFO_BAR_HEIGHT = 130

# AGV颜色（8台AGV各不同颜色，更鲜艳）
AGV_COLORS = [
    (46, 204, 113),   # 绿色
    (52, 152, 219),   # 蓝色
    (155, 89, 182),   # 紫色
    (230, 126, 34),   # 橙色
    (231, 76, 60),    # 红色
    (26, 188, 156),   # 青色
    (241, 196, 15),   # 黄色
    (149, 165, 166),  # 灰色
]

# AGV状态颜色
AGV_STATUS_COLORS = {
    "IDLE": (200, 200, 200),              # 灰色 - 空闲
    "MOVING_TO_PICKUP": (100, 200, 255),  # 浅蓝 - 前往取货
    "LOADING": (255, 255, 100),           # 黄色 - 装载中
    "MOVING_TO_DELIVERY": (100, 255, 100),# 浅绿 - 前往送货
    "UNLOADING": (255, 150, 100),         # 橙色 - 卸货中
    "CHARGING": (255, 215, 0),            # 金色 - 充电中
    "MOVING_TO_CHARGE": (255, 200, 50),   # 橙金 - 前往充电
}

# AGV状态文字标签
AGV_STATUS_LABELS = {
    "IDLE": "空闲",
    "MOVING_TO_PICKUP": "取货",
    "LOADING": "装载",
    "MOVING_TO_DELIVERY": "送货",
    "UNLOADING": "卸货",
    "CHARGING": "充电",
    "MOVING_TO_CHARGE": "去充电",
}


class WarehouseRenderer:
    """
    仓库环境渲染器
    
    使用 Pygame 可视化展示仓库网格环境，包括：
    - 网格地图（进货口蓝色、出货口红色、空地灰色）
    - 障碍物（黑色方块，实时移动）
    - AGV（彩色方块，显示ID和装载状态）
    - 步数和障碍物数量信息
    
    Usage:
        >>> env = WarehouseEnv()
        >>> renderer = WarehouseRenderer(env)
        >>> env.reset()
        >>> for _ in range(100):
        ...     env.step()
        ...     renderer.render()
        ...     renderer.handle_events()
        >>> renderer.close()
    """
    
    def __init__(self, env: WarehouseEnv, fps: int = FPS):
        """
        初始化渲染器
        
        Args:
            env: 仓库环境实例
            fps: 渲染帧率
        """
        self.env = env
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.running = False
        
        # AGV控制器引用（由外部设置）
        self.agv_controller = None
        
        # 出货口占用状态 {position: True/False}（由外部设置）
        self.unloading_zone_status: Dict[Tuple[int, int], bool] = {}
        
        # 初始化 Pygame
        pygame.init()
        pygame.display.set_caption("AGV仓库环境仿真 - 50×50网格")
        
        # 创建窗口（可调整大小）
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        
        # 字体 - 使用系统字体支持中文
        # 尝试多个常见中文字体路径
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",     # 黑体
            "C:/Windows/Fonts/simsun.ttc",     # 宋体
            "C:/Windows/Fonts/yahei.ttf",      # 微软雅黑(备选)
        ]
        self.font_name = None
        for fp in font_paths:
            if os.path.exists(fp):
                self.font_name = fp
                break
        
        if self.font_name:
            self.font_large = pygame.font.Font(self.font_name, 36)
            self.font_medium = pygame.font.Font(self.font_name, 28)
            self.font_small = pygame.font.Font(self.font_name, 22)
            self.font_tiny = pygame.font.Font(self.font_name, 18)
        else:
            # 回退到默认字体（可能不支持中文）
            self.font_large = pygame.font.Font(None, 36)
            self.font_medium = pygame.font.Font(None, 28)
            self.font_small = pygame.font.Font(None, 22)
            self.font_tiny = pygame.font.Font(None, 18)
        
        self.logger = logging.getLogger("AGVProject.Renderer")
        self.logger.info(f"渲染器初始化完成: {WINDOW_WIDTH}x{WINDOW_HEIGHT}, 格子大小={CELL_SIZE}")
    
    def set_agv_controller(self, controller):
        """设置AGV控制器引用"""
        self.agv_controller = controller
    
    def set_unloading_zone_status(self, status: Dict[Tuple[int, int], bool]):
        """
        设置出货口占用状态
        
        Args:
            status: 出货口占用状态字典 {position: True(有货)/False(无货)}
        """
        self.unloading_zone_status = status
    
    def render(self):
        """
        渲染当前环境状态
        
        绘制网格、障碍物、AGV和信息栏。
        """
        if not self.running:
            self.running = True
        
        # 清空屏幕
        self.screen.fill(COLOR_BACKGROUND)
        
        # 绘制网格
        self._draw_grid()
        
        # 绘制障碍物
        self._draw_obstacles()
        
        # 绘制AGV
        self._draw_agvs()
        
        # 绘制信息栏
        self._draw_info_bar()
        
        # 更新显示
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _draw_grid(self):
        """
        绘制网格地图
        
        每个格子根据类型着色：
        - 空地: 浅灰色
        - 进货口: 蓝色（最左列和最右列）
        - 出货口: 红色（中间两列交错）
        """
        grid = self.env.get_grid_for_render()
        
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                # 计算像素位置
                px = GRID_OFFSET_X + x * CELL_SIZE
                py = GRID_OFFSET_Y + y * CELL_SIZE
                
                # 根据单元格类型选择颜色
                cell_type = grid[y][x]
                if cell_type == CELL_LOADING:
                    color = COLOR_LOADING
                elif cell_type == CELL_UNLOADING:
                    color = COLOR_UNLOADING
                elif cell_type == CELL_CHARGING:
                    color = COLOR_CHARGING
                elif cell_type == CELL_OBSTACLE:
                    color = COLOR_OBSTACLE
                else:
                    color = COLOR_EMPTY
                
                # 绘制填充矩形
                pygame.draw.rect(
                    self.screen, color,
                    (px, py, CELL_SIZE, CELL_SIZE)
                )
                
                # 绘制网格线
                pygame.draw.rect(
                    self.screen, COLOR_GRID_LINE,
                    (px, py, CELL_SIZE, CELL_SIZE),
                    1  # 线宽
                )
        
        # 在出货口（红色格子）上绘制"有货/无货"状态文字
        if self.unloading_zone_status:
            for y in range(MAP_HEIGHT):
                for x in range(MAP_WIDTH):
                    cell_type = grid[y][x]
                    if cell_type == CELL_LOADING:  # 红色 = 出货口（仓库出货，AGV来取货）
                        pos = (x, y)
                        has_goods = self.unloading_zone_status.get(pos, False)
                        status_text = "有货" if has_goods else "无货"
                        text_color = (200, 50, 50) if has_goods else (100, 180, 100)
                        
                        px = GRID_OFFSET_X + x * CELL_SIZE
                        py = GRID_OFFSET_Y + y * CELL_SIZE
                        
                        # 绘制文字
                        text_surf = self.font_tiny.render(status_text, True, text_color)
                        text_rect = text_surf.get_rect(
                            center=(px + CELL_SIZE // 2, py + CELL_SIZE // 2)
                        )
                        self.screen.blit(text_surf, text_rect)
        
        # 绘制边框
        border_rect = pygame.Rect(
            GRID_OFFSET_X, GRID_OFFSET_Y,
            MAP_WIDTH * CELL_SIZE, MAP_HEIGHT * CELL_SIZE
        )
        pygame.draw.rect(self.screen, (100, 100, 100), border_rect, 3)
    
    def _draw_obstacles(self):
        """
        绘制障碍物
        
        障碍物用黑色方块表示。
        """
        obstacles = self.env.get_obstacle_info()
        
        for obs in obstacles:
            x, y = obs["position"]
            
            # 计算像素位置（稍微内缩以显示在网格内）
            px = GRID_OFFSET_X + x * CELL_SIZE + 2
            py = GRID_OFFSET_Y + y * CELL_SIZE + 2
            size = CELL_SIZE - 4
            
            # 绘制障碍物（黑色）
            pygame.draw.rect(
                self.screen, COLOR_OBSTACLE,
                (px, py, size, size)
            )
    
    def _draw_agvs(self):
        """
        绘制AGV
        
        每台AGV用不同颜色表示，已装载的AGV有白色外圈标记。
        在AGV上方显示ID编号和状态标签。
        """
        if self.agv_controller is None:
            return
        
        agv_positions = self.agv_controller.get_agv_positions()
        
        for agv_id, pos in agv_positions.items():
            x, y = pos
            
            # 计算像素位置
            px = GRID_OFFSET_X + x * CELL_SIZE
            py = GRID_OFFSET_Y + y * CELL_SIZE
            
            # 获取AGV颜色
            color = AGV_COLORS[agv_id % len(AGV_COLORS)]
            
            # 获取AGV状态
            status_str = "IDLE"
            is_loaded = False
            if agv_id in self.agv_controller.agvs:
                agv = self.agv_controller.agvs[agv_id]
                status_str = agv.status.name if hasattr(agv.status, 'name') else str(agv.status)
                is_loaded = agv.is_loaded
            
            # 根据状态选择边框颜色
            border_color = AGV_STATUS_COLORS.get(status_str, (200, 200, 200))
            
            # 绘制AGV主体（圆形，更显眼）
            center_x = px + CELL_SIZE // 2
            center_y = py + CELL_SIZE // 2
            radius = CELL_SIZE // 2 - 1
            
            # 绘制外发光效果（状态颜色光环）
            glow_radius = radius + 3
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*border_color, 80), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (center_x - glow_radius, center_y - glow_radius))
            
            # 绘制AGV主体（圆形）
            pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
            pygame.draw.circle(self.screen, border_color, (center_x, center_y), radius, 2)
            
            # 如果已装载，绘制金色内圈
            if is_loaded:
                inner_radius = radius - 4
                pygame.draw.circle(self.screen, (255, 215, 0), (center_x, center_y), inner_radius, 2)
                # 在中心画一个小星星或点
                pygame.draw.circle(self.screen, (255, 215, 0), (center_x, center_y), 3)

            # 电池条（AGV下方）
            bat_pct = agv.battery / 100.0 if agv_id in self.agv_controller.agvs else 1.0
            bar_w = max(4, CELL_SIZE - 2)
            bar_h = max(2, CELL_SIZE // 8)
            bar_x = px + 1
            bar_y = py + CELL_SIZE - bar_h - 1
            pygame.draw.rect(self.screen, (60, 60, 60),
                             (bar_x, bar_y, bar_w, bar_h))
            if bat_pct > 0.5:
                bc = (int(255 * (1 - bat_pct) * 2), 200, 50)
            elif bat_pct > 0.2:
                bc = (255, int(255 * (bat_pct - 0.2) / 0.3 * 0.8 + 50), 50)
            else:
                bc = (255, 50, 50)
            fw = int(bar_w * bat_pct)
            if fw > 0:
                pygame.draw.rect(self.screen, bc,
                                 (bar_x, bar_y, fw, bar_h))

            # 在AGV上显示ID
            if CELL_SIZE >= 10:
                id_text = self.font_tiny.render(str(agv_id), True, (255, 255, 255))
                text_rect = id_text.get_rect(
                    center=(center_x, center_y)
                )
                self.screen.blit(id_text, text_rect)
            
            # 在AGV上方显示状态标签
            status_label = AGV_STATUS_LABELS.get(status_str, status_str)
            if status_str != "IDLE":
                label_color = border_color
                label_text = self.font_tiny.render(status_label, True, label_color)
                label_rect = label_text.get_rect(
                    center=(center_x, py - 4)
                )
                # 绘制标签背景
                label_bg = pygame.Surface((label_rect.width + 4, label_rect.height + 2))
                label_bg.fill((255, 255, 255))
                label_bg.set_alpha(180)
                self.screen.blit(label_bg, (label_rect.x - 2, label_rect.y - 1))
                self.screen.blit(label_text, label_rect)
    
    def _draw_info_bar(self):
        """
        绘制底部信息栏
        
        显示当前步数、障碍物数量、AGV状态统计、操作提示等信息。
        """
        info_y = WINDOW_HEIGHT - INFO_BAR_HEIGHT
        
        # 信息栏背景
        pygame.draw.rect(
            self.screen, (230, 230, 230),
            (0, info_y, WINDOW_WIDTH, INFO_BAR_HEIGHT)
        )
        pygame.draw.line(
            self.screen, (180, 180, 180),
            (0, info_y), (WINDOW_WIDTH, info_y), 2
        )
        
        # 获取环境状态
        step = self.env.current_step
        num_obstacles = len(self.env.obstacles)
        
        # 获取AGV状态统计
        idle_count = 0
        moving_count = 0
        loading_count = 0
        tasks_completed = 0
        
        if self.agv_controller:
            stats = self.agv_controller.get_statistics()
            idle_count = stats['agvs']['idle']
            moving_count = stats['agvs']['moving']
            charging_count = stats['agvs'].get('charging', 0)
            loading_count = stats['agvs'].get('loading', 0)
            tasks_completed = stats['tasks_completed']
        
        # 步数信息
        step_text = self.font_large.render(
            f"步数: {step}", True, (50, 50, 50)
        )
        self.screen.blit(step_text, (30, info_y + 10))
        
        # 障碍物信息
        obs_text = self.font_small.render(
            f"障碍物: {num_obstacles}/10", True, (50, 50, 50)
        )
        self.screen.blit(obs_text, (30, info_y + 50))
        
        # AGV状态统计
        agv_text = self.font_small.render(
            f"AGV: 空闲{idle_count} 移动{moving_count} 充电{charging_count} 装卸{loading_count}",
            True, (50, 50, 50)
        )
        self.screen.blit(agv_text, (200, info_y + 10))
        
        # 完成任务数
        task_text = self.font_small.render(
            f"已完成任务: {tasks_completed}", True, (50, 50, 50)
        )
        self.screen.blit(task_text, (200, info_y + 50))
        
        # 地图尺寸信息
        map_text = self.font_small.render(
            f"地图: {MAP_WIDTH}x{MAP_HEIGHT}", True, (100, 100, 100)
        )
        self.screen.blit(map_text, (500, info_y + 10))
        
        # 图例
        legend_items = [
            ("装货口", COLOR_LOADING),
            ("卸货口", COLOR_UNLOADING),
            ("充电站", COLOR_CHARGING),
            ("障碍物", COLOR_OBSTACLE),
            ("AGV", AGV_COLORS[0]),
        ]
        
        legend_x = 500
        legend_y = info_y + 45
        for i, (label, color) in enumerate(legend_items):
            # 色块
            pygame.draw.rect(
                self.screen, color,
                (legend_x + i * 110, legend_y, 16, 16)
            )
            pygame.draw.rect(
                self.screen, (150, 150, 150),
                (legend_x + i * 110, legend_y, 16, 16), 1
            )
            # 标签
            label_text = self.font_tiny.render(label, True, (80, 80, 80))
            self.screen.blit(label_text, (legend_x + i * 110 + 22, legend_y + 1))
        
        # 操作提示
        hint_text = self.font_small.render(
            "ESC退出 | R重置 | SPACE暂停/继续", True, (120, 120, 120)
        )
        hint_rect = hint_text.get_rect(right=WINDOW_WIDTH - 30, bottom=info_y + INFO_BAR_HEIGHT - 15)
        self.screen.blit(hint_text, hint_rect)
        
        # 绘制每个AGV的详细状态（在信息栏底部）
        if self.agv_controller:
            self._draw_agv_status_list(info_y)
    
    def _draw_agv_status_list(self, info_y: int):
        """
        绘制每个AGV的详细状态列表
        
        Args:
            info_y: 信息栏顶部Y坐标
        """
        agv_list_y = info_y + 75
        agv_list_x = 30
        
        # 标题
        title_text = self.font_tiny.render("AGV状态:", True, (80, 80, 80))
        self.screen.blit(title_text, (agv_list_x, agv_list_y))
        
        # 每行显示4个AGV
        agvs_per_row = 4
        agv_width = (WINDOW_WIDTH - 60) // agvs_per_row
        
        for i, (agv_id, agv) in enumerate(self.agv_controller.agvs.items()):
            row = i // agvs_per_row
            col = i % agvs_per_row
            
            x = agv_list_x + col * agv_width + 60
            y = agv_list_y + row * 22
            
            # AGV颜色方块
            color = AGV_COLORS[agv_id % len(AGV_COLORS)]
            pygame.draw.rect(self.screen, color, (x, y + 2, 12, 12))
            
            # 状态文字
            status_str = agv.status.name if hasattr(agv.status, 'name') else str(agv.status)
            status_label = AGV_STATUS_LABELS.get(status_str, status_str)
            
            # 装载标记（使用普通ASCII字符避免方框显示问题）
            load_mark = "[载]" if agv.is_loaded else "[空]"
            
            # 位置信息
            pos_str = f"({agv.position[0]},{agv.position[1]})"
            
            bat_str = f" {agv.battery:.0f}%"
            agv_info = f"AGV{agv_id}:{status_label}{load_mark}{pos_str}{bat_str}"
            info_color = (50, 50, 50)
            info_text = self.font_tiny.render(agv_info, True, info_color)
            self.screen.blit(info_text, (x + 16, y))
    
    def handle_events(self) -> bool:
        """
        处理 Pygame 事件
        
        Returns:
            是否继续运行（False表示用户关闭窗口）
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                
                elif event.key == pygame.K_r:
                    # 重置环境
                    self.env.reset()
                    if self.agv_controller:
                        self.agv_controller.reset()
                    self.logger.info("环境已重置")
                
                elif event.key == pygame.K_SPACE:
                    # 暂停/继续（由外部控制）
                    pass
        
        return True
    
    def close(self):
        """关闭渲染器，释放资源"""
        self.running = False
        pygame.quit()
        self.logger.info("渲染器已关闭")


# ==========================================
# 独立运行测试
# ==========================================

def run_renderer_demo():
    """
    独立运行渲染器演示
    
    创建环境并启动 Pygame 窗口，展示完整的仓库可视化。
    """
    import logging
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("仓库环境可视化演示")
    print("=" * 60)
    print(f"地图尺寸: {MAP_WIDTH}x{MAP_HEIGHT}")
    print(f"窗口尺寸: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    print(f"格子大小: {CELL_SIZE}px")
    print(f"帧率: {FPS}FPS")
    print("=" * 60)
    print("操作说明:")
    print("  ESC - 退出")
    print("  R   - 重置环境")
    print("  SPACE - 暂停/继续")
    print("=" * 60)
    
    # 创建环境
    env = WarehouseEnv()
    env.reset()
    
    # 创建渲染器
    renderer = WarehouseRenderer(env)
    
    # 仿真循环
    paused = False
    running = True
    step_count = 0
    
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                elif event.key == pygame.K_r:
                    env.reset()
                    step_count = 0
                    print("环境已重置")
                
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"{'已暂停' if paused else '继续运行'}")
        
        if not paused:
            # 执行一步环境更新
            env.step()
            step_count += 1
        
        # 渲染当前状态
        renderer.render()
    
    # 关闭渲染器
    renderer.close()
    print(f"\n演示结束，共运行 {step_count} 步")


if __name__ == "__main__":
    run_renderer_demo()
