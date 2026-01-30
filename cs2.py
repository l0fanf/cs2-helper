import os
import threading
import json
import tkinter as tk
import subprocess
import shlex
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
import winreg
import pandas as pd
import numpy as np
import random
import math
import shutil
import requests
import gzip
import time
import webbrowser
import traceback  # 新增：用于获取详细错误堆栈

# --- 依赖检测 ---
HAS_OPENAI = False
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    pass

HAS_PARSER = False
try:
    from demoparser2 import DemoParser
    HAS_PARSER = True
except ImportError:
    pass

HAS_PIL = False
try:
    from PIL import Image, ImageTk, ImageDraw, ImageFilter
    HAS_PIL = True
except ImportError:
    pass

HAS_ZSTD = False
try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    pass

# [Pygame 依赖检测]
HAS_PYGAME = False
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    pass


# --- 商业版配置 ---
APP_NAME = "CS2 FLOW STATE PRO" 
VERSION = "20.2.4 (Auto-Scale Map Fix)"
API_KEY = "sk-420c70471e9b48c78b0d24bece0a5772" 
API_BASE = "https://api.deepseek.com"

# [FACEIT API KEY]
FACEIT_API_KEY = "fbdf43a3-134e-41f9-a722-545bc4d69828" 

# --- 配色方案 ---
C_BG_BASE = "#050505"       
C_SURFACE = "#121212"       
C_SURFACE_HOVER = "#1e1e1e"
C_HIGHLIGHT = "#f0f0f0"     
C_DIM = "#666666"           
C_ACCENT_1 = "#ff453a"      
C_ACCENT_2 = "#32d74b"      
C_ACCENT_3 = "#0a84ff"      
C_FACEIT_ORANGE = "#ff5500" 
C_PRO_GOLD = "#C9A227"
C_GOLD_DIM = "#6f5a1a"
C_GOLD_SOFT = "#e2c66a"
C_GLASS = "#0e0e0e"
C_GLASS_2 = "#141414"
C_BORDER = "#242424"
C_BORDER_GOLD = "#3a2f12"
C_T_SIDE = "#E8B976"   
C_CT_SIDE = "#6BA4FF"  

# Pygame RGB
PYGAME_C_T = (232, 185, 118)
PYGAME_C_CT = (107, 164, 255)

# 字体配置 
FONT_HERO = ("Helvetica", 56, "bold")
FONT_H1 = ("Helvetica", 24, "bold")
FONT_H2 = ("Helvetica", 16, "bold")
FONT_BODY = ("Helvetica", 13)
FONT_MONO = ("Consolas", 11)

# --- 2D Radar 资源 ---
# 说明：旧版 2D demo 坐标体系（MAP_RADAR_DATA + 线性换算）已废弃并删除。
# 新版 2D Radar 从本地 maps/ 读取雷达图片；坐标系将按资源元数据重建。
RADAR_ASSET_DIR = os.path.join(os.getcwd(), "maps")

# --- 道具教学数据 ---
GRENADE_DB = {
    "de_mirage": [
        {"title": "Window Smoke from Spawn", "url": "https://www.youtube.com/results?search_query=cs2+mirage+window+smoke+spawn"},
        {"title": "Top Connector Smoke", "url": "https://www.youtube.com/results?search_query=cs2+mirage+top+connector+smoke"},
        {"title": "Jungle/Stairs Smoke", "url": "https://www.youtube.com/results?search_query=cs2+mirage+jungle+smoke"},
        {"title": "B Market Window Smoke", "url": "https://www.youtube.com/results?search_query=cs2+mirage+market+window+smoke"},
        {"title": "B Short Rush Smoke", "url": "https://www.youtube.com/results?search_query=cs2+mirage+b+short+smoke"},
    ],
    "de_dust2": [
        {"title": "Xbox Smoke from T Spawn", "url": "https://www.youtube.com/results?search_query=cs2+dust2+xbox+smoke"},
        {"title": "Long A Corner Smoke", "url": "https://www.youtube.com/results?search_query=cs2+dust2+long+corner+smoke"},
    ],
    "de_inferno": [], "de_nuke": [], "de_ancient": [] 
}

# --- 多语言字典 ---
LANG = {
    "en": {
        "start": "START ANALYSIS",
        "waiting": "Waiting for demo...",
        "processing": "Processing data stream...",
        "ready": "Analysis Ready",
        "tab_dash": "OVERVIEW",
        "tab_round": "TIMELINE",
        "tab_ai": "NEURAL ENGINE",
        "tab_faceit": "FACEIT HUB",
        "tab_util": "UTILITY LAB",
        "tab_replay": "SMOOTH PLAYER",
        "copy": "Copy Output",
        "ai_gen": "Initialize Neural Analysis",
        "err_parser": "Engine Core Missing",
        "err_path": "Path Not Found",
        "prompt_sys": "You are an elite CS2 Performance Coach.",
        "prompt_user": "Analyze this match.",
        "watch": "WATCH ROUND",
        "login_title": "FLOW STATE PRO LOGIN",
        "login_btn": "ACTIVATE LICENSE",
        "faceit_link": "SYNC MATCHES",
        "faceit_placeholder": "Faceit Nickname",
        "filter_map": "All Maps",
        "filter_res": "All Results",
        "util_sel": "Select Map",
        "load_replay": "LAUNCH ENGINE",
        "replay_loading": "Parsing Demo & Initializing Engine...",
        "download_map": "Downloading Map Radar...",
        "err_pygame": "Pygame Library Missing. Install with: pip install pygame",
        "download_failed": "Download failed. Open Faceit Match Room?",
        "manual_guide": "Manual Mode: Download .dem file from Faceit, put into /downloads/ folder, or select it manually in Timeline tab."
    },
    "zh": {
        "start": "开启心流分析",
        "waiting": "等待数据载入...",
        "processing": "正在解析神经流...",
        "ready": "解析就绪",
        "tab_dash": "全景视图",
        "tab_round": "时间轴",
        "tab_ai": "神经中枢",
        "tab_faceit": "FACEIT 赛事", 
        "tab_util": "道具实验室",
        "tab_replay": "丝滑回放引擎",
        "copy": "复制报告",
        "ai_gen": "激活 AI 深度复盘",
        "err_parser": "核心引擎缺失",
        "err_path": "未检测到 CS2 路径",
        "prompt_sys": "你是一位顶级的 CS2 电竞战术分析师。",
        "prompt_user": "请根据数据生成深度报告。",
        "watch": "回放本局",
        "login_title": "FLOW STATE 专业版登录",
        "login_btn": "激活许可证",
        "faceit_link": "同步比赛数据",
        "faceit_placeholder": "Faceit 昵称",
        "filter_map": "所有地图",
        "filter_res": "所有结果",
        "util_sel": "选择地图",
        "load_replay": "启动引擎",
        "replay_loading": "正在解析数据并初始化引擎...",
        "download_map": "正在下载地图雷达...",
        "err_pygame": "缺少 Pygame 库，请安装：pip install pygame",
        "download_failed": "自动下载失败。是否打开 Faceit 网页手动下载 Demo？",
        "manual_guide": "手动说明：在网页下载 Demo 后，将 .dem 文件放入 downloads 文件夹，或在 [时间轴] 页面手动选择该文件进行分析。"
    }
}

LOCAL_AI_PHRASES = {
    "carry": ["Dominating presence.", "Zone state achieved."],
    "carry_zh": ["统治级表现。", "进入心流状态。"],
    "death_early": ["Over-extension detected.", "Timing error."],
    "death_early_zh": ["走位过于激进。", "时间差计算错误。"],
    "trade": ["Efficient exchange.", "Space created."],
    "trade_zh": ["有效的战术互换。", "为队友拉开了空间。"],
    "normal": ["Standard engagement.", "Rotation valid."],
    "normal_zh": ["常规交火。", "回防路线正确。"]
}

# --- 配置管理器 ---
class ConfigManager:
    def __init__(self):
        self.file = "config.json"
        self.data = self.load()
    def load(self):
        if os.path.exists(self.file):
            try:
                with open(self.file, 'r') as f: return json.load(f)
            except: return {}
        return {}
    def save(self):
        with open(self.file, 'w') as f: json.dump(self.data, f)
    def get(self, key, default=None): return self.data.get(key, default)
    def set(self, key, value):
        self.data[key] = value
        self.save()

CONFIG = ConfigManager()

# --- 数学工具函数 ---
def mapped_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def lerp(a, b, t):
    return a + (b - a) * t

def lerp_angle(a, b, t):
    diff = (b - a + 180) % 360 - 180
    return (a + diff * t) % 360

# --- 新版 2D Radar 播放器 (路线 B: Bbox) ---
class PygameRadarPlayer:
    def __init__(self, map_name, replay_data):
        if not HAS_PYGAME:
            raise RuntimeError("Pygame not installed.")

        self.WINDOW_SIZE = (1024, 1024)
        self.DOT_SIZE = 5
        self.FPS = 60
        self.TICKRATE = 64

        # 从 replay_data 解包
        self.df = replay_data.get("df")
        self.bbox = replay_data.get("bbox")
        self.team_names = replay_data.get("team_names", ["Team A", "Team B"])
        self.rounds = replay_data.get("rounds") or []
        self.kills_df = replay_data.get("kills")

        # rounds: list[dict] {start:int, freeze_end:int|None, end:int|None}
        if not self.rounds:
            # 兜底兼容旧字段
            rs = replay_data.get("round_starts") or []
            fe = replay_data.get("freeze_ends") or []
            self.rounds = []
            for i, st in enumerate(rs):
                self.rounds.append({"start": int(st), "freeze_end": int(fe[i]) if i < len(fe) else None, "end": None})

        self.num_rounds = len(self.rounds)
        self.last_round = self.num_rounds - 1

        pygame.init()
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption(f"2D Replay: {self.team_names[0]} vs {self.team_names[1]}")
        
        # 加载地图
        minimap_path = os.path.join(RADAR_ASSET_DIR, f"{map_name}.png")
        if not os.path.exists(minimap_path):
             minimap_path = os.path.join(RADAR_ASSET_DIR, f"{map_name}_radar.png") # 兼容旧命名

        try:
            self.minimap_original = pygame.image.load(minimap_path)
        except pygame.error:
            self.minimap_original = pygame.Surface(self.WINDOW_SIZE)
            self.minimap_original.fill((30, 30, 30))

        self.minimap = pygame.transform.scale(self.minimap_original, self.WINDOW_SIZE)

        # 地图有效绘制区域（像素裁黑边）
        self._map_valid_rect = None
        try:
            self._map_valid_rect = self._compute_valid_rect(self.minimap_original)
        except Exception:
            self._map_valid_rect = None

        # 兜底留边
        self.map_margin = 12

        # 击杀提示（右上角）
        self.killfeed = []  # list[dict]: {"t":float,"text":str}
        self.killfeed_ttl = 4.0

        # 当前回合标记（用于回合刷新）
        self._last_round_index = 0

        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.tick = 0
        self.round = 0
        self.speed = 1

        # 预计算 tick 列表，用于“找最近 tick”时避免 merge_asof 闪烁
        try:
            self.df["tick"] = self.df["tick"].astype(np.int64)
        except Exception:
            pass
        self._unique_ticks = self.df["tick"].drop_duplicates().sort_values().to_numpy(dtype=np.int64)

        # 预处理击杀事件（用于右上角 killfeed）
        self._kills = []
        if self.kills_df is not None:
            try:
                kdf = self.kills_df.copy()
                if "tick" in kdf.columns:
                    kdf["tick"] = kdf["tick"].astype(np.int64)
                else:
                    kdf = None

                if kdf is not None:
                    for _, r in kdf.iterrows():
                        atk = str(r.get("attacker_name") or "?")
                        vic = str(r.get("user_name") or "?")
                        t = int(r.get("tick") or 0)
                        if t > 0 and atk and vic:
                            self._kills.append((t, f"{atk} killed {vic}"))
                    self._kills.sort(key=lambda x: x[0])
            except Exception:
                self._kills = []

        self._kill_idx = 0

    def draw_hud(self):
        # 纯 pygame HUD（不依赖 pygame_gui）
        # 快捷键：SPACE 暂停/播放，↑/↓ 调速，←/→ 切回合，ESC 退出
        try:
            font = pygame.font.SysFont("Consolas", 18)
        except Exception:
            font = pygame.font.Font(None, 18)

        scr_w, scr_h = self.screen.get_size()
        hud_lines = [
            f"Round: {self.round + 1}/{self.num_rounds}",
            f"Speed: {self.speed}x", 
            f"Tick: {int(self.tick)}",
            "SPACE Pause | ↑/KP+ Faster | ↓/KP- Slower | ←/KP4 Prev | →/KP6 Next | ESC Quit",
        ]

        # 背景条
        pad = 8
        line_h = 22
        box_h = pad * 2 + line_h * len(hud_lines)
        box_w = 520
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(10, scr_h - box_h - 10, box_w, box_h))

        y = scr_h - box_h
        for i, line in enumerate(hud_lines):
            surf = font.render(line, True, (230, 230, 230))
            self.screen.blit(surf, (20, scr_h - box_h - 2 + pad + i * line_h))

    def _is_black_pixel(self, px):
        if len(px) == 4:
            r, g, b, a = px
            if a <= 16:
                return True
        else:
            r, g, b = px
        return (r + g + b) <= 60

    def _snap_to_valid_pixel(self, x_scr, y_scr):
        """如果点落在黑区，则沿向地图中心方向向内寻找最近有效像素，并返回新的屏幕坐标。"""
        if self.minimap_original is None:
            return x_scr, y_scr

        scr_w, scr_h = self.screen.get_size()
        if scr_w <= 0 or scr_h <= 0:
            return x_scr, y_scr

        ix = int(x_scr / scr_w * self.minimap_original.get_width())
        iy = int(y_scr / scr_h * self.minimap_original.get_height())
        iw = self.minimap_original.get_width()
        ih = self.minimap_original.get_height()

        ix = max(0, min(iw - 1, ix))
        iy = max(0, min(ih - 1, iy))

        try:
            px = self.minimap_original.get_at((ix, iy))
        except Exception:
            return x_scr, y_scr

        if not self._is_black_pixel(px):
            return x_scr, y_scr

        cx, cy = iw / 2.0, ih / 2.0
        dx = cx - ix
        dy = cy - iy
        norm = (dx * dx + dy * dy) ** 0.5
        if norm < 1e-6:
            return x_scr, y_scr
        dx /= norm
        dy /= norm

        max_steps = 80
        step = 2.0
        for i in range(1, max_steps + 1):
            tx = int(ix + dx * step * i)
            ty = int(iy + dy * step * i)
            if tx < 0 or ty < 0 or tx >= iw or ty >= ih:
                break
            try:
                tpx = self.minimap_original.get_at((tx, ty))
            except Exception:
                break
            if not self._is_black_pixel(tpx):
                nx = int(tx / iw * scr_w)
                ny = int(ty / ih * scr_h)
                return nx, ny

        return x_scr, y_scr

    def _compute_valid_rect(self, surf):
        """在原始雷达图上自动裁黑边/透明边，返回 pygame.Rect(x,y,w,h)"""
        w, h = surf.get_size()
        try:
            surf = surf.convert_alpha()
        except Exception:
            pass

        def is_valid(px):
            if len(px) == 4:
                r, g, b, a = px
                if a <= 16:
                    return False
            else:
                r, g, b = px
            return (r + g + b) > 60

        left, right = 0, w - 1
        top, bottom = 0, h - 1

        # 扫描 top
        for y in range(h):
            ok = False
            for x in range(w):
                if is_valid(surf.get_at((x, y))):
                    ok = True
                    break
            if ok:
                top = y
                break
        # 扫描 bottom
        for y in range(h - 1, -1, -1):
            ok = False
            for x in range(w):
                if is_valid(surf.get_at((x, y))):
                    ok = True
                    break
            if ok:
                bottom = y
                break
        # 扫描 left
        for x in range(w):
            ok = False
            for y in range(top, bottom + 1):
                if is_valid(surf.get_at((x, y))):
                    ok = True
                    break
            if ok:
                left = x
                break
        # 扫描 right
        for x in range(w - 1, -1, -1):
            ok = False
            for y in range(top, bottom + 1):
                if is_valid(surf.get_at((x, y))):
                    ok = True
                    break
            if ok:
                right = x
                break

        # 兜底
        if right <= left or bottom <= top:
            return pygame.Rect(0, 0, w, h)

        return pygame.Rect(left, top, right - left + 1, bottom - top + 1)

    def coords_to_screen(self, x, y):
        if not self.bbox or self.bbox[2] == self.bbox[0] or self.bbox[3] == self.bbox[1]:
            return -100, -100

        scr_w, scr_h = self.screen.get_size()

        # 有效区域：优先用图片裁剪出的 valid_rect；否则用 margin 兜底
        if self._map_valid_rect is not None:
            vr = self._map_valid_rect
            # 将原图 valid_rect 按当前窗口缩放
            sx = scr_w / max(1, self.minimap_original.get_width())
            sy = scr_h / max(1, self.minimap_original.get_height())
            vx = vr.x * sx
            vy = vr.y * sy
            vw = max(1, vr.width * sx)
            vh = max(1, vr.height * sy)
        else:
            m = int(self.map_margin)
            vx, vy = m, m
            vw = max(1, scr_w - 2 * m)
            vh = max(1, scr_h - 2 * m)

        x_s = (x - self.bbox[0]) / (self.bbox[2] - self.bbox[0]) * vw + vx
        y_s = (y - self.bbox[1]) / (self.bbox[3] - self.bbox[1]) * vh + vy
        return int(x_s), int(scr_h - y_s)  # Y 轴翻转

    def run(self):
        if self.df is None or self.df.empty:
            print("Replay data is empty, cannot run.")
            return

        self.tick = float(self.rounds[0]["start"]) if self.rounds else 0.0

        while self.running:
            time_delta = self.clock.tick(self.FPS) / 1000.0

            if not self.paused:
                self.tick += self.TICKRATE / self.FPS * self.speed
            
            # 自动推进到下一回合（基于 rounds）
            if self.round < self.last_round and self.rounds:
                next_start = float(self.rounds[self.round + 1].get("start", 0))
                if next_start > 0 and self.tick >= next_start:
                    self.round += 1
                    self.tick = float(self.rounds[self.round].get("start", self.tick))

            # 新回合刷新
            if self.round != self._last_round_index:
                self._last_round_index = self.round
                self.killfeed = []
                # 重置击杀索引到当前回合起点附近（粗略定位）
                start_tick = int(self.rounds[self.round].get("start", 0)) if self.rounds and self.round < len(self.rounds) else 0
                self._kill_idx = 0
                while self._kill_idx < len(self._kills) and self._kills[self._kill_idx][0] < start_tick:
                    self._kill_idx += 1
            
            # freeze time（基于 rounds 的 freeze_end）
            if self.rounds and self.round < len(self.rounds):
                fe = self.rounds[self.round].get("freeze_end")
                st = self.rounds[self.round].get("start")
                if fe is not None and st is not None:
                    try:
                        fe_i = float(fe)
                        st_i = float(st)
                        if st_i <= self.tick < fe_i:
                            self.tick = fe_i
                    except Exception:
                        pass

            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.VIDEORESIZE:
                    self.WINDOW_SIZE = (event.w, event.h)
                    self.screen = pygame.display.set_mode(self.WINDOW_SIZE, pygame.RESIZABLE)
                    self.minimap = pygame.transform.scale(self.minimap_original, self.WINDOW_SIZE)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    # 速度控制（主键盘 & 小键盘）
                    elif event.key in (pygame.K_UP, pygame.K_KP_PLUS, pygame.K_EQUALS):
                        self.speed = min(64, self.speed * 2)
                    elif event.key in (pygame.K_DOWN, pygame.K_KP_MINUS, pygame.K_MINUS):
                        self.speed = max(0.25, self.speed / 2)
                    # 回合切换（方向键 & 小键盘）
                    elif event.key in (pygame.K_LEFT, pygame.K_KP4, pygame.K_KP_LEFT):
                        self.round = max(0, self.round - 1)
                        if self.rounds and self.round < len(self.rounds):
                            self.tick = float(self.rounds[self.round].get("start", self.tick))
                    elif event.key in (pygame.K_RIGHT, pygame.K_KP6, pygame.K_KP_RIGHT):
                        self.round = min(self.last_round, self.round + 1)
                        if self.rounds and self.round < len(self.rounds):
                            self.tick = float(self.rounds[self.round].get("start", self.tick))
                

            # 渲染（硬清屏，避免残影）
            self.screen.fill((0, 0, 0))
            self.screen.blit(self.minimap, (0, 0))
            
            # 找到当前 tick 的帧（优先精确 tick，其次找最近 tick）
            current_tick_val = int(self.tick)

            tick_to_render = current_tick_val
            frame_to_render = self.df[self.df["tick"] == tick_to_render]
            if frame_to_render.empty:
                # 兜底：找最近 tick（不使用 merge_asof，避免 dtype/性能问题）
                ticks = self._unique_ticks
                if ticks.size > 0:
                    idx = int(np.searchsorted(ticks, tick_to_render))
                    if idx <= 0:
                        tick_to_render = int(ticks[0])
                    elif idx >= ticks.size:
                        tick_to_render = int(ticks[-1])
                    else:
                        before = int(ticks[idx - 1])
                        after = int(ticks[idx])
                        tick_to_render = before if (tick_to_render - before) <= (after - tick_to_render) else after
                    frame_to_render = self.df[self.df["tick"] == tick_to_render]

            # 绘制玩家点（带 name 标签）
            if not frame_to_render.empty:
                scr_w, scr_h = self.screen.get_size()
                try:
                    name_font = pygame.font.SysFont("Consolas", 12)
                except Exception:
                    name_font = pygame.font.Font(None, 12)

                for _, row in frame_to_render.iterrows():
                    if not row.get('is_alive'):
                        continue
                    x, y = self.coords_to_screen(row["X"], row["Y"])
                    x, y = self._snap_to_valid_pixel(x, y)
                    # clamp：限制到地图有效区域（优先 valid_rect，其次 margin）
                    if self._map_valid_rect is not None:
                        vr = self._map_valid_rect
                        sx = scr_w / max(1, self.minimap_original.get_width())
                        sy = scr_h / max(1, self.minimap_original.get_height())
                        vx = vr.x * sx
                        vy = vr.y * sy
                        vw = max(1, vr.width * sx)
                        vh = max(1, vr.height * sy)
                        if x < vx or x >= (vx + vw) or y < vy or y >= (vy + vh):
                            continue
                    else:
                        m = int(self.map_margin)
                        if x < m or y < m or x >= (scr_w - m) or y >= (scr_h - m):
                            continue

                    color = PYGAME_C_CT if row["team_num"] == 3 else PYGAME_C_T
                    pygame.draw.circle(self.screen, (255,255,255), (x, y), self.DOT_SIZE + 1)
                    pygame.draw.circle(self.screen, color, (x, y), self.DOT_SIZE)

                    # name
                    nm = str(row.get("name") or "")
                    if nm:
                        nm = nm.strip()
                        if len(nm) > 10:
                            nm = nm[:10]
                        surf = name_font.render(nm, True, (255, 255, 255))
                        self.screen.blit(surf, (x + self.DOT_SIZE + 2, y - 6))

            # 处理击杀提示（将当前 tick 前发生的击杀加入队列）
            now_sec = pygame.time.get_ticks() / 1000.0
            while self._kill_idx < len(self._kills) and self._kills[self._kill_idx][0] <= tick_to_render:
                _, txt = self._kills[self._kill_idx]
                self.killfeed.append({"t": now_sec, "text": txt})
                # 最多保留 6 条
                if len(self.killfeed) > 6:
                    self.killfeed = self.killfeed[-6:]
                self._kill_idx += 1

            # 过期清理
            self.killfeed = [k for k in self.killfeed if (now_sec - float(k.get("t", 0))) <= float(self.killfeed_ttl)]

            # 右上角绘制 killfeed
            try:
                kf_font = pygame.font.SysFont("Consolas", 16)
            except Exception:
                kf_font = pygame.font.Font(None, 16)

            scr_w, scr_h = self.screen.get_size()
            y0 = 10
            for k in reversed(self.killfeed):
                surf = kf_font.render(str(k.get("text", "")), True, (255, 255, 255))
                self.screen.blit(surf, (scr_w - surf.get_width() - 12, y0))
                y0 += 18

            # HUD
            self.draw_hud()
            pygame.display.update()

        pygame.quit()

# --- 图形引擎 (背景特效) ---
class AuroraCanvas(tk.Canvas):
    def __init__(self, master, width, height, **kwargs):
        super().__init__(master, width=width, height=height, highlightthickness=0, bg=C_BG_BASE, **kwargs)
        self.width = width
        self.height = height

        self._bg_img = None
        self._bg_id = None

        self.glow_img = None
        self.glow_id = None
        self.glow2_img = None
        self.glow2_id = None

        self._mx = width // 2
        self._my = height // 2
        self._gx = self._mx
        self._gy = self._my

        # 粒子拖尾已移除（避免诡异观感），仅保留双层光晕
        self._particles = []
        self._last_emit_t = 0.0

        if HAS_PIL:
            self.create_resources()
            self.bind("<Configure>", self.on_resize)

    def create_resources(self):
        # 背景：暗金渐变 + 微噪点（克制）
        w = max(1, self.width)
        h = max(1, self.height)
        bg = Image.new('RGBA', (w, h), (5, 5, 6, 255))
        px = bg.load()
        # 线性渐变（左上更亮一点，右下更暗）
        for y in range(h):
            for x in range(w):
                t = (x / max(1, w - 1)) * 0.6 + (y / max(1, h - 1)) * 0.4
                base = int(10 + (1 - t) * 12)
                r = base
                g = base
                b = base + 2
                px[x, y] = (r, g, b, 255)
        # 微噪点
        try:
            import random
            for _ in range(int(w * h * 0.015)):
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
                rr, gg, bb, aa = px[x, y]
                n = random.randint(-8, 8)
                px[x, y] = (max(0, min(255, rr + n)), max(0, min(255, gg + n)), max(0, min(255, bb + n)), aa)
        except Exception:
            pass
        bg = bg.filter(ImageFilter.GaussianBlur(radius=0.6))
        self._bg_img = ImageTk.PhotoImage(bg)
        if self._bg_id is None:
            self._bg_id = self.create_image(0, 0, anchor='nw', image=self._bg_img)
        else:
            self.itemconfig(self._bg_id, image=self._bg_img)

        # 光晕资源（双层：暗金主光 + 冷灰辅光）
        self.glow_img = ImageTk.PhotoImage(self._make_glow(900, (201, 162, 39), 90))
        self.glow2_img = ImageTk.PhotoImage(self._make_glow(700, (120, 130, 150), 55))

        if self.glow_id is None:
            self.glow_id = self.create_image(self.width//2, self.height//2, image=self.glow_img)
        else:
            self.itemconfig(self.glow_id, image=self.glow_img)

        if self.glow2_id is None:
            self.glow2_id = self.create_image(self.width//2, self.height//2, image=self.glow2_img)
        else:
            self.itemconfig(self.glow2_id, image=self.glow2_img)

        self.tag_lower(self._bg_id)
        if self.glow2_id: self.tag_raise(self.glow2_id)
        if self.glow_id: self.tag_raise(self.glow_id)

    def _make_glow(self, size, rgb, max_alpha):
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        cx = size // 2
        max_r = size // 2
        r0, g0, b0 = rgb
        for radius in range(max_r, 0, -6):
            a = int((1 - (radius / max_r))**2 * max_alpha)
            draw.ellipse((cx - radius, cx - radius, cx + radius, cx + radius), fill=(r0, g0, b0, a))
        return image.filter(ImageFilter.GaussianBlur(radius=60))

    def on_resize(self, event):
        self.width = event.width
        self.height = event.height
        self.config(width=self.width, height=self.height)
        if HAS_PIL:
            self.create_resources()

    def update_mouse_position(self, x, y):
        self._mx = x
        self._my = y
        # 平滑插值
        self._gx = int(self._gx + (self._mx - self._gx) * 0.18)
        self._gy = int(self._gy + (self._my - self._gy) * 0.18)
        if self.glow_id:
            self.coords(self.glow_id, self._gx, self._gy)
        if self.glow2_id:
            self.coords(self.glow2_id, self._gx + 30, self._gy + 20)

        # 粒子拖尾已移除

class EndelCard(ctk.CTkFrame):
    def __init__(self, master, r_num, stats, details, lang, watch_callback, **kwargs):
        super().__init__(master, fg_color=C_SURFACE, corner_radius=16, border_width=0, **kwargs)
        self.details = details
        self.is_expanded = False
        self.lang = lang
        self.watch_callback = watch_callback
        self.tick = stats.get('start_tick', 0)
        self.bind("<Button-1>", self.toggle)
        self.bind("<Enter>", lambda e: self.on_hover(True))
        self.bind("<Leave>", lambda e: self.on_hover(False))
        self.grid_columnconfigure(1, weight=1)
        self.pill = ctk.CTkLabel(self, text=f"{r_num}", font=("Helvetica", 12, "bold"), width=32, height=32, corner_radius=16, fg_color="#222", text_color="#fff")
        self.pill.grid(row=0, column=0, padx=15, pady=15)
        self.pill.bind("<Button-1>", self.toggle)
        color = C_HIGHLIGHT
        if "白给" in stats['tag'] or "Rushed" in stats['tag']: color = C_ACCENT_1
        elif "CARRY" in stats['tag']: color = C_PRO_GOLD 
        info_text = f"{stats['tag']}  •  {stats['kills']}K / {stats['deaths']}D"
        self.lbl_info = ctk.CTkLabel(self, text=info_text, font=("Helvetica", 14, "bold"), text_color=color, anchor="w")
        self.lbl_info.grid(row=0, column=1, sticky="w")
        self.lbl_info.bind("<Button-1>", self.toggle)
        self.btn_watch = ctk.CTkButton(self, text="▶", width=30, height=30, corner_radius=15, fg_color="#222", hover_color=C_ACCENT_3, text_color="white", font=("Arial", 14), command=self.on_watch_click)
        self.btn_watch.grid(row=0, column=2, padx=(0, 10))
        self.lbl_insight = ctk.CTkLabel(self, text=stats['ai_comment'], font=("Helvetica", 12), text_color=C_DIM, anchor="e")
        self.lbl_insight.grid(row=0, column=3, padx=20)
        self.lbl_insight.bind("<Button-1>", self.toggle)
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        if details:
            for line in details:
                row = ctk.CTkFrame(self.body, fg_color="transparent", height=20)
                row.pack(fill="x", padx=20, pady=2)
                ctk.CTkLabel(row, text=line['time'], font=FONT_MONO, text_color=C_DIM, width=50).pack(side="left")
                ctk.CTkLabel(row, text=line['msg'], font=("Helvetica", 12), text_color="#ccc" if line['type'] == 'kill' else C_ACCENT_1, anchor="w").pack(side="left", fill="x")
            if stats['advice']:
                ctk.CTkFrame(self.body, height=1, fg_color="#333").pack(fill="x", padx=20, pady=10)
                ctk.CTkLabel(self.body, text=f"⚡ COACH: {stats['advice']}", font=("Helvetica", 12, "italic"), text_color=C_ACCENT_3, anchor="w").pack(fill="x", padx=25, pady=(0, 10))
    def on_watch_click(self):
        if self.watch_callback: self.watch_callback(self.tick)
    def on_hover(self, is_hover):
        if not self.is_expanded: self.configure(fg_color=C_SURFACE_HOVER if is_hover else C_SURFACE)
    def toggle(self, event=None):
        if self.is_expanded:
            self.body.grid_forget()
            self.configure(fg_color=C_SURFACE)
            self.lbl_info.configure(text_color=C_HIGHLIGHT if "CARRY" not in self.lbl_info.cget("text") else C_PRO_GOLD)
        else:
            self.body.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 15))
            self.configure(fg_color="#1a1a1a") 
        self.is_expanded = not self.is_expanded

class FaceitMatchCard(ctk.CTkFrame):
    def __init__(self, master, match_data, analyze_callback, **kwargs):
        super().__init__(master, fg_color=C_SURFACE, corner_radius=12, **kwargs)
        self.match_data = match_data
        self.grid_columnconfigure(2, weight=1)
        res_color = C_ACCENT_2 if match_data['result'] == 'WIN' else C_ACCENT_1
        ctk.CTkFrame(self, width=4, height=40, fg_color=res_color).grid(row=0, column=0, rowspan=2, padx=(10, 5), pady=10)
        ctk.CTkLabel(self, text=match_data['map'], font=("Helvetica", 14, "bold"), text_color="white").grid(row=0, column=1, sticky="w", padx=5)
        ctk.CTkLabel(self, text=match_data['score'], font=("Consolas", 14, "bold"), text_color=res_color).grid(row=1, column=1, sticky="w", padx=5)
        ctk.CTkLabel(self, text=f"K/D: {match_data['kd']}  •  HS: {match_data['hs']}%", font=("Helvetica", 12), text_color=C_DIM).grid(row=0, column=2, rowspan=2, sticky="e", padx=20)
        has_demo = bool(match_data.get('demo_url'))
        self.btn_analyze = ctk.CTkButton(self, text="DOWNLOAD" if has_demo else "EXPIRED", width=100, height=28, fg_color="#222", hover_color=C_FACEIT_ORANGE if has_demo else "#333", state="normal" if has_demo else "disabled", font=("Helvetica", 11, "bold"), command=lambda: analyze_callback(match_data))
        self.btn_analyze.grid(row=0, column=3, rowspan=2, padx=15)

class LoginWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Login")
        self.geometry("400x500")
        self.configure(fg_color=C_BG_BASE)
        ctk.CTkLabel(self, text="FLOW STATE", font=FONT_H1, text_color=C_HIGHLIGHT).pack(pady=(80, 10))
        ctk.CTkLabel(self, text="PRO EDITION", font=("Helvetica", 12, "bold"), text_color=C_PRO_GOLD).pack(pady=(0, 40))
        self.entry_key = ctk.CTkEntry(self, placeholder_text="License Key (Try: DEMO)", width=250, height=40, border_width=0, fg_color=C_SURFACE)
        self.entry_key.pack(pady=10)
        ctk.CTkButton(self, text="ACTIVATE", width=250, height=45, fg_color=C_HIGHLIGHT, text_color="black", font=("Helvetica", 13, "bold"), command=self.verify).pack(pady=20)
        self.verified = False
    def verify(self):
        key = self.entry_key.get().strip().upper()
        if key == "DEMO" or key.startswith("PRO-"):
            self.verified = True
            self.destroy()
        else: messagebox.showerror("Error", "Invalid License Key.\nPlease subscribe for $5/mo.")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.running = True 
        self.lang = "en"
        saved_path = CONFIG.get("cs2_path")
        self.cs2_path = saved_path if saved_path else self.find_cs2_auto()
        if self.cs2_path and self.cs2_path != saved_path: CONFIG.set("cs2_path", self.cs2_path)
        self.current_demo_path = None
        self.match_data = None
        self.cached_matches = [] 
        self.replay_map_name = "de_mirage"
        self.replay_map_file = None
        self.title(APP_NAME)
        self.geometry("1280x850")
        ctk.set_appearance_mode("Dark")
        self.configure(fg_color=C_BG_BASE)
        self.anim_t = 0
        self.anim_running = True
        self.setup_background() 
        self.build_ui()          
        if not self.cs2_path: self.notify(LANG[self.lang]["err_path"], C_ACCENT_1)
        if not HAS_PYGAME: self.notify(LANG[self.lang]["err_pygame"], C_ACCENT_1)
        ctk.CTkLabel(self.nav, text="PRO", font=("Helvetica", 10, "bold"), text_color="black", fg_color=C_PRO_GOLD, corner_radius=4).place(x=160, y=55)
        self.bind("<Motion>", self.on_mouse_move)
        self.animate_loop()

    def destroy(self):
        self.running = False
        super().destroy()
    def setup_background(self):
        if HAS_PIL:
            self.bg_canvas = AuroraCanvas(self, width=1280, height=850)
            self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)
            tk.Misc.lower(self.bg_canvas)
    def on_mouse_move(self, event):
        if HAS_PIL and hasattr(self, 'bg_canvas'): self.bg_canvas.update_mouse_position(event.x, event.y)
    def build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.nav = ctk.CTkFrame(self, width=220, fg_color=C_SURFACE, corner_radius=24, border_width=0)
        self.nav.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
        self.nav.grid_propagate(False)
        ctk.CTkLabel(self.nav, text="FLOW\nSTATE", font=FONT_H1, text_color=C_HIGHLIGHT, justify="left", anchor="w").pack(fill="x", padx=30, pady=(40, 40))
        self.btn_tabs = {}
        self.create_nav_btn("tab_dash", LANG[self.lang]["tab_dash"], True)
        self.create_nav_btn("tab_faceit", LANG[self.lang]["tab_faceit"])
        self.create_nav_btn("tab_round", LANG[self.lang]["tab_round"])
        self.create_nav_btn("tab_replay", LANG[self.lang]["tab_replay"]) 
        self.create_nav_btn("tab_util", LANG[self.lang]["tab_util"])     
        self.create_nav_btn("tab_ai", LANG[self.lang]["tab_ai"])
        bottom_frame = ctk.CTkFrame(self.nav, fg_color="transparent")
        bottom_frame.pack(side="bottom", fill="x", padx=20, pady=30)
        self.btn_lang = ctk.CTkButton(bottom_frame, text="CN / EN", width=80, height=32, fg_color="#222", hover_color="#333", corner_radius=16, font=("Helvetica", 10, "bold"), command=self.toggle_lang)
        self.btn_lang.pack(anchor="w", pady=10)
        self.entry_id = ctk.CTkEntry(bottom_frame, placeholder_text="STEAM ID", fg_color="#0a0a0a", border_width=0, text_color="#888", height=35, corner_radius=8)
        self.entry_id.pack(fill="x", pady=(0, 5))
        saved_id = CONFIG.get("steam_id")
        if saved_id: self.entry_id.insert(0, saved_id)
        ctk.CTkButton(bottom_frame, text="AUTO DETECT ID", height=20, fg_color="transparent", text_color="#444", font=("Helvetica", 10), hover_color=C_SURFACE, anchor="w", command=self.auto_id).pack(fill="x")
        self.content = ctk.CTkFrame(self, fg_color=C_SURFACE, corner_radius=24, border_width=0)
        self.content.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(1, weight=1)
        header_frame = ctk.CTkFrame(self.content, fg_color="transparent", height=100)
        header_frame.grid(row=0, column=0, sticky="ew", padx=40, pady=(40, 20))
        self.lbl_header = ctk.CTkLabel(header_frame, text="Ready.", font=FONT_HERO, text_color=C_DIM, anchor="w")
        self.lbl_header.pack(side="left")
        self.pages_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self.pages_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=0)
        self.pages = {
            "tab_dash": ctk.CTkFrame(self.pages_frame, fg_color="transparent"),
            "tab_faceit": ctk.CTkFrame(self.pages_frame, fg_color="transparent"),
            "tab_round": ctk.CTkScrollableFrame(self.pages_frame, fg_color="transparent", scrollbar_button_color="#222", scrollbar_button_hover_color="#333"),
            "tab_replay": ctk.CTkFrame(self.pages_frame, fg_color="transparent"),
            "tab_util": ctk.CTkFrame(self.pages_frame, fg_color="transparent"),
            "tab_ai": ctk.CTkFrame(self.pages_frame, fg_color="transparent")
        }
        self.build_dashboard(self.pages["tab_dash"])
        self.build_faceit_page(self.pages["tab_faceit"])
        self.build_replay_page(self.pages["tab_replay"])
        self.build_util_page(self.pages["tab_util"])
        self.build_ai_page(self.pages["tab_ai"])
        self.show_page("tab_dash")

    def create_nav_btn(self, key, text, active=False):
        btn_frame = ctk.CTkFrame(self.nav, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=4)
        ind = ctk.CTkFrame(btn_frame, width=4, height=24, corner_radius=2, fg_color=C_HIGHLIGHT if active else "transparent")
        ind.pack(side="left", padx=(0, 10))
        btn = ctk.CTkButton(btn_frame, text=text, fg_color="transparent", text_color=C_HIGHLIGHT if active else C_DIM, anchor="w", font=("Helvetica", 13, "bold" if active else "normal"), hover_color=C_SURFACE_HOVER, height=40, command=lambda: self.show_page(key))
        btn.pack(side="left", fill="x", expand=True)
        self.btn_tabs[key] = {"btn": btn, "ind": ind}

    def show_page(self, key):
        for p in self.pages.values(): p.pack_forget()
        for k, v in self.btn_tabs.items(): 
            v["btn"].configure(text_color=C_DIM, font=("Helvetica", 13, "normal"))
            v["ind"].configure(fg_color="transparent")
        self.pages[key].pack(fill="both", expand=True, padx=20, pady=20)
        self.btn_tabs[key]["btn"].configure(text_color=C_HIGHLIGHT, font=("Helvetica", 13, "bold"))
        self.btn_tabs[key]["ind"].configure(fg_color=C_HIGHLIGHT)
        self.lbl_header.configure(text={"tab_dash": "Overview", "tab_round": "Timeline", "tab_ai": "Neural Engine", "tab_faceit": "Faceit Hub", "tab_util": "Utility Lab", "tab_replay": "Smooth Replay"}.get(key, ""))

    def build_util_page(self, parent):
        ctrl = ctk.CTkFrame(parent, fg_color="transparent")
        ctrl.pack(fill="x", pady=(0, 20))
        self.combo_util_map = ctk.CTkComboBox(ctrl, values=list(GRENADE_DB.keys()), width=200, command=self.load_util_list)
        self.combo_util_map.set("de_mirage")
        self.combo_util_map.pack(side="left", padx=10)
        self.util_list_frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self.util_list_frame.pack(fill="both", expand=True)
        self.load_util_list("de_mirage")
    def load_util_list(self, map_name):
        for w in self.util_list_frame.winfo_children(): w.destroy()
        for item in GRENADE_DB.get(map_name, []):
            f = ctk.CTkFrame(self.util_list_frame, fg_color=C_SURFACE, corner_radius=12)
            f.pack(fill="x", pady=5, padx=5)
            ctk.CTkLabel(f, text="▶", font=("Arial", 16), text_color=C_FACEIT_ORANGE).pack(side="left", padx=15, pady=15)
            ctk.CTkLabel(f, text=item["title"], font=("Helvetica", 14, "bold"), text_color="white").pack(side="left")
            ctk.CTkButton(f, text="WATCH", width=80, fg_color="#222", hover_color="#333", command=lambda u=item["url"]: webbrowser.open(u)).pack(side="right", padx=15)

    def build_replay_page(self, parent):
        center = ctk.CTkFrame(parent, fg_color="transparent")
        center.pack(expand=True)
        ctk.CTkLabel(center, text="SMOOTH REPLAY ENGINE", font=("Helvetica", 24, "bold"), text_color="white").pack(pady=20)
        ctk.CTkLabel(center, text="Powered by Pygame & Auto-Scaling Converter", font=("Helvetica", 14), text_color=C_DIM).pack(pady=(0, 40))
        self.btn_launch_replay = ctk.CTkButton(center, text=LANG[self.lang]["load_replay"], font=("Helvetica", 16, "bold"), height=60, width=250, fg_color=C_FACEIT_ORANGE, hover_color="#ff7700", command=lambda: self.run_thread(self.load_replay_data))
        self.btn_launch_replay.pack()
        self.lbl_replay_status = ctk.CTkLabel(center, text="", text_color=C_DIM)
        self.lbl_replay_status.pack(pady=20)


    def load_replay_data(self):
        if not HAS_PYGAME: 
            return self.notify(LANG[self.lang]["err_pygame"], C_ACCENT_1)
        if not HAS_PARSER:
            return self.notify(LANG[self.lang]["err_parser"], C_ACCENT_1)
        
        if not self.current_demo_path: 
            self.current_demo_path = filedialog.askopenfilename(filetypes=[("CS2 Demo", "*.dem")])
            if not self.current_demo_path: 
                return self.notify("No Demo Selected", C_ACCENT_1)
        
        self.btn_launch_replay.configure(state="disabled", text="LOADING...")
        self.notify(LANG[self.lang]["replay_loading"], C_HIGHLIGHT)
        self.lbl_replay_status.configure(text="Initializing parser...")
        self.update_idletasks()
        
        try:
            # 解析demo文件
            parser = DemoParser(self.current_demo_path)
            header = parser.parse_header()
            map_name = header.get("map_name", "de_mirage")
            if not map_name or map_name == "":
                map_name = "de_mirage"
            self.replay_map_name = map_name
            
            # 路线B：不再下载雷达图，直接使用本地 maps/ 中的图片
            self.replay_map_file = None
            
            # 解析 tick 数据
            self.lbl_replay_status.configure(text="Parsing ticks...")
            self.update_idletasks()

            wanted_props = [
                "tick", "X", "Y", "team_num", "player_name", "health", "steamid",
                "m_angEyeAngles_y", "view_x",
            ]
            df = parser.parse_ticks(wanted_props)

            if df.empty:
                raise ValueError("No player data found in demo")

            # 过滤掉死亡的玩家，并降采样（每2帧取1帧）
            # 路线B需要“每个tick所有玩家”来避免缺人/闪烁：不要过滤死亡、不要降采样
            df = df.copy()

            # 解析投掷物/道具 & 掉落物（用于 2D 展示）
            self.lbl_replay_status.configure(text="Parsing grenades...")
            self.update_idletasks()
            try:
                grenades_df = parser.parse_grenades(extra=["tick", "X", "Y"])  # extra 字段若不支持会抛异常
            except TypeError:
                grenades_df = parser.parse_grenades()
            except Exception:
                grenades_df = None

            self.lbl_replay_status.configure(text="Parsing item drops...")
            self.update_idletasks()
            try:
                item_drops_df = parser.parse_item_drops()
            except Exception:
                item_drops_df = None

            # 解析常见事件：击杀、炸弹（不同 demo/版本事件名可能不一致，失败则忽略）
            self.lbl_replay_status.configure(text="Parsing events...")
            self.update_idletasks()
            def safe_parse_event(name, player=None, other=None):
                try:
                    return parser.parse_event(name, player=player, other=other)
                except Exception:
                    return None

            death_df = safe_parse_event(
                "player_death",
                player=["attacker_steamid", "user_steamid", "attacker_name", "user_name"],
                other=["tick", "headshot", "weapon"],
            )

            bomb_planted_df = safe_parse_event("bomb_planted", player=["user_steamid", "user_name"], other=["tick"])
            bomb_defused_df = safe_parse_event("bomb_defused", player=["user_steamid", "user_name"], other=["tick"])
            bomb_exploded_df = safe_parse_event("bomb_exploded", other=["tick"])

            # 路线B：构建用于 2D Radar 的 dataframe（按 tick 渲染）
            # 保留 tick、X、Y、team_num、player_name、health（用于 is_alive）
            replay_df = df[["tick", "X", "Y", "team_num", "player_name", "health"]].copy()
            replay_df.rename(columns={"player_name": "name"}, inplace=True)
            replay_df["is_alive"] = (replay_df["health"] > 0).astype(int)
            replay_df.sort_values("tick", inplace=True)

            # bbox（路线B增强）：使用分位数去离群，避免 nuke 等地图异常点拉爆缩放
            q_low = 0.01
            q_high = 0.99
            try:
                xs = replay_df["X"].astype(float)
                ys = replay_df["Y"].astype(float)
                bbox = (
                    float(xs.quantile(q_low)),
                    float(ys.quantile(q_low)),
                    float(xs.quantile(q_high)),
                    float(ys.quantile(q_high)),
                )
            except Exception:
                bbox = (
                    float(replay_df["X"].min()),
                    float(replay_df["Y"].min()),
                    float(replay_df["X"].max()),
                    float(replay_df["Y"].max()),
                )

            # Faceit 回合结构：保留刀局/选边，跳过 warmup
            # 以 match_start_tick 作为 cut（round_announce_match_start 最早 tick），round_freeze_end 用作 round start
            try:
                ms = parser.parse_event("round_announce_match_start")
                cut_tick = int(ms["tick"].min()) if ms is not None and not ms.empty and "tick" in ms.columns else int(replay_df["tick"].min())
            except Exception:
                cut_tick = int(replay_df["tick"].min())

            rounds = []
            try:
                fe = parser.parse_event("round_freeze_end")
                if fe is not None and not fe.empty and "tick" in fe.columns:
                    starts = [int(t) for t in fe["tick"].tolist() if int(t) >= cut_tick]
                else:
                    starts = []
            except Exception:
                starts = []

            try:
                re = parser.parse_event("round_end")
                if re is not None and not re.empty and "tick" in re.columns:
                    ends = [int(t) for t in re["tick"].tolist() if int(t) >= cut_tick]
                else:
                    ends = []
            except Exception:
                ends = []

            if not starts:
                starts = [int(replay_df["tick"].min())]

            # end 与 start 对齐：end 通常比 start 少 1（最后一局可能缺），用 None/下一局 start 兜底
            for i, st in enumerate(starts):
                fe_tick = st
                end_tick = ends[i] if i < len(ends) else None
                rounds.append({"start": int(st), "freeze_end": int(fe_tick), "end": int(end_tick) if end_tick is not None else None})

            # 同时保留旧字段（兼容）
            round_start_ticks = [r["start"] for r in rounds]
            freeze_end_ticks = [r["freeze_end"] for r in rounds]

            # 队伍名称（尽量从 demo header 或 tick 数据推断；拿不到就用默认）
            team_names = ["Team A", "Team B"]

            self.replay_radar_data = {
                "df": replay_df,
                "bbox": bbox,
                "team_names": team_names,
                "rounds": rounds,
                "round_starts": round_start_ticks,
                "freeze_ends": freeze_end_ticks,
                "kills": death_df,
            }

            self.lbl_replay_status.configure(text="Ready")
            self.notify("2D Radar Ready. Launching...", C_ACCENT_2)
            self.update_idletasks()

            threading.Thread(target=self.start_pygame_window, daemon=True).start()
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.notify(error_msg, C_ACCENT_1)
            self.lbl_replay_status.configure(text=error_msg)
            import traceback
            print(f"Replay loading error: {traceback.format_exc()}")
        finally: 
            self.btn_launch_replay.configure(state="normal", text=LANG[self.lang]["load_replay"])

    def start_pygame_window(self):
        """在新线程中启动新版 2D Radar 窗口（路线B：bbox）"""
        try:
            if not HAS_PYGAME:
                raise RuntimeError(LANG[self.lang]["err_pygame"])
            if not hasattr(self, 'replay_radar_data') or not self.replay_radar_data:
                raise RuntimeError("Replay radar data missing")

            player = PygameRadarPlayer(self.replay_map_name, self.replay_radar_data)
            player.run()
        except Exception as e:
            error_msg = f"Pygame window error: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            self.after(0, lambda: self.notify(error_msg, C_ACCENT_1))

    def build_dashboard(self, parent):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(expand=True, fill="both")
        self.btn_start = ctk.CTkButton(container, text=LANG[self.lang]["start"], font=("Helvetica", 18, "bold"), height=80, width=240, corner_radius=40, fg_color=C_HIGHLIGHT, text_color="#000", hover_color="#ddd", command=lambda: self.run_thread(self.analyze_core))
        self.btn_start.pack(pady=(60, 60))
        grid = ctk.CTkFrame(container, fg_color="transparent")
        grid.pack(fill="x", padx=20)
        grid.grid_columnconfigure((0,1,2), weight=1)
        self.lbl_kd = self.create_stat(grid, "K/D RATIO", "0.00", 0)
        self.lbl_hs = self.create_stat(grid, "HEADSHOT %", "0%", 1)
        self.lbl_rating = self.create_stat(grid, "IMPACT RATING", "0.00", 2)
    def create_stat(self, parent, title, val, col):
        f = ctk.CTkFrame(parent, fg_color="#181818", corner_radius=24, height=150)
        f.grid(row=0, column=col, padx=10, sticky="ew")
        f.grid_propagate(False)
        ctk.CTkLabel(f, text=title, font=("Helvetica", 11, "bold"), text_color="#555").pack(pady=(30, 10))
        lbl = ctk.CTkLabel(f, text=val, font=("Helvetica", 42, "bold"), text_color=C_HIGHLIGHT)
        lbl.pack(pady=(0, 20))
        return lbl
    def build_faceit_page(self, parent):
        inp = ctk.CTkFrame(parent, fg_color="transparent")
        inp.pack(fill="x", pady=20)
        self.entry_faceit = ctk.CTkEntry(inp, placeholder_text=LANG[self.lang]["faceit_placeholder"], width=300, height=40, font=("Helvetica", 13))
        self.entry_faceit.pack(side="left", padx=(0, 10))
        self.btn_faceit_link = ctk.CTkButton(inp, text=LANG[self.lang]["faceit_link"], height=40, fg_color=C_FACEIT_ORANGE, hover_color="#e64d00", font=("Helvetica", 12, "bold"), command=lambda: self.run_thread(self.link_faceit_account))
        self.btn_faceit_link.pack(side="left")
        filt = ctk.CTkFrame(parent, fg_color="transparent")
        filt.pack(fill="x", pady=(0, 15))
        self.combo_map = ctk.CTkComboBox(filt, values=["All Maps", "de_mirage", "de_inferno", "de_ancient", "de_nuke", "de_anubis", "de_vertigo", "de_dust2"], width=150, command=self.apply_filters)
        self.combo_map.set("All Maps")
        self.combo_map.pack(side="left", padx=(0, 10))
        self.combo_res = ctk.CTkComboBox(filt, values=["All Results", "WIN", "LOSS"], width=150, command=self.apply_filters)
        self.combo_res.set("All Results")
        self.combo_res.pack(side="left")
        self.faceit_list = ctk.CTkScrollableFrame(parent, fg_color="transparent", scrollbar_button_color="#222")
        self.faceit_list.pack(fill="both", expand=True)
    def build_ai_page(self, parent):
        btns = ctk.CTkFrame(parent, fg_color="transparent")
        btns.pack(fill="x", pady=(0, 20))
        self.btn_ai_gen = ctk.CTkButton(btns, text=LANG[self.lang]["ai_gen"], height=45, corner_radius=22, fg_color="#222", hover_color="#333", border_width=0, font=("Helvetica", 13, "bold"), command=lambda: self.run_thread(self.call_ai))
        self.btn_ai_gen.pack(side="left")
        ctk.CTkButton(btns, text="COPY", width=80, height=45, corner_radius=22, fg_color="#111", hover_color="#222", font=("Helvetica", 11), command=self.copy_ai).pack(side="right")
        self.txt_ai = ctk.CTkTextbox(parent, font=("Georgia", 15), fg_color="#0a0a0a", text_color="#ccc", corner_radius=16, border_width=0, padx=20, pady=20)
        self.txt_ai.pack(fill="both", expand=True)

    def animate_loop(self):
        if not getattr(self, 'running', False) or not self.winfo_exists(): return
        self.anim_t += 0.03
        val = int(100 + (255 - 100) * ((math.sin(self.anim_t) + 1) / 2)) 
        try:
            if "..." in self.lbl_header.cget("text") or "Ready" in self.lbl_header.cget("text"): self.lbl_header.configure(text_color=f"#{val:02x}{val:02x}{val:02x}")
        except: pass
        if self.running: self.after(50, self.animate_loop)

    def cascade_load_rounds(self, cards, index=0):
        if not self.running: return
        if index >= len(cards):
            self.notify(LANG[self.lang]["ready"], C_ACCENT_2)
            return
        card, opts = cards[index]
        card.pack(**opts)
        self.after(20, lambda: self.cascade_load_rounds(cards, index + 1))

    def notify(self, msg, color=C_DIM):
        try: self.lbl_header.configure(text=msg, text_color=color)
        except: pass
    def find_cs2_auto(self):
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam")
            path, _ = winreg.QueryValueEx(key, "InstallPath")
            cs2 = os.path.join(path, "steamapps", "common", "Counter-Strike Global Offensive", "game", "bin", "win64", "cs2.exe")
            if os.path.exists(cs2): return cs2
            alt = os.path.join(path, "steamapps", "common", "Counter-Strike Global Offensive")
            if os.path.exists(alt): return alt
        except: return None
    def auto_id(self):
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam\ActiveProcess")
            val, _ = winreg.QueryValueEx(key, "ActiveUser")
            if val:
                sid = str(val + 76561197960265728)
                self.entry_id.delete(0, "end"); self.entry_id.insert(0, sid)
                self.notify("ID Detected", C_ACCENT_2); CONFIG.set("steam_id", sid)
        except: self.notify("Steam Not Found", C_ACCENT_1)
    def toggle_lang(self):
        self.lang = "zh" if self.lang == "en" else "en"
        d = LANG[self.lang]
        for k, v in self.btn_tabs.items(): v["btn"].configure(text=d[k])
        self.btn_start.configure(text=d["start"]); self.btn_ai_gen.configure(text=d["ai_gen"]); self.btn_faceit_link.configure(text=d["faceit_link"]); self.entry_faceit.configure(placeholder_text=d["faceit_placeholder"]); self.combo_map.set(d["filter_map"]); self.combo_res.set(d["filter_res"]); self.combo_util_map.set(d["util_sel"]); self.btn_launch_replay.configure(text=d["load_replay"])
        self.notify(d["ready"] if self.match_data else "", C_DIM)
    def copy_ai(self):
        self.clipboard_clear(); self.clipboard_append(self.txt_ai.get("0.0", "end"))
        self.notify("Copied to Clipboard", C_ACCENT_2)
    def run_thread(self, func): threading.Thread(target=func).start()

    def watch_round(self, tick):
        if not self.current_demo_path: return self.notify("No Demo Loaded", C_ACCENT_1)
        if not self.cs2_path or not os.path.exists(self.cs2_path): return self.notify("CS2 Path Not Found", C_ACCENT_1)
        try:
            root = os.path.dirname(os.path.dirname(os.path.dirname(self.cs2_path))) 
            cdir = os.path.join(root, "csgo")
            if not os.path.exists(cdir) and "Counter-Strike Global Offensive" in self.cs2_path: cdir = os.path.join(self.cs2_path.split("Counter-Strike Global Offensive")[0] + "Counter-Strike Global Offensive", "game", "csgo")
            if not os.path.exists(cdir): return self.notify("Cannot find /csgo/ folder", C_ACCENT_1)
            tname, tpath, count = "flowstate.dem", os.path.join(cdir, "flowstate.dem"), 0
            while os.path.exists(tpath):
                count += 1; tname = f"flowstate_{count}.dem"; tpath = os.path.join(cdir, tname)
            self.notify(f"Copying to {tname}...", C_ACCENT_3); self.update_idletasks()
            shutil.copy2(self.current_demo_path, tpath)
            cstr = f"playdemo {tname.replace('.dem', '')}; demo_goto {max(0, tick - 640)}"
            self.clipboard_clear(); self.clipboard_append(cstr); self.update()
            messagebox.showinfo("Ready", f"DEMO COPIED: {tname}\n\n1. Open Console (~)\n2. Paste Command (Ctrl+V)")
            subprocess.Popen([self.cs2_path, "-game", "csgo"])
        except Exception as e: self.notify(f"Copy Error: {e}", C_ACCENT_1)

    def link_faceit_account(self):
        name = self.entry_faceit.get().strip()
        if not name: return self.notify("Nickname Required", C_ACCENT_1)
        self.notify(f"Searching Faceit: {name}...", C_HIGHLIGHT)
        if not FACEIT_API_KEY: return self.notify("API Key Missing", C_ACCENT_1)
        h = {"Authorization": f"Bearer {FACEIT_API_KEY}"}
        try:
            r = requests.get(f"https://open.faceit.com/data/v4/players?nickname={name}", headers=h)
            if r.status_code != 200: return self.notify("Player Not Found", C_ACCENT_1)
            pid = r.json()['player_id']
            self.notify("Fetching Matches...", C_ACCENT_3)
            matches = requests.get(f"https://open.faceit.com/data/v4/players/{pid}/history?game=cs2&offset=0&limit=20", headers=h).json().get('items', [])
            self.cached_matches = []
            for i, m in enumerate(matches):
                self.notify(f"Parsing {i+1}/{len(matches)}...", C_HIGHLIGHT); self.update_idletasks()
                mn, sc, res, kd, hs, durl = "Unknown", "- / -", "UNKNOWN", "-", "-", ""
                try:
                    rs = requests.get(f"https://open.faceit.com/data/v4/matches/{m['match_id']}/stats", headers=h)
                    if rs.status_code == 200:
                        rd = rs.json()['rounds'][0]
                        mn, sc = rd['round_stats']['Map'], rd['round_stats']['Score']
                        for t in rd['teams']:
                            for p in t['players']:
                                if p['player_id'] == pid:
                                    kd, hs, res = p['player_stats'].get('K/D Ratio', '0.0'), p['player_stats'].get('Headshots %', '0'), "WIN" if str(p['player_stats'].get('Result', '0')) == "1" else "LOSS"
                                    break
                except: pass
                # history API 不再返回 demo_url，需要单独请求 matches API
                try:
                    match_details = requests.get(f"https://open.faceit.com/data/v4/matches/{m['match_id']}", headers=h).json()
                    if 'demo_url' in match_details and match_details['demo_url']:
                        durl = match_details['demo_url'][0]
                except Exception:
                    durl = ""
                self.cached_matches.append({"id": m['match_id'], "map": mn, "score": sc, "result": res, "kd": str(kd), "hs": str(hs), "demo_url": durl})
            self.apply_filters(); self.notify(f"Synced {len(self.cached_matches)} Matches", C_ACCENT_2)
        except Exception as e: self.notify(f"API Error: {e}", C_ACCENT_1)

    def apply_filters(self, event=None):
        sm, sr = self.combo_map.get(), self.combo_res.get()
        for w in self.faceit_list.winfo_children(): w.destroy()
        for m in self.cached_matches:
            if sm not in ["All Maps", "所有地图"] and m['map'] != sm: continue
            if sr not in ["All Results", "所有结果"] and m['result'] != sr: continue
            FaceitMatchCard(self.faceit_list, m, self.download_and_analyze_faceit).pack(fill="x", padx=10, pady=5)

    def download_and_analyze_faceit(self, match_data):
        rid = match_data['id']
        cid = rid.split('-')[-1] if '-' in rid else rid
        ddir = os.path.join(os.getcwd(), "downloads")
        if not os.path.exists(ddir): os.makedirs(ddir)
        dpath = os.path.join(ddir, f"faceit_{cid}.dem")
        if os.path.exists(dpath) and os.path.getsize(dpath) > 1024*1024:
            self.current_demo_path = dpath; self.analyze_core(override_path=dpath); self.show_page("tab_round")
            return
        self.notify("Downloading...", C_HIGHLIGHT)
        success = False
        for s in [{"url": f"https://api.faceit.com/match/v2/match/{cid}/demo", "h": {"User-Agent": "Mozilla/5.0"}}, {"url": match_data.get('demo_url'), "h": {}}]:
            if not s['url']: continue
            try:
                with requests.get(s['url'], stream=True, verify=False, timeout=30) as r:
                    if r.status_code == 404: continue
                    r.raise_for_status()
                    tarc = os.path.join(ddir, f"temp_{cid}.archive")
                    with open(tarc, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
                    if HAS_ZSTD:
                        with open(tarc, 'rb') as ifh, open(dpath, 'wb') as ofh: zstd.ZstdDecompressor().copy_stream(ifh, ofh)
                    else:
                        with gzip.open(tarc, 'rb') as fin, open(dpath, 'wb') as fout: shutil.copyfileobj(fin, fout)
                    try: os.remove(tarc)
                    except: pass
                    success = True; break 
            except: continue
        if success:
            self.current_demo_path = dpath; self.analyze_core(override_path=dpath); self.show_page("tab_round")
        else:
            self.notify("DL Failed", C_ACCENT_1)
            if messagebox.askyesno("Download Blocked", LANG[self.lang]["download_failed"]):
                webbrowser.open(f"https://www.faceit.com/en/cs2/room/{rid}")
                messagebox.showinfo("Manual Guide", LANG[self.lang]["manual_guide"])

    def analyze_core(self, override_path=None):
        # --- [关键诊断修改开始] ---
        try:
            if not HAS_PARSER: 
                return self.notify(LANG[self.lang]["err_parser"], C_ACCENT_1)
            
            tid = self.entry_id.get().strip()
            if not tid: 
                return self.notify("ID Required", C_ACCENT_1)
            
            CONFIG.set("steam_id", tid)
            path = override_path if override_path else filedialog.askopenfilename(filetypes=[("CS2 Demo", "*.dem")])
            if not path: return
            
            # 检查物理路径是否存在且可读
            if not os.path.exists(path):
                messagebox.showerror("File Error", f"Cannot find demo at: {path}")
                return

            self.current_demo_path = path
            self.notify(LANG[self.lang]["processing"], C_HIGHLIGHT)
            for w in self.pages["tab_round"].winfo_children(): w.destroy()
            self.replay_data = None
            
            # 初始化解析器
            parser = DemoParser(path)
            
            # 解析数据
            df_k = parser.parse_event("player_death", other=["tick", "headshot", "weapon"], player=["attacker_steamid", "user_steamid", "attacker_name", "user_name"])
            df_r = parser.parse_event("round_start", other=["tick"])
            
            if df_r.empty: 
                start_tick = df_k['tick'].min() if not df_k.empty else 0
                ticks = [start_tick]
            else: 
                start_tick = df_r['tick'].min()
                ticks = df_r['tick'].tolist() + [float('inf')]
                
            df_k['round'] = pd.cut(df_k['tick'], bins=ticks, labels=range(1, len(ticks)), right=False)
            total_k, total_d, total_hs = 0, 0, 0
            ai_rounds, cards_to_show = [], []
            max_r = int(df_k['round'].max()) if not df_k['round'].isnull().all() else 0
            
            for r in range(1, max_r + 1):
                r_data = df_k[df_k['round'] == r].sort_values('tick')
                if r_data.empty: continue
                my_k = r_data[(r_data['attacker_steamid'] == str(tid)) & (r_data['user_steamid'] != str(tid))]
                my_d = r_data[r_data['user_steamid'] == str(tid)]
                k_count, d_count = len(my_k), len(my_d)
                total_k += k_count; total_d += d_count; total_hs += my_k['headshot'].sum()
                round_start_t = ticks[r-1]
                details = []
                for _, row in my_k.iterrows():
                    s = (row['tick'] - start_tick) / 64
                    details.append({"type": "kill", "time": f"{int(s//60)}:{int(s%60):02d}", "msg": f"{row.get('user_name', 'Enemy')} ({row.get('weapon','gun')})"})
                if d_count > 0:
                    s = (my_d.iloc[0]['tick'] - start_tick) / 64
                    details.append({"type": "death", "time": f"{int(s//60)}:{int(s%60):02d}", "msg": f"Killed by {my_d.iloc[0].get('attacker_name', 'Enemy')}"})
                tag, advice, ai_comment, suffix = "NORMAL", "", "", "_zh" if self.lang == "zh" else ""
                if k_count >= 3: 
                    tag = "🌟 CARRY"; ai_comment = random.choice(LOCAL_AI_PHRASES[f"carry{suffix}"])
                elif k_count == 0 and d_count > 0:
                    if (my_d.iloc[0]['tick'] - round_start_t) / 64 < 20: 
                        tag = "❌ Rushed"; advice = "Don't dry peek." if self.lang=="en" else "开局切勿干拉。"; ai_comment = random.choice(LOCAL_AI_PHRASES[f"death_early{suffix}"])
                    else: 
                        tag = "💀 DIED"; ai_comment = "Repositioning needed." if self.lang=="en" else "需要调整站位。"
                elif k_count == 1 and d_count == 1: 
                    tag = "⚖️ TRADE"; ai_comment = random.choice(LOCAL_AI_PHRASES[f"trade{suffix}"])
                else: ai_comment = random.choice(LOCAL_AI_PHRASES[f"normal{suffix}"])
                
                card = EndelCard(self.pages["tab_round"], r, {"kills": k_count, "deaths": d_count, "tag": tag, "advice": advice, "ai_comment": ai_comment, "start_tick": round_start_t}, details, self.lang, self.watch_round)
                cards_to_show.append((card, {"fill": "x", "pady": 6, "padx": 10}))
                ai_rounds.append({"r": r, "k": k_count, "d": d_count, "tag": tag})
                
            self.lbl_kd.configure(text=f"{total_k / total_d if total_d > 0 else total_k:.2f}")
            self.lbl_hs.configure(text=f"{(total_hs/total_k*100) if total_k>0 else 0:.1f}%")
            self.match_data = {"id": tid, "rounds": ai_rounds, "stats": {"kd": total_k/total_d if total_d > 0 else total_k}}
            if not override_path: self.show_page("tab_round")
            self.after(100, lambda: self.cascade_load_rounds(cards_to_show))
            
        except Exception as e:
            # 获取详细错误堆栈并弹窗显示（关键调试步骤）
            err_trace = traceback.format_exc()
            messagebox.showerror("Engine Error", f"Analysis failed!\n\nReason: {e}\n\nTraceback:\n{err_trace}")
            self.notify(f"Error: {e}", C_ACCENT_1)
        # --- [关键诊断修改结束] ---

    def call_ai(self):
        if not self.match_data: return
        if not HAS_OPENAI: return self.notify("OpenAI Lib Missing", C_ACCENT_1)
        self.txt_ai.delete("0.0", "end"); self.txt_ai.insert("0.0", "...\n")
        try:
            client = OpenAI(api_key=API_KEY, base_url=API_BASE)
            response = client.chat.completions.create(model="deepseek-chat", messages=[{"role":"system", "content": LANG[self.lang]["prompt_sys"] + " Be concise."}, {"role":"user", "content": LANG[self.lang]["prompt_user"] + f"\nDATA: {json.dumps(self.match_data)}"}], stream=True, temperature=1.3)
            self.txt_ai.delete("0.0", "end")
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content: self.txt_ai.insert("end", content); self.txt_ai.see("end"); self.txt_ai.update_idletasks()
            self.notify("Neural Report Ready", C_ACCENT_2)
        except Exception as e: self.txt_ai.insert("end", f"\nError: {e}"); self.notify("Failed", C_ACCENT_1)

if __name__ == "__main__":
    for folder in ["maps", "downloads"]:
        if not os.path.exists(folder): os.makedirs(folder)
    login = LoginWindow(); login.mainloop()
    if login.verified: app = App(); app.mainloop()