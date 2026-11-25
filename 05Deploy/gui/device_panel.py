"""
device_panel.py
设备连通示意图 (纯界面层, 不接 PLC)

对外接口:
  panel = DevicePanel(parent, on_execute=回调)
  panel.set_states(n_chiller, n_cdz, n_pump=2, system_on=True)
      → 冷水机/CDZ/冷冻水泵 前 n 台点亮(绿), 其余熄灭(暗)
  panel.set_system(True/False)  → 单独控制系统运行灯(红=运行)
"""

import tkinter as tk
from pathlib import Path

# Pillow 可选依赖: 有则贴PNG图标, 无则回退矢量绘制
try:
    from PIL import Image, ImageTk, ImageEnhance
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runtime_paths import DEPLOY_DIR
ICON_DIR = DEPLOY_DIR / 'icons'

# ── 配色 (与 app.py 工业灰风格一致) ──
PANEL    = '#363636'
CARD     = '#2b2b2b'
BORDER   = '#555555'
FG       = '#e0e0e0'
FG_DIM   = '#9e9e9e'
PIPE_W   = '#5a9fd4'   # 冷冻水管(蓝)
PIPE_A   = '#4fa39a'   # 空气侧(青)
GHOST    = '#6a6a6a'   # 冷却水侧摆设(灰)
LAMP_ON  = '#4caf50'   # 运行(绿)
LAMP_OFF = '#3a3a3a'   # 待机(暗)
LAMP_SYS = '#e05a5a'   # 系统运行灯(工业红=运行)

FONT = 'Microsoft YaHei'

# 基准画布尺寸(设计坐标系) + 缩放系数
BASE_W, BASE_H = 700, 560
S = 1.25


class DevicePanel(tk.Frame):
    """设备连通示意图 + 状态灯。父容器直接 pack/grid 本 Frame 即可。"""

    def __init__(self, parent, on_execute=None, on_lamp_test=None,on_stop=None, **kw):
        """
        on_execute: 可选回调, "根据算法推荐启停亮灯"按钮点击时触发。
                    device_panel 本身不读日志/不碰数据, 真正取台数点灯的
                    逻辑由外部(app.py)通过该回调实现, 以保持纯界面层。
        """
        super().__init__(parent, bg=PANEL, **kw)
        self.on_execute = on_execute
        self.on_lamp_test = on_lamp_test
        self.on_stop = on_stop
        self.on_execute = on_execute
        self.on_lamp_test = on_lamp_test
        self.S = S
        self.canvas = tk.Canvas(self, width=self._c(BASE_W),
                                height=self._c(BASE_H), bg=PANEL,
                                highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        self._icons = {}         # (文件名, 宽, 高) → PhotoImage 缓存
                                 # 必须挂在实例上, 局部变量会被GC导致图标空白

        self.lamp_chiller = []   # 4
        self.lamp_cdz     = []   # 6
        self.lamp_pump    = []   # 2
        self.lamp_system  = None # 1

        # 动态缩放: 记忆灯状态/按钮状态(重绘后恢复), 防抖任务句柄
        self._last_states = (0, 0, 0, False)
        self._buttons_enabled = True
        self._resize_job = None
        # 绘图内容在宽画布中的水平偏移
        self._offset_x = 0
        self._pending_s = self.S
        self._pending_width = self._c(BASE_W)

        self._draw_all()
        # 容器尺寸变化 → 防抖重绘(S 随容器动态计算)
        self.canvas.bind('<Configure>', self._on_resize)

        self.on_execute = on_execute
        self.on_lamp_test = on_lamp_test
        self.on_stop = on_stop

    # ── 缩放助手: 所有坐标/字号统一乘 S ──
    def _c(self, v):
        return int(round(v * self.S))

    def _font(self, size, weight=None):
        sz = self._c(size)
        return (FONT, sz) if weight is None else (FONT, sz, weight)

    def _line(self, x0, y0, x1, y1, **kw):
        return self.canvas.create_line(self._c(x0), self._c(y0),
                                       self._c(x1), self._c(y1), **kw)

    def _oval(self, x0, y0, x1, y1, **kw):
        return self.canvas.create_oval(self._c(x0), self._c(y0),
                                       self._c(x1), self._c(y1), **kw)

    def _rect(self, x0, y0, x1, y1, **kw):
        return self.canvas.create_rectangle(self._c(x0), self._c(y0),
                                            self._c(x1), self._c(y1), **kw)

    def _poly(self, *coords, **kw):
        sc = [self._c(v) for v in coords]
        return self.canvas.create_polygon(sc, **kw)

    def _text(self, x, y, text, size, weight=None, **kw):
        return self.canvas.create_text(self._c(x), self._c(y), text=text,
                                       font=self._font(size, weight), **kw)

    def _round_rect(self, x0, y0, x1, y1, r=6, **kw):
        pts = [x0+r, y0, x1-r, y0, x1, y0, x1, y0+r, x1, y1-r, x1, y1,
               x1-r, y1, x0+r, y1, x0, y1, x0, y1-r, x0, y0+r, x0, y0]
        return self._poly(*pts, smooth=True, **kw)

    def _lamp(self, cx, cy, r=6, fill=LAMP_OFF):
        return self._oval(cx-r, cy-r, cx+r, cy+r,
                          fill=fill, outline=BORDER, width=1)

    # ── 图标助手: 尺寸按设计坐标传入, 内部乘 S 缩放 ──
    def _icon(self, name, w, h, brightness=1.0):
        if not _HAS_PIL:
            return None

        key = (name, w, h, brightness)
        if key not in self._icons:
            try:
                img = Image.open(ICON_DIR / name).convert('RGBA')
                if brightness != 1.0:
                    img = ImageEnhance.Brightness(img).enhance(brightness)
                img = img.resize((self._c(w), self._c(h)), Image.LANCZOS)
                self._icons[key] = ImageTk.PhotoImage(img)
            except Exception as e:
                print(f'[device_panel] 图标加载失败 {name}: {e}, 回退矢量绘制')
                self._icons[key] = None
        return self._icons[key]

    def _image(self, cx, cy, ico):
        """在设计坐标 (cx,cy) 居中贴图"""
        return self.canvas.create_image(self._c(cx), self._c(cy), image=ico)

    # ── 绘制入口 ──
    def _draw_all(self):
        self._draw_pipes()          # 所有管路(在最底层)
        self._draw_machineroom()    # 机房(汇聚点)
        self._draw_chillers()       # 冷水机 x4 (受控)
        self._draw_chilled_pumps()  # 冷冻水泵 x2 (受控)
        self._draw_cdz()            # 空调 CDZ x6 (受控)
        self._draw_cooling_side()   # 冷却塔 x2 + 并联冷却水泵 x2
        self._draw_labels()         # 供/回水、循环标注
        self._draw_bottom_bar()     # 底部一排
        self.set_states(0, 0, 0, system_on=False)   # 初始待机

    # ── 管路 (先画, 位于底层) ──
    def _draw_pipes(self):
        # 冷冻水供水：冷水机组 → 两台并联水泵 → 机房
        self._line(285, 150, 315, 150, fill=PIPE_W, width=3)
        # 左侧分流管
        self._line(315, 130, 315, 170, fill=PIPE_W, width=3)
        # 上下两条并联支路
        self._line(315, 130, 425, 130, fill=PIPE_W, width=3)
        self._line(315, 170, 425, 170, fill=PIPE_W, width=3)
        # 右侧汇流管
        self._line(425, 130, 425, 170, fill=PIPE_W, width=3)
        # 汇流后进入机房
        self._line(425, 150, 452, 150, fill=PIPE_W, width=3, arrow='last')

        # 冷冻水: 回水(机房→冷水机)
        self._line(452, 232, 285, 232, fill=PIPE_W, width=3, arrow='last')
        # 空气侧: CDZ → 机房(底部)
        self._line(400, 356, 400, 322, fill=PIPE_A, width=3)
        self._line(400, 322, 536, 322, fill=PIPE_A, width=3)
        self._line(536, 322, 536, 300, fill=PIPE_A, width=3, arrow='last')
        # 空气侧: 机房 → 室外 (出风, 热空气排出)
        self._line(620, 170, 686, 170, fill=PIPE_A, width=3, arrow='last')
        # 回风：机房 → 空调区域
        self._line(580, 300, 580, 348,
                   fill=PIPE_A, width=3, arrow='last')
        # 新风：室外 → 回风管汇合点
        self._line(686, 326, 580, 326,
                   fill=PIPE_A, width=3, arrow='last')

        # ── 冷却水侧 ──
        # 高温冷却水：冷水机组 → 两台并联冷却水泵
        self._line(150, 155, 140, 155, fill=PIPE_W, width=3)

        # 水泵入口分流管
        self._line(140, 130, 140, 180, fill=PIPE_W, width=3)

        # 两条并联水泵支路
        self._line(140, 130, 95, 130, fill=PIPE_W, width=3)
        self._line(140, 180, 95, 180, fill=PIPE_W, width=3)

        # 水泵出口汇流管
        self._line(95, 130, 95, 180, fill=PIPE_W, width=3)

        # 汇流后只用一个箭头指向冷却塔组
        self._line(95, 155, 80, 155,
                   fill=PIPE_W, width=3, arrow='last')

        # 冷却塔组只用一条回水管返回冷水机组
        self._line(80, 260, 145, 260,
                   fill=PIPE_W, width=3, arrow='last')

    # ── 机房 (液冷+风冷汇聚点) ──
    def _draw_machineroom(self):
        x0, y0, x1, y1 = 452, 95, 620, 300
        self._round_rect(x0, y0, x1, y1, r=8, fill=CARD, outline=BORDER)
        self._text((x0+x1)//2, y0+22, '机房', 14, 'bold', fill=FG)
        ico = self._icon('server_room.png', 118, 108)
        if ico:
            # 图标替代原机柜矩形示意, 居中放在机房框下半部
            self._image((x0+x1)//2, y1-72, ico)
        else:
            for xr in range(x0+22, x1-18, 26):
                self._rect(xr, y1-88, xr+14, y1-18, outline=FG_DIM, width=1)

    # ── 冷水机 x4 (受控, 左列竖排) ──
    def _draw_chillers(self):
        self._text(160, 56, '冷水机组', 11, 'bold', fill=PIPE_W, anchor='w')
        ys = [70, 128, 186, 244]
        x0, x1 = 160, 285
        ico = self._icon('chiller.png', 55, 55)
        for i, y in enumerate(ys):
            icon_cx = 174
            card_x0, card_x1 = 198, 285
            # 右侧只绘制状态灯和文字框
            self._round_rect(card_x0, y, card_x1, y + 46, r=6,
                             fill=CARD, outline=BORDER)
            # 图标放在文字框左侧
            if ico:
                self._image(icon_cx, y + 23, ico)
            else:
                self._rect(155, y + 4, 193, y + 42,
                           outline=PIPE_W, width=2)

            lamp = self._lamp(card_x0 + 11, y + 12)
            self._text(card_x0 + 22, y + 24, f'冷水机#{i + 1}',
                       8, fill=FG, anchor='w')
            self.lamp_chiller.append(lamp)

    # ── 冷冻水泵 x2 (受控, 供水管上) ──
    def _draw_chilled_pumps(self):
        ico = self._icon('chilled_pump.png', 32, 32)
        positions = [
            (365, 130, 105),  # 泵中心x、泵中心y、状态信息y
            (365, 170, 195),
        ]

        for i, (cx, cy, info_y) in enumerate(positions):
            if ico:
                self._image(cx, cy, ico)
            else:
                self._oval(cx - 15, cy - 15, cx + 15, cy + 15,
                           fill=CARD, outline=PIPE_W, width=2)
                self._poly(cx - 6, cy - 7, cx - 6, cy + 7, cx + 8, cy,
                           fill=FG_DIM, outline='')

            lamp = self._lamp(cx - 18, info_y)
            self._text(cx - 7, info_y, f'冷冻水泵{i + 1}', 8, fill=FG, anchor='w')
            self.lamp_pump.append(lamp)

    # ── 空调 CDZ x6 (受控, 独立空气侧) ──
    def _draw_cdz(self):
        self._text(160, 332, '风冷空调', 11, 'bold', fill=PIPE_W, anchor='w')
        cols = [160, 330, 500]
        rows = [360, 414]
        ico = self._icon('cdz.png', 42, 42)
        idx = 0
        for ry in rows:
            for cx0 in cols:
                x0, x1 = cx0, cx0 + 140
                y0, y1 = ry, ry + 48
                card_x0 = x0 + 48

                self._round_rect(card_x0, y0, x1, y1, r=6,
                                 fill=CARD, outline=BORDER)

                if ico:
                    self._image(x0 + 22, y0 + 24, ico)
                else:
                    self._oval(x0 + 3, y0 + 5, x0 + 41, y0 + 43,
                               outline='#eeee00', width=2)

                lamp = self._lamp(card_x0 + 11, y0 + 12)
                self._text(card_x0 + 22, y0 + 24, f'空调#{idx + 1}',
                           8, fill=FG, anchor='w')
                self.lamp_cdz.append(lamp)
                idx += 1

    # ── 冷却塔 + 冷却水泵 (摆设: 灰色, 无灯) ──
    def _draw_cooling_side(self):
        """两台冷却塔和两台并联冷却水泵，仅用于流程示意。"""

        # 两台冷却塔上下排列
        tower_positions = [
            (52, 140),
            (52, 215),
        ]
        ico_t = self._icon(
            'cooling_tower.png', 70, 60, brightness=1.8
        )

        for i, (cx, cy) in enumerate(tower_positions):
            if ico_t:
                self._image(cx, cy, ico_t)
            else:
                # 无图标时的梯形冷却塔
                self._poly(
                    cx - 22, cy - 17,
                    cx + 22, cy - 17,
                    cx + 18, cy + 17,
                    cx - 18, cy + 17,
                    fill='', outline=PIPE_W, width=2
                )
                self._line(
                    cx - 18, cy + 5, cx + 18, cy + 5,
                    fill=PIPE_W, width=1
                )

            self._text(
                cx, cy + 31, f'冷却塔{i + 1}',
                7, fill=PIPE_W
            )

        # 两台并联冷却水泵
        pump_positions = [
            (118, 130),
            (118, 180),
        ]

        ico_p = self._icon('cooling_pump.png', 30, 30)

        for i, (cx, cy) in enumerate(pump_positions):
            if ico_p:
                self._image(cx, cy, ico_p)
            else:
                self._oval(
                    cx - 14, cy - 14, cx + 14, cy + 14,
                    fill=CARD, outline=PIPE_W, width=2
                )
                self._poly(
                    cx - 5, cy - 6,
                    cx - 5, cy + 6,
                    cx + 7, cy,
                    fill=FG_DIM, outline=''
                )

            label_y = cy - 20 if i == 0 else cy + 20
            self._text(
                cx, label_y, f'冷却泵{i + 1}',
                7, fill=FG_DIM
            )

    # ── 供/回水等文字标注 ──
    def _draw_labels(self):
        self._text(375, 90, '冷板供水 (低温)', 8, fill=PIPE_W)
        self._text(368, 244, '冷板回水 (高温)', 8, fill=PIPE_W)
        self._text(410, 334, '送风（冷）', 8, fill=PIPE_A, anchor='w')
        self._text(624, 156, '排风 (热)', 8, fill=PIPE_A, anchor='w')
        self._text(568, 342, '回风（热）', 8,fill=PIPE_A, anchor='e')
        self._text(633, 312, '新风', 8,fill=PIPE_A)
        # 冷却塔组标题
        self._text(
            12, 100, '冷却塔组',
            11, 'bold', fill=PIPE_W, anchor='w'
        )

        # 并联水泵供水方向
        self._text(
            120, 95, '高温冷却水',
            7, fill=PIPE_W
        )

        # 公共回水管
        self._text(
            112, 275, '低温冷却水',
            7, fill=PIPE_W
        )
    # ── 底部一排: 系统运行 / 设备运行 / 设备待机 / 执行按钮 ──
    def _draw_bottom_bar(self):
        y = 515
        # 左侧提示灯区域
        self.lamp_system = self._lamp(32, y)
        self._text(46, y, '系统运行', 9, fill=FG, anchor='w')

        self._lamp(125, y, fill=LAMP_ON)
        self._text(139, y, '设备运行', 9, fill=FG_DIM, anchor='w')

        self._lamp(220, y, fill=LAMP_OFF)
        self._text(234, y, '设备待机', 9, fill=FG_DIM, anchor='w')

        # 亮灯测试按钮
        self.btn_lamp_test = tk.Button(
            self, text='亮灯测试', font=self._font(10, 'bold'),
            bg='#4a5a6a', fg='white', relief='flat',
            padx=self._c(14), pady=self._c(6),
            activebackground='#5a6a7a', activeforeground='white',
            command=self._on_lamp_test_click
        )
        self.canvas.create_window(
            self._c(BASE_W - 260), self._c(y),
            anchor='e', window=self.btn_lamp_test
        )

        # 根据算法执行按钮
        self.btn_execute = tk.Button(
            self, text='根据算法推荐启停设备', font=self._font(10, 'bold'),
            bg='#3a6ea5', fg='white', relief='flat',
            padx=self._c(14), pady=self._c(6),
            activebackground='#4a7eb5', activeforeground='white',
            command=self._on_execute_click
        )
        self.canvas.create_window(
            self._c(BASE_W - 70), self._c(y),
            anchor='e', window=self.btn_execute
        )

        # 停止按钮
        self.btn_stop = tk.Button(
            self, text='停止', font=self._font(10, 'bold'),
            bg='#a83a3a', fg='white', relief='flat',
            padx=self._c(12), pady=self._c(6),
            activebackground='#c04848', activeforeground='white',
            command=self._on_stop_click
        )
        self.canvas.create_window(
            self._c(BASE_W - 5), self._c(y),
            anchor='e', window=self.btn_stop
        )

    def _on_execute_click(self):
        if callable(self.on_execute):
            self.on_execute()

    def _on_stop_click(self):
        if callable(self.on_stop):
            self.on_stop()

    def _on_lamp_test_click(self):
        if callable(self.on_lamp_test):
            self.on_lamp_test()

    # ── 动态缩放: 容器尺寸变化 → 重算 S → 防抖重绘 ──
    def _on_resize(self, event):
        if event.width < 50 or event.height < 50:
            return

        new_s = min(event.width / BASE_W, event.height / BASE_H)
        new_s = max(0.7, min(1.6, new_s))

        # 按缩放后的设计宽度计算左右相等的空白
        new_offset_x = max(
            0,
            (event.width - BASE_W * new_s) / 2
        )

        scale_unchanged = abs(new_s - self.S) / self.S < 0.02
        offset_unchanged = abs(new_offset_x - self._offset_x) < 2

        if scale_unchanged and offset_unchanged:
            return

        self._pending_s = new_s
        self._pending_width = event.width

        if self._resize_job is not None:
            self.after_cancel(self._resize_job)

        self._resize_job = self.after(150, self._redraw)

    def _redraw(self):
        self._resize_job = None

        # 重绘前保存状态，防止缩放时灯状态被初始化覆盖
        last_states = self._last_states
        buttons_enabled = self._buttons_enabled

        self.S = self._pending_s

        self.canvas.delete('all')

        for btn_name in ('btn_execute', 'btn_lamp_test', 'btn_stop'):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.destroy()

        self.lamp_chiller = []
        self.lamp_cdz = []
        self.lamp_pump = []
        self.lamp_system = None
        self._icons = {}

        # 先按照原始坐标重画
        self._draw_all()

        # 再把全部图形和按钮水平移动到画布中间
        content_width = BASE_W * self.S
        self._offset_x = max(
            0,
            (self._pending_width - content_width) / 2
        )

        if self._offset_x > 0:
            self.canvas.move(
                'all',
                int(round(self._offset_x)),
                0
            )

        # 恢复重绘前的灯和按钮状态
        self.set_states(
            *last_states[:3],
            system_on=last_states[3]
        )
        self.set_buttons_enabled(buttons_enabled)

    # ── 对外: 点灯 ──
    def set_states(self, n_chiller, n_cdz, n_pump=2, system_on=True):
        """按推荐台数点亮受控设备的状态灯(开一台亮一个, 其余熄灭)。"""
        self._last_states = (n_chiller, n_cdz, n_pump, system_on)
        self._set_group(self.lamp_chiller, n_chiller)
        self._set_group(self.lamp_cdz, n_cdz)
        self._set_group(self.lamp_pump, n_pump)
        self.set_system(system_on)

    def set_system(self, on: bool):
        if self.lamp_system is not None:
            self.canvas.itemconfig(self.lamp_system,
                                   fill=LAMP_SYS if on else LAMP_OFF)

    def _set_group(self, lamps, n):
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = 0
        for i, lamp in enumerate(lamps):
            self.canvas.itemconfig(lamp, fill=LAMP_ON if i < n else LAMP_OFF)

    def set_single(self, kind, index, on):
        """巡检用: 单独点亮/熄灭界面上某一个设备灯"""
        table = {'chiller': self.lamp_chiller,
                 'cdz': self.lamp_cdz,
                 'pump': self.lamp_pump}
        lamps = table.get(kind)
        if lamps is None or not (0 <= index < len(lamps)):
            return
        self.canvas.itemconfig(lamps[index],
                               fill=LAMP_ON if on else LAMP_OFF)

    def set_buttons_enabled(self, enabled: bool):
        """巡检期间禁用/恢复两个按钮"""
        self._buttons_enabled = enabled
        state = 'normal' if enabled else 'disabled'
        self.btn_execute.config(state=state)
        self.btn_lamp_test.config(state=state)

# ── 独立测试: 直接运行本文件可预览并循环点灯 ──
if __name__ == '__main__':
    root = tk.Tk()
    root.title('DevicePanel 预览')
    root.configure(bg=PANEL)

    def demo_execute():
        print('[执行按钮] 被点击 → 此处将由 app.py 读日志取台数并点灯')
        panel.set_states(n_chiller=3, n_cdz=5, n_pump=2, system_on=True)

    panel = DevicePanel(root, on_execute=demo_execute)
    panel.pack(fill='both', expand=True, padx=10, pady=10)

    panel.set_states(n_chiller=2, n_cdz=6, n_pump=2, system_on=True)

    combos = [(2, 6, 2), (3, 4, 2), (1, 2, 2), (4, 6, 2), (0, 0, 0)]
    state = {'i': 0}

    def cycle():
        nc, ncdz, npu = combos[state['i'] % len(combos)]
        panel.set_states(nc, ncdz, npu, system_on=(nc > 0))
        state['i'] += 1
        root.after(1500, cycle)

    root.after(1500, cycle)
    root.mainloop()
