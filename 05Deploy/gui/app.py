"""
app.py
数据中心冷却优化控制 - 主界面
目标: 1280x800 工业触摸屏, 工业灰风格
"""

import sys
import json
import queue
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

GUI_DIR    = Path(__file__).resolve().parent
sys.path.insert(0, str(GUI_DIR.parent))          # 开发期: 05Deploy 进 path
sys.path.insert(0, str(GUI_DIR))
sys.path.insert(0, str(GUI_DIR.parent / 'plc'))
from runtime_paths import DEPLOY_DIR
import run_hourly
from worker import ComputeWorker
from device_panel import DevicePanel
import electricity_pricing as ep
import plc_mapping

PRICE_CFG_PATH = DEPLOY_DIR / 'price_config.json'
# 电价档位固定顺序(由低到高), 供电价条展示
PRICE_ORDER = ['深谷', '低谷', '平时', '高峰', '尖峰']

AUTH_CFG_PATH = DEPLOY_DIR / 'auth_config.json'

# ── 登录页文案(写死, 要改名称就改这两行) ──
LOGIN_SCHOOL  = '上海海洋大学'            # ← 学校全称改这里
LOGIN_COMPANY = '上海维扩机电工程有限公司'     # ← 企业全称改这里
LOGIN_LOGO    = 'school_logo.png'   # ← 图标文件名(在 icons 文件夹内)

#工业灰配色
BG        = '#2b2b2b'   # 主背景
PANEL     = '#363636'   # 面板背景
PANEL_LT  = '#424242'   # 浅面板(参考量)
CARD_DEC  = '#1f1f1f'   # 决策量卡片(深)
FG        = '#e0e0e0'   # 主文字
FG_DIM    = '#9e9e9e'   # 次要文字
ACCENT    = '#4caf50'   # 省电高亮(绿)
ACCENT_B  = '#5a9fd4'   # 数值高亮(蓝)
BORDER    = '#555555'

FONT      = 'Microsoft YaHei'
F_TITLE   = (FONT, 14, 'bold')
F_LABEL   = (FONT, 10)
F_VALUE   = (FONT, 13, 'bold')
F_BIG     = (FONT, 40, 'bold')
F_SMALL   = (FONT, 9)

# 环境面板六项天气(顺序即显示顺序)
WEATHER_FIELDS = [
    ('temp_C',        '室外℃'),
    ('humidity_pct',  '湿度%'),
    ('wind_speed_ms', '风速m/s'),
    ('pressure_hPa',  '气压hPa'),
    ('dew_point_C',   '露点℃'),
    ('temp_wet_C',    '湿球℃'),
]


class App:
    def __init__(self, root):
        self.root = root
        self.worker = ComputeWorker()
        self.auto_on = False
        self.next_trigger = None

        root.title('数据中心冷却优化控制系统')
        root.geometry('1280x750')
        root.configure(bg=BG)
        root.after(0, lambda: root.state('zoomed'))
        self._lamp_test_job = None
        self._build_login()  # 先进登录页, 验证通过才建主界面

    def _enter_main(self):
        self._build_log()  # 先 pack: 底部日志栏优先锁定空间
        self._build_tabs()  # 后 pack: Notebook 吃剩余高度

        self._poll_queues()
        self._tick_countdown()
        self.refresh_chart()

        # 软件启动即点亮系统运行灯
        self.device_panel.set_system(True)
        self._highlight_current_period()

    # ── 登录页: 同窗口门厅, 验证通过后销毁并进入主界面 ──
    def _build_login(self):
        self.login_frame = tk.Frame(self.root, bg=BG)
        self.login_frame.pack(fill='both', expand=True)

        box = tk.Frame(self.login_frame, bg=BG)
        box.place(relx=0.5, rely=0.5, anchor='center')  # 整列垂直水平居中

        # 校徽图标(缺失/PIL不可用则优雅降级为不显示)
        try:
            from PIL import Image, ImageTk
            img = Image.open(DEPLOY_DIR / 'icons' / LOGIN_LOGO)
            img.thumbnail((96, 96))
            self._login_logo = ImageTk.PhotoImage(img)  # 挂实例防GC
            tk.Label(box, image=self._login_logo, bg=BG).pack(pady=(0, 12))
        except Exception:
            pass

        # 两家单位, 同字号同色, 完全平级
        for name in (LOGIN_SCHOOL, LOGIN_COMPANY):
            tk.Label(box, text=name, font=(FONT, 19, 'bold'),
                     bg=BG, fg=FG).pack(pady=3)

        form = tk.Frame(box, bg=BG)
        form.pack(pady=(36, 0))

        tk.Label(form, text='账号', font=(FONT, 13), bg=BG,
                 fg=FG_DIM).grid(row=0, column=0, padx=(0, 10), pady=8, sticky='e')
        self.ent_user = tk.Entry(form, font=(FONT, 14), width=18,
                                 bg=PANEL_LT, fg=FG, relief='flat',
                                 insertbackground=FG)
        self.ent_user.grid(row=0, column=1, ipady=7, pady=8)

        tk.Label(form, text='密码', font=(FONT, 13), bg=BG,
                 fg=FG_DIM).grid(row=1, column=0, padx=(0, 10), pady=8, sticky='e')
        self.ent_pwd = tk.Entry(form, font=(FONT, 14), width=18, show='*',
                                bg=PANEL_LT, fg=FG, relief='flat',
                                insertbackground=FG)
        self.ent_pwd.grid(row=1, column=1, ipady=7, pady=8)
        self.ent_pwd.bind('<Return>', lambda e: self._do_login())  # 回车即登录

        tk.Button(box, text='登  录', font=(FONT, 14, 'bold'),
                  width=18, pady=8, bg='#3a6ea5', fg='white', relief='flat',
                  activebackground='#4a7eb5', activeforeground='white',
                  command=self._do_login).pack(pady=(22, 0))

        self.lbl_login_err = tk.Label(box, text='', font=(FONT, 11),
                                      bg=BG, fg='#e05a5a')
        self.lbl_login_err.pack(pady=(10, 0))

        self.ent_user.focus_set()

    def _do_login(self):
        # 读外置账密(读不到用默认兜底, 不把人锁在门外)
        try:
            with open(AUTH_CFG_PATH, encoding='utf-8') as f:
                auth = json.load(f)
        except Exception:
            auth = {}
        user = auth.get('username', 'admin')
        pwd = auth.get('password', '123456')

        if self.ent_user.get().strip() == user and self.ent_pwd.get() == pwd:
            self.login_frame.destroy()
            self._enter_main()
        else:
            self.lbl_login_err.config(text='账号或密码错误')


    # ── 顶部控制栏 ──
    def _build_top(self, parent):
        bar = tk.Frame(parent, bg=PANEL, height=56)
        bar.pack(fill='x', side='top')
        bar.pack_propagate(False)

        self.btn_run = tk.Button(
            bar, text='立即执行', font=F_LABEL, width=10, height=2,
            bg='#3a6ea5', fg='white', relief='flat', command=self.manual_run)
        self.btn_run.pack(side='left', padx=10, pady=8)

        self.btn_auto = tk.Button(
            bar, text='启动自动', font=F_LABEL, width=10, height=2,
            bg='#4a4a4a', fg='white', relief='flat', command=self.toggle_auto)
        self.btn_auto.pack(side='left', padx=5, pady=8)

        self.lbl_status = tk.Label(
            bar, text='就绪', font=F_LABEL, bg=PANEL, fg=FG)
        self.lbl_status.pack(side='left', padx=20)

        self.lbl_source = tk.Label(
            bar, text='数据源: 历史回放(模拟)', font=F_SMALL,
            bg=PANEL, fg=FG_DIM)
        self.lbl_source.pack(side='right', padx=15)

        self.lbl_countdown = tk.Label(
            bar, text='', font=F_LABEL, bg=PANEL, fg=ACCENT_B)
        self.lbl_countdown.pack(side='right', padx=10)

    # ── 主体: 两个标签页 ──
    def _build_tabs(self):
        # 让 Notebook 融入深色主题
        style = ttk.Style()
        try:
            style.theme_use('default')
        except tk.TclError:
            pass
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=PANEL, foreground=FG_DIM,
                        padding=(18, 8), font=(FONT, 11))
        style.map('TNotebook.Tab',
                  background=[('selected', PANEL_LT)],
                  foreground=[('selected', FG)])

        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=8, pady=6)

        tab1 = tk.Frame(nb, bg=BG)
        tab2 = tk.Frame(nb, bg=BG)
        nb.add(tab1, text='运行监控')
        nb.add(tab2, text='设备与执行')

        self._build_body(tab1)
        self._build_tab2(tab2)

    # ── Tab1 主体: 左栏 + 右栏 ──
    def _build_body(self, parent):
        self._build_top(parent)  # ← 顶栏挪进本页顶部
        body = tk.Frame(parent, bg=BG)
        body.pack(fill='both', expand=True)

        left = tk.Frame(body, bg=BG, width=440)
        left.pack(side='left', fill='both', padx=(0, 6))
        left.pack_propagate(False)

        right = tk.Frame(body, bg=BG)
        right.pack(side='right', fill='both', expand=True)

        self._build_left(left)
        self._build_right(right)

    # ── Tab2: 设备连通图 + 执行按钮 ──
    def _build_tab2(self, parent):
        self.device_panel = DevicePanel(
            parent,
            on_execute=self.exec_plan,
            on_lamp_test=self.lamp_test,
            on_stop=self.stop_all
        )
        self.device_panel.pack(fill='both', expand=True)

    def _build_left(self, parent):
        # 环境(实时天气) + 负载预测
        top = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER,
                       highlightthickness=1)
        top.pack(fill='x', pady=(0, 4))
        self.v_q = {}

        # 第1行: 环境数据来源标注
        tk.Label(top, text='天气数据来源: 和风天气', font=F_SMALL,
                 bg=PANEL, fg=FG_DIM, anchor='w').pack(fill='x', padx=8, pady=(3, 0))

        # 第2行: 六项实时天气
        row_wx = tk.Frame(top, bg=PANEL)
        row_wx.pack(fill='x', pady=(2, 2))
        for key, label in WEATHER_FIELDS:
            self._mini_stat(row_wx, label, key, self.v_q, FG)

        # 第3行: 负载预测标注 + Q10/Q50/Q90
        tk.Label(top, text='负载预测', font=F_SMALL,
                 bg=PANEL, fg=FG_DIM, anchor='w').pack(fill='x', padx=8)
        row_q = tk.Frame(top, bg=PANEL)
        row_q.pack(fill='x', pady=(2, 2))
        for key, label in [('Q10_kW', 'Q10'), ('Q50_kW', 'Q50'), ('Q90_kW', 'Q90')]:
            self._mini_stat(row_q, label, key, self.v_q, ACCENT_B)

        # 设备状态
        dev = self._panel(parent, '设备状态')
        dev.pack(fill='both', expand=True, pady=(0, 0))

        self.v_dev = {}
        # 三个设备块按各自格子行数配权重(2:2:1), grid 弹性分配
        blocks_box = tk.Frame(dev, bg=PANEL)
        blocks_box.pack(fill='both', expand=True)
        blocks_box.grid_columnconfigure(0, weight=1)
        for r, w in [(0, 2), (1, 2), (2, 1)]:
            blocks_box.grid_rowconfigure(r, weight=w)
        self._dev_block(blocks_box, 0, '冷水机', [
            ('台数', 'N_chiller', '决策'), ('PLR', 'PLR', '参考'),
            ('COP', 'COP', '参考'), ('功耗kW', 'P_chiller', '参考'),
            ('供水℃', 'T_mandata', '参考'), ('回水℃', 'T_ritorno', '参考'),
        ])
        self._dev_block(blocks_box, 1, '空调', [
            ('总台数', 'N_cdz', '决策'), ('FC台数', 'N_FC', '决策'),
            ('单台转速(m³/h)', 's_fan', '决策'), ('功耗kW', 'P_cdz', '参考'),
        ])
        self._dev_block(blocks_box, 2, '水泵', [
            ('频率(Hz)', 'f_pump', '决策'), ('流量(m³/h)', 'Portata', '参考'),
            ('功耗kW', 'P_pump', '参考'),
        ])

    def _build_right(self, parent):
        # 电费成果区
        cost = self._panel(parent, '优化成果')
        cost.pack(fill='x', pady=(0, 6))

        inner = tk.Frame(cost, bg=PANEL)
        inner.pack(fill='x', pady=8)

        # 大号省电百分比
        save_box = tk.Frame(inner, bg=PANEL)
        save_box.pack(side='left', padx=(10, 8))
        tk.Label(save_box, text='预计省电', font=F_LABEL,
                 bg=PANEL, fg=FG_DIM).pack()
        self.v_save_pct = tk.Label(save_box, text='--', font=F_BIG,
                                    bg=PANEL, fg=ACCENT , width=6)
        self.v_save_pct.pack()

        # 电费对比
        cmp_box = tk.Frame(inner, bg=PANEL)
        cmp_box.pack(side='left', padx=10)
        self.v_cost = {}
        for key, label, color in [
            ('cost_yuan', '优化电费', ACCENT_B),
            ('cost_baseline', '保守基线', FG_DIM),
            ('saving_yuan', '省电金额', ACCENT)]:
            r = tk.Frame(cmp_box, bg=PANEL)
            r.pack(anchor='w', pady=3)
            tk.Label(r, text=f'{label}:', font=F_LABEL, bg=PANEL,
                     fg=FG_DIM, width=8, anchor='w').pack(side='left')
            v = tk.Label(r, text='--', font=F_VALUE, bg=PANEL, fg=color, width=7, anchor='w')
            v.pack(side='left')
            self.v_cost[key] = v
            tk.Label(r, text='元/h', font=F_SMALL, bg=PANEL,
                     fg=FG_DIM).pack(side='left', padx=4)
        # 弹性留白: 吸收多余宽度, 使电价条贴右缘而非溢出
        tk.Frame(inner, bg=PANEL).pack(side='left', expand=True, fill='x')

        # 分时电价条(填充优化成果右侧空白区)
        self._build_price_inline(inner)

        # 趋势图
        chart = self._panel(parent, '历史趋势 (最近24小时)')
        chart.pack(fill='both', expand=True)
        self.fig = Figure(figsize=(7, 3.2), dpi=80, facecolor=PANEL)
        self.ax = self.fig.add_subplot(111)
        self.ax2 = self.ax.twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)

    # ── 分时电价条(五档全展开, 可改可存) ──
    def _build_price_inline(self, parent):
        prices = self._load_prices()

        box = tk.Frame(parent, bg=PANEL)
        box.pack(side='right', padx=(10, 10), fill='y')

        tk.Label(box, text='分时电价 (元/kWh)', font=F_LABEL,
                 bg=PANEL, fg=FG_DIM).pack(anchor='w', pady=(0, 6))

        row = tk.Frame(box, bg=PANEL)
        row.pack()

        self.v_price = {}        # 档位 → Entry
        self.price_blocks = {}   # 档位 → (外框, 标签)  供高亮切换
        for name in PRICE_ORDER:
            cell = tk.Frame(row, bg=PANEL_LT, highlightbackground=BORDER,
                            highlightthickness=1)
            cell.pack(side='left', padx=2)
            lbl = tk.Label(cell, text=name, font=F_SMALL,
                           bg=PANEL_LT, fg=ACCENT_B, width=5)
            lbl.pack(padx=3, pady=(3, 0))
            ent = tk.Entry(cell, width=6, font=F_SMALL, justify='center',
                           bg=CARD_DEC, fg=FG, relief='flat',
                           insertbackground=FG)
            ent.insert(0, f'{prices.get(name, 0):.4f}')
            ent.pack(padx=3, pady=(0, 3))
            self.v_price[name] = ent
            self.price_blocks[name] = (cell, lbl)

        btn = tk.Button(box, text='保存电价', font=F_SMALL,
                        bg='#3a6ea5', fg='white', relief='flat',
                        command=self._save_prices)
        btn.pack(anchor='w', pady=(8, 0))
        self.lbl_price_now = tk.Label(box, text='', font=F_SMALL,
                                      bg=PANEL, fg=FG_DIM)
        self.lbl_price_now.pack(anchor='w', pady=(3, 0))

    def _load_prices(self) -> dict:
        try:
            with open(PRICE_CFG_PATH, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return dict(ep.PRICES)   # 兜底: 用模块默认

    def _save_prices(self):
        try:
            new = {}
            for name, ent in self.v_price.items():
                new[name] = round(float(ent.get()), 4)
            # 先序列化成字符串(失败则不碰文件), 成功后才落盘
            text = json.dumps(new, ensure_ascii=False, indent=4)
            with open(PRICE_CFG_PATH, 'w', encoding='utf-8') as f:
                f.write(text)
            self._append_log(f'电价已保存: {new}')
            self._highlight_current_period()
        except ValueError:
            self._append_log('保存失败: 电价必须为数字')
        except Exception as e:
            self._append_log(f'保存失败: {e}')

    def _highlight_current_period(self):
        """把当前时段的色块加深高亮, 其余复原"""
        if not hasattr(self, 'price_blocks'):
            return
        now = datetime.now()
        try:
            cur = ep.get_period(now.hour, ep.get_day_type(now), now.month)
        except Exception:
            cur = None
        for name, (cell, lbl) in self.price_blocks.items():
            if name == cur:
                cell.config(bg=ACCENT_B)
                lbl.config(bg=ACCENT_B, fg='white')
            else:
                cell.config(bg=PANEL_LT)
                lbl.config(bg=PANEL_LT, fg=ACCENT_B)
        if cur and hasattr(self, 'lbl_price_now'):
            price = self._load_prices().get(cur, 0)
            self.lbl_price_now.config(text=f'当前时段: {cur}  {price:.4f} 元/kWh')

    # ── 执行按钮: 读日志最新方案 → 点灯(PLC 留桩) ──
    def exec_plan(self):
        try:
            df = pd.read_csv(run_hourly.LOG_PATH)
            if len(df) == 0:
                self._append_log('无可执行方案: 日志为空, 请先执行计算')
                return
            row = df.iloc[-1].to_dict()
            n_chiller = int(round(float(row['N_chiller'])))
            n_cdz     = int(round(float(row['N_cdz'])))
            n_pump    = int(round(float(row.get('N_pump', 2))))
            self.device_panel.set_states(n_chiller, n_cdz, n_pump,
                                         system_on=True)
            self._append_log(
                f'已按方案启动: 冷水机{n_chiller} 空调{n_cdz} 水泵{n_pump}')
            # ── PLC 点灯(模拟模式下打印, SIMULATION=False 时真写) ──
            plc_mapping.update_lamps(n_chiller, n_cdz, n_pump)

        except Exception as e:
            self._append_log(f'执行失败: {e}')

    # ── 亮灯测试: 12 灯逐个巡检(每灯1秒), 结束全灭 ──
    LAMP_TEST_SEQ = ([('chiller', i) for i in range(4)] +
                     [('pump', i) for i in range(2)] +
                     [('cdz', i) for i in range(6)])

    def stop_all(self):
        """停止当前点灯流程，并关闭界面及 PLC 的所有灯。"""

        # 取消正在等待执行的亮灯测试步骤
        if self._lamp_test_job is not None:
            try:
                self.root.after_cancel(self._lamp_test_job)
            except tk.TclError:
                pass
            self._lamp_test_job = None

        # 熄灭冷水机、空调、水泵以及系统运行灯
        self.device_panel.set_states(
            n_chiller=0,
            n_cdz=0,
            n_pump=0,
            system_on=False
        )

        # 恢复执行和亮灯测试按钮
        self.device_panel.set_buttons_enabled(True)

        try:
            # 关闭实际 PLC 输出；模拟模式下会执行对应的模拟逻辑
            plc_mapping.all_lamps_off()
            self._append_log('已停止：所有设备灯已关闭')
        except Exception as e:
            self._append_log(f'停止失败：界面灯已关闭，PLC关闭失败: {e}')

    def lamp_test(self):
        self.device_panel.set_buttons_enabled(False)
        self._append_log('亮灯测试开始: 冷水机#1-4 → 泵1-2 → CDZ#1-6, 每灯1秒')
        self._lamp_test_step(0)

    def _lamp_test_step(self, i):
        self._lamp_test_job = None
        try:
            if i > 0:
                k, idx = self.LAMP_TEST_SEQ[i - 1]
                self.device_panel.set_single(k, idx, False)
                plc_mapping.test_single_lamp(k, idx, False)
            if i >= len(self.LAMP_TEST_SEQ):
                plc_mapping.all_lamps_off()
                self.device_panel.set_states(0, 0, 0, system_on=True)
                self.device_panel.set_buttons_enabled(True)
                self._append_log('亮灯测试完成 ✓')
                return
            k, idx = self.LAMP_TEST_SEQ[i]
            self.device_panel.set_single(k, idx, True)
            plc_mapping.test_single_lamp(k, idx, True)
            self._append_log(f'测试灯: {k}#{idx + 1}')
            self._lamp_test_job = self.root.after(
                1000, lambda: self._lamp_test_step(i + 1)
            )
        except Exception as e:
            self._append_log(f'亮灯测试中止: {e}')
            self.device_panel.set_states(0, 0, 0, system_on=True)
            self.device_panel.set_buttons_enabled(True)


    # ── 底部日志(可折叠) ──
    def _build_log(self):
        self.log_open = False
        bar = tk.Frame(self.root, bg=PANEL)
        bar.pack(fill='x', side='bottom')

        self.btn_log = tk.Button(
            bar, text='▲ 执行日志', font=F_SMALL, bg=PANEL, fg=FG,
            relief='flat', command=self.toggle_log)
        self.btn_log.pack(side='left', padx=8, pady=2)

        self.log_frame = tk.Frame(self.root, bg='#1a1a1a', height=120)
        self.log_text = tk.Text(
            self.log_frame, bg='#1a1a1a', fg=FG_DIM, font=F_SMALL,
            height=7, relief='flat', wrap='word')
        self.log_text.pack(fill='both', expand=True, padx=4, pady=2)

    # ── UI 工具方法 ──
    def _panel(self, parent, title):
        p = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER,
                     highlightthickness=1)
        tk.Label(p, text=title, font=F_TITLE, bg=PANEL, fg=FG,
                 anchor='w').pack(fill='x', padx=8, pady=(2, 2))
        return p

    def _mini_stat(self, parent, label, key, store, color):
        box = tk.Frame(parent, bg=PANEL)
        box.pack(side='left', expand=True, fill='x')
        tk.Label(box, text=label, font=F_SMALL, bg=PANEL,
                 fg=FG_DIM).pack()
        v = tk.Label(box, text='--', font=F_VALUE, bg=PANEL, fg=color)
        v.pack()
        store[key] = v

    def _dev_block(self, parent, row, name, fields):
        block = tk.Frame(parent, bg=PANEL)
        block.grid(row=row, column=0, sticky='nsew', padx=8, pady=3)
        tk.Label(block, text=name, font=(FONT, 11, 'bold'),
                 bg=PANEL, fg=ACCENT_B, anchor='w').pack(fill='x')

        grid = tk.Frame(block, bg=PANEL)
        grid.pack(fill='both', expand=True, pady=2)
        n_rows = (len(fields) + 2) // 3
        for r in range(n_rows):
            grid.grid_rowconfigure(r, weight=1)
        for i, (label, key, kind) in enumerate(fields):
            cell_bg = CARD_DEC if kind == '决策' else PANEL_LT
            cell = tk.Frame(grid, bg=cell_bg, highlightbackground=BORDER,
                            highlightthickness=1)
            cell.grid(row=i // 3, column=i % 3, sticky='nsew', padx=2, pady=2)
            grid.grid_columnconfigure(i % 3, weight=1)
            tk.Label(cell, text=label, font=F_SMALL, bg=cell_bg,
                     fg=FG_DIM).pack(anchor='w', padx=4, expand=True)
            v = tk.Label(cell, text='--', font=F_VALUE, bg=cell_bg,
                         fg=FG if kind == '决策' else FG_DIM)
            v.pack(anchor='w', padx=4, pady=(0, 2), expand=True)
            self.v_dev[key] = v

    # ── 事件 ──
    def manual_run(self):
        if self.worker.is_running:
            return
        if not run_hourly.BUFFER_PATH.exists():
            self.lbl_status.config(text='冷启动初始化中 ...')
        else:
            self.lbl_status.config(text='计算中 ... (同整点只保留最新)')
        self.worker.run_async()

    def toggle_auto(self):
        self.auto_on = not self.auto_on
        if self.auto_on:
            self.btn_auto.config(text='停止自动', bg='#a05a5a')
            self._schedule_next()
        else:
            self.btn_auto.config(text='启动自动', bg='#4a4a4a')
            self.next_trigger = None
            self.lbl_countdown.config(text='')

    def toggle_log(self):
        self.log_open = not self.log_open
        if self.log_open:
            # 浮层展开: place 到底部日志按钮栏正上方, 不挤占布局(盖在内容之上)
            self.log_frame.place(relx=0, rely=1.0, anchor='sw',
                                 relwidth=1.0, y=-24, height=120)
            self.log_frame.lift()          # 提到最上层, 确保盖住内容
            self.btn_log.config(text='▼ 执行日志')
        else:
            self.log_frame.place_forget()
            self.btn_log.config(text='▲ 执行日志')

    # ── 定时与轮询 ──
    def _schedule_next(self):
        now = datetime.now()
        nxt = (now + timedelta(hours=1)).replace(
            minute=0, second=0, microsecond=0)
        self.next_trigger = nxt

    def _tick_countdown(self):
        if self.auto_on and self.next_trigger:
            remain = (self.next_trigger - datetime.now()).total_seconds()
            if remain <= 0:
                self.manual_run()
                self._schedule_next()
            else:
                h = int(remain // 3600)
                m = int((remain % 3600) // 60)
                s = int(remain % 60)
                self.lbl_countdown.config(
                    text=f'下次触发: {h:02d}:{m:02d}:{s:02d}')
        self._highlight_current_period()
        self.root.after(1000, self._tick_countdown)

    def _poll_queues(self):
        while not self.worker.log_queue.empty():
            self._append_log(self.worker.log_queue.get())
        while not self.worker.result_queue.empty():
            self._on_result(self.worker.result_queue.get())
        self.root.after(200, self._poll_queues)

    def _on_result(self, res):
        if not res['ok']:
            self.lbl_status.config(text=f'失败: {res["error"]}')
            return
        self.lbl_status.config(text='完成')
        log, detail, temps = res['log'], res['detail'], res['temps']

        # 负载/环境 (容错: 旧日志缺列或值为NaN时显示 --)
        for key, widget in self.v_q.items():
            val = log.get(key)
            if val is None or pd.isna(val):
                widget.config(text='--')
            elif key == 'humidity_pct':
                widget.config(text=f'{float(val):.0f}')
            else:
                widget.config(text=f'{float(val):.1f}')

        # 设备(合并 log + detail + temps)
        merged = {**log, **detail, **temps}
        for key, widget in self.v_dev.items():
            val = merged.get(key)
            if val is None:
                widget.config(text='--')
            elif key == 'f_pump':  # % → Hz (100%=50Hz)
                widget.config(text=f'{float(val) * 0.5:.1f}')
            elif key == 's_fan':  # % → m³/h/台 (100%=19423)
                widget.config(text=f'{float(val) / 100 * 19423:.0f}')
            elif isinstance(val, float):
                widget.config(text=f'{val:.2f}' if val < 100 else f'{val:.0f}')
            else:
                widget.config(text=str(val))

        # 电费
        self.v_save_pct.config(text=f'{float(log["saving_pct"]):.1f}%')
        for key, widget in self.v_cost.items():
            widget.config(text=f'{float(log[key]):.1f}')

        self.refresh_chart()

    def _append_log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert('end', f'[{ts}] {msg}\n')
        self.log_text.see('end')

    # ── 趋势图 ──
    def refresh_chart(self):
        self.ax.clear()
        self.ax2.clear()
        try:
            df = pd.read_csv(run_hourly.LOG_PATH).tail(24)
        except Exception:
            df = pd.DataFrame()

        N = 24
        self.ax.set_xlim(-0.5, N - 0.5)          # 固定24格

        if len(df) > 0:
            x = range(len(df))
            ts = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert('Asia/Shanghai')
            hour_labels = [t.strftime('%H') for t in ts]

            self.ax.plot(x, df['Q50_kW'], color=ACCENT_B, lw=2,
                         marker='o', markersize=4, label='负载Q50')
            self.ax.set_ylabel('负载 kW', color=ACCENT_B, fontsize=11)
            self.ax.tick_params(colors=FG_DIM, labelsize=10)
            # 负载Y轴: 下限取整到比最小值更低的刻度
            import numpy as np
            q_min, q_max = df['Q50_kW'].min(), df['Q50_kW'].max()
            if q_max > q_min:
                step = 20  # 负载刻度间隔
                lo = np.floor(q_min / step) * step
                hi = np.ceil(q_max / step) * step 
                self.ax.set_ylim(lo, hi)
                self.ax.set_yticks(np.arange(lo, hi + step, step))

            self.ax2.plot(x, df['saving_yuan'], color=ACCENT, lw=2,
                          marker='o', markersize=4, label='省电')  # ← 用 self.ax2
            self.ax2.set_ylabel('省电 元/h', color=ACCENT, fontsize=11)
            self.ax2.yaxis.set_label_position('right')  # 标签强制右侧
            self.ax2.yaxis.tick_right()  # 刻度强制右侧
            self.ax2.tick_params(colors=FG_DIM, labelsize=10)
            self.ax2.set_facecolor(PANEL)
            # 省电Y轴: 下限取整到比最小值更低的刻度
            import numpy as np
            s_min, s_max = df['saving_yuan'].min(), df['saving_yuan'].max()
            if s_max > s_min:
                step = 50  # 省电刻度间隔
                lo = np.floor(s_min / step) * step
                hi = np.ceil(s_max / step) * step
                self.ax2.set_ylim(lo, hi)
                self.ax2.set_yticks(np.arange(lo, hi + step, step))

            # X轴24个固定刻度, 有数据的位置填小时标签
            full_labels = ['' for _ in range(N)]
            for i, lbl in enumerate(hour_labels):
                if i < N:
                    full_labels[i] = lbl
            self.ax.set_xticks(range(N))
            self.ax.set_xticklabels(full_labels, fontsize=9)
        else:
            self.ax.set_xticks(range(N))
            self.ax.set_xticklabels(['' for _ in range(N)])

        self.ax.set_facecolor(PANEL)
        self.ax.set_xlabel('最近24小时 (北京时间整点)', color=FG_DIM, fontsize=11)
        for spine in self.ax.spines.values():
            spine.set_color(BORDER)
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
