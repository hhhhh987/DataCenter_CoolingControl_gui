"""
plc_config.py
PLC 连接参数 + 指示灯地址映射
硬件: SIMATIC S7-200 SMART (CPU ST20 + EM QR16 继电器输出)

地址说明:
  - 13 路灯全部外接在 EM QR16 扩展模块上, 以下为实际接线地址(已现场确认)。
  - 实际接线顺序是乱的(如冷水机#1 在 Q9.5、#4 在 Q8.0), 这是接线事实;
    列表顺序 = 设备编号顺序(#1→#N), 与界面/算法编号一一对应,
    update_lamps 按"开 n 台点亮列表前 n 个"工作, 因此列表顺序不可按地址重排!
  - 系统运行灯已取消(PLC 侧无此灯); 界面上的红色"系统运行"为纯软件指示。
"""

# ── 连接参数: 从 exe/项目旁的 plc_settings.json 读取 ──
import os
import sys
import json

def _settings_path():
    """打包后: exe 所在目录; 开发时: 05Deploy 目录(本文件上一级)"""
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, 'plc_settings.json')

# 默认值(安全兜底: 读不到配置 → 模拟模式, 绝不乱连)
SIMULATION = True
PLC_IP     = '192.168.2.1'
try:
    with open(_settings_path(), encoding='utf-8') as _f:
        _s = json.load(_f)
    SIMULATION = bool(_s.get('simulation', True))
    PLC_IP     = str(_s.get('plc_ip', PLC_IP))
except Exception:
    pass    # 用上方默认值

PLC_RACK = 0
PLC_SLOT = 1

# ── 指示灯输出地址映射(EM QR16, 实际接线地址) ─────────
# 列表顺序 = 设备编号顺序, 勿按地址大小重排

LAMP_CHILLER = ['Q9.5',   # 冷水机#1
                'Q9.6',   # 冷水机#2
                'Q9.7',   # 冷水机#3
                'Q8.0']   # 冷水机#4

LAMP_CDZ     = ['Q8.3',   # 空调 CDZ#1
                'Q8.4',   # 空调 CDZ#2
                'Q8.5',   # 空调 CDZ#3
                'Q9.2',   # 空调 CDZ#4
                'Q9.3',   # 空调 CDZ#5
                'Q9.4']   # 空调 CDZ#6

LAMP_PUMP    = ['Q8.1',   # 冷冻水泵1
                'Q8.2']   # 冷冻水泵2

# ── 真实功耗读取地址(不启用, 保留占位) ────────────────
REG_POWER_CHILLER = None
REG_POWER_CDZ     = None
REG_POWER_PUMP    = None