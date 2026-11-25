"""
plc_mapping.py
业务语义层: 算法推荐值 → 指示灯信号; PLC → 真实功耗
上层(app/data_collector)只调用本文件, 不直接碰 plc_client

系统灯说明: PLC 侧已取消系统运行灯(硬件无此灯)。
update_lamps 的 system_on 参数保留仅为接口兼容(app 调用处不用改),
本层不再向 PLC 写系统灯; 界面上的红色"系统运行"由 device_panel 自行显示。
"""

import plc_config as cfg
from plc_client import get_client


def update_lamps(n_chiller, n_cdz, n_pump=2, system_on=True):
    """
    根据推荐设备台数点亮对应指示灯(开一台亮一个)
    n_chiller: 冷水机台数 → 点亮 LAMP_CHILLER 前 n_chiller 个
    n_cdz:     空调台数   → 点亮 LAMP_CDZ 前 n_cdz 个
    n_pump:    水泵台数   → 点亮 LAMP_PUMP 前 n_pump 个(固定2)
    system_on: 仅接口兼容, PLC 无系统灯, 本参数在此层被忽略
    注: 各 LAMP_* 列表按设备编号排序(接线地址是乱序的), 勿改顺序。
    """
    client = get_client()

    # 冷水机: 前 n_chiller 个亮, 其余灭
    for i, addr in enumerate(cfg.LAMP_CHILLER):
        client.write_bit(addr, i < n_chiller)

    # 空调
    for i, addr in enumerate(cfg.LAMP_CDZ):
        client.write_bit(addr, i < n_cdz)

    # 水泵
    for i, addr in enumerate(cfg.LAMP_PUMP):
        client.write_bit(addr, i < n_pump)


def all_lamps_off():
    """熄灭所有指示灯(系统停止时)"""
    client = get_client()
    for addr in (cfg.LAMP_CHILLER + cfg.LAMP_CDZ + cfg.LAMP_PUMP):
        client.write_bit(addr, False)


def read_real_power():
    """
    从 PLC 读真实功耗(不启用, 保留接口)
    模拟模式或地址未配置时返回 None → 上层走历史回放
    """
    if cfg.SIMULATION or cfg.REG_POWER_CHILLER is None:
        return None
    return None
def test_single_lamp(kind, index, on):
    """
    巡检用: 单独点亮/熄灭某一个灯
    kind: 'chiller' / 'cdz' / 'pump'; index: 0 起; on: True亮/False灭
    """
    table = {'chiller': cfg.LAMP_CHILLER,
             'cdz':     cfg.LAMP_CDZ,
             'pump':    cfg.LAMP_PUMP}
    lamps = table.get(kind)
    if lamps is None or not (0 <= index < len(lamps)):
        return
    get_client().write_bit(lamps[index], bool(on))

# ── 测试 ───────────────────────────────────────────────

if __name__ == '__main__':
    print('=== 模拟点灯测试: 冷水机2台, 空调6台, 水泵2台 ===')
    print('(预期: 冷水机亮 Q9.5/Q9.6, 空调全亮, 水泵亮 Q8.1/Q8.2)')
    update_lamps(n_chiller=2, n_cdz=6, n_pump=2)
    print('\n=== 全部熄灭 ===')
    all_lamps_off()