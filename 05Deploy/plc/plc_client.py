"""
plc_client.py
PLC 通信底层封装(连接/读/写)

模式:
  - SIMULATION=True  → 不连真机, 所有写操作仅打印(开发/验证用)
  - SIMULATION=False → 通过 python-snap7 连接真实 S7-200 SMART

S7-200 SMART 连接要点:
  不能用 snap7 标准的 connect(ip, rack, slot)(那是 S7-300/400/1200/1500 的方式),
  必须用 TSAP 方式: set_connection_params(ip, 0x0200, 0x0200) 后再 connect。

写灯原理:
  指示灯地址如 'Q9.5' → 输出过程映像区(PA区) 第9字节第5位。
  写单个位采用"读该字节→改目标位→写回"(read-modify-write),
  避免覆盖同字节其它位(同字节可能挂着别的灯)。

依赖: pip install python-snap7  (仅真实模式需要; 模拟模式零依赖)
"""

import plc_config as cfg


def _parse_q_address(address: str):
    """'Q9.5' → (byte=9, bit=5)。非法格式抛 ValueError"""
    addr = address.strip().upper()
    if not addr.startswith('Q') or '.' not in addr:
        raise ValueError(f'非法输出点地址: {address} (应形如 Q9.5)')
    byte_s, bit_s = addr[1:].split('.', 1)
    byte_idx, bit_idx = int(byte_s), int(bit_s)
    if not (0 <= bit_idx <= 7):
        raise ValueError(f'位号越界: {address} (bit 应为 0~7)')
    return byte_idx, bit_idx


class PLCClient:
    """S7-200 SMART 通信封装. 模拟模式下仅打印, 不连真机"""

    def __init__(self):
        self.connected = False
        self._client = None

    # ── 连接 ──────────────────────────────────────────
    def connect(self):
        if cfg.SIMULATION:
            self.connected = True
            print(f'[模拟] PLC 已连接 ({cfg.PLC_IP})')
            return True

        # ── 真实模式: snap7 + TSAP(S7-200 SMART 专用连法) ──
        import snap7
        self._snap7 = snap7
        self._client = snap7.client.Client()
        # S7-200 SMART: 本地/远程 TSAP 均为 0x0200
        self._client.set_connection_params(cfg.PLC_IP, 0x0200, 0x0200)
        self._client.connect(cfg.PLC_IP, cfg.PLC_RACK, cfg.PLC_SLOT)
        self.connected = self._client.get_connected()
        if self.connected:
            print(f'PLC 已连接 ({cfg.PLC_IP})')
        else:
            print(f'PLC 连接失败 ({cfg.PLC_IP})')
        return self.connected

    def disconnect(self):
        if cfg.SIMULATION:
            self.connected = False
            return
        if self._client is not None:
            try:
                self._client.disconnect()
            finally:
                self.connected = False

    def _ensure_connected(self):
        """真实模式下掉线自动重连一次"""
        if self.connected and self._client is not None \
                and self._client.get_connected():
            return True
        return self.connect()

    # ── 写输出点(指示灯) ──────────────────────────────
    def write_bit(self, address, value):
        """
        写单个输出点(指示灯亮/灭)
        address: 如 'Q9.5'; value: True=亮 / False=灭
        真实模式: PA区 read-modify-write, 不影响同字节其它位
        """
        if cfg.SIMULATION:
            state = '亮' if value else '灭'
            print(f'[模拟] 写 {address} = {state}')
            return True

        byte_idx, bit_idx = _parse_q_address(address)
        self._ensure_connected()
        # snap7 2.x: Areas 在 snap7.type; 旧版在 snap7.types — 双兼容
        try:
            from snap7.type import Areas  # 新版 2.x
        except ImportError:
            from snap7.types import Areas  # 旧版 0.x/1.x
        from snap7.util import set_bool

        data = self._client.read_area(Areas.PA, 0, byte_idx, 1)
        set_bool(data, 0, bit_idx, bool(value))
        self._client.write_area(Areas.PA, 0, byte_idx, data)
        return True

    # ── 读寄存器(功耗等, 当前不启用) ──────────────────
    def read_register(self, address):
        """读输入寄存器(真实功耗等). 功能未启用, 统一返回 None"""
        return None


# ── 模块级单例 ────────────────────────────────────────
_client = None

def get_client():
    global _client
    if _client is None:
        _client = PLCClient()
        _client.connect()
    return _client


# ── 独立测试 ──────────────────────────────────────────
if __name__ == '__main__':
    # 地址解析自检(不需连接)
    for a, expect in [('Q9.5', (9, 5)), ('Q8.0', (8, 0)), ('q9.7', (9, 7))]:
        got = _parse_q_address(a)
        assert got == expect, (a, got, expect)
    print('地址解析自检通过: Q9.5→(9,5), Q8.0→(8,0), q9.7→(9,7)')

    c = get_client()
    print('\n=== write_bit 测试(当前模式:',
          '模拟' if cfg.SIMULATION else '真实', ') ===')
    c.write_bit('Q9.5', True)
    c.write_bit('Q9.5', False)