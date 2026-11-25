"""
worker.py
后台计算线程: 在子线程跑 run_once, 结果经队列传回主线程
避免 GA 优化 + 预测阻塞 GUI
"""
import sys
import queue
import threading
from pathlib import Path

import pandas as pd

GUI_DIR    = Path(__file__).resolve().parent
DEPLOY_DIR = GUI_DIR.parent

sys.path.insert(0, str(DEPLOY_DIR))
sys.path.insert(0, str(GUI_DIR))

import run_hourly
import state_reader


class ComputeWorker:
    """封装后台计算: 一次只跑一个任务, 结果/状态经队列回传"""

    def __init__(self):
        self.result_queue = queue.Queue()   # 计算结果
        self.log_queue    = queue.Queue()   # 状态消息
        self._running = False

    @property
    def is_running(self):
        return self._running

    def run_async(self):
        """启动后台线程(若已有任务在跑则忽略)"""
        if self._running:
            self.log_queue.put('上一个任务尚未完成, 已忽略本次触发')
            return
        self._running = True
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        """子线程主体: 跑 run_once → 读日志 → 组装结果"""
        try:
            self.log_queue.put('开始计算 ...')
            run_hourly.run_once()   # print 输出到控制台

            # run_once 已写入 hourly_log.csv, 读最后一行
            log_row = self._read_last_log()

            # 反算设备详情 + 参考温度
            models = run_hourly.get_models()
            detail = state_reader.get_device_detail(log_row, models)
            ts = pd.Timestamp(log_row['timestamp'])
            temps = state_reader.get_reference_temps(ts)

            self.result_queue.put({
                'ok':     True,
                'log':    log_row,      # 15个日志字段
                'detail': detail,       # COP/PLR/功耗等
                'temps':  temps,        # 供回水参考温度
            })
            self.log_queue.put('计算完成 ✓')
        except Exception as e:
            self.log_queue.put(f'计算失败: {e}')
            self.result_queue.put({'ok': False, 'error': str(e)})
        finally:
            self._running = False

    def _read_last_log(self):
        """读 hourly_log.csv 最后一行为 dict"""
        df = pd.read_csv(run_hourly.LOG_PATH)
        if len(df) == 0:
            raise RuntimeError('日志为空, run_once 可能未成功写入')
        return df.iloc[-1].to_dict()


if __name__ == '__main__':
    import time

    worker = ComputeWorker()
    worker.run_async()

    while worker.is_running or not worker.result_queue.empty():
        while not worker.log_queue.empty():
            print('[状态]', worker.log_queue.get())
        if not worker.result_queue.empty():
            res = worker.result_queue.get()
            print('\n[结果]', '成功' if res['ok'] else '失败')
            if res['ok']:
                print('  省电:', res['log'].get('saving_pct'), '%')
                print('  COP:', res['detail'].get('COP'))
                print('  供水温度:', res['temps'].get('T_mandata'))
            break
        time.sleep(0.2)