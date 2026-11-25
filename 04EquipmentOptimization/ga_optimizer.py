"""
ga_optimizer.py
改进遗传算法优化冷源系统: 自适应交叉变异 + Q90鲁棒约束 + 切换惩罚
染色体: [N_chiller, f_pump, N_cdz, N_FC, s_fan]
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '05Deploy'))
from runtime_paths import PROJECT_ROOT
ROOT = str(PROJECT_ROOT)
OPT_DIR = os.path.join(ROOT, '04EquipmentOptimization')
WX_PATH = os.path.join(ROOT, '01dataFresh', 'weather.parquet')

# GA 超参数
POP_SIZE   = 60
N_GEN      = 30
TOURNAMENT = 3
ELITE_SIZE = 2

# 自适应交叉/变异率的 4 个常数(Srinivas 1994)
K1, K2, K3, K4 = 1.0, 0.5, 0.8, 0.2

# 惩罚系数
LAMBDA_SUPPLY = 50.0   # 供给不足惩罚(元/kW)
LAMBDA_SWITCH = 5.0    # 切换惩罚(元/次)

# 决策变量边界 [min, max]
BOUNDS = {
    'N_chiller': (1, 4),
    'f_pump':    (30.0, 100.0),
    'N_cdz':     (1, 6),
    'N_FC':      (0, 4),
    's_fan':     (60.0, 100.0),
}
DISCRETE_IDX = [0, 2, 3]   # N_chiller, N_cdz, N_FC 是整数
N_VARS = 5

# 物理常数
RHO_AIR = 1.2
CP_AIR  = 1.005

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── PART 1: 加载设备模型 ──────────────────────────────────

def load_models() -> dict:
    print('加载设备模型 ...')

    with open(os.path.join(OPT_DIR, 'chiller_results', 'chiller_model.pkl'), 'rb') as f:
        chiller = pickle.load(f)
    with open(os.path.join(OPT_DIR, 'pump_results', 'pump_model.pkl'), 'rb') as f:
        pump = pickle.load(f)
    with open(os.path.join(OPT_DIR, 'cdz_results', 'cdz_model.pkl'), 'rb') as f:
        cdz = pickle.load(f)

    models = {
        'chiller': chiller,
        'pump': pump,
        'cdz': cdz,
        'Q_rated':       chiller['physics']['Q_rated'],
        'V_rated':       cdz['physics']['V_rated'],
        'T_ret_const':   cdz['constants']['T_return_const'],
        'T_sup_DX':      cdz['constants']['T_supply_DX_const'],
    }
    print(f'  冷水机: Q_rated={models["Q_rated"]:.0f} kW, N_total={chiller["physics"]["N_total"]}')
    print(f'  空调:   V_rated={models["V_rated"]:.0f} m³/h, T_ret={models["T_ret_const"]}°C')
    return models


# ── PART 2: CDZ 制冷量物理公式 ────────────────────────────

def compute_q_air(N_cdz, N_FC, s_fan, T_amb, models):
    """物理公式计算 Q_air(不是模型预测)"""
    V = models['V_rated']
    T_ret = models['T_ret_const']
    T_sup = models['T_sup_DX']
    N_DX = N_cdz - N_FC

    Q_FC = N_FC * V / 3600 * RHO_AIR * CP_AIR * max(0, T_ret - T_amb)
    Q_DX = N_DX * V / 3600 * RHO_AIR * CP_AIR * max(0, T_ret - T_sup)
    return Q_FC + Q_DX


# ── PART 3: 适应度函数 ────────────────────────────────────

def fitness(individual, context, models):
    """
    评估一个候选解的总电费(越小越好)

    context = {
        'Tot_ict_Q50': float,     Q50 预测(或实际值)
        'Tot_ict_Q90': float,     Q90 预测(鲁棒约束用)
        'T_amb': float,
        'price': float,           当前小时电价(元/kWh)
        'prev_N_chiller': int,    上一小时的冷水机台数(切换惩罚用)
    }
    """
    N_chiller = int(round(individual[0]))
    f_pump    = individual[1]
    N_cdz     = int(round(individual[2]))
    N_FC      = int(round(individual[3]))
    s_fan     = individual[4]

    T_amb = context['T_amb']
    Q50   = context['Tot_ict_Q50']
    Q90   = context['Tot_ict_Q90']
    price = context['price']

    # ── 约束修复(确保可行) ──
    N_FC = min(N_FC, 4, N_cdz)
    if T_amb >= models['T_ret_const'] - 2:
        N_FC = 0

    # ── Step 1: 空调侧 ──
    Q_air = compute_q_air(N_cdz, N_FC, s_fan, T_amb, models)

    cdz_m = models['cdz']['unified_model']
    X_cdz = np.array([[Q_air, T_amb, N_cdz, N_FC, s_fan]])
    X_cdz_poly = cdz_m['poly'].transform(X_cdz)
    P_cdz = max(0, cdz_m['model'].predict(X_cdz_poly)[0])

    # ── Step 2: 热平衡 ──
    Q_liquid = max(0, Q50 - Q_air)
    Q_liquid_Q90 = max(0, Q90 - Q_air)

    # ── Step 3: 供给不足惩罚(Q90 鲁棒) ──
    Q_cap = N_chiller * models['Q_rated']
    penalty_supply = 0
    if Q_liquid_Q90 > Q_cap:
        penalty_supply = LAMBDA_SUPPLY * (Q_liquid_Q90 - Q_cap)

    # ── Step 4: 冷水机 ──
    if Q_liquid > 0 and Q_cap > 0:
        PLR = min(Q_liquid / Q_cap, 1.5)
        ch = models['chiller']['cop_model']
        X_ch = np.array([[PLR, T_amb]])
        X_ch_poly = ch['poly'].transform(X_ch)
        COP = max(0.5, ch['model'].predict(X_ch_poly)[0])
        P_chiller = Q_liquid / COP
    else:
        P_chiller = 0

    # ── Step 5: 水泵 ──
    Portata = Q_liquid * 0.1435
    pm = models['pump']['model']
    X_pump = np.array([[f_pump, Portata]])
    X_pump_poly = pm['poly'].transform(X_pump)
    P_pump = max(0, pm['model'].predict(X_pump_poly)[0])

    # ── Step 6: 总电费 + 惩罚 ──
    P_total = P_chiller + P_pump + P_cdz
    cost = P_total * price

    # 切换惩罚
    penalty_switch = 0
    prev = context.get('prev_N_chiller')
    if prev is not None:
        penalty_switch = LAMBDA_SWITCH * abs(N_chiller - prev)

    return cost + penalty_supply + penalty_switch,

def evaluate_detail(individual, context, models):
    """复用 fitness 物理公式, 返回完整中间量(供前端展示, 不参与优化)"""
    N_chiller = int(round(individual[0]))
    f_pump    = individual[1]
    N_cdz     = int(round(individual[2]))
    N_FC      = int(round(individual[3]))
    s_fan     = individual[4]

    T_amb = context['T_amb']
    Q50   = context['Tot_ict_Q50']

    N_FC = min(N_FC, 4, N_cdz)
    if T_amb >= models['T_ret_const'] - 2:
        N_FC = 0

    # 空调侧
    Q_air = compute_q_air(N_cdz, N_FC, s_fan, T_amb, models)
    cdz_m = models['cdz']['unified_model']
    X_cdz = np.array([[Q_air, T_amb, N_cdz, N_FC, s_fan]])
    P_cdz = max(0, cdz_m['model'].predict(cdz_m['poly'].transform(X_cdz))[0])

    # 热平衡
    Q_liquid = max(0, Q50 - Q_air)
    Q_cap = N_chiller * models['Q_rated']

    # 冷水机
    if Q_liquid > 0 and Q_cap > 0:
        PLR = min(Q_liquid / Q_cap, 1.5)
        ch = models['chiller']['cop_model']
        X_ch = np.array([[PLR, T_amb]])
        COP = max(0.5, ch['model'].predict(ch['poly'].transform(X_ch))[0])
        P_chiller = Q_liquid / COP
    else:
        PLR = 0.0
        COP = 0.0
        P_chiller = 0.0

    # 水泵
    Portata = Q_liquid * 0.1435
    pm = models['pump']['model']
    X_pump = np.array([[f_pump, Portata]])
    P_pump = max(0, pm['model'].predict(pm['poly'].transform(X_pump))[0])

    return {
        'N_chiller':  N_chiller,
        'PLR':        round(PLR, 3),
        'COP':        round(COP, 3),
        'P_chiller':  round(P_chiller, 1),
        'Q_liquid':   round(Q_liquid, 1),
        'Q_cap':      round(Q_cap, 1),
        'N_cdz':      N_cdz,
        'N_FC':       N_FC,
        'P_cdz':      round(P_cdz, 1),
        'Q_air':      round(Q_air, 1),
        'f_pump':     round(f_pump, 1),
        'Portata':    round(Portata, 1),
        'P_pump':     round(P_pump, 1),
        'P_total':    round(P_chiller + P_pump + P_cdz, 1),
    }

def fitness_batch(pop, context, models):
    """向量化评估整个种群"""
    n = len(pop)
    arr = np.array(pop)

    N_chiller = np.round(arr[:, 0]).astype(int).clip(1, 4)
    f_pump    = arr[:, 1]
    N_cdz     = np.round(arr[:, 2]).astype(int).clip(1, 6)
    N_FC      = np.round(arr[:, 3]).astype(int).clip(0, 4)
    s_fan     = arr[:, 4]

    T_amb = context['T_amb']
    Q50   = context['Tot_ict_Q50']
    Q90   = context['Tot_ict_Q90']
    price = context['price']

    # 约束修复
    N_FC = np.minimum(N_FC, np.minimum(4, N_cdz))
    if T_amb >= models['T_ret_const'] - 2:
        N_FC[:] = 0
    N_DX = N_cdz - N_FC

    # CDZ 制冷量(向量化物理公式)
    V = models['V_rated']
    T_ret = models['T_ret_const']
    T_sup = models['T_sup_DX']
    Q_FC = N_FC * V / 3600 * RHO_AIR * CP_AIR * max(0, T_ret - T_amb)
    Q_DX = N_DX * V / 3600 * RHO_AIR * CP_AIR * max(0, T_ret - T_sup)
    Q_air = Q_FC + Q_DX

    # CDZ 功耗(批量 predict)
    cdz_m = models['cdz']['unified_model']
    X_cdz = np.column_stack([Q_air, np.full(n, T_amb), N_cdz, N_FC, s_fan])
    P_cdz = np.clip(cdz_m['model'].predict(cdz_m['poly'].transform(X_cdz)), 0, None)

    # 热平衡
    Q_liquid = np.clip(Q50 - Q_air, 0, None)
    Q_liquid_Q90 = np.clip(Q90 - Q_air, 0, None)

    # 供给不足惩罚
    Q_cap = N_chiller * models['Q_rated']
    penalty_supply = np.where(
        Q_liquid_Q90 > Q_cap,
        LAMBDA_SUPPLY * (Q_liquid_Q90 - Q_cap),
        0
    )

    # 冷水机(批量 predict)
    PLR = np.where(Q_cap > 0, np.clip(Q_liquid / Q_cap, 0.01, 1.5), 0.5)
    ch = models['chiller']['cop_model']
    X_ch = np.column_stack([PLR, np.full(n, T_amb)])
    COP = np.clip(ch['model'].predict(ch['poly'].transform(X_ch)), 0.5, 15)
    P_chiller = np.where(Q_liquid > 0, Q_liquid / COP, 0)

    # 水泵(批量 predict)
    Portata = Q_liquid * 0.1435
    pm = models['pump']['model']
    X_pump = np.column_stack([f_pump, Portata])
    P_pump = np.clip(pm['model'].predict(pm['poly'].transform(X_pump)), 0, None)

    # 总电费
    P_total = P_chiller + P_pump + P_cdz
    cost = P_total * price + penalty_supply

    # 切换惩罚
    prev = context.get('prev_N_chiller')
    if prev is not None:
        cost += LAMBDA_SWITCH * np.abs(N_chiller - prev)

    return cost.tolist()


# ── PART 4: GA 算子 ───────────────────────────────────────

def random_individual():
    bounds = list(BOUNDS.values())
    ind = [random.uniform(lo, hi) for lo, hi in bounds]
    for i in DISCRETE_IDX:
        ind[i] = round(ind[i])
    return ind


def tournament_select(pop, fits, k=TOURNAMENT):
    candidates = random.sample(range(len(pop)), k)
    best = min(candidates, key=lambda i: fits[i])
    return deepcopy(pop[best])


def sbx_crossover(p1, p2, eta=20):
    """模拟二进制交叉(SBX)"""
    c1, c2 = deepcopy(p1), deepcopy(p2)
    bounds = list(BOUNDS.values())
    for i in range(N_VARS):
        if random.random() > 0.5:
            continue
        lo, hi = bounds[i]
        if abs(p1[i] - p2[i]) < 1e-10:
            continue
        u = random.random()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (eta + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        c1[i] = max(lo, min(hi, c1[i]))
        c2[i] = max(lo, min(hi, c2[i]))
    return c1, c2


def polynomial_mutate(ind, eta=20):
    """多项式变异"""
    bounds = list(BOUNDS.values())
    for i in range(N_VARS):
        lo, hi = bounds[i]
        u = random.random()
        if u < 0.5:
            delta = (2 * u) ** (1 / (eta + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        ind[i] = ind[i] + delta * (hi - lo)
        ind[i] = max(lo, min(hi, ind[i]))
    return ind


def adaptive_rates(fits, f_parent, f_individual):
    """Srinivas 1994 自适应交叉/变异率"""
    f_max = max(fits)
    f_avg = np.mean(fits)

    if f_max - f_avg < 1e-10:
        return K3, K4

    if f_parent >= f_avg:
        Pc = K1 * (f_max - f_parent) / (f_max - f_avg)
    else:
        Pc = K3

    if f_individual >= f_avg:
        Pm = K2 * (f_max - f_individual) / (f_max - f_avg)
    else:
        Pm = K4

    return min(Pc, 1.0), min(Pm, 1.0)


def fix_discrete(ind):
    for i in DISCRETE_IDX:
        lo, hi = list(BOUNDS.values())[i]
        ind[i] = max(lo, min(hi, round(ind[i])))
    return ind


# ── PART 5: 单小时优化 ───────────────────────────────────

def optimize_hour(context, models):
    """
    对单个小时运行 GA,返回最优解和最优电费

    返回: (best_individual, best_fitness, gen_history)
    """
    pop = [random_individual() for _ in range(POP_SIZE)]
    fits = fitness_batch(pop, context, models)

    best_ever = deepcopy(pop[np.argmin(fits)])
    best_fit_ever = min(fits)
    gen_history = [best_fit_ever]

    for gen in range(N_GEN):
        # 精英保留
        elite_idx = np.argsort(fits)[:ELITE_SIZE]
        new_pop = [deepcopy(pop[i]) for i in elite_idx]

        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(pop, fits)
            p2 = tournament_select(pop, fits)

            # # 原来(每对父母重新算 fitness)
            # f_parent = max(fitness(p1, context, models)[0],
            #                fitness(p2, context, models)[0])
            # Pc, Pm = adaptive_rates(fits, f_parent, f_parent)
            # 用种群统计值代替逐对重算(每代已有 fits 列表)
            Pc, Pm = adaptive_rates(fits, fits[np.argmin(fits)], np.mean(fits))

            if random.random() < Pc:
                c1, c2 = sbx_crossover(p1, p2)
            else:
                c1, c2 = deepcopy(p1), deepcopy(p2)

            if random.random() < Pm:
                c1 = polynomial_mutate(c1)
            if random.random() < Pm:
                c2 = polynomial_mutate(c2)

            new_pop.extend([fix_discrete(c1), fix_discrete(c2)])

        pop = new_pop[:POP_SIZE]
        fits = fitness_batch(pop, context, models)

        gen_best = min(fits)
        if gen_best < best_fit_ever:
            best_fit_ever = gen_best
            best_ever = deepcopy(pop[np.argmin(fits)])
        gen_history.append(best_fit_ever)

    best_ever = fix_discrete(best_ever)
    # SA 局部搜索(对 GA 最优解做精细调优)
    T_sa = 10.0
    T_min = 0.1
    cooling = 0.9
    current = deepcopy(best_ever)
    current_fit = best_fit_ever

    while T_sa > T_min:
        neighbor = deepcopy(current)
        # 随机扰动一个变量
        idx = random.randint(0, N_VARS - 1)
        lo, hi = list(BOUNDS.values())[idx]
        neighbor[idx] += random.gauss(0, (hi - lo) * 0.05)
        neighbor[idx] = max(lo, min(hi, neighbor[idx]))
        neighbor = fix_discrete(neighbor)

        new_fit = fitness(neighbor, context, models)[0]
        delta = new_fit - current_fit

        if delta < 0 or random.random() < np.exp(-delta / T_sa):
            current = neighbor
            current_fit = new_fit
            if current_fit < best_fit_ever:
                best_ever = deepcopy(current)
                best_fit_ever = current_fit

        T_sa *= cooling
    return best_ever, best_fit_ever, gen_history


# ── PART 6: 历史回测 ──────────────────────────────────────

def backtest(n_hours=None):
    """
    历史数据回测: GA 优化 vs 实际运行

    n_hours: 回测小时数(None=全部, 建议先用 200~500 测试)
    """
    print('\n[回测] 加载数据 ...')
    models = load_models()

    # 加载历史数据
    df = pd.read_parquet(os.path.join(OPT_DIR, 'cooling_with_modes.parquet'))
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # 合并天气
    wx = pd.read_parquet(WX_PATH)
    wx['timestamp'] = pd.to_datetime(wx['timestamp'], utc=True)
    t_col = next(c for c in wx.columns if c.lower() in ('temp_c', 't_amb'))
    wx = wx[['timestamp', t_col]].rename(columns={t_col: 'temp_C'})
    df = df.merge(wx, on='timestamp', how='inner')

    # 添加电价
    from electricity_pricing import add_prices
    df = add_prices(df)

    # 过滤有效数据
    df = df[(df['Tot_ict_kW'] > 100) & (df['Tot_cdz_kW'] > 0)].reset_index(drop=True)
    if n_hours:
        df = df.head(n_hours)
    print(f'  回测小时数: {len(df):,}')

    results = []
    prev_N_chiller = None

    for i, row in df.iterrows():
        Q50 = row['Tot_ict_kW']
        Q90 = Q50 * 1.05    # 回测用实际值 + 5% 裕度模拟 Q90
        T_amb = row['temp_C']
        price = row['price_yuan']

        context = {
            'Tot_ict_Q50': Q50,
            'Tot_ict_Q90': Q90,
            'T_amb': T_amb,
            'price': price,
            'prev_N_chiller': prev_N_chiller,
        }

        best_ind, best_cost, _ = optimize_hour(context, models)
        prev_N_chiller = int(round(best_ind[0]))

        # 实际功耗和电费
        P_actual = row['Tot_chiller_kW'] + row['Tot_qpompe_kW'] + row['Tot_cdz_kW']
        cost_actual = P_actual * price

        results.append({
            'timestamp':    row['timestamp'],
            'P_actual':     P_actual,
            'cost_actual':  cost_actual,
            'P_optimal':    best_cost / price if price > 0 else 0,
            'cost_optimal': best_cost,
            'N_chiller':    int(round(best_ind[0])),
            'f_pump':       best_ind[1],
            'N_cdz':        int(round(best_ind[2])),
            'N_FC':         int(round(best_ind[3])),
            's_fan':        best_ind[4],
            'T_amb':        T_amb,
            'price':        price,
        })

        if (i + 1) % 50 == 0:
            savings = 1 - sum(r['cost_optimal'] for r in results) / sum(r['cost_actual'] for r in results)
            print(f'  {i+1}/{len(df)}  累计节费率: {savings*100:.1f}%')

    df_res = pd.DataFrame(results)

    # 汇总
    total_actual  = df_res['cost_actual'].sum()
    total_optimal = df_res['cost_optimal'].sum()
    savings_pct   = (1 - total_optimal / total_actual) * 100

    P_actual_mean  = df_res['P_actual'].mean()
    P_optimal_mean = df_res['P_optimal'].mean()
    power_save_pct = (1 - P_optimal_mean / P_actual_mean) * 100

    print(f'\n{"="*50}')
    print(f'回测结果({len(df_res)} 小时)')
    print(f'{"="*50}')
    print(f'实际总电费:   {total_actual:,.0f} 元')
    print(f'优化总电费:   {total_optimal:,.0f} 元')
    print(f'节费:         {total_actual - total_optimal:,.0f} 元 ({savings_pct:.1f}%)')
    print(f'实际平均功耗: {P_actual_mean:.1f} kW')
    print(f'优化平均功耗: {P_optimal_mean:.1f} kW')
    print(f'节电:         {power_save_pct:.1f}%')

    # 保存结果
    out_path = os.path.join(OPT_DIR, 'backtest_results.parquet')
    df_res.to_parquet(out_path, index=False)
    print(f'结果已保存: {out_path}')

    # 对比图
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax1 = axes[0]
    ax1.plot(df_res['cost_actual'].values, lw=0.5, alpha=0.7, label='实际电费')
    ax1.plot(df_res['cost_optimal'].values, lw=0.5, alpha=0.7, label='优化电费')
    ax1.set_ylabel('电费 (元/h)')
    ax1.set_title(f'GA 优化 vs 实际  节费 {savings_pct:.1f}%')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    savings_ts = (df_res['cost_actual'] - df_res['cost_optimal']).values
    ax2.fill_between(range(len(savings_ts)), savings_ts, alpha=0.5, color='green')
    ax2.set_xlabel('小时')
    ax2.set_ylabel('节省 (元/h)')
    ax2.set_title('逐小时节省')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OPT_DIR, 'backtest_comparison.png'), bbox_inches='tight')
    plt.close(fig)
    print(f'对比图: backtest_comparison.png')

    return df_res


# ── main ──────────────────────────────────────────────────

if __name__ == '__main__':
    # import sys
    # if len(sys.argv) > 1 and sys.argv[1] == 'backtest':
    #     n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    #     backtest(n_hours=n)
    # else:
    #     # 单小时演示
    #     models = load_models()
    #     context = {
    #         'Tot_ict_Q50': 850,
    #         'Tot_ict_Q90': 900,
    #         'T_amb': 25.0,
    #         'price': 1.2287,
    #         'prev_N_chiller': None,
    #     }
    #     best, cost, history = optimize_hour(context, models)
    #     print(f'\n最优解:')
    #     print(f'  N_chiller = {int(round(best[0]))} 台冷水机')
    #     print(f'  f_pump    = {best[1]:.1f}%    水泵频率')
    #     print(f'  N_cdz     = {int(round(best[2]))} 台空调')
    #     print(f'  N_FC      = {int(round(best[3]))} 台FreeCooling空调')
    #     print(f'  s_fan     = {best[4]:.1f}%    风扇转速')
    #     print(f'  总电费    = {cost:.2f} 元/h')
    backtest(n_hours=1000)