import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# 设置随机种子
np.random.seed(2050)
random.seed(2050)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 核心模型: 包含物理参数、扰动与应急逻辑
# ==========================================
class RobustLogisticsModel:
    def __init__(self):
        # --- 全局目标 ---
        self.target_payload = 1e8 * 1000  # 目标: 1亿吨 (kg)
        self.max_years = 120              # 仿真时间窗口
        
        self.ref_cost = 5e12              # 归一化参考成本 $5T
        self.ref_time = 40                # 归一化参考时间 40年
        self.alpha = 0.5                  # 成本与时间的权衡权重
        
        # --- 太空电梯 (SE) 参数 ---
        self.se_cap_design = 179000 * 3 * 1000 # 3个港口设计年运力 (kg)
        
        # 成本细项
        self.se_c_base_maint = 2.4e8      # 基础年维护费 ($2.4B)
        self.se_c_wear_max = 5.0e8        # 满负荷下的额外磨损费 ($5B)
        self.se_c_unit_ops = 50           # 初始运营单价 $/kg
        self.se_gamma = 2.5
        
        # 扰动与灾难参数
        self.p_snap = 0.003               # 年断裂概率
        self.c_snap = 6.0e10              # 断裂重建费 ($60B)
        self.t_snap = 270                 # 断裂停运天数
        
        # 物理扰动因子 (Coriolis & Tether Perturbation)
        self.beta_storm = 0.95            # 太阳风暴可用率
        self.eta_cor_mean = 0.90          # 科里奥利力导致的平均效率
        self.eta_tether_mean = 0.92       # 缆绳摆动导致的平均效率
        
        # --- 赖特定律 (学习曲线) ---
        self.q0_se = 2.5e6 * 1000
        self.q0_rocket = 5e7 * 1000
        self.lr_se = 0.15
        self.lr_rocket = 0.20

        # --- 十大火箭发射场数据 (Data from Excel) ---
        self.sites_data = [
            {'name': 'India (Satish)', 'pl': 145000, 'vc': 300*145000, 'fc': 150e6, 'L': 4},
            {'name': 'China (Taiyuan)', 'pl': 125000, 'vc': 320*125000, 'fc': 200e6, 'L': 5},
            {'name': 'USA (Texas)',    'pl': 145000, 'vc': 320*145000, 'fc': 350e6, 'L': 6},
            {'name': 'USA (Florida)',  'pl': 140000, 'vc': 350*140000, 'fc': 400e6, 'L': 8},
            {'name': 'Kazakhstan',     'pl': 120000, 'vc': 380*120000, 'fc': 250e6, 'L': 4},
            {'name': 'New Zealand',    'pl': 125000, 'vc': 400*125000, 'fc': 100e6, 'L': 3},
            {'name': 'Fr. Guiana',     'pl': 150000, 'vc': 450*150000, 'fc': 300e6, 'L': 3},
            {'name': 'USA (Calif)',    'pl': 135000, 'vc': 450*135000, 'fc': 300e6, 'L': 4},
            {'name': 'USA (Virginia)', 'pl': 130000, 'vc': 480*130000, 'fc': 150e6, 'L': 2},
            {'name': 'USA (Alaska)',   'pl': 100000, 'vc': 500*100000, 'fc': 100e6, 'L': 2},
        ]
        self.num_sites = len(self.sites_data)
        # 物理极限 (年发射次数)
        self.site_limits_annual = np.array([s['L'] * 365 for s in self.sites_data])

        # 火箭灾难参数
        self.p_pad = 0.0003       # 炸台概率 (单次发射)
        self.c_pad = 2.0e8        # 炸台重建费 ($200M)
        self.t_pad = 60           # 炸台停运天数

    def run_simulation(self, chromosome, mode='monte_carlo', force_events=None):
        """
        运行仿真。
        force_events: 字典，如 {'snap_year': 10, 'explode_site': 3, 'explode_year': 5}
        """
        T_finish = self.max_years
        total_cost = 0
        cum_payload = 0
        
        # 详细日志
        log = {
            'year': [], 'payload_se': [], 'payload_rocket': [], 
            'cost_annual': [], 'events': [], 'site_details': []
        }
        
        curr_q_se = self.q0_se
        curr_q_rocket = self.q0_rocket
        
        for t in range(self.max_years):
            event_desc = []
            
            # 1. 灾难判定
            is_snap = False
            is_pad_exploded = np.zeros(self.num_sites, dtype=bool)
            
            if mode == 'scenario' and force_events:
                if t == force_events.get('snap_year', -1):
                    is_snap = True
                    event_desc.append("【严重】缆绳断裂")
                if t == force_events.get('explode_year', -1):
                    s_idx = force_events.get('explode_site', 0)
                    is_pad_exploded[s_idx] = True
                    event_desc.append(f"【警告】{self.sites_data[s_idx]['name']} 爆炸")
            elif mode == 'monte_carlo':
                is_snap = np.random.random() < self.p_snap
                is_pad_exploded = np.random.random(self.num_sites) < self.p_pad * 100 
            
            # 2. 太空电梯运行 (SE)
            plan_u = chromosome[t, 0] # 计划负荷率
            
            # 扰动影响：克利尼奥力 & 缆绳扰动 (加入随机波动)
            eta_dynamic = np.random.normal(self.eta_cor_mean * self.eta_tether_mean, 0.05)
            eta_dynamic = np.clip(eta_dynamic, 0.5, 0.95)
            
            # 实际可用性
            avail_days = 365
            repair_cost_se = 0
            if is_snap:
                avail_days = max(0, 365 - self.t_snap)
                repair_cost_se = self.c_snap # 计入意外成本
            
            # SE 实际运力
            real_payload_se = plan_u * self.se_cap_design * self.beta_storm * eta_dynamic * (avail_days/365.0)
            
            # SE 成本计算
            lr_se = max((curr_q_se / self.q0_se) ** (-self.lr_se), 0.15)
            c_ops_se = real_payload_se * self.se_c_unit_ops * lr_se
            c_maint_se = self.se_c_base_maint + self.se_c_wear_max * (plan_u ** self.se_gamma)
            
            total_se_cost_year = c_ops_se + c_maint_se + repair_cost_se
            
            # 3. 火箭系统运行 (Rocket)
            plan_n = chromosome[t, 1:] # 各基地计划发射数
            real_payload_rocket = 0
            total_rocket_cost_year = 0
            site_status = []
            
            lr_rocket = max((curr_q_rocket / self.q0_rocket) ** (-self.lr_rocket), 0.25)
            
            for i in range(self.num_sites):
                site = self.sites_data[i]
                n_target = plan_n[i]
                
                # --- 紧急预案 A: 电梯断裂 ---
                if is_snap:
                    n_target = self.site_limits_annual[i] # 强制拉满
                
                # --- 紧急预案 B: 发射台爆炸 ---
                c_pad_fix = 0
                if is_pad_exploded[i]:
                    avail_days_r = 365 - self.t_pad
                    limit_reduced = int(site['L'] * avail_days_r)
                    n_actual = min(n_target, limit_reduced)
                    c_pad_fix = self.c_pad # 计入意外成本
                else:
                    n_actual = n_target
                
                site_status.append({
                    'name': site['name'],
                    'launches': n_actual,
                    'is_exploded': is_pad_exploded[i]
                })
                
                if n_actual > 0:
                    c_launch = n_actual * site['vc'] * lr_rocket
                    c_fix = site['fc']
                    total_rocket_cost_year += (c_launch + c_fix + c_pad_fix)
                    
                    # 运力 (扣除0.5%的飞行故障)
                    real_payload_rocket += n_actual * site['pl'] * 0.995
            
            # 4. 汇总与更新
            total_cost_year = total_se_cost_year + total_rocket_cost_year
            total_cost += total_cost_year
            cum_payload += (real_payload_se + real_payload_rocket)
            
            curr_q_se += real_payload_se
            curr_q_rocket += real_payload_rocket
            
            log['year'].append(t + 1)
            log['payload_se'].append(real_payload_se)
            log['payload_rocket'].append(real_payload_rocket)
            log['cost_annual'].append(total_cost_year)
            log['events'].append(", ".join(event_desc) if event_desc else "正常运行")
            log['site_details'].append(site_status)
            
            if cum_payload >= self.target_payload:
                T_finish = t + 1
                break
                
        # 惩罚未完成
        if cum_payload < self.target_payload:
            total_cost += 1e16
            
        return total_cost, T_finish, log

# ==========================================
# 2. 遗传算法求解器
# ==========================================
class GeneticSolver:
    # 1. 在 __init__ 中增加 mc_samples 参数，默认设为 50
    def __init__(self, model, pop_size=40, generations=40, mc_samples=50):
        self.model = model
        self.pop_size = pop_size
        self.gen = generations
        self.mc_samples = mc_samples # <--- 保存这个参数
        self.T = model.max_years
        
        # ... (初始化种群的代码保持不变) ...
        self.pop = []
        for _ in range(pop_size):
            chrom = np.zeros((self.T, 11))
            chrom[:, 0] = np.random.uniform(0.6, 0.95, self.T)
            for k in range(10):
                limit = model.site_limits_annual[k]
                chrom[:, k+1] = np.random.randint(0, limit*0.5, self.T)
            self.pop.append(chrom)
        self.pop = np.array(self.pop)
        self.best_solution = None

    # 2. 修改 get_fitness，使用 self.mc_samples
    def get_fitness(self, chrom):
        costs = []
        times = []
        # 将原来的 range(10) 改为 range(self.mc_samples)
        for _ in range(self.mc_samples): 
            c, t, _ = self.model.run_simulation(chrom, mode='monte_carlo')
            costs.append(c)
            times.append(t)
        
        var_cost = np.percentile(costs, 90)
        var_time = np.percentile(times, 90)
        
        score = self.model.alpha * (var_cost/self.model.ref_cost) + \
                (1-self.model.alpha) * (var_time/self.model.ref_time)
        return score, var_cost, var_time

    def run(self):
        print(f"开始遗传算法优化 (代数: {self.gen}, 种群: {self.pop_size})...")
        for g in range(self.gen):
            scores = []
            metrics = []
            for ind in self.pop:
                s, vc, vt = self.get_fitness(ind)
                scores.append(s)
                metrics.append((vc, vt))
            
            best_idx = np.argmin(scores)
            if self.best_solution is None or scores[best_idx] < self.best_solution['score']:
                self.best_solution = {
                    'score': scores[best_idx],
                    'chrom': self.pop[best_idx].copy(),
                    'metrics': metrics[best_idx]
                }
            
            if g % 10 == 0:
                print(f"Gen {g}: 最佳评分={scores[best_idx]:.4f} | 耗时={metrics[best_idx][1]}年 | 成本=${metrics[best_idx][0]/1e12:.2f}T")
            
            # 进化操作
            new_pop = [self.pop[best_idx].copy()]
            indices = np.argsort(scores)[:int(self.pop_size*0.4)]
            parents = self.pop[indices]
            
            while len(new_pop) < self.pop_size:
                p1 = parents[np.random.randint(len(parents))]
                p2 = parents[np.random.randint(len(parents))]
                child = p1.copy()
                mask = np.random.rand(self.T, 11) < 0.5
                child[mask] = p2[mask]
                if np.random.rand() < 0.2:
                    r = np.random.randint(self.T)
                    c = np.random.randint(11)
                    if c == 0: child[r,c] = np.random.rand()
                    else: child[r,c] = np.random.randint(0, self.model.site_limits_annual[c-1])
                new_pop.append(child)
            self.pop = np.array(new_pop)
            
        return self.best_solution

# ==========================================
# 3. 结果生成与报告输出
# ==========================================
if __name__ == "__main__":
    model = RobustLogisticsModel()
    solver = GeneticSolver(model, pop_size=30, generations=30)
    best_sol = solver.run()
    
    best_chrom = best_sol['chrom']

    print("\n" + "="*60)
    print("      正在进行 1000 次蒙特卡洛压力测试 (High-Fidelity Validation)      ")
    print("="*60)
    
    val_costs = []
    val_times = []
    
    # 循环 1000 次
    for i in range(1000):
        c, t, _ = model.run_simulation(best_chrom, mode='monte_carlo')
        val_costs.append(c)
        val_times.append(t)
        # 打印进度条 (每100次显示一次)
        if (i+1) % 100 == 0:
            print(f"进度: {i+1}/1000 完成...")
            
    # 计算精确的 VaR (90%) 和 平均值
    var90_cost = np.percentile(val_costs, 90)
    mean_cost = np.mean(val_costs)
    var90_time = np.percentile(val_times, 90)
    mean_time = np.mean(val_times)
    
    print(f"\n【最终鲁棒性验证结果 (N=1000)】")
    print(f"  - 平均成本 (Mean): ${mean_cost/1e12:.3f} T")
    print(f"  - 风险成本 (VaR 90%): ${var90_cost/1e12:.3f} T (即90%概率下成本不超过此值)")
    print(f"  - 平均耗时: {mean_time:.1f} 年")
    print(f"  - 风险耗时 (VaR 90%): {var90_time} 年")

    # [新增] 绘制 1000 次模拟的成本分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(np.array(val_costs)/1e12, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(var90_cost/1e12, color='red', linestyle='--', linewidth=2, label=f'VaR 90%: ${var90_cost/1e12:.2f}T')
    plt.axvline(mean_cost/1e12, color='green', linestyle='-', linewidth=2, label=f'Mean: ${mean_cost/1e12:.2f}T')
    plt.xlabel('总成本 (万亿美元)')
    plt.ylabel('频次 (Frequency)')
    plt.title('1000次蒙特卡洛模拟下的成本分布 (鲁棒性验证)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    
    # 运行一次确定性场景用于展示 (第15年断裂，第5年德州爆炸)
    scenario = {'snap_year': 15, 'explode_site': 2, 'explode_year': 5}
    cost, t_fin, log = model.run_simulation(best_sol['chrom'], mode='scenario', force_events=scenario)
    
    print("\n" + "="*60)
    print("                1亿吨物资运输 - 综合调度报告                ")
    print("="*60)
    
    # --- [关键修正] 1. 总体运力分配输出 ---
    total_se_load = np.sum(log['payload_se'])
    total_rk_load = np.sum(log['payload_rocket'])
    total_all = total_se_load + total_rk_load
    
    print(f"\n【1. 总体物资分配 (Total Payload Allocation)】")
    print(f"  - 目标运量: 100.00 百万吨")
    print(f"  - 实际总运量: {total_all/1e9:.3f} 亿吨")
    print(f"  ------------------------------------------")
    print(f"  > 太空电梯承担: {total_se_load/1e9:.3f} 亿吨 ({total_se_load/total_all*100:.1f}%)")
    print(f"  > 火箭系统承担: {total_rk_load/1e9:.3f} 亿吨 ({total_rk_load/total_all*100:.1f}%)")
    print(f"  ------------------------------------------")
    print(f"  - 完工时间: {t_fin} 年")
    print(f"  - 总计成本: ${cost/1e12:.3f} 万亿美元")
    
    # --- [关键修正] 2. 每5年简报 (优化格式) ---
    print(f"\n【2. 每 5 年运行情况简报】")
    print(f"{'年份':<6} | {'事件状态':<15} | {'SE运量(万吨)':<12} | {'火箭运量(万吨)':<12} | {'活跃基地数':<10} | {'主力基地(Top1)'}")
    print("-" * 90)
    
    for t in range(0, t_fin, 5): 
        idx = t
        if idx >= len(log['year']): break
        
        y_label = log['year'][idx]
        evt = log['events'][idx]
        se_load = log['payload_se'][idx] / 1e7 # 万吨
        rk_load = log['payload_rocket'][idx] / 1e7 # 万吨
        
        sites = log['site_details'][idx]
        active_count = sum(1 for s in sites if s['launches'] > 0)
        
        top_site = max(sites, key=lambda x: x['launches'])
        top_site_str = f"{top_site['name'][:10]} ({top_site['launches']})"
        
        evt_str = evt[:15] + ".." if len(evt)>15 else evt
        print(f"Year {y_label:<2} | {evt_str:<15} | {se_load:<12.2f} | {rk_load:<12.2f} | {active_count:<10} | {top_site_str}")
        
    # --- [关键修正] 3. 灾难年份特别详情 (统一输出，避免重复) ---
    print("\n【3. 关键灾难年份详情与应急响应】")
    
    # 德州爆炸详情
    expl_idx = scenario['explode_year']
    if expl_idx < t_fin:
        print(f">>> 第 {expl_idx+1} 年 [火箭爆炸事件]:")
        sites = log['site_details'][expl_idx]
        tx_site = sites[2]
        print(f"    - 受损基地: {tx_site['name']}")
        print(f"    - 状态: {'爆炸(已停运)' if tx_site['is_exploded'] else '正常'}")
        print(f"    - 实际发射: {tx_site['launches']} 次 (远低于计划值)")
        # 看看谁发射最多（补位）
        top_site = max(sites, key=lambda x: x['launches'])
        print(f"    - 应急分流: 主力已转移至 {top_site['name']} (发射 {top_site['launches']} 次)")

    # 缆绳断裂详情
    snap_idx = scenario['snap_year']
    if snap_idx < t_fin:
        print(f"\n>>> 第 {snap_idx+1} 年 [太空电梯断裂事件]:")
        se_load = log['payload_se'][snap_idx]
        rk_load = log['payload_rocket'][snap_idx]
        # 对比前一年
        prev_load = log['payload_rocket'][snap_idx-1] if snap_idx>0 else 0
        
        print(f"    - 太空电梯运力: {se_load/1e7:.2f} 万吨 (严重瘫痪)")
        print(f"    - 火箭系统运力: {rk_load/1e7:.2f} 万吨 (前一年: {prev_load/1e7:.2f} 万吨)")
        print(f"    - 应急响应: 运力激增 +{(rk_load - prev_load)/1e7:.2f} 万吨")
        print(f"    - 启动指令: <全基地饱和发射> 已执行")

    # --- 绘图: 累积运量进度 ---
    years = log['year']
    cum_payload = np.cumsum(np.array(log['payload_se']) + np.array(log['payload_rocket']))
    
    plt.figure(figsize=(10, 5))
    plt.plot(years, cum_payload/1e9, 'g-o', linewidth=2, label='累计运量')
    plt.axhline(y=0.1, color='r', linestyle='--', label='目标 (1亿吨)')
    plt.xlabel('年份')
    plt.ylabel('累计运量 (亿吨)')
    plt.title(f'项目进度曲线 (完工: {t_fin}年)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 绘图: 堆叠图展示灾难响应 ---
    
    se_data = np.array(log['payload_se']) / 1e9
    rk_data = np.array(log['payload_rocket']) / 1e9
    
    plt.figure(figsize=(12, 6))
    plt.bar(years, se_data, label='太空电梯 (含扰动)', color='#2ecc71', alpha=0.8)
    plt.bar(years, rk_data, bottom=se_data, label='火箭系统 (含应急)', color='#e74c3c', alpha=0.8)
    
    # 标注事件
    if snap_idx < t_fin:
        plt.annotate('缆绳断裂\n火箭接管', xy=(snap_idx+1, rk_data[snap_idx]+se_data[snap_idx]), 
                     xytext=(snap_idx+1, se_data[snap_idx]+0.04),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.xlabel('年份')
    plt.ylabel('年运量 (亿吨)')
    plt.title(f'1亿吨物资运输计划 (含灾难响应演示)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()