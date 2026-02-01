import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set font for English display (Arial or similar)
plt.rcParams['font.family'] = 'Arial' 
plt.rcParams['axes.unicode_minus'] = False

class MoonWaterLogistics:
    def __init__(self):
        # 1. Rocket Launch Site Data
        data = {
            'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Location': [
                'India (Satish Dhawan)', 'China (Taiyuan)', 'USA (Texas)', 
                'USA (Florida)', 'Kazakhstan (Baikonur)', 'New Zealand (Mahia)', 
                'French Guiana (Kourou)', 'USA (California)', 'USA (Virginia)', 
                'USA (Alaska)'
            ],
            'Cost_per_kg': [300, 320, 320, 350, 380, 400, 450, 450, 480, 500],
            'Payload_tons': [145, 125, 145, 140, 120, 125, 150, 135, 130, 100],
            'Max_Launch_per_day': [4, 5, 6, 8, 4, 3, 3, 4, 2, 2],
            'Annual_Fixed_Cost_M': [150, 200, 350, 400, 250, 100, 300, 300, 150, 100]
        }
        self.rocket_sites = pd.DataFrame(data)
        self.rocket_sites = self.rocket_sites.sort_values(by='Cost_per_kg')
        self.site_used = {id: False for id in self.rocket_sites['ID']}

        # 2. System Parameters
        self.days = 365           
        self.population = 100000  
        self.w_req = 50           
        self.eta_recycle = 0.98   
        
        # Daily Net Demand (~100 tons)
        self.daily_demand = self.population * self.w_req * (1 - self.eta_recycle) 
        
        # Inventory Parameters
        self.I_min = 3000 * 1000  # Safety Stock: 3000 tons
        self.I_max = 12000 * 1000 
        self.I_init = 0           # Initial Inventory
        self.I_target = self.I_min + 500 * 1000 # Target Level
        
        self.h_holding = 0.05     
        self.C_SE = 20            
        
        self.tau_R = 2            # Rocket Lead Time
        self.tau_SE = 20          # SE Lead Time
        
        self.inventory = [self.I_init] * (self.days + 1)
        self.pipeline_R = []   
        self.pipeline_SE = []  
        
        self.history = {
            'day': [],
            'inventory': [],
            'order_R': [],
            'order_SE': [],
            'cost_transport': [],
            'cost_holding': []
        }

    def get_rocket_cost_and_dispatch(self, amount_needed_kg):
        if amount_needed_kg <= 0:
            return 0, 0
        remaining_load = amount_needed_kg
        total_cost = 0
        actual_load = 0
        
        for _, site in self.rocket_sites.iterrows():
            if remaining_load <= 0: break
            site_id = site['ID']
            payload_kg = site['Payload_tons'] * 1000
            cost_per_kg = site['Cost_per_kg']
            max_launches = site['Max_Launch_per_day']
            launches = 0
            while launches < max_launches and remaining_load > 0:
                launches += 1
                load = payload_kg 
                self.site_used[site_id] = True
                cost_launch = load * cost_per_kg
                total_cost += cost_launch
                actual_load += load
                remaining_load -= load
        return total_cost, actual_load

    def run_simulation(self):
        total_transport_cost = 0
        total_holding_cost = 0
        
        for t in range(1, self.days + 1):
            # 1. Arrivals
            arrived_R = sum([amt for day, amt in self.pipeline_R if day == t])
            arrived_SE = sum([amt for day, amt in self.pipeline_SE if day == t])
            
            # 2. Update Inventory
            inv_prev = self.inventory[t-1]
            inv_curr = inv_prev + arrived_R + arrived_SE - self.daily_demand
            if inv_curr < 0: inv_curr = 0
            self.inventory[t] = inv_curr
            
            # 3. Decision Logic (Coordinated)
            order_R = 0
            order_SE = 0
            
            # --- Logic A: Rocket (Emergency Response) ---
            future_demand_short = self.daily_demand * self.tau_R
            short_term_arrivals = sum([amt for day, amt in self.pipeline_SE if t < day <= t + self.tau_R]) + \
                                  sum([amt for day, amt in self.pipeline_R if t < day <= t + self.tau_R])
            
            projected_inv_short = inv_curr - future_demand_short + short_term_arrivals
            
            if projected_inv_short < self.I_min:
                shortage = self.I_min - projected_inv_short
                order_R = shortage * 1.05
            
            cost_today_R = 0
            if order_R > 0:
                cost, actual_load = self.get_rocket_cost_and_dispatch(order_R)
                if actual_load > 0:
                    self.pipeline_R.append((t + self.tau_R, actual_load))
                    cost_today_R = cost
            
            # --- Logic B: Space Elevator (Daily Maintenance) ---
            pipeline_all = sum([amt for day, amt in self.pipeline_SE if day > t]) + \
                           sum([amt for day, amt in self.pipeline_R if day > t])
            
            future_demand_long = self.daily_demand * self.tau_SE
            projected_inv_long = inv_curr - future_demand_long + pipeline_all
            
            gap = self.I_target - projected_inv_long
            base_order = self.daily_demand
            adjustment = gap / self.tau_SE 
            
            # Damping Control
            order_SE = base_order + adjustment
            order_SE = max(base_order * 0.5, min(order_SE, base_order * 1.5))
            
            cost_today_SE = 0
            if order_SE > 0:
                cost_today_SE = order_SE * self.C_SE
                self.pipeline_SE.append((t + self.tau_SE, order_SE))
            
            # 5. Holding Cost
            holding_mass = max(0, inv_curr - self.I_min) 
            cost_holding_today = holding_mass * self.h_holding
            
            total_transport_cost += (cost_today_R + cost_today_SE)
            total_holding_cost += cost_holding_today
            
            self.history['day'].append(t)
            self.history['inventory'].append(inv_curr)
            self.history['order_R'].append(order_R)
            self.history['order_SE'].append(order_SE)
            self.history['cost_transport'].append(cost_today_R + cost_today_SE)
            self.history['cost_holding'].append(cost_holding_today)

        return total_transport_cost, total_holding_cost

    def calculate_total_cost(self, transport_cost, holding_cost):
        fixed_cost = 0
        for site_id, used in self.site_used.items():
            if used:
                site_info = self.rocket_sites[self.rocket_sites['ID'] == site_id].iloc[0]
                fixed_cost += site_info['Annual_Fixed_Cost_M'] * 1e6
        return transport_cost + holding_cost + fixed_cost, fixed_cost

    def analyze_strategy(self):
        df = pd.DataFrame(self.history)
        strategy_log = []
        
        rocket_days = df[df['order_R'] > 0].copy()
        if not rocket_days.empty:
            rocket_days['group'] = (rocket_days['day'] != rocket_days['day'].shift(1) + 1).cumsum()
            for _, group in rocket_days.groupby('group'):
                strategy_log.append({
                    'Start Day': group['day'].min(),
                    'End Day': group['day'].max(),
                    'Method': 'Rocket',
                    'Purpose': 'Emergency / Gap Filling',
                    'Amount (Tons)': group['order_R'].sum() / 1000
                })
        
        se_days = df[df['order_SE'] > 0]
        if not se_days.empty:
             strategy_log.append({
                'Start Day': se_days['day'].min(),
                'End Day': se_days['day'].max(),
                'Method': 'Space Elevator',
                'Purpose': 'Daily Routine Supply',
                'Amount (Tons)': se_days['order_SE'].sum() / 1000
            })
        return pd.DataFrame(strategy_log).sort_values('Start Day')

    def plot_results(self):
        df_res = pd.DataFrame(self.history)
        
        # Plot 1: Inventory Curve
        plt.figure(figsize=(12, 6))
        plt.plot(df_res['day'], df_res['inventory']/1000, color='#1f77b4', linewidth=2.5, label='Actual Inventory')
        plt.axhline(self.I_min/1000, color='r', linestyle='--', linewidth=2, label='Safety Stock Level (3000 Tons)')
        
        plt.text(200, (self.I_min/1000) + 200, 'Stable Maintenance Zone', fontsize=12, color='green', ha='center')
        
        plt.ylabel('Inventory Level (Tons)', fontsize=12)
        plt.xlabel('Time (Days)', fontsize=12)
        plt.title('Moon Colony Water Inventory Dynamics', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show() 
        
        # Plot 2: Transport Volume
        plt.figure(figsize=(12, 6))
        days = df_res['day']
        rocket_orders = df_res['order_R'] / 1000
        se_orders = df_res['order_SE'] / 1000
        
        plt.bar(days, se_orders, color='#2ca02c', alpha=0.8, label='Space Elevator (Daily)', width=1.0)
        plt.bar(days, rocket_orders, bottom=se_orders, color='#ff7f0e', alpha=0.9, label='Rocket (Emergency)', width=1.0)
        
        plt.ylim(0, 300) 
        
        # Annotation for Rocket
        rocket_max = rocket_orders.max()
        if rocket_max > 0:
            rocket_events = df_res[df_res['order_R'] > 0]
            for day in rocket_events['day']:
                if day < 10: 
                    val = rocket_orders[day-1] 
                    if val > 300: 
                        plt.annotate(f'Rocket Launch\n{val:.0f} Tons', 
                                     xy=(day, 280), xytext=(day+15, 250),
                                     arrowprops=dict(facecolor='red', shrink=0.05),
                                     fontsize=10, color='red')
        
        plt.ylabel('Daily Transport Volume (Tons)', fontsize=12)
        plt.xlabel('Time (Days)', fontsize=12)
        plt.title('Daily Supply Transport Strategy', fontsize=14, fontweight='bold')
        
        plt.yticks([0, 50, 100, 150, 200, 250, 300])
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_pie_chart(self):
        df_res = pd.DataFrame(self.history)
        total_R = df_res['order_R'].sum()
        total_SE = df_res['order_SE'].sum()
        
        labels = ['Rocket\n(Emergency)', 'Space Elevator\n(Daily Routine)']
        sizes = [total_R, total_SE]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.1, 0)
        
        plt.figure(figsize=(9, 7))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.2f%%', shadow=True, startangle=140, textprops={'fontsize': 14})
        plt.title('Annual Water Transport Ratio', fontsize=16)
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    model = MoonWaterLogistics()
    trans_cost, hold_cost = model.run_simulation()
    total_cost, fixed_cost = model.calculate_total_cost(trans_cost, hold_cost)
    strategy_df = model.analyze_strategy()

    print("="*50)
    print("       Moon Water Logistics Optimization Report       ")
    print("="*50)
    print(f"1. Cost Breakdown (Unit: Million USD):")
    print(f"   - Variable Transport Cost: ${trans_cost/1e6:,.2f} M")
    print(f"   - Inventory Holding Cost:  ${hold_cost/1e6:,.2f} M")
    print(f"   - Site Fixed Cost:         ${fixed_cost/1e6:,.2f} M")
    print(f"   -------------------------------------------")
    print(f"   - Total Annual Cost:       ${total_cost/1e6:,.2f} M")
    
    print("\n2. Detailed Transport Strategy Execution:")
    print(strategy_df.to_markdown(index=False, numalign="left", stralign="left"))
    
    
    model.plot_results()
    model.plot_pie_chart()