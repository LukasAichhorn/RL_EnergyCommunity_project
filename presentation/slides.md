---
marp: true
theme: default
---

<style>
section {
  font-size: 22px;
}
h1 {
  font-size: 42px;
}
h2 {
  font-size: 32px;
}
h3 {
  font-size: 26px;
}
code {
  font-size: 18px;
}
pre {
  font-size: 16px;
}
table {
  font-size: 20px;
}
</style>

# RL Battery Control System

**Training an AI Agent to Optimize Community Battery Management**

Minimizing Grid Dependency through Reinforcement Learning

Lukas Aichhorn | Johannes Goelles

---

## Project Outline

### Goal
Train an **RL agent** to control a community battery system that **minimizes grid energy usage** by optimizing charge/discharge decisions

### Approach
1. **Environment**: Gymnasium-compatible battery control simulation
2. **Algorithm**: PPO (Proximal Policy Optimization) from Stable-Baselines3
3. **Training**: 500k timesteps on real community data
4. **Infrastructure**: Modal cloud platform with A10G GPU acceleration

### Success Metric
Reduce grid dependency by intelligently storing surplus and discharging during deficit

---

## The Data

### Source: Real Community Metering Data
- **Duration**: 1 year of historical data
- **Interval**: 15-minute timesteps (96 per day)
- **Total**: ~35,000 timesteps (365 days)

### Features per Timestep:
| Feature | Description | Range |
|---------|-------------|-------|
| `total_consumption` | Community energy usage | 0-40 kWh |
| `total_production` | Community energy generation | 0-60 kWh |
| `surplus_production` | Excess production available | 0-60 kWh |

### Split:
- **Training**: 80% (287 days)
- **Test**: 20% (72 days)

---

## PPO Algorithm

**Proximal Policy Optimization** - State-of-the-art RL for continuous control

### Why PPO?
- ✅ **Stable** - Clipped updates prevent policy collapse
- ✅ **Sample efficient** - Learns from fewer interactions
- ✅ **Continuous actions** - Native support for charge/discharge rates

### How it Works:
```
Agent observes state → Chooses action → Gets reward
           ↓
Updates policy (with clipping to prevent large jumps)
```

**Clipped Objective:**
```python
L_CLIP = min(ratio × advantage, clip(ratio, 1-ε, 1+ε) × advantage)
```

---

## Environment Design

### State Space (What Agent Sees):
- Battery SOC (0-1)
- Current consumption (kW)
- Current production (kW)
- Recent history (4 timesteps)

### Action Space:
- **Continuous: -50 to +50 kW**
  - Positive = Charge battery
  - Negative = Discharge battery

### Reward Function:
```python
reward = -grid_usage
  + bonus for charging when surplus exists
  + bonus for discharging when deficit exists
  - penalty for impossible actions
```

---

## Training Configuration

```python
PPO(
    policy="MlpPolicy",           # Neural network
    learning_rate=3e-4,            # → decays to 1e-5
    gamma=0.99,                    # Values future rewards
    clip_range=0.2,                # PPO stability
    ent_coef=0.1,                  # Exploration → 0.01
    device="cuda",                 # GPU acceleration
    n_steps=2048,                  # Per update
    n_epochs=10                    # Optimization passes
)
```

### Training Details:
- **500,000 timesteps** (~5,200 episodes)
- **Continuous battery state** across episodes
- **Learning rate + entropy decay** over time
- **A10G GPU** on Modal cloud (~15 minutes)

---

## Why PPO for Battery Control?

| Challenge | Solution |
|-----------|----------|
| Continuous actions | Native continuous action support |
| Long-term planning | High gamma (0.99) values future |
| Physical constraints | Clipping prevents unsafe policies |
| Exploration vs exploitation | Entropy bonus with decay |

**Result:** Agent learns to maximize battery utilization and minimize grid usage! ⚡🔋

---

## Model Performance: Grid Usage Reduction

![Grid Usage Comparison](../evaluation_results_3/grid_usage_comparison.png)

### Key Results:
- **Significant reduction** in grid energy usage
- Agent learns to **anticipate** consumption patterns
- **Optimal timing** of charge/discharge cycles

---

## Battery State of Charge (SOC) Behavior

![Battery SOC](../evaluation_results_3/battery_soc_full_period.png)

### What the Agent Learned:
- ✅ **Charge** when surplus production exists (solar peaks)
- ✅ **Discharge** during high consumption periods (evening/morning)
- ✅ **Maintain reserves** for anticipated deficit periods
- ✅ **Full utilization** of battery capacity (0-100% SOC)

---

## Agent Behavior Across Episodes

![All Episodes Continuous](../evaluation_results_3/all_episodes_continuous.png)

### Observations:
- **Consistent strategy** across multiple test episodes
- **Aggressive charging** during surplus (production peaks)
- **Strategic discharge** during deficit periods
- **Continuous learning** maintains battery state between episodes

---

## Comprehensive Performance Analysis

![Comprehensive Plots](../evaluation_results_3/comprehensive_plots.png)

### Multi-Metric Evaluation:
1. **Top Left**: Production vs Consumption patterns
2. **Top Right**: Battery charge/discharge actions
3. **Bottom Left**: Grid usage over time
4. **Bottom Right**: State of charge dynamics

**Agent successfully balances** short-term needs with long-term optimization!

---

## Detailed Episode Analysis

![Evaluation Plots](../evaluation_results_3/evaluation_plots.png)

### Agent Decision-Making:
- **Action values**: Ranges from -50 to +50 kW (full action space)
- **SOC tracking**: Maintains healthy battery levels
- **Grid usage minimization**: Reduces dependency significantly
- **Reward accumulation**: Positive trend indicates successful learning

---

## Key Takeaways

### What the Agent Does:
1. **Morning/Evening**: Discharges battery during high consumption
2. **Midday**: Charges battery when solar production peaks
3. **Night**: Manages reserves strategically for next day
4. **Continuous**: Maintains state across episodes for realistic simulation

### Performance Highlights:
- ✅ **Reduced grid dependency** through optimal battery management
- ✅ **Full action range utilization** (-50 to +50 kW)
- ✅ **Anticipatory behavior** learned from consumption patterns
- ✅ **Stable policy** converged after 500k timesteps

### Technical Success:
- Modal + GPU: Fast training (~15 min)
- PPO: Stable continuous control
- Real data: Generalizes to unseen test periods

---

## Conclusion & Future Work

### Achievements:
🎯 Successfully trained RL agent to minimize grid usage  
🔋 Agent learned intelligent charge/discharge strategies  
📊 Evaluated on real community data with promising results  

### Future Improvements:
1. **Reward function tuning**: Penalize weak discharge more heavily
2. **Exploration increase**: Higher `log_std_init` for full action range
3. **Alternative algorithms**: Test A2C for faster convergence
4. **Multi-agent**: Scale to multiple batteries/communities
5. **Real-world deployment**: Test with live data streams

### Impact:
**Renewable energy communities can use RL to optimize battery systems autonomously!** 🌍⚡

---

## Thank You!

### Questions?

**Project Repository**: [Add your repo link here]

**Key Technologies**:
- Gymnasium (Environment)
- Stable-Baselines3 (PPO)
- Modal (Cloud Training)
- Python + NumPy/Pandas

---
