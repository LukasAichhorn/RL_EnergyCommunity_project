# Reward Function Changes - Detailed Explanation

## Problem Identified

The agent was using very small actions (0.1-0.2 kW) even when large excess production (50+ kWh) was available. This happened because:

1. **Original reward function only penalized grid usage**: `reward = -grid_usage`
2. **No incentive to store excess energy**: When production > consumption, the agent got the same reward whether it charged 0.1 kW or 50 kW (as long as grid usage was 0)
3. **Agent learned minimal actions are "good enough"**: Small actions don't hurt the immediate reward, so the agent converged to a conservative policy

### Example of the Problem:

**Scenario**: Production = 100 kWh, Consumption = 50 kWh (excess = 50 kWh)

**With 0.2 kW action**:
- Charges: 0.2 × 0.25 × 0.95 = 0.0475 kWh
- Grid usage: 0 (production covers consumption)
- Reward: -0 = 0

**With 50 kW action**:
- Charges: 50 × 0.25 × 0.95 = 11.875 kWh
- Grid usage: 0 (production still covers consumption)
- Reward: -0 = 0

**Result**: Same reward, so agent has no reason to use larger actions!

## Changes Made

### 1. Modified Reward Function (`battery_env.py`)

**Location**: `_calculate_reward()` method (lines 257-286)

**What Changed**:
- **Before**: `reward = -grid_usage` (only penalizes grid usage)
- **After**: `reward = -grid_usage + storage_bonus + soc_bonus` (penalizes grid + rewards storage)

**New Reward Components**:

1. **Primary Reward (Grid Penalty)**: `-grid_usage`
   - Still the main objective
   - Penalizes energy drawn from grid
   - Range: typically [-100, 0] depending on consumption

2. **Storage Bonus**: `0.1 × energy_charged_kwh`
   - **Purpose**: Reward storing excess energy in battery
   - **Why 0.1?**: Makes it a bonus (10% of stored energy), not the primary reward
   - **Effect**: Agent now gets rewarded for charging more when excess is available
   - **Example**: Charging 10 kWh gives +1.0 bonus, charging 0.1 kWh gives +0.01 bonus

3. **SOC Bonus**: `0.05 × battery_soc`
   - **Purpose**: Small reward for maintaining battery charge
   - **Why 0.05?**: Very small (5% of SOC) to keep it secondary
   - **Effect**: Prevents agent from keeping battery empty
   - **Example**: SOC at 0.5 gives +0.025 bonus, SOC at 0.9 gives +0.045 bonus

**Why These Scales?**:
- Storage bonus (0.1) is 10x smaller than typical grid penalty (10-100)
- This ensures grid minimization remains primary, but storage is incentivized
- SOC bonus (0.05) is even smaller to prevent over-optimization on SOC alone

### 2. Increased Exploration (`train_agent.py`)

**Location**: PPO algorithm configuration (line 196)

**What Changed**:
- **Before**: `ent_coef=0.01` (low exploration)
- **After**: `ent_coef=0.05` (5x more exploration)

**Why This Matters**:
- **Entropy coefficient** controls how much the policy explores vs exploits
- Higher entropy = more random actions during training
- This allows the agent to try larger actions (10 kW, 20 kW, 50 kW) and discover they're beneficial
- Without this, the agent might get stuck in a local minimum of small actions

**Trade-off**:
- More exploration = slower convergence but better final policy
- 0.05 is a moderate increase (not too high to cause instability)

## Expected Effects

### Before Changes:
- Agent uses 0.1-0.2 kW actions
- Most excess production goes unused
- SOC changes very slowly (linear appearance)
- Agent is overly conservative

### After Changes:
- Agent should use larger actions (5-50 kW) when excess is available
- More excess production gets stored in battery
- SOC changes more dynamically
- Agent learns to be more aggressive when beneficial

## Example Reward Calculation

**Scenario**: Production = 100 kWh, Consumption = 50 kWh, Battery at 50% SOC

**Action: 0.2 kW (old behavior)**:
- Energy charged: 0.0475 kWh
- Grid usage: 0
- Reward = -0 + (0.1 × 0.0475) + (0.05 × 0.5) = **0.02975**

**Action: 50 kW (new desired behavior)**:
- Energy charged: 11.875 kWh
- Grid usage: 0
- Reward = -0 + (0.1 × 11.875) + (0.05 × 0.5) = **1.2125**

**Difference**: 40x higher reward for using full capacity!

## Testing the Changes

After retraining with these changes, you should see:
1. **Larger actions**: Actions in the 5-50 kW range, not just 0.1-0.2 kW
2. **More dynamic SOC**: SOC changes more noticeably over time
3. **Better grid usage**: More excess energy stored = less grid dependency later
4. **Higher rewards**: Episode rewards should increase due to storage bonuses

## Notes

- The reward scales (0.1 and 0.05) can be tuned if needed
- If agent becomes too aggressive, reduce storage_bonus scale
- If agent still too conservative, increase storage_bonus scale
- Monitor training to ensure rewards are learning properly

