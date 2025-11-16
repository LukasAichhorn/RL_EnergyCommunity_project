# Battery Control RL Environment Documentation

## Overview

The agent controls a community battery to minimize grid energy usage. It observes consumption/production patterns and decides when to charge or discharge the battery.

## State Space (Observations)

The agent observes **4 normalized values** (0-1 range):

1. **`total_consumption`** - Community energy consumption (kWh)
   - Normalized by participant count (scaled to mean ~156.5 participants)
   - Range: ~0-50 kWh (after normalization)

2. **`total_production`** - Community energy production (kWh)
   - Also normalized by participant count
   - Range: ~0-113 kWh

3. **`surplus_production`** - Excess production (production - consumption)
   - Available for charging or sharing

4. **`battery_soc`** - Battery state of charge (0.0-1.0)
   - 0.0 = empty, 1.0 = full

## Action Space

**Single continuous value**: Charge/discharge rate in kW

- **Range**: `[-50 kW, +50 kW]` (default)
- **Positive** = Charge (e.g., +30 kW)
- **Negative** = Discharge (e.g., -20 kW)
- **Zero** = Do nothing (0 kW)

### Example Actions:
```
+50 kW  → Charge at maximum rate
+25 kW  → Charge at half rate
  0 kW  → No action
-25 kW  → Discharge at half rate
-50 kW  → Discharge at maximum rate
```

## Action Constraints

The environment enforces physical and operational limits:

### 1. Battery Capacity Limits
- Can't charge if battery is full (SOC = 1.0)
- Can't discharge if battery is empty (SOC = 0.0)

### 2. Charge Rate Limits
- **Hardware limit**: max 50 kW (default)
- **Space limit**: depends on current SOC
- **Production limit**: can only charge from excess production

### 3. Discharge Rate Limits
- **Hardware limit**: max 50 kW (default)
- **Energy limit**: depends on current SOC

### 4. **No Grid Charging** ⚠️
- Charging **only allowed** when: `production > consumption`
- If trying to charge with no excess production → action set to 0

**Example:**
```
Production: 5 kWh, Consumption: 10 kWh
→ No charging allowed (all production needed for consumption)

Production: 20 kWh, Consumption: 10 kWh
→ Can charge up to 10 kWh (excess production)
```

## Battery Dynamics

### When Charging (action > 0):
```
Energy stored = action × 0.25 hours × 0.95 efficiency
SOC increase = energy_stored / battery_capacity
```

**Example:** Charge at +40 kW
- Energy stored: 40 × 0.25 × 0.95 = **9.5 kWh**
- SOC increase: 9.5 / 100 = **0.095 (9.5%)**

### When Discharging (action < 0):
```
Energy delivered = |action| × 0.25 hours / 0.95 efficiency
SOC decrease = energy_delivered / battery_capacity
```

**Example:** Discharge at -40 kW
- Energy delivered: 40 × 0.25 / 0.95 = **10.53 kWh**
- SOC decrease: 10.53 / 100 = **0.1053 (10.53%)**

## Reward Function

**Goal**: Minimize grid energy usage

### Reward Calculation:

```python
# Step 1: Calculate available energy
available_energy = total_production + energy_discharged_kwh

# Step 2: Calculate own coverage (how much consumption we can cover)
own_coverage = min(total_consumption, available_energy)

# Step 3: Calculate grid usage (what we need from grid)
grid_usage = total_consumption - own_coverage

# Step 4: Reward (negative grid usage)
reward = -grid_usage
```

### Reward Characteristics:
- **Range**: typically `[-100, 0]` (depends on consumption levels)
- **Best**: 0 (zero grid usage)
- **Worst**: large negative values (high grid dependency)
- **Higher reward** = less grid dependency

### Reward Examples:

#### Scenario 1: Perfect Self-Sufficiency ✅
```
Consumption: 20 kWh
Production: 15 kWh
Battery discharge: 5 kWh

Available energy: 15 + 5 = 20 kWh
Own coverage: min(20, 20) = 20 kWh
Grid usage: 20 - 20 = 0 kWh
Reward: -0 = 0 ✓ (best!)
```

#### Scenario 2: Need Grid ❌
```
Consumption: 30 kWh
Production: 10 kWh
Battery discharge: 5 kWh

Available energy: 10 + 5 = 15 kWh
Own coverage: min(30, 15) = 15 kWh
Grid usage: 30 - 15 = 15 kWh
Reward: -15 ✗
```

#### Scenario 3: Smart Charging (Storing Excess) ✅
```
Consumption: 10 kWh
Production: 20 kWh
Battery charge: 8 kWh (from excess)

Available energy: 20 + 0 = 20 kWh
Own coverage: min(10, 20) = 10 kWh
Grid usage: 10 - 10 = 0 kWh
Reward: -0 = 0 ✓ (good strategy!)
```

## Episode Structure

- **Episode length**: 96 steps = 1 day (15-minute intervals)
- **Each step**: Agent observes → acts → receives reward → next timestep
- **Episode ends**: After 96 steps or when data runs out

## Data Normalization

**Important**: Data is normalized by participant count:

1. Calculate per-participant averages (divide by participant count)
2. Scale back to mean participant count (~156.5)
3. Result: Values account for changing participant numbers over time

This ensures the agent learns **consumption/production patterns**, not participant count changes.

## What the Agent Learns

### ✅ Good Strategies:
- Charge when production > consumption (store excess)
- Discharge when consumption > production (use stored energy)
- Keep battery ready for peak consumption periods
- Balance SOC to be available when needed

### ❌ Bad Strategies:
- Charging when no excess production (blocked anyway)
- Discharging when production already covers consumption
- Keeping battery full/empty when it should be used
- Not using battery when needed

## Environment Parameters

### Default Configuration:
- **Battery capacity**: 100 kWh
- **Max charge rate**: 50 kW
- **Max discharge rate**: 50 kW
- **Efficiency**: 0.95 (95%)
- **Timestep duration**: 0.25 hours (15 minutes)
- **Episode length**: 96 steps (1 day)

### Data Configuration:
- **Normalization**: By participant count (default: enabled)
- **Reference participants**: Mean participant count (~156.5)
- **Train/test split**: Configurable (default: 0.8/0.2 or 1.0 for full dataset)

## Summary

The agent learns to **minimize grid dependency** by optimizing battery usage based on consumption and production patterns. The reward function directly penalizes grid usage, encouraging the agent to:

1. Use production efficiently
2. Store excess production in battery
3. Discharge battery when needed to cover consumption gaps
4. Never charge from grid (enforced constraint)

The environment provides a realistic simulation of community battery control with proper physical constraints and operational rules.

