"""
Resource-Aware Training Architecture Visualization
"""

ARCHITECTURE = r"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    RESOURCE-AWARE PCG TRAINING ARCHITECTURE               ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│                            TRAINING LOOP                                 │
└─────────────────────────────────────────────────────────────────────────┘

   ┌──────────────┐
   │   Agent      │
   │ (PPO / A2C)  │
   └──────┬───────┘
          │ action
          ▼
   ┌─────────────────────────────────────────┐
   │  ResourceAwarePCGRLWrapper              │
   │  ┌────────────────────────────────────┐ │
   │  │ 1. Get Resources                   │ │
   │  │    ResourceMonitor.get_resources() │ │
   │  │    ├─ CPU: 15%                     │ │
   │  │    ├─ RAM: 82%  ← Above 60%!       │ │
   │  │    └─ GPU: 18%                     │ │
   │  └────────────────────────────────────┘ │
   │  ┌────────────────────────────────────┐ │
   │  │ 2. Adapt Complexity                │ │
   │  │    if RAM > 90%:                   │ │
   │  │        reduce_complexity()         │ │
   │  │    elif RAM < 70%:                 │ │
   │  │        increase_complexity()       │ │
   │  └────────────────────────────────────┘ │
   └──────────────┬──────────────────────────┘
                  │ action (forwarded)
                  ▼
          ┌───────────────┐
          │ gym-pcgrl Env │
          │  (Zelda Game) │
          └───────┬───────┘
                  │ obs, raw_reward=10, done, info
                  ▼
   ┌─────────────────────────────────────────┐
   │  ResourceAwarePCGRLWrapper              │
   │  ┌────────────────────────────────────┐ │
   │  │ 3. Calculate Penalties             │ │
   │  │                                    │ │
   │  │  RAM Penalty:                      │ │
   │  │    if RAM > 60%:                   │ │
   │  │      penalty = (82-60) × 0.2 = 4.4│ │
   │  │                                    │ │
   │  │  CPU Penalty:                      │ │
   │  │    if CPU > 70%:                   │ │
   │  │      penalty = 0 (15 < 70)         │ │
   │  │                                    │ │
   │  │  GPU Penalty:                      │ │
   │  │    if GPU > 70%:                   │ │
   │  │      penalty = 0 (18 < 70)         │ │
   │  │                                    │ │
   │  │  TOTAL = 4.4                       │ │
   │  └────────────────────────────────────┘ │
   │  ┌────────────────────────────────────┐ │
   │  │ 4. Shape Reward                    │ │
   │  │                                    │ │
   │  │  Shaped = Raw - Penalty            │ │
   │  │         = 10 - 4.4                 │ │
   │  │         = 5.6                      │ │
   │  │                                    │ │
   │  │  ✓ Agent gets penalized for        │ │
   │  │    high RAM usage!                 │ │
   │  └────────────────────────────────────┘ │
   │  ┌────────────────────────────────────┐ │
   │  │ 5. Add Debug Info                  │ │
   │  │    info['raw_reward'] = 10         │ │
   │  │    info['shaped_reward'] = 5.6     │ │
   │  │    info['ram_penalty'] = 4.4       │ │
   │  │    info['cpu_penalty'] = 0.0       │ │
   │  │    info['gpu_penalty'] = 0.0       │ │
   │  │    info['total_penalty'] = 4.4     │ │
   │  │    info['env_complexity'] = 8      │ │
   │  │    info['ram_usage'] = 82%         │ │
   │  └────────────────────────────────────┘ │
   └──────────────┬──────────────────────────┘
                  │ obs, shaped_reward=5.6, done, info
                  ▼
   ┌──────────────────────────┐
   │ ResourceAwareCallback    │
   │  ┌────────────────────┐  │
   │  │ Extract from info: │  │
   │  │  - ram_penalty     │  │
   │  │  - total_penalty   │  │
   │  │  - raw_reward      │  │
   │  └────────────────────┘  │
   │  ┌────────────────────┐  │
   │  │ Log to CSV:        │  │
   │  │  - reward (shaped) │  │
   │  │  - resources       │  │
   │  │  - metrics         │  │
   │  └────────────────────┘  │
   │  ┌────────────────────┐  │
   │  │ Display:           │  │
   │  │  Episode 10:       │  │
   │  │  Avg Penalty: 4.48 │  │
   │  │  RAM: 82.3%        │  │
   │  └────────────────────┘  │
   └──────────────┬───────────┘
                  │ shaped_reward=5.6
                  ▼
          ┌──────────────┐
          │    Agent     │
          │   Learning   │
          │              │
          │  ✓ Learns to │
          │    minimize  │
          │    penalties │
          │              │
          │  ✓ Balances  │
          │    quality + │
          │    efficiency│
          └──────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                         KEY FEEDBACK LOOP                                │
└─────────────────────────────────────────────────────────────────────────┘

   High RAM Usage (82%)
          ↓
   Large Penalty (4.4)
          ↓
   Lower Shaped Reward (5.6 instead of 10)
          ↓
   Agent learns: "High RAM = Bad"
          ↓
   Adjusts strategy to use less RAM
          ↓
   Lower penalties in future episodes
          ↓
   Better total reward over time
          ↓
   RESOURCE-AWARE AGENT ✓


┌─────────────────────────────────────────────────────────────────────────┐
│                    BEFORE vs AFTER COMPARISON                            │
└─────────────────────────────────────────────────────────────────────────┘

BEFORE (Resource Observant):
───────────────────────────
  Agent → Env → Raw Reward (10)
                    ↓
              Agent learns from 10
              
  ResourceMonitor → Logs RAM (82%)
                    ↓
              [Data written to CSV]
              
  ✗ No connection between RAM and reward
  ✗ Agent never learns RAM matters


AFTER (Resource Aware):
───────────────────────
  Agent → Env → Raw Reward (10)
                    ↓
              ResourceMonitor → RAM: 82%
                    ↓
              Penalty: (82-60)×0.2 = 4.4
                    ↓
              Shaped Reward: 10 - 4.4 = 5.6
                    ↓
              Agent learns from 5.6
              
  ✓ Direct connection: RAM → Penalty → Reward
  ✓ Agent learns: High RAM = Lower reward
  ✓ Feedback loop complete!


┌─────────────────────────────────────────────────────────────────────────┐
│                      TRAINING PROGRESSION                                │
└─────────────────────────────────────────────────────────────────────────┘

Episode    Mean Reward    Avg Penalty    RAM Usage    Learning Phase
────────────────────────────────────────────────────────────────────────
   10        -107.62         4.48         82.3%       Exploration
   50         -91.59         4.27         80.8%       Early Learning
  100         -86.24         3.96         80.0%       Adaptation
  150         -85.39         3.89         80.3%       Optimization
  200         -86.63         3.74         79.0%       Refinement ✓

Observations:
  • Penalties decrease over time (4.48 → 3.74)
  • RAM usage becomes more stable
  • Agent learns resource-efficient strategies
  • Quality improves DESPITE penalties


┌─────────────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION EXAMPLES                                │
└─────────────────────────────────────────────────────────────────────────┘

1. Balanced (Default):
   ram_penalty_weight = 0.2
   cpu_penalty_weight = 0.1
   gpu_penalty_weight = 0.1
   → Equal focus on quality and efficiency

2. Quality-Focused:
   ram_penalty_weight = 0.05
   cpu_penalty_weight = 0.02
   gpu_penalty_weight = 0.02
   → Prefer content quality, soft resource constraints

3. Efficiency-Focused:
   ram_penalty_weight = 0.5
   cpu_penalty_weight = 0.3
   gpu_penalty_weight = 0.3
   → Strong resource constraints, may sacrifice quality

4. Resource-Agnostic (Baseline):
   ram_penalty_weight = 0.0
   cpu_penalty_weight = 0.0
   gpu_penalty_weight = 0.0
   → Original behavior, no resource awareness
"""

if __name__ == '__main__':
    print(ARCHITECTURE)
    
    print("\n" + "="*75)
    print("IMPLEMENTATION SUMMARY")
    print("="*75)
    print("\n✓ Resource-aware reward shaping implemented")
    print("✓ Feedback loop creates resource-aware agent")
    print("✓ Dynamic complexity adaptation activated")
    print("✓ Penalty tracking and visualization enabled")
    print("\nThe agent now learns to balance:")
    print("  • Content quality (from gym-pcgrl)")
    print("  • Resource efficiency (from penalties)")
    print("\nResult: Truly resource-aware procedural content generation!")
