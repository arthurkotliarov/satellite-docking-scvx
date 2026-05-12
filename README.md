# Satellite Docking via Sequential Convexification (SCvx)

**ETH Zurich — Planning and Decision Making for Autonomous Robots (PDM4AR) | Fall Semester 2025**

Nonlinear trajectory optimization for autonomous satellite docking using Sequential Convex Approximation (SCvx). The planner navigates a satellite from an arbitrary initial state to a docking target while avoiding static planets and moving asteroids, under thrust and attitude constraints.

---

## Overview

The core challenge is to compute a dynamically feasible, collision-free trajectory for a satellite with nonlinear dynamics in real time, including online replanning when tracking error accumulates or the horizon runs out.

The approach formulates trajectory optimization as a sequence of convex subproblems (SOCP), each solved around a linearization of the nonlinear dynamics. A trust region mechanism ensures convergence.

---

## Algorithm: Sequential Convex Approximation (SCvx)

```
Initialize: straight-line guess X_bar, U_bar, p_bar (final time)
─────────────────────────────────────────────────────────────────
For each iteration:
  1. Discretize nonlinear dynamics around (X_bar, U_bar, p_bar)
     using First-Order Hold (FOH) via ODE integration
  2. Linearize obstacle constraints (planets, asteroids)
     as supporting half-spaces around current trajectory
  3. Solve convex subproblem (SOCP) via CLARABEL / ECOS:
       minimize  fuel + time + slack penalties
       subject to  linearized dynamics
                   trust region constraints
                   thrust limits
                   boundary conditions (initial + goal state)
                   collision avoidance half-spaces
                   docking cone constraints (if applicable)
  4. Update trust region radii based on step size
  5. Check convergence → stop if step < threshold
─────────────────────────────────────────────────────────────────
Output: optimal state trajectory X*, input sequence U*, final time t_f*
```

---

## Implementation

### Satellite Dynamics (`satellite.py`)
6-state rigid body model `[x, y, ψ, vx, vy, dψ]` with dual thruster inputs `[F_L, F_R]`. Dynamics and Jacobians (A, B, F) derived symbolically via SymPy and compiled to NumPy functions for efficient evaluation.

### Discretization (`discretization.py`)
Two discretization schemes implemented:
- **Zero-Order Hold (ZOH)** — constant input assumption between knots
- **First-Order Hold (FOH)** — linearly interpolated input, produces separate `B⁺` and `B⁻` matrices

Both integrate the variational equations alongside the state ODE to compute exact discrete-time Jacobians `(A_bar, B_bar, F_bar, r_bar)`.

### Planner (`planner.py`)
Full SCvx implementation with:
- **Trust region management** — per-channel radii for states, inputs and final time; adaptive grow/shrink based on step size
- **Static obstacle avoidance** — planets modeled as circles; linearized as half-spaces at each knot
- **Dynamic obstacle avoidance** — moving asteroids with predicted positions at each time step; margin inflated with velocity magnitude
- **Docking constraints** — half-space cone constraints to enforce approach within docking arms
- **State/input scaling** — normalizes optimization variables to `[0,1]` for numerical stability
- **Solver switching** — automatically switches from CLARABEL to ECOS when moving obstacles are present

### Agent (`agent.py`)
Closed-loop tracking agent with:
- Initial trajectory computed at episode start
- Online replanning triggered by position tracking error > 0.8 m or remaining horizon < 1 s
- First-Order Hold interpolation for smooth command extraction

---

## Repository Structure

```
satellite-docking-scvx/
├── README.md
├── src/
│   ├── agent.py            # Simulation agent & replanning logic
│   ├── planner.py          # SCvx trajectory optimizer
│   ├── discretization.py   # ZOH & FOH discretization
│   └── satellite.py        # Symbolic satellite dynamics & Jacobians
├── configs/
│   ├── config_1_public.yaml
│   ├── config_2_public.yaml
│   └── config_3_public.yaml    # Scenario configuration files
└── media/
    ├── config_1_public.mp4
    ├── config_2_public.mp4
    └── config_3_public.mp4     # Simulation recordings
```

---

## Scenarios

| Scenario | Description |
|---|---|
| `config_1_public` | Basic point-to-point docking, static obstacles |
| `config_2_public` | Multiple planets, tighter approach corridor |
| `config_3_public` | Moving asteroids, dynamic avoidance required |

---

## Dependencies

```
python >= 3.10
cvxpy
numpy
scipy
sympy
dg_commons  # ETH PDM4AR simulation framework
```

---

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `K` | 50 | Discretization knots |
| `N_sub` | 5 | ODE substeps per interval |
| `lambda_nu` | 350 | Dynamics slack penalty |
| `lambda_obs` | 950 | Obstacle slack penalty |
| `max_iterations` | 25 | SCvx iteration cap |
| `tr_radius_x` | 3.0 | Initial state trust region |
| `stop_crit` | 1e-3 | Convergence threshold |

---

*ETH Zurich — Planning and Decision Making for Autonomous Robots (PDM4AR)*
*Supervisor: Prof. Dr. Emilio Frazzoli*
