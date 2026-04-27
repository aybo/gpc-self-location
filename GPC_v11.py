#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generative Predictive Coding of Self-Location — v11
===========================================================================
Paper: "Generative Predictive Coding of Self-Location in Space"
       — Erdeniz & Yıldırım

Four simulation regimes corresponding to Paper §Pathological Implications:
  1. Healthy self-location  (π_ego ≈ π_allo ≈ 1, intact T_θ)
  2. Full-bodily illusion   (π_ego >> π_allo)
  3. Out-of-body experience (π_ego < π_min, π_allo dominates)
  4. Disorientation         (T_θ impaired, precisions intact)
Requirements
numpy>=1.24,<3.0
pandas>=2.0,<3.0
matplotlib>=3.7,<4.0

here the cliff refers to the obstacle in text cliff* → obstacle* 

Dependencies: numpy, pandas, matplotlib
Usage:
  python GPC_v11.py --reproduce-paper                  # regenerate ALL paper figures
  python GPC_v11.py --batch-all                        # 50 runs × 500 steps (paper §3.6)
  python GPC_v11.py --batch-all --replay               # + offline replay/preplay
  python GPC_v11.py --batch-all --active-inference     # EFE policy enabled (now actually works)
  python GPC_v11.py --quantile-mode                    # 30 runs × 400 steps (paper §3.7)
  python GPC_v11.py --compare-all                      # 1 run/condition for visual demo
"""

import math, sys, os, time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("  Note: pandas not found. CSV export disabled. Install with: pip install pandas")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("  Note: matplotlib not found. Plot generation disabled. Install with: pip install matplotlib")

# ═══════════════════════════════════════════════════════════════════════════════
# §1  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
ARENA = 10.0
LANDMARKS = np.array([[2.0, 2.0], [8.0, 2.0], [2.0, 8.0], [8.0, 8.0]])
GOAL = np.array([8.5, 8.4])
DT = 0.1  # κ = dt (paper Notation Appendix)
NL = 4
START_POS = np.array([1.5, 1.5])
START_THETA = 0.4
DEFAULT_CLIFFS = [
    {"cx": 4.0, "cy": 3.5, "r": 1.0},
    {"cx": 6.5, "cy": 5.5, "r": 1.2},
    {"cx": 5.0, "cy": 7.5, "r": 0.9},
]

# Formal precision thresholds (paper Figure 4)
PI_MIN = 0.05          # Below this → OBE regime
PI_CONFLICT = 3.0      # |log(π_ego/π_allo)| > this → pathological regime

# ═══════════════════════════════════════════════════════════════════════════════
# §2  MATH UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def wrap_angle(a): return (a + np.pi) % (2 * np.pi) - np.pi
def rotation_matrix(theta):
    c, s = math.cos(-theta), math.sin(-theta)
    return np.array([[c, -s], [s, c]], dtype=float)
def mv(R, v): return R @ v
def safe_norm(v, eps=1e-9): return float(np.sqrt(np.sum(np.asarray(v)**2) + eps**2))
def norm2(a, b): return math.sqrt(a*a + b*b + 1e-18)
def clamp(x, lo, hi): return max(lo, min(hi, x))

def symmetrise(S): return 0.5 * (S + S.T)
def stabilise_cov(S, jitter=1e-7):
    """Ensure covariance is symmetric positive-definite."""
    S = symmetrise(np.array(S, dtype=float))
    vals = np.linalg.eigvalsh(S)
    if float(np.min(vals)) < jitter:
        S = S + np.eye(S.shape[0]) * (jitter - float(np.min(vals)) + 1e-9)
    return symmetrise(S)

# ═══════════════════════════════════════════════════════════════════════════════
# §3  CLIFF HAZARD SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
def cliff_repulsion(pos, cliffs, margin=0.6, strength=2.5):
    fx, fy = 0.0, 0.0
    for c in cliffs:
        dx, dy = pos[0]-c["cx"], pos[1]-c["cy"]
        d = math.sqrt(dx*dx+dy*dy+1e-12)
        bd = c["r"]+margin
        if d < bd:
            p = (bd-d)/bd; m = strength*p*p
            fx += m*dx/d; fy += m*dy/d
    return np.array([fx, fy])

def inside_cliff(pos, cliffs):
    for c in cliffs:
        dx, dy = pos[0]-c["cx"], pos[1]-c["cy"]
        if dx*dx+dy*dy < c["r"]**2: return c
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# §4  CONDITION PRESETS (Paper §Pathological Implications, Figure 4)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ConditionPreset:
    key: str; label: str; color: str
    pi_ego: float; pi_allo: float; t_impair: float
    max_step_pos: float; max_step_theta: float
    desc: str
    dynamic_precision: bool = False
    precision_adapt_rate: float = 0.02

CONDITION_PRESETS: Dict[str, ConditionPreset] = {
    "healthy": ConditionPreset(
        key="healthy", label="Healthy (balanced)", color="#2563eb",
        pi_ego=1.0, pi_allo=1.0, t_impair=0.0,
        max_step_pos=0.18, max_step_theta=0.25,
        dynamic_precision=True, precision_adapt_rate=0.02,
        desc="π_ego≈π_allo≈1. Intact T_θ. Dynamic precision adaptation enabled."),
    "bodily": ConditionPreset(
        key="bodily", label="Full-Bodily Illusion (π_ego>>π_allo)", color="#d97706",
        pi_ego=6.0, pi_allo=0.05, t_impair=0.0,
        max_step_pos=0.40, max_step_theta=0.50,
        desc="π_ego=6>>π_allo=0.05. Sensory PE dominates → jittery, map-unstable."),
    "obe": ConditionPreset(
        key="obe", label="OBE (π_ego<π_min)", color="#dc2626",
        pi_ego=0.02, pi_allo=6.0, t_impair=0.0,
        max_step_pos=0.18, max_step_theta=0.25,
        desc="π_ego=0.02<π_min. Allocentric prior dominates → mislocalized."),
    "disorientation": ConditionPreset(
        key="disorientation", label="Disorientation (T_impair=0.85)", color="#7c3aed",
        pi_ego=1.0, pi_allo=1.0, t_impair=0.85,
        max_step_pos=0.35, max_step_theta=0.45,
        desc="T_impair=0.85. RSC broken, precisions intact → chaotic."),
}

DEFAULTS = {
    "base_prec_dist": 1.2, "base_prec_bearing": 3.0,
    "state_prec_dist": 0.8, "state_prec_bearing": 2.0,
    "prior_prec_pos": 0.2, "prior_prec_theta": 0.1,
    "speed": 1.0, "turn_gain": 3.0,
    "motor_noise_xy": 0.01, "motor_noise_theta": 0.008,
    "obs_noise_dist": 0.08, "obs_noise_bearing": 0.03,
}

# ═══════════════════════════════════════════════════════════════════════════════
# §5  RSC TRANSFORM T_θ = R(θ)(x_target − x_allo) + b
# ═══════════════════════════════════════════════════════════════════════════════
def rsc_transform(mu_allo, mu_theta, bias, t_impair, rng, use_noise=True):
    eff = mu_theta
    if t_impair > 0:
        eff *= (1.0-0.5*t_impair); eff += 0.4*t_impair
        if use_noise: eff += rng.normal(0.0, 1.2*t_impair)
    R = rotation_matrix(eff)
    preds = np.zeros((NL, 2))
    for i, lm in enumerate(LANDMARKS):
        rel = lm - mu_allo; body = mv(R, rel)
        bx, by = body[0]+bias[0], body[1]+bias[1]
        preds[i] = [max(math.sqrt(bx*bx+by*by), 1e-6), wrap_angle(math.atan2(by, bx))]
    return preds

def jacobian_rsc(mu_allo, mu_theta, bias, t_impair, rng):
    """J = ∂g/∂μ_s ∈ ℝ^(2NL×3), noise-free finite differences."""
    eps = 1e-3
    flat = lambda p: np.concatenate([p[:, 0], p[:, 1]])
    J = np.zeros((2*NL, 3))
    for col in range(3):
        if col < 2:
            mp, mm = mu_allo.copy(), mu_allo.copy()
            mp[col] += eps; mm[col] -= eps
            pf = flat(rsc_transform(mp, mu_theta, bias, t_impair, rng, False))
            mf = flat(rsc_transform(mm, mu_theta, bias, t_impair, rng, False))
        else:
            pf = flat(rsc_transform(mu_allo, wrap_angle(mu_theta+eps), bias, t_impair, rng, False))
            mf = flat(rsc_transform(mu_allo, wrap_angle(mu_theta-eps), bias, t_impair, rng, False))
        diff = (pf - mf) / (2*eps)
        for r in range(NL, 2*NL): diff[r] = wrap_angle(diff[r])
        J[:, col] = diff
    return J

# ═══════════════════════════════════════════════════════════════════════════════
# §6  (reserved)
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# §7  SIMULATION STATE
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class SimState:
    true_pos: np.ndarray; true_theta: float
    mu_allo: np.ndarray; mu_theta: float
    mu_allo_prior: np.ndarray; mu_theta_prior: float
    mu_ego: np.ndarray; bias: np.ndarray
    # Posterior uncertainty Σ_s ∈ ℝ^(3×3) for (x, y, θ)
    Sigma_s: np.ndarray
    # Dynamic effective precisions
    pi_ego_eff: float; pi_allo_eff: float
    smoothed_eps_o: float  # EMA of mean |ε_o|
    # Histories
    true_path: List; belief_path: List
    F_hist: List; err_hist: List; acc_o_hist: List; comp_hist: List
    drift_hist: List; cliff_penalty_hist: List
    # Prediction error histories
    eps_o_mean_hist: List    # mean |ε_o| per step
    eps_s_mean_hist: List    # mean |ε_s| per step
    eps_o_dist_hist: List    # mean ε_o distance component
    eps_o_bear_hist: List    # mean ε_o bearing component
    # Uncertainty & info gain histories
    uncertainty_hist: List   # tr(Σ_s) per step
    info_gain_hist: List     # 0.5*(log|Σ_prior| - log|Σ_post|)
    # Precision histories
    pi_ego_eff_hist: List
    pi_allo_eff_hist: List
    # acc_s as diagnostic only
    acc_s_diagnostic_hist: List
    # Distance tracking
    dist_to_goal_hist: List
    dist_traveled: float
    step: int; cliff_hits: int
    reached_goal: bool; reached_goal_step: int

@dataclass
class SimParams:
    pi_ego: float; pi_allo: float; t_impair: float
    prior_prec_pos: float; prior_prec_theta: float
    base_prec_dist: float; base_prec_bearing: float
    state_prec_dist: float; state_prec_bearing: float
    speed: float; turn_gain: float
    motor_noise_xy: float; motor_noise_theta: float
    obs_noise_dist: float; obs_noise_bearing: float
    max_step_pos: float; max_step_theta: float
    cliffs: List[dict]; cliff_avoid_gain: float
    # Dynamic precision
    dynamic_precision: bool = False
    precision_adapt_rate: float = 0.02
    # Active inference
    use_active_inference: bool = False
    n_policy_samples: int = 5
    policy_horizon: int = 5
    efe_gamma: float = 4.0
    pragmatic_weight: float = 2.4
    epistemic_weight: float = 0.45
    wall_avoid_weight: float = 1.3

def make_params(preset, overrides=None):
    d = dict(DEFAULTS)
    d["pi_ego"] = preset.pi_ego; d["pi_allo"] = preset.pi_allo
    d["t_impair"] = preset.t_impair
    d["max_step_pos"] = preset.max_step_pos; d["max_step_theta"] = preset.max_step_theta
    d["cliffs"] = [dict(c) for c in DEFAULT_CLIFFS]; d["cliff_avoid_gain"] = 2.5
    d["dynamic_precision"] = preset.dynamic_precision
    d["precision_adapt_rate"] = preset.precision_adapt_rate
    if overrides: d.update(overrides)
    return SimParams(**d)

def create_sim_state(start_pos=None, start_theta=None):
    sp = start_pos if start_pos is not None else START_POS.copy()
    st = start_theta if start_theta is not None else START_THETA
    return SimState(
        true_pos=sp.copy(), true_theta=st,
        mu_allo=sp.copy(), mu_theta=st,
        mu_allo_prior=sp.copy(), mu_theta_prior=st,
        mu_ego=np.zeros((NL, 2)), bias=np.zeros(2),
        Sigma_s=np.diag([0.45**2, 0.45**2, 0.25**2]).astype(float),
        pi_ego_eff=1.0, pi_allo_eff=1.0, smoothed_eps_o=0.5,
        true_path=[sp.copy()], belief_path=[sp.copy()],
        F_hist=[], err_hist=[], acc_o_hist=[], comp_hist=[],
        drift_hist=[], cliff_penalty_hist=[],
        eps_o_mean_hist=[], eps_s_mean_hist=[],
        eps_o_dist_hist=[], eps_o_bear_hist=[],
        uncertainty_hist=[], info_gain_hist=[],
        pi_ego_eff_hist=[], pi_allo_eff_hist=[],
        acc_s_diagnostic_hist=[],
        dist_to_goal_hist=[], dist_traveled=0.0,
        step=0, cliff_hits=0, reached_goal=False, reached_goal_step=-1)

# ═══════════════════════════════════════════════════════════════════════════════
# §8  ACTIVE INFERENCE POLICY — Expected Free Energy G
# ═══════════════════════════════════════════════════════════════════════════════
# Paper §Active Inference: G(π) = E_Q[log Q(s) − log P(o,s|π)]
# Decomposed: epistemic (info gain) + pragmatic (goal proximity)
# P(π) ∝ exp(−γ G(π))

def evaluate_policy(mu_xy, mu_theta, Sigma, angle_offset, p, dt=DT):
    """Evaluate one candidate policy by rolling out for p.policy_horizon steps."""
    w_o = np.concatenate([np.full(NL, p.pi_ego*p.base_prec_dist),
                          np.full(NL, p.pi_ego*p.base_prec_bearing)])
    total_G = 0.0
    _mu = mu_xy.copy(); _th = float(mu_theta); _Sig = Sigma.copy()
    Q_proc = np.diag([(p.motor_noise_xy*dt)**2, (p.motor_noise_xy*dt)**2,
                       (p.motor_noise_theta*dt)**2])

    for h in range(p.policy_horizon):
        # Goal-directed + offset policy
        gv = GOAL - _mu; gd = safe_norm(gv)
        des = math.atan2(gv[1], gv[0]) + angle_offset
        hErr = wrap_angle(des - _th)
        omega = p.turn_gain * hErr
        v = p.speed * math.tanh(gd)

        # Transition
        _th_new = wrap_angle(_th + omega*dt)
        _mu_new = _mu + np.array([v*dt*math.cos(_th_new), v*dt*math.sin(_th_new)])
        _mu_new = np.clip(_mu_new, 0.2, ARENA-0.2)

        # Propagate uncertainty: Σ_prior = A Σ A^T + Q
        A = np.eye(3); A[0,2] = -v*dt*math.sin(_th_new); A[1,2] = v*dt*math.cos(_th_new)
        Sig_prior = stabilise_cov(A @ _Sig @ A.T + Q_proc)

        # Jacobian uses noise-free finite differences; rng is unused but
        # passed for API consistency with the noisy-RSC code path.
        J = jacobian_rsc(_mu_new, _th_new, np.zeros(2), p.t_impair,
                         np.random.default_rng())

        # [C1] Fixed in v11: removed stray trailing comma. In v10 this read
        # `Pi_o = np.diag(...),` — the comma silently turned Pi_o into a
        # 1-element tuple, causing every J.T @ Pi_o @ J to raise ValueError
        # that was masked by `except Exception:` in sim_step(). Result:
        # active inference never executed; heuristic fallback always used.
        Pi_o = np.diag(np.maximum(w_o, 1e-3))

        # Posterior uncertainty: Σ_post = (Σ_prior^-1 + J^T Π_o J)^-1
        try:
            info_mat = J.T @ Pi_o @ J
            Sig_post = stabilise_cov(np.linalg.inv(np.linalg.inv(Sig_prior) + info_mat))
        except np.linalg.LinAlgError:
            Sig_post = Sig_prior

        # Epistemic value: info gain = 0.5*(log|Σ_prior| - log|Σ_post|)
        sp, lp = np.linalg.slogdet(Sig_prior)
        sq, lq = np.linalg.slogdet(Sig_post)
        ig = max(0.0, 0.5*(lp - lq)) if sp > 0 and sq > 0 else 0.0

        # Pragmatic value: proximity to goal
        prag = (safe_norm(GOAL - _mu_new) / ARENA)**2

        # Wall penalty
        mx, my = float(_mu_new[0]), float(_mu_new[1])
        wall = max(0, 0.7-mx)**2 + max(0, mx-(ARENA-0.7))**2 + \
               max(0, 0.7-my)**2 + max(0, my-(ARENA-0.7))**2

        discount = 0.96**h
        total_G += discount * (p.pragmatic_weight*prag + p.wall_avoid_weight*wall
                               - p.epistemic_weight*ig)
        _mu = _mu_new; _th = _th_new; _Sig = Sig_post

    return total_G

def active_inference_policy(s, p):
    """
    Select action via expected free energy minimization.
    Paper: P(π) ∝ exp(-γ G(π))
    """
    offsets = np.array([-0.95, -0.45, 0.0, 0.45, 0.95])
    Gs = np.array([evaluate_policy(s.mu_allo, s.mu_theta, s.Sigma_s, off, p)
                   for off in offsets])

    # Softmax: P(π) ∝ exp(-γG)
    shifted = -p.efe_gamma * (Gs - np.min(Gs))
    shifted = np.clip(shifted, -60, 60)
    probs = np.exp(shifted); probs /= probs.sum()

    if not np.all(np.isfinite(probs)):
        # Fallback to heuristic
        gv = GOAL - s.mu_allo; gd = safe_norm(gv)
        des = math.atan2(gv[1], gv[0])
        return p.speed*math.tanh(gd), p.turn_gain*wrap_angle(des-s.mu_theta)

    # Weighted action
    best_off = float(np.sum(probs * offsets))
    gv = GOAL - s.mu_allo; gd = safe_norm(gv)
    rep = cliff_repulsion(s.mu_allo, p.cliffs, margin=0.7, strength=p.cliff_avoid_gain)
    dd = np.array([gv[0]/(gd+1e-9)+rep[0], gv[1]/(gd+1e-9)+rep[1]])
    dd /= safe_norm(dd)
    des = math.atan2(dd[1], dd[0]) + best_off
    omega = p.turn_gain * wrap_angle(des - s.mu_theta)
    v = p.speed * math.tanh(gd)
    return v, omega

# ═══════════════════════════════════════════════════════════════════════════════
# §9  SIMULATION STEP
# ═══════════════════════════════════════════════════════════════════════════════
def sim_step(s, p, rng):
    # ── Dynamic precision adaptation ──
    if p.dynamic_precision and s.step > 0:
        alpha = p.precision_adapt_rate
        s.pi_ego_eff = p.pi_ego * (1.0 + alpha * s.smoothed_eps_o)
        s.pi_allo_eff = p.pi_allo * max(0.05, 1.0 - alpha * s.smoothed_eps_o * 0.5)
    else:
        s.pi_ego_eff = p.pi_ego
        s.pi_allo_eff = p.pi_allo

    pi_ego = s.pi_ego_eff
    pi_allo = s.pi_allo_eff

    # ── Policy selection ──
    if p.use_active_inference and s.step > 5 and s.step % 8 == 0:
        try:
            v, omega = active_inference_policy(s, p)
        except (np.linalg.LinAlgError, ValueError):
            # [C2] v11: typed catch (was bare `except Exception:` in v10).
            # Legitimate LinAlgError can happen on near-singular Σ (rare);
            # ValueError is kept because numpy raises it on shape/broadcast
            # issues that can occur transiently in degenerate states. Other
            # exception types (TypeError, AttributeError, IndexError, …)
            # now propagate so programming bugs are not silently masked.
            gv = GOAL - s.mu_allo; gd = norm2(gv[0], gv[1])
            dd = np.array([gv[0]/(gd+1e-9), gv[1]/(gd+1e-9)])
            rep = cliff_repulsion(s.mu_allo, p.cliffs, margin=0.7, strength=p.cliff_avoid_gain)
            dd += rep; dd /= safe_norm(dd)
            omega = p.turn_gain * wrap_angle(math.atan2(dd[1], dd[0]) - s.mu_theta)
            v = p.speed * math.tanh(gd)
    else:
        gv = GOAL - s.mu_allo; gd = norm2(gv[0], gv[1])
        dd = np.array([gv[0]/(gd+1e-9), gv[1]/(gd+1e-9)])
        rep = cliff_repulsion(s.mu_allo, p.cliffs, margin=0.7, strength=p.cliff_avoid_gain)
        dd += rep; dd /= safe_norm(dd)
        omega = p.turn_gain * wrap_angle(math.atan2(dd[1], dd[0]) - s.mu_theta)
        v = p.speed * math.tanh(gd)

    # Save prior BEFORE transition
    s.mu_allo_prior = s.mu_allo.copy(); s.mu_theta_prior = s.mu_theta
    Sigma_prior = s.Sigma_s.copy()

    # Belief transition
    s.mu_theta = wrap_angle(s.mu_theta + omega*DT)
    s.mu_allo[0] += v*DT*math.cos(s.mu_theta)
    s.mu_allo[1] += v*DT*math.sin(s.mu_theta)
    s.mu_allo = np.clip(s.mu_allo, 0.2, ARENA-0.2)

    # Propagate uncertainty through transition
    A = np.eye(3); A[0,2] = -v*DT*math.sin(s.mu_theta); A[1,2] = v*DT*math.cos(s.mu_theta)
    Q = np.diag([(p.motor_noise_xy*DT)**2, (p.motor_noise_xy*DT)**2,
                  (p.motor_noise_theta*DT)**2])
    s.Sigma_s = stabilise_cov(A @ Sigma_prior @ A.T + Q)

    # True state transition (external η — hidden)
    prev_true = s.true_pos.copy()
    s.true_theta = wrap_angle(s.true_theta + omega*DT + rng.normal(0, p.motor_noise_theta))
    s.true_pos[0] += v*DT*math.cos(s.true_theta) + rng.normal(0, p.motor_noise_xy)
    s.true_pos[1] += v*DT*math.sin(s.true_theta) + rng.normal(0, p.motor_noise_xy)
    lo, hi = 0.2, ARENA-0.2
    if s.true_pos[0]<lo or s.true_pos[0]>hi:
        s.true_theta = wrap_angle(math.pi-s.true_theta); s.true_pos[0] = clamp(s.true_pos[0],lo,hi)
    if s.true_pos[1]<lo or s.true_pos[1]>hi:
        s.true_theta = wrap_angle(-s.true_theta); s.true_pos[1] = clamp(s.true_pos[1],lo,hi)
    hit = inside_cliff(s.true_pos, p.cliffs)
    cliff_pen = 0.0
    if hit:
        dx,dy = s.true_pos[0]-hit["cx"], s.true_pos[1]-hit["cy"]
        d = math.sqrt(dx*dx+dy*dy+1e-12)
        s.true_pos[0] = hit["cx"]+(dx/d)*(hit["r"]+0.05)
        s.true_pos[1] = hit["cy"]+(dy/d)*(hit["r"]+0.05)
        s.true_theta = wrap_angle(s.true_theta+math.pi*0.5+rng.normal(0,0.3))
        cliff_pen = 1.0; s.cliff_hits += 1
    s.dist_traveled += norm2(s.true_pos[0]-prev_true[0], s.true_pos[1]-prev_true[1])
    dGoal = norm2(s.true_pos[0]-GOAL[0], s.true_pos[1]-GOAL[1])
    s.dist_to_goal_hist.append(dGoal)
    if not s.reached_goal and dGoal < 1.0:
        s.reached_goal = True; s.reached_goal_step = s.step

    # Observation — from true body position
    # Paper §P(o|x_ego): egocentric likelihood from TPJ
    Rt = rotation_matrix(s.true_theta)
    obs_ego = np.zeros((NL, 2))
    for i, lm in enumerate(LANDMARKS):
        body = mv(Rt, lm-s.true_pos)
        obs_ego[i] = [max(math.sqrt(body[0]**2+body[1]**2)+rng.normal(0,p.obs_noise_dist), 0.01),
                      wrap_angle(math.atan2(body[1],body[0])+rng.normal(0,p.obs_noise_bearing))]

    # RSC prediction g(μ_s)
    pred_ego = rsc_transform(s.mu_allo, s.mu_theta, s.bias, p.t_impair, rng, True)

    # Sensory PE: ε_o = o − g(μ_s)
    eps_o = np.zeros((NL, 2))
    for i in range(NL):
        eps_o[i, 0] = obs_ego[i, 0] - pred_ego[i, 0]
        eps_o[i, 1] = wrap_angle(obs_ego[i, 1] - pred_ego[i, 1])

    # Precision weights (using effective precisions)
    w_o_dist = pi_ego * p.base_prec_dist
    w_o_bear = pi_ego * p.base_prec_bearing

    # TPJ update
    for i in range(NL):
        s.mu_ego[i, 0] = max(s.mu_ego[i, 0] + DT*w_o_dist*eps_o[i, 0], 0.02)
        s.mu_ego[i, 1] = wrap_angle(s.mu_ego[i, 1] + DT*w_o_bear*eps_o[i, 1])

    # State PE: ε_s = μ_ego − T_θ(μ_allo)
    eps_s = np.zeros((NL, 2))
    for i in range(NL):
        eps_s[i, 0] = s.mu_ego[i, 0] - pred_ego[i, 0]
        eps_s[i, 1] = wrap_angle(s.mu_ego[i, 1] - pred_ego[i, 1])

    # Jacobian + allocentric update:
    # μ_s ← μ_s + κ (∂g/∂μ_s)ᵀ Π_o ε_o − κ Π_prior (μ − μ_prior)
    J = jacobian_rsc(s.mu_allo, s.mu_theta, s.bias, p.t_impair, rng)
    eps_flat = np.concatenate([eps_o[:, 0], eps_o[:, 1]])
    w_o = np.concatenate([np.full(NL, w_o_dist), np.full(NL, w_o_bear)])
    weighted = eps_flat * w_o
    delta = np.zeros(3)
    for c in range(3):
        for r in range(2*NL): delta[c] += J[r, c] * weighted[r]
        delta[c] *= DT
    # Complexity gradient
    delta[0] -= DT * pi_allo * p.prior_prec_pos * (s.mu_allo[0]-s.mu_allo_prior[0])
    delta[1] -= DT * pi_allo * p.prior_prec_pos * (s.mu_allo[1]-s.mu_allo_prior[1])
    delta[2] -= DT * pi_allo * p.prior_prec_theta * wrap_angle(s.mu_theta-s.mu_theta_prior)
    # Stability clamp
    dn1 = math.sqrt(delta[0]**2+delta[1]**2)
    if dn1 > p.max_step_pos: delta[:2] *= p.max_step_pos/dn1
    delta[2] = clamp(delta[2], -p.max_step_theta, p.max_step_theta)
    s.mu_allo += delta[:2]; s.mu_theta = wrap_angle(s.mu_theta+delta[2])
    s.mu_allo = np.clip(s.mu_allo, 0.2, ARENA-0.2)

    # Posterior uncertainty update: Σ_post = (Σ_prior⁻¹ + (∂g/∂μ)ᵀ Π_o (∂g/∂μ))⁻¹
    try:
        Pi_o = np.diag(np.maximum(w_o, 1e-3))
        info_mat = J.T @ Pi_o @ J
        Sig_post = stabilise_cov(np.linalg.inv(np.linalg.inv(s.Sigma_s) + info_mat))
        # Info gain
        sp, lp = np.linalg.slogdet(s.Sigma_s)
        sq, lq = np.linalg.slogdet(Sig_post)
        info_gain = max(0.0, 0.5*(lp-lq)) if sp>0 and sq>0 else 0.0
        s.Sigma_s = Sig_post
    except np.linalg.LinAlgError:
        info_gain = 0.0

    # Bias learning: Δb ∝ −∂F/∂θ
    me = np.array([np.mean(eps_o[:, 0]), np.mean(eps_o[:, 1])])
    s.bias[0] = clamp(s.bias[0]+0.005*pi_ego*me[0], -1, 1)
    s.bias[1] = clamp(s.bias[1]+0.005*pi_ego*me[1], -1, 1)

    # Free energy: F = accuracy + complexity (paper §2.1.3, Eq. 14)
    acc_o = 0.5 * float(np.sum(w_o * eps_flat**2))
    comp = 0.5 * pi_allo * (
        p.prior_prec_pos*((s.mu_allo[0]-s.mu_allo_prior[0])**2 +
                           (s.mu_allo[1]-s.mu_allo_prior[1])**2)
        + p.prior_prec_theta*wrap_angle(s.mu_theta-s.mu_theta_prior)**2)
    F = acc_o + comp

    # acc_s as DIAGNOSTIC only
    w_s = np.concatenate([np.full(NL, pi_ego*p.state_prec_dist),
                          np.full(NL, pi_ego*p.state_prec_bearing)])
    eps_s_flat = np.concatenate([eps_s[:, 0], eps_s[:, 1]])
    acc_s_diag = 0.5 * float(np.sum(w_s * eps_s_flat**2))

    err = norm2(s.true_pos[0]-s.mu_allo[0], s.true_pos[1]-s.mu_allo[1])
    drift = norm2(s.mu_allo[0]-s.mu_allo_prior[0], s.mu_allo[1]-s.mu_allo_prior[1])

    # Update smoothed PE for dynamic precision
    mean_abs_eps_o = float(np.mean(np.abs(eps_o)))
    s.smoothed_eps_o = 0.95*s.smoothed_eps_o + 0.05*mean_abs_eps_o

    # Track everything
    s.true_path.append(s.true_pos.copy())
    s.belief_path.append(s.mu_allo.copy())
    s.F_hist.append(F); s.err_hist.append(err)
    s.acc_o_hist.append(acc_o); s.comp_hist.append(comp)
    s.drift_hist.append(drift); s.cliff_penalty_hist.append(cliff_pen)
    s.eps_o_mean_hist.append(float(np.mean(np.abs(eps_o))))
    s.eps_s_mean_hist.append(float(np.mean(np.abs(eps_s))))
    s.eps_o_dist_hist.append(float(np.mean(np.abs(eps_o[:, 0]))))
    s.eps_o_bear_hist.append(float(np.mean(np.abs(eps_o[:, 1]))))
    s.uncertainty_hist.append(float(np.trace(s.Sigma_s)))
    s.info_gain_hist.append(info_gain)
    s.pi_ego_eff_hist.append(pi_ego)
    s.pi_allo_eff_hist.append(pi_allo)
    s.acc_s_diagnostic_hist.append(acc_s_diag)
    s.step += 1
    return s

# ═══════════════════════════════════════════════════════════════════════════════
# §10  OFFLINE REPLAY & SIMULATION — Paper §Generative Model + Figure 1 Step 3
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReplayResult:
    replay_path: np.ndarray
    egocentric_obs: List[np.ndarray]
    replay_order: str
    replay_length: int
    mean_local_fe: float
    local_fe_trace: np.ndarray
    eps_o_trace: np.ndarray
    eps_s_trace: np.ndarray
    belief_path: np.ndarray
    min_dist_to_goal: float
    reached_goal: bool


def _replay_transition_step(mu_pos, mu_theta, waypoint, params, rng):
    """
    Transition model P(s_t|s_{t-1}) — uses the AGENT'S learned parameters.
    """
    vec = waypoint - mu_pos
    dist = safe_norm(vec)
    if dist > 0.1:
        desired = math.atan2(vec[1], vec[0])
    else:
        desired = math.atan2(GOAL[1] - mu_pos[1], GOAL[0] - mu_pos[0])

    heading_err = wrap_angle(desired - mu_theta)
    omega = params.turn_gain * heading_err
    v = params.speed * math.tanh(dist)

    new_theta = wrap_angle(mu_theta + omega * DT + rng.normal(0, params.motor_noise_theta))
    new_pos = mu_pos.copy()
    new_pos[0] += v * DT * math.cos(new_theta) + rng.normal(0, params.motor_noise_xy)
    new_pos[1] += v * DT * math.sin(new_theta) + rng.normal(0, params.motor_noise_xy)
    new_pos = np.clip(new_pos, 0.2, ARENA - 0.2)

    return new_pos, new_theta, v, omega


def _replay_observation_step(mu_pos, mu_theta, bias, t_impair, params, rng):
    """
    Observation model at a counterfactual location — RSC/TPJ restores
    egocentric perspectives for generative simulation.
    """
    pred_ego = rsc_transform(mu_pos, mu_theta, bias, t_impair, rng, use_noise=False)

    synth_obs = np.zeros((NL, 2))
    for i in range(NL):
        synth_obs[i, 0] = max(pred_ego[i, 0] + rng.normal(0, params.obs_noise_dist), 0.01)
        synth_obs[i, 1] = wrap_angle(pred_ego[i, 1] + rng.normal(0, params.obs_noise_bearing))

    return synth_obs, pred_ego


def _replay_belief_update(mu_pos, mu_theta, mu_pos_prior, mu_theta_prior,
                          bias, mu_ego, synth_obs, pred_ego,
                          params, pi_ego, pi_allo, rng):
    """
    [A1] Full belief update at a counterfactual location, now including
    the complexity gradient to match the online update in sim_step().

    μ_s ← μ_s + κ (∂g/∂μ_s)ᵀ Π_o ε_o − κ Π_prior (μ − μ_prior)

    mu_pos_prior / mu_theta_prior: the position BEFORE the transition
    step, serving as the "prior" that the complexity term pulls toward.
    """
    # Sensory PE: ε_o = o_synth − g(μ_s)
    eps_o = np.zeros((NL, 2))
    for i in range(NL):
        eps_o[i, 0] = synth_obs[i, 0] - pred_ego[i, 0]
        eps_o[i, 1] = wrap_angle(synth_obs[i, 1] - pred_ego[i, 1])

    # State PE: ε_s = μ_ego − T_θ(μ_allo)
    eps_s = np.zeros((NL, 2))
    for i in range(NL):
        eps_s[i, 0] = mu_ego[i, 0] - pred_ego[i, 0]
        eps_s[i, 1] = wrap_angle(mu_ego[i, 1] - pred_ego[i, 1])

    # Precision weights
    w_o_dist = pi_ego * params.base_prec_dist
    w_o_bear = pi_ego * params.base_prec_bearing

    # Jacobian at replayed position
    J = jacobian_rsc(mu_pos, mu_theta, bias, params.t_impair, rng)
    eps_flat = np.concatenate([eps_o[:, 0], eps_o[:, 1]])
    w_o = np.concatenate([np.full(NL, w_o_dist), np.full(NL, w_o_bear)])
    weighted = eps_flat * w_o

    # Accuracy gradient: δμ = κ J^T Π_o ε_o
    delta = np.zeros(3)
    for c in range(3):
        for r in range(2 * NL):
            delta[c] += J[r, c] * weighted[r]
        delta[c] *= DT

    # [A1] Complexity gradient: −κ Π_prior (μ − μ_prior)
    # Matches the online update in sim_step()
    delta[0] -= DT * pi_allo * params.prior_prec_pos * (mu_pos[0] - mu_pos_prior[0])
    delta[1] -= DT * pi_allo * params.prior_prec_pos * (mu_pos[1] - mu_pos_prior[1])
    delta[2] -= DT * pi_allo * params.prior_prec_theta * wrap_angle(mu_theta - mu_theta_prior)

    # Apply update (mental position shift during replay)
    updated_pos = mu_pos.copy()
    updated_pos[0] += clamp(delta[0], -params.max_step_pos, params.max_step_pos)
    updated_pos[1] += clamp(delta[1], -params.max_step_pos, params.max_step_pos)
    updated_theta = wrap_angle(mu_theta + clamp(delta[2], -params.max_step_theta, params.max_step_theta))
    updated_pos = np.clip(updated_pos, 0.2, ARENA - 0.2)

    # [A2] Free energy: accuracy + complexity using (μ − μ_prior)
    acc_o = 0.5 * float(np.sum(w_o * eps_flat**2))
    comp = 0.5 * pi_allo * (
        params.prior_prec_pos * ((mu_pos[0] - mu_pos_prior[0])**2 +
                                  (mu_pos[1] - mu_pos_prior[1])**2) +
        params.prior_prec_theta * wrap_angle(mu_theta - mu_theta_prior)**2)
    F_local = acc_o + comp

    mean_eps_o = float(np.mean(np.abs(eps_o)))
    mean_eps_s = float(np.mean(np.abs(eps_s)))

    return updated_pos, updated_theta, F_local, mean_eps_o, mean_eps_s


def generate_replay_path(seed_positions, seed_thetas, state, params,
                         replay_order="forward", replay_horizon=60,
                         rng=None, goal=None):
    """
    Sample a trajectory from the agent's generative model:
    s̃_1:T ~ P(s_1:T | θ_updated)
    """
    if rng is None:
        rng = np.random.default_rng()
    if goal is None:
        goal = GOAL.copy()
    n = len(seed_positions)
    if n < 2:
        return ReplayResult(
            np.array([[1.5, 1.5]]), [], replay_order, 1, 0.0,
            np.array([0.0]), np.array([0.0]), np.array([0.0]),
            np.array([[1.5, 1.5]]), float("inf"), False)

    # Determine replay ordering
    if replay_order == "reverse":
        idx = np.arange(n - 1, -1, -1)
    elif replay_order == "noisy":
        idx = rng.permutation(n)
    else:  # forward
        idx = np.arange(n)

    # Initialize from stored belief trajectory
    mu_pos = seed_positions[idx[0]].copy()
    mu_theta = float(seed_thetas[min(idx[0], len(seed_thetas) - 1)])
    mu_ego = state.mu_ego.copy()
    bias = state.bias.copy()
    pi_ego = state.pi_ego_eff
    pi_allo = state.pi_allo_eff

    belief_pos = mu_pos.copy()

    # Trace arrays
    positions = [mu_pos.copy()]
    beliefs = [belief_pos.copy()]
    ego_obs_list = []
    fe_trace = []
    eo_trace = []
    es_trace = []

    # [A1] Track prior position for complexity gradient
    prev_pos = mu_pos.copy()
    prev_theta = float(mu_theta)

    for si in range(replay_horizon):
        wi = idx[min(si + 1, len(idx) - 1)]
        waypoint = seed_positions[min(wi, n - 1)]

        # (1) TRANSITION: P(s_t|s_{t-1}) with agent's parameters
        mu_pos, mu_theta, v, omega = _replay_transition_step(
            mu_pos, mu_theta, waypoint, params, rng)

        if replay_order == "noisy":
            mu_pos += rng.normal(0, 0.04, 2)
            mu_pos = np.clip(mu_pos, 0.2, ARENA - 0.2)

        positions.append(mu_pos.copy())

        # (2) OBSERVATION: RSC/TPJ restores egocentric perspective
        synth_obs, pred_ego = _replay_observation_step(
            mu_pos, mu_theta, bias, params.t_impair, params, rng)
        ego_obs_list.append(synth_obs.copy())

        # (3) BELIEF UPDATE: F-minimization over internal states
        #     [A1] Now includes complexity gradient with prior tracking
        belief_pos, mu_theta, F_local, mean_eo, mean_es = _replay_belief_update(
            mu_pos, mu_theta, prev_pos, prev_theta,
            bias, mu_ego, synth_obs, pred_ego,
            params, pi_ego, pi_allo, rng)

        beliefs.append(belief_pos.copy())
        fe_trace.append(F_local)
        eo_trace.append(mean_eo)
        es_trace.append(mean_es)

        # Update prior for next step
        prev_pos = mu_pos.copy()
        prev_theta = float(mu_theta)

    path = np.array(positions)
    belief_path = np.array(beliefs)
    md = float(np.min(np.linalg.norm(path - goal[None, :], axis=1)))

    return ReplayResult(
        replay_path=path,
        egocentric_obs=ego_obs_list,
        replay_order=replay_order,
        replay_length=len(path),
        mean_local_fe=float(np.mean(fe_trace)) if fe_trace else 0.0,
        local_fe_trace=np.array(fe_trace),
        eps_o_trace=np.array(eo_trace),
        eps_s_trace=np.array(es_trace),
        belief_path=belief_path,
        min_dist_to_goal=md,
        reached_goal=md <= 0.6)


def generate_preplay_path(state, params, horizon=60, rng=None, goal=None):
    """
    Paper Figure 1 Step 3: Novel candidate routes via generative sampling.
    """
    if rng is None:
        rng = np.random.default_rng()
    if goal is None:
        goal = GOAL.copy()

    mu_pos = state.mu_allo.copy()
    mu_theta = float(state.mu_theta)
    mu_ego = state.mu_ego.copy()
    bias = state.bias.copy()
    pi_ego = state.pi_ego_eff
    pi_allo = state.pi_allo_eff

    dir_offset = rng.normal(0, 0.8)

    positions = [mu_pos.copy()]
    beliefs = [mu_pos.copy()]
    ego_obs_list = []
    fe_trace, eo_trace, es_trace = [], [], []

    # [A1] Track prior position for complexity gradient
    prev_pos = mu_pos.copy()
    prev_theta = float(mu_theta)

    for si in range(horizon):
        vec = goal - mu_pos
        gd = safe_norm(vec)
        desired = math.atan2(vec[1], vec[0]) + dir_offset * max(0, 1 - si / 25)

        heading_err = wrap_angle(desired - mu_theta)
        omega = params.turn_gain * heading_err
        v = params.speed * math.tanh(gd)
        mu_theta = wrap_angle(mu_theta + omega * DT + rng.normal(0, params.motor_noise_theta))
        mu_pos[0] += v * DT * math.cos(mu_theta) + rng.normal(0, params.motor_noise_xy)
        mu_pos[1] += v * DT * math.sin(mu_theta) + rng.normal(0, params.motor_noise_xy)
        mu_pos = np.clip(mu_pos, 0.2, ARENA - 0.2)
        positions.append(mu_pos.copy())

        # RSC/TPJ egocentric perspective
        synth_obs, pred_ego = _replay_observation_step(
            mu_pos, mu_theta, bias, params.t_impair, params, rng)
        ego_obs_list.append(synth_obs.copy())

        # [A1] Belief update with complexity gradient
        belief_pos, mu_theta, F_local, mean_eo, mean_es = _replay_belief_update(
            mu_pos, mu_theta, prev_pos, prev_theta,
            bias, mu_ego, synth_obs, pred_ego,
            params, pi_ego, pi_allo, rng)
        beliefs.append(belief_pos.copy())
        fe_trace.append(F_local)
        eo_trace.append(mean_eo)
        es_trace.append(mean_es)

        # Update prior for next step
        prev_pos = mu_pos.copy()
        prev_theta = float(mu_theta)

    path = np.array(positions)
    md = float(np.min(np.linalg.norm(path - goal[None, :], axis=1)))

    return ReplayResult(
        replay_path=path,
        egocentric_obs=ego_obs_list,
        replay_order="preplay",
        replay_length=len(path),
        mean_local_fe=float(np.mean(fe_trace)) if fe_trace else 0.0,
        local_fe_trace=np.array(fe_trace),
        eps_o_trace=np.array(eo_trace),
        eps_s_trace=np.array(es_trace),
        belief_path=np.array(beliefs),
        min_dist_to_goal=md,
        reached_goal=md <= 0.6)


def consolidate_after_replay(state, replay_results, params, lr=0.01):
    """
    Paper §Generative Model: "updated parameters support offline generative
    simulation" + §Timescale separation: "θ-updates at slow timescale τ_learn"
    """
    if not replay_results:
        return state

    best = min(replay_results, key=lambda r: r.mean_local_fe)
    fe_values = [r.mean_local_fe for r in replay_results]
    fe_mean = float(np.mean(fe_values))

    # (1) Update RSC bias toward best replay's terminal direction
    if len(best.replay_path) > 5:
        final_dir = best.replay_path[-1] - best.replay_path[-5]
        nd = safe_norm(final_dir)
        if nd > 1e-6:
            state.bias[0] = clamp(state.bias[0] + lr * final_dir[0] / nd, -1, 1)
            state.bias[1] = clamp(state.bias[1] + lr * final_dir[1] / nd, -1, 1)

    # (2) Update posterior uncertainty from replay evidence
    n_good = sum(1 for r in replay_results if r.reached_goal)
    if n_good > 0:
        shrink = max(0.8, 1.0 - 0.05 * n_good)
        state.Sigma_s = stabilise_cov(state.Sigma_s * shrink)

    # (3) Update smoothed PE for dynamic precision adaptation
    state.smoothed_eps_o = 0.8 * state.smoothed_eps_o + 0.2 * min(fe_mean, 5.0)

    return state

# ═══════════════════════════════════════════════════════════════════════════════
# §11  VARIABLE STARTING LOCATIONS
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class StartConfig:
    random_starts: bool = True
    spatial_bounds: Tuple = (0.5, 0.5, 3.5, 3.5)
    randomize_targets: bool = False
    target_bounds: Tuple = (6.0, 6.0, 9.5, 9.5)
    seed: int = 42

def sample_start_position(rng, bounds, cliffs, max_attempts=50):
    x0, y0, x1, y1 = bounds
    for _ in range(max_attempts):
        pos = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if inside_cliff(pos, cliffs) is None: return pos
    return np.array([(x0+x1)/2, (y0+y1)/2])

# ═══════════════════════════════════════════════════════════════════════════════
# §12  SINGLE-RUN SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class SimResult:
    condition_key: str
    true_path: np.ndarray; belief_path: np.ndarray
    F_hist: np.ndarray; err_hist: np.ndarray
    acc_o_hist: np.ndarray; comp_hist: np.ndarray
    drift_hist: np.ndarray; cliff_penalty_hist: np.ndarray
    final_err: float; final_F: float; cliff_hits: int
    start_pos: np.ndarray; target_pos: np.ndarray
    path_length: float; converged: bool; convergence_step: int
    final_true_pos: np.ndarray; final_belief_pos: np.ndarray
    allocentric_estimate: np.ndarray; egocentric_estimate: np.ndarray
    eps_o_mean_hist: np.ndarray; eps_s_mean_hist: np.ndarray
    eps_o_dist_hist: np.ndarray; eps_o_bear_hist: np.ndarray
    uncertainty_hist: np.ndarray; info_gain_hist: np.ndarray
    pi_ego_eff_hist: np.ndarray; pi_allo_eff_hist: np.ndarray
    acc_s_diagnostic_hist: np.ndarray
    dist_to_goal_hist: np.ndarray; dist_traveled: float
    reached_goal: bool; reached_goal_step: int
    replay_results: Optional[List[ReplayResult]] = None
    preplay_results: Optional[List[ReplayResult]] = None
    consolidation_applied: bool = False

def compute_path_length(path):
    if len(path) < 2: return 0.0
    return float(np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))))

def find_convergence_step(err_hist, threshold=0.5):
    below = err_hist < threshold
    if not np.any(below): return False, len(err_hist)
    for i in range(len(below)-10):
        if np.all(below[i:i+10]): return True, int(i)
    return True, int(np.argmax(below))

def run_single(preset, n_steps=500, seed=42, cliffs=None, cliff_avoid_gain=2.5,
               start_pos=None, start_theta=None, target_pos=None,
               enable_replay=False, replay_horizon=60, n_replay_paths=5,
               use_active_inference=False, policy_horizon=5):
    rng = np.random.default_rng(seed)
    overrides = {"cliff_avoid_gain": cliff_avoid_gain}
    if cliffs is not None: overrides["cliffs"] = cliffs
    # [B1] Plumb the active-inference flag through to SimParams so that
    # sim_step() actually exercises the EFE policy when requested.
    overrides["use_active_inference"] = bool(use_active_inference)
    overrides["policy_horizon"] = int(policy_horizon)
    params = make_params(preset, overrides)
    sp = start_pos if start_pos is not None else START_POS.copy()
    st = start_theta if start_theta is not None else START_THETA
    state = create_sim_state(sp, st)
    goal = target_pos if target_pos is not None else GOAL.copy()

    for _ in range(n_steps):
        state = sim_step(state, params, rng)

    tp = np.array(state.true_path); bp = np.array(state.belief_path)
    eh = np.array(state.err_hist) if state.err_hist else np.array([float("nan")])
    pl = compute_path_length(tp)
    conv, cs = find_convergence_step(eh)

    replay_results = None; preplay_results = None; consolidated = False
    if enable_replay and len(bp) > 10:
        bth = np.zeros(len(bp))
        for i in range(1, len(bp)):
            d = bp[i]-bp[i-1]; bth[i] = math.atan2(d[1], d[0])
        bth[0] = bth[1] if len(bth) > 1 else 0.0

        replay_results = []
        for order in ["forward", "reverse", "noisy"]:
            for _ in range(n_replay_paths):
                rr = generate_replay_path(
                    bp, bth, state, params,
                    replay_order=order, replay_horizon=replay_horizon,
                    rng=np.random.default_rng(seed+1000+len(replay_results)),
                    goal=goal)
                replay_results.append(rr)

        preplay_results = []
        for _ in range(n_replay_paths):
            pp = generate_preplay_path(
                state, params,
                horizon=replay_horizon,
                rng=np.random.default_rng(seed+2000+len(preplay_results)),
                goal=goal)
            preplay_results.append(pp)

        all_replay = replay_results + preplay_results
        state = consolidate_after_replay(state, all_replay, params)
        consolidated = True

    return SimResult(
        condition_key=preset.key, true_path=tp, belief_path=bp,
        F_hist=np.array(state.F_hist), err_hist=eh,
        acc_o_hist=np.array(state.acc_o_hist), comp_hist=np.array(state.comp_hist),
        drift_hist=np.array(state.drift_hist),
        cliff_penalty_hist=np.array(state.cliff_penalty_hist),
        final_err=state.err_hist[-1] if state.err_hist else float("nan"),
        final_F=state.F_hist[-1] if state.F_hist else float("nan"),
        cliff_hits=state.cliff_hits, start_pos=sp.copy(), target_pos=goal.copy(),
        path_length=pl, converged=conv, convergence_step=cs,
        final_true_pos=state.true_pos.copy(), final_belief_pos=state.mu_allo.copy(),
        allocentric_estimate=state.mu_allo.copy(),
        egocentric_estimate=np.mean(state.mu_ego, axis=0),
        eps_o_mean_hist=np.array(state.eps_o_mean_hist),
        eps_s_mean_hist=np.array(state.eps_s_mean_hist),
        eps_o_dist_hist=np.array(state.eps_o_dist_hist),
        eps_o_bear_hist=np.array(state.eps_o_bear_hist),
        uncertainty_hist=np.array(state.uncertainty_hist),
        info_gain_hist=np.array(state.info_gain_hist),
        pi_ego_eff_hist=np.array(state.pi_ego_eff_hist),
        pi_allo_eff_hist=np.array(state.pi_allo_eff_hist),
        acc_s_diagnostic_hist=np.array(state.acc_s_diagnostic_hist),
        dist_to_goal_hist=np.array(state.dist_to_goal_hist),
        dist_traveled=state.dist_traveled,
        reached_goal=state.reached_goal, reached_goal_step=state.reached_goal_step,
        replay_results=replay_results, preplay_results=preplay_results,
        consolidation_applied=consolidated)

# ═══════════════════════════════════════════════════════════════════════════════
# §13  BATCH SIMULATION + REPORTING
# ═══════════════════════════════════════════════════════════════════════════════
def run_batch(conditions=None, n_runs=100, n_steps=500, base_seed=42,
              cliffs=None, start_config=None, enable_replay=False,
              csv_path=None, verbose=True,
              use_active_inference=False, policy_horizon=5):
    if cliffs is None: cliffs = [dict(c) for c in DEFAULT_CLIFFS]
    if conditions is None: conditions = list(CONDITION_PRESETS.keys())
    if start_config is None: start_config = StartConfig(seed=base_seed)
    srng = np.random.default_rng(start_config.seed)
    all_rows, all_results, summary = [], {}, {}

    for ci, ck in enumerate(conditions):
        if ck not in CONDITION_PRESETS:
            print(f"  WARNING: Unknown condition '{ck}'"); continue
        preset = CONDITION_PRESETS[ck]; results = []
        if verbose: print(f"  [{ci+1}/{len(conditions)}] {n_runs}× {preset.label} ({n_steps} steps)...")
        t0 = time.time()
        for i in range(n_runs):
            sp = sample_start_position(srng, start_config.spatial_bounds, cliffs) if start_config.random_starts else START_POS.copy()
            st = float(srng.uniform(-math.pi, math.pi)) if start_config.random_starts else START_THETA
            tp = GOAL.copy()
            r = run_single(preset, n_steps, base_seed+ci*n_runs+i, cliffs, 2.5,
                           sp, st, tp, enable_replay,
                           use_active_inference=use_active_inference,
                           policy_horizon=policy_horizon)
            results.append(r)
            row = {"trial":i,"condition":ck,
                   "start_x":float(r.start_pos[0]),"start_y":float(r.start_pos[1]),
                   "final_err":r.final_err,"final_F":r.final_F,
                   "path_length":r.path_length,"dist_traveled":r.dist_traveled,
                   "converged":r.converged,"convergence_step":r.convergence_step,
                   "cliff_hits":r.cliff_hits,
                   "target_reached":r.reached_goal,"steps_to_goal":r.reached_goal_step,
                   "final_eps_o":float(r.eps_o_mean_hist[-1]) if len(r.eps_o_mean_hist) else float("nan"),
                   "final_eps_s":float(r.eps_s_mean_hist[-1]) if len(r.eps_s_mean_hist) else float("nan"),
                   "final_uncertainty":float(r.uncertainty_hist[-1]) if len(r.uncertainty_hist) else float("nan"),
                   "mean_info_gain":float(np.mean(r.info_gain_hist)) if len(r.info_gain_hist) else 0,
                   "final_pi_ego_eff":float(r.pi_ego_eff_hist[-1]) if len(r.pi_ego_eff_hist) else preset.pi_ego,
                   "final_pi_allo_eff":float(r.pi_allo_eff_hist[-1]) if len(r.pi_allo_eff_hist) else preset.pi_allo,
                   "consolidation_applied":r.consolidation_applied,
                   "active_inference":bool(use_active_inference)}
            all_rows.append(row)
        if verbose: print(f"      Done in {time.time()-t0:.1f}s")
        all_results[ck] = results

        mn = lambda a: float(np.nanmean(a)); sd = lambda a: float(np.nanstd(a))
        fe = np.array([r.final_err for r in results])
        fF = np.array([r.final_F for r in results])
        pl2 = np.array([r.dist_traveled for r in results])
        ch = np.array([r.cliff_hits for r in results])
        eo = np.array([r.eps_o_mean_hist[-1] if len(r.eps_o_mean_hist) else np.nan for r in results])
        es = np.array([r.eps_s_mean_hist[-1] if len(r.eps_s_mean_hist) else np.nan for r in results])
        ig = np.array([np.mean(r.info_gain_hist) if len(r.info_gain_hist) else 0 for r in results])
        unc = np.array([r.uncertainty_hist[-1] if len(r.uncertainty_hist) else np.nan for r in results])
        sr = np.mean([r.reached_goal for r in results])*100

        eo_curves = [r.eps_o_mean_hist[:n_steps] for r in results[:20] if len(r.eps_o_mean_hist)>=n_steps]
        es_curves = [r.eps_s_mean_hist[:n_steps] for r in results[:20] if len(r.eps_s_mean_hist)>=n_steps]
        err_curves = [r.err_hist[:n_steps] for r in results[:20] if len(r.err_hist)>=n_steps]
        F_curves = [r.F_hist[:n_steps] for r in results[:20] if len(r.F_hist)>=n_steps]
        ig_curves = [r.info_gain_hist[:n_steps] for r in results[:20] if len(r.info_gain_hist)>=n_steps]
        unc_curves = [r.uncertainty_hist[:n_steps] for r in results[:20] if len(r.uncertainty_hist)>=n_steps]

        summary[ck] = {
            "label":preset.label,"color":preset.color,"n_runs":n_runs,
            "mean_err":mn(fe),"std_err":sd(fe),"mean_F":mn(fF),"std_F":sd(fF),
            "mean_dist":mn(pl2),"std_dist":sd(pl2),
            "mean_cliff":mn(ch),"success_rate":sr,
            "mean_eps_o":mn(eo),"std_eps_o":sd(eo),
            "mean_eps_s":mn(es),"std_eps_s":sd(es),
            "mean_info_gain":mn(ig),"mean_uncertainty":mn(unc),
            "eo_curves":eo_curves,"es_curves":es_curves,
            "err_curves":err_curves,"F_curves":F_curves,
            "ig_curves":ig_curves,"unc_curves":unc_curves,
        }

    df = pd.DataFrame(all_rows) if HAS_PANDAS else all_rows
    if csv_path and HAS_PANDAS:
        df.to_csv(csv_path, index=False)
        if verbose: print(f"\n  CSV: {csv_path}")
    return df, summary, all_results

# ═══════════════════════════════════════════════════════════════════════════════
# §14  REPORTING + PUBLICATION PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
def print_summary(summary, n_runs, n_steps, replay):
    print("\n"+"="*120)
    print(f"  GPC v11 BATCH REPORT — {n_runs} runs × {n_steps} steps")
    print("="*120)
    hdr = f"  {'Condition':<38} {'Error(μ±σ)':<16} {'ε_o(μ±σ)':<16} {'ε_s(μ±σ)':<16} {'Dist':<10} {'InfoGain':<10} {'Uncert':<10} {'Succ%':<8}"
    print(hdr); print("-"*120)
    for k, s in summary.items():
        line = f"  {s['label']:<38} "
        line += f"{s['mean_err']:.3f}±{s['std_err']:.3f}   "
        line += f"{s['mean_eps_o']:.3f}±{s['std_eps_o']:.3f}   "
        line += f"{s['mean_eps_s']:.3f}±{s['std_eps_s']:.3f}   "
        line += f"{s['mean_dist']:7.1f}   "
        line += f"{s['mean_info_gain']:.4f}   "
        line += f"{s['mean_uncertainty']:.4f}   "
        line += f"{s['success_rate']:.1f}%"
        print(line)
    print("="*120)

def _draw_cliffs(ax, cliffs):
    if not HAS_MPL:
        return
    for c in cliffs:
        ax.add_patch(Circle((c["cx"],c["cy"]),c["r"],facecolor="#fca5a555",edgecolor="#991b1b88",ls="--",lw=1.5))

def generate_plots(summary, all_results, n_runs, n_steps, enable_replay=False, save_dir="."):
    if not HAS_MPL:
        print("  Skipping plots (matplotlib not installed)")
        return []
    saved = []; conds = list(summary.keys()); nc = len(conds)
    colors = [summary[k]["color"] for k in conds]
    labels = [summary[k]["label"].split("(")[0].strip()[:15] for k in conds]

    def mean_curve(curves):
        if not curves: return np.array([])
        L = min(len(c) for c in curves)
        return np.mean([c[:L] for c in curves], axis=0)

    # ── FIG 1: Prediction Error Dynamics ──
    fig, axes = plt.subplots(2, nc, figsize=(5*nc, 8), facecolor="#f8fafc")
    if nc == 1: axes = axes.reshape(2,1)
    fig.suptitle("Prediction Error Dynamics — Paper §2.1.4\n"
                 "ε_o = o − g(μ_s)  |  ε_s = μ_ego − T_θ(μ_allo)", fontsize=24, fontweight="bold", y=0.98)
    for i, k in enumerate(conds):
        s = summary[k]; c = s["color"]
        mc_eo = mean_curve(s["eo_curves"]); mc_es = mean_curve(s["es_curves"])
        ax = axes[0, i]; ax.set_title(labels[i], fontsize=18, color=c, fontweight="bold")
        if len(mc_eo): ax.plot(mc_eo, lw=1.5, color=c, label="|ε_o|")
        if len(mc_es): ax.plot(mc_es, lw=1.5, color=c, ls="--", alpha=0.7, label="|ε_s|")
        ax.set_ylabel("Mean |ε|", fontsize=17, fontweight="bold"); ax.grid(alpha=0.15)
        ax.tick_params(labelsize=14)
        if i == 0: ax.legend(fontsize=13)
        ax.text(0.95, 0.90, f"ε_o={s['mean_eps_o']:.3f}\nε_s={s['mean_eps_s']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=12, fontfamily="monospace",
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"))

        mc_ao = mean_curve(s.get("F_curves",[])); mc_err = mean_curve(s["err_curves"])
        ax = axes[1, i]
        if len(mc_ao): ax.plot(mc_ao, lw=1, color="#1e293b", label="F")
        if len(mc_err): ax.plot(mc_err, lw=1.5, color=c, ls="--", label="‖η−μ‖")
        ax.set_xlabel("Step", fontsize=17, fontweight="bold"); ax.set_ylabel("F / Error", fontsize=17, fontweight="bold")
        ax.grid(alpha=0.15); ax.tick_params(labelsize=14)
        if i == 0: ax.legend(fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.92])
    p1 = os.path.join(save_dir, "fig1_prediction_errors.png")
    fig.savefig(p1, dpi=300, bbox_inches="tight"); plt.close(fig); saved.append(p1)

    # ── FIG 2: Uncertainty & Info Gain ──
    fig, axes = plt.subplots(2, nc, figsize=(5*nc, 7), facecolor="#f8fafc")
    if nc == 1: axes = axes.reshape(2,1)
    fig.suptitle("Posterior Uncertainty & Information Gain — Paper §2.1.3, Q(x)≈N(μ,Σ)",
                 fontsize=24, fontweight="bold", y=0.98)
    for i, k in enumerate(conds):
        s = summary[k]; c = s["color"]
        mc_u = mean_curve(s["unc_curves"]); mc_ig = mean_curve(s["ig_curves"])
        ax = axes[0, i]; ax.set_title(labels[i], fontsize=18, color=c, fontweight="bold")
        if len(mc_u): ax.plot(mc_u, lw=1.5, color=c)
        ax.set_ylabel("tr(Σ_s)", fontsize=17, fontweight="bold"); ax.grid(alpha=0.15)
        ax.tick_params(labelsize=14)
        ax = axes[1, i]
        if len(mc_ig): ax.plot(mc_ig, lw=1.5, color=c)
        ax.set_ylabel("Info Gain", fontsize=17, fontweight="bold")
        ax.set_xlabel("Step", fontsize=17, fontweight="bold")
        ax.grid(alpha=0.15); ax.tick_params(labelsize=14)
    fig.tight_layout(rect=[0,0,1,0.92])
    p2 = os.path.join(save_dir, "fig2_uncertainty_info_gain.png")
    fig.savefig(p2, dpi=300, bbox_inches="tight"); plt.close(fig); saved.append(p2)

    # ── FIG 3: Condition Comparison Bars ──
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), facecolor="#f8fafc")
    fig.suptitle(f"Cross-Condition Comparison — {n_runs} runs × {n_steps} steps",
                 fontsize=24, fontweight="bold", y=0.98)
    metrics = [
        ("Mean Error", [summary[k]["mean_err"] for k in conds]),
        ("Mean |ε_o|", [summary[k]["mean_eps_o"] for k in conds]),
        ("Mean |ε_s|", [summary[k]["mean_eps_s"] for k in conds]),
        ("Success %", [summary[k]["success_rate"] for k in conds]),
        ("Dist Traveled", [summary[k]["mean_dist"] for k in conds]),
        ("Free Energy", [summary[k]["mean_F"] for k in conds]),
        ("Info Gain", [summary[k]["mean_info_gain"] for k in conds]),
        ("Uncertainty", [summary[k]["mean_uncertainty"] for k in conds]),
    ]
    for idx, (title, vals) in enumerate(metrics):
        ax = axes[idx//4, idx%4]
        bars = ax.bar(labels, vals, color=colors, alpha=0.6, edgecolor="black", lw=0.5)
        ax.set_title(title, fontsize=18, fontweight="bold"); ax.grid(alpha=0.15, axis="y")
        ax.tick_params(axis="x", rotation=30, labelsize=13)
        ax.tick_params(axis="y", labelsize=14)
        vmax = max(vals + [1])
        ax.set_ylim(top=vmax * 1.18)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01*vmax,
                    f"{val:.3f}" if val < 10 else f"{val:.1f}", ha="center", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.93])
    p3 = os.path.join(save_dir, "fig3_condition_comparison.png")
    fig.savefig(p3, dpi=300, bbox_inches="tight"); plt.close(fig); saved.append(p3)

    # ── FIG 4: Best run per condition ──
    fig, axes = plt.subplots(1, nc, figsize=(5.5*nc, 6.5), facecolor="white")
    if nc == 1: axes = [axes]
    fig.suptitle("Figure 4. Best-Performing Trajectory per Condition\n"
                 "Lowest final localisation error ‖η − μ‖ across all runs",
                 fontsize=24, fontweight="bold", y=0.99)
    for i, k in enumerate(conds):
        ax = axes[i]; res = all_results[k]
        best = min(res, key=lambda r: r.final_err)
        ax.set_xlim(-0.2, ARENA+0.2); ax.set_ylim(-0.2, ARENA+0.2); ax.set_aspect("equal")
        ax.set_facecolor("#fafbfd")
        ax.plot([0,ARENA,ARENA,0,0],[0,0,ARENA,ARENA,0], color="#cbd5e1", lw=1)
        ax.set_title(f"{labels[i]}\nerr = {best.final_err:.3f}", fontsize=18, color=colors[i], fontweight="bold")
        ax.grid(alpha=0.1)
        _draw_cliffs(ax, DEFAULT_CLIFFS)
        for ci2, c in enumerate(DEFAULT_CLIFFS):
            ax.text(c["cx"], c["cy"], "cliff", ha="center", va="center", fontsize=10, color="#b91c1c", alpha=0.5)
        ax.scatter(LANDMARKS[:,0], LANDMARKS[:,1], s=50, marker="s", color="#16a34a", zorder=10, edgecolors="white", lw=0.8)
        for li, lm in enumerate(LANDMARKS):
            ax.annotate(f"L{li+1}", (lm[0]+0.2, lm[1]+0.25), fontsize=11, color="#15803d", fontweight="bold")
        ax.scatter([GOAL[0]], [GOAL[1]], s=120, marker="*", color="#eab308", zorder=10, edgecolors="#854d0e", lw=0.8)
        ax.annotate("Goal", (GOAL[0]-0.7, GOAL[1]+0.4), fontsize=12, color="#854d0e", fontweight="bold")
        ax.plot(best.true_path[:,0], best.true_path[:,1], "-", lw=2.2, color="#3b82f6", alpha=0.75, label="True body (η)", zorder=5)
        ax.plot(best.belief_path[:,0], best.belief_path[:,1], "--", lw=2, color=colors[i], alpha=0.65, label="Belief (μ_allo)", zorder=5)
        ax.scatter([best.true_path[0,0]], [best.true_path[0,1]], s=60, marker="o", color="#22c55e", zorder=11, edgecolors="white", lw=1.2)
        ax.annotate("Start", (best.true_path[0,0]+0.2, best.true_path[0,1]-0.4), fontsize=11, color="#15803d", fontweight="bold")
        ax.set_xlabel("x (arena units)", fontsize=17, fontweight="bold"); ax.set_ylabel("y (arena units)", fontsize=17, fontweight="bold")
        ax.tick_params(labelsize=14)
        if i == 0: ax.legend(fontsize=13, loc="lower right", framealpha=0.9)
    fig.tight_layout(rect=[0,0,1,0.89])
    p4 = os.path.join(save_dir, "fig4_best_trajectories.png")
    fig.savefig(p4, dpi=300, bbox_inches="tight"); plt.close(fig); saved.append(p4)

    # ── FIG 5: Best / Median / Worst ──
    fig, axes = plt.subplots(3, nc, figsize=(5.5*nc, 16), facecolor="white")
    if nc == 1: axes = axes.reshape(3, 1)
    fig.suptitle("Figure 5. Trajectory Comparison: Best, Median, and Worst Performance\n"
                 "Ranked by final localisation error ‖η − μ‖",
                 fontsize=24, fontweight="bold", y=0.995)
    row_labels = ["Best", "Median", "Worst"]
    for i, k in enumerate(conds):
        res = all_results[k]
        sorted_res = sorted(res, key=lambda r: r.final_err)
        picks = [sorted_res[0], sorted_res[len(sorted_res)//2], sorted_res[-1]]
        for ri, (run_r, rlabel) in enumerate(zip(picks, row_labels)):
            ax = axes[ri, i]
            ax.set_xlim(-0.2, ARENA+0.2); ax.set_ylim(-0.2, ARENA+0.2); ax.set_aspect("equal")
            ax.set_facecolor("#fafbfd")
            ax.plot([0,ARENA,ARENA,0,0],[0,0,ARENA,ARENA,0], color="#cbd5e1", lw=0.8)
            ax.set_title(f"{rlabel}: {labels[i]}\nerr = {run_r.final_err:.3f}  |  dist = {run_r.dist_traveled:.1f}",
                         fontsize=17, fontweight="bold", color=colors[i])
            ax.grid(alpha=0.08)
            _draw_cliffs(ax, DEFAULT_CLIFFS)
            for ci2, c in enumerate(DEFAULT_CLIFFS):
                ax.text(c["cx"], c["cy"], "cliff", ha="center", va="center", fontsize=10, color="#b91c1c", alpha=0.4)
            ax.scatter(LANDMARKS[:,0], LANDMARKS[:,1], s=40, marker="s", color="#16a34a", zorder=10, edgecolors="white", lw=0.6)
            ax.scatter([GOAL[0]], [GOAL[1]], s=100, marker="*", color="#eab308", zorder=10, edgecolors="#854d0e", lw=0.6)
            if ri == 0 and i == 0:
                ax.annotate("Goal", (GOAL[0]-0.7, GOAL[1]+0.4), fontsize=12, color="#854d0e", fontweight="bold")
            ax.plot(run_r.true_path[:,0], run_r.true_path[:,1], "-", lw=2, color="#3b82f6", alpha=0.7, zorder=5)
            ax.plot(run_r.belief_path[:,0], run_r.belief_path[:,1], "--", lw=1.8, color=colors[i], alpha=0.6, zorder=5)
            ax.scatter([run_r.true_path[0,0]], [run_r.true_path[0,1]], s=45, marker="o", color="#22c55e", zorder=11, edgecolors="white", lw=1)
            ax.scatter([run_r.true_path[-1,0]], [run_r.true_path[-1,1]], s=45, marker="o", color="#3b82f6", zorder=11, edgecolors="white", lw=1)
            err_col = "#16a34a" if run_r.final_err < 0.3 else ("#d97706" if run_r.final_err < 1.0 else "#dc2626")
            ax.text(0.97, 0.03, f"err={run_r.final_err:.3f}\nε_o={(run_r.eps_o_mean_hist[-1] if len(run_r.eps_o_mean_hist) else 0):.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=12, fontfamily="monospace",
                    color=err_col, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor=err_col, lw=0.8, boxstyle="round,pad=0.3"))
            ax.tick_params(labelsize=13)
            if i == 0: ax.set_ylabel(f"{rlabel}\ny (arena)", fontsize=17, fontweight="bold")
            if ri == 2: ax.set_xlabel("x (arena units)", fontsize=17, fontweight="bold")
            if ri == 0 and i == 0:
                from matplotlib.lines import Line2D
                leg = [Line2D([0],[0],color="#3b82f6",lw=2,label="True body (η)"),
                       Line2D([0],[0],color="#888",lw=2,ls="--",label="Belief (μ_allo)")]
                ax.legend(handles=leg, fontsize=12, loc="lower right", framealpha=0.9)
    fig.tight_layout(rect=[0,0,1,0.94])
    p5 = os.path.join(save_dir, "fig5_best_median_worst.png")
    fig.savefig(p5, dpi=300, bbox_inches="tight"); plt.close(fig); saved.append(p5)

    # ── FIG 6: Offline Replay ──
    if enable_replay:
        replay_runs = {}
        for k in conds:
            for r in all_results[k]:
                if r.replay_results and r.preplay_results:
                    replay_runs[k] = r; break
        if replay_runs:
            n_rk = len(replay_runs)
            fig, axes = plt.subplots(1, n_rk, figsize=(6*n_rk, 6), facecolor="white")
            if n_rk == 1: axes = [axes]
            fig.suptitle("Figure 6. Offline Hippocampal Replay & Generative Preplay\n"
                         "Paper §2.1.6, Figure 1 Step 3: "
                         r"$\tilde{s}_{1:T} \sim P(s_{1:T} \mid \theta_{\mathrm{updated}})$",
                         fontsize=24, fontweight="bold", y=0.99)
            rp_cols = {"forward":"#6366f1","reverse":"#f97316","noisy":"#6b7280","preplay":"#10b981"}
            rp_dash = {"forward":(6,3),"reverse":(4,4),"noisy":(2,3),"preplay":(8,3)}
            rp_lw = {"forward":1.3,"reverse":1.3,"noisy":1.0,"preplay":2.0}

            for idx, k in enumerate(replay_runs):
                ax = axes[idx]; r = replay_runs[k]
                cc = summary[k]["color"]
                ax.set_xlim(-0.3, ARENA+0.3); ax.set_ylim(-0.3, ARENA+0.3)
                ax.set_aspect("equal"); ax.set_facecolor("#fafbfd")
                ax.plot([0,ARENA,ARENA,0,0],[0,0,ARENA,ARENA,0], color="#cbd5e1", lw=1)
                ax.grid(alpha=0.08)

                for c in DEFAULT_CLIFFS:
                    ax.add_patch(Circle((c["cx"],c["cy"]),c["r"],facecolor="#fee2e211",edgecolor="#f8717144",ls="--",lw=1))

                ax.scatter(LANDMARKS[:,0], LANDMARKS[:,1], s=45, marker="s", color="#16a34a", zorder=10, edgecolors="white", lw=0.7)
                for li, lm in enumerate(LANDMARKS):
                    ax.annotate(f"L{li+1}", (lm[0]+0.2, lm[1]+0.25), fontsize=9, color="#15803d", fontweight="bold")

                ax.scatter([GOAL[0]], [GOAL[1]], s=200, marker="*", color="#eab308", zorder=12, edgecolors="#854d0e", lw=1)
                ax.annotate("Goal", (GOAL[0]-0.8, GOAL[1]+0.45), fontsize=11, color="#854d0e", fontweight="bold")

                ax.plot(r.belief_path[:,0], r.belief_path[:,1], "-", lw=2.5, color=cc, alpha=0.7, zorder=5, label="Online trajectory (μ)")
                ax.scatter([r.belief_path[0,0]], [r.belief_path[0,1]], s=60, marker="o", color="#22c55e", zorder=13, edgecolors="white", lw=1.2)
                ax.annotate("Start", (r.belief_path[0,0]+0.2, r.belief_path[0,1]-0.4), fontsize=10, color="#15803d", fontweight="bold")

                end_x, end_y = r.belief_path[-1,0], r.belief_path[-1,1]
                ax.scatter([end_x], [end_y], s=180, marker="D", color=cc, zorder=14, edgecolors="white", lw=2)
                ax.annotate("Replay starts\n(ȧ = 0)", xy=(end_x, end_y),
                            xytext=(end_x + (1.8 if end_x < 6 else -3), end_y + (-2 if end_y > 4 else 1.5)),
                            fontsize=7, fontweight="bold", color="#1e1b4b",
                            arrowprops=dict(arrowstyle="-|>", color="#1e1b4b", lw=1.3, connectionstyle="arc3,rad=-0.2"),
                            bbox=dict(facecolor="#ede9fe", edgecolor="#6d28d9", lw=1, boxstyle="round,pad=0.25"), zorder=15)

                for rp in r.replay_results:
                    ax.plot(rp.replay_path[:,0], rp.replay_path[:,1],
                            color=rp_cols[rp.replay_order], lw=rp_lw[rp.replay_order],
                            alpha=0.5, dashes=rp_dash[rp.replay_order], zorder=4)
                    ax.scatter([rp.replay_path[-1,0]], [rp.replay_path[-1,1]],
                               s=14, color=rp_cols[rp.replay_order], zorder=6, alpha=0.7)

                for pi2, pp in enumerate(r.preplay_results):
                    ax.plot(pp.replay_path[:,0], pp.replay_path[:,1],
                            color=rp_cols["preplay"], lw=rp_lw["preplay"],
                            alpha=0.55, dashes=rp_dash["preplay"], zorder=4)
                    ax.scatter([pp.replay_path[-1,0]], [pp.replay_path[-1,1]],
                               s=20, color=rp_cols["preplay"], zorder=6, alpha=0.7)
                    if pi2 == 0 and len(pp.replay_path) > 10:
                        mid = pp.replay_path[10]
                        ax.annotate("novel\ncandidate", (mid[0], mid[1]),
                                    fontsize=5.5, color="#059669", fontweight="bold", alpha=0.8)

                ax.set_title(f"{summary[k]['label'].split('(')[0].strip()}", fontsize=18, fontweight="bold", color=cc)
                ax.set_xlabel("x (arena units)", fontsize=17, fontweight="bold")
                if idx == 0: ax.set_ylabel("y (arena units)", fontsize=17, fontweight="bold")
                ax.tick_params(labelsize=14)

                if idx == 0:
                    from matplotlib.lines import Line2D as L2D
                    leg_el = [
                        L2D([0],[0], color=cc, lw=2.5, label="Online trajectory"),
                        L2D([0],[0], color="none", marker="D", markerfacecolor=cc, markeredgecolor="white", markersize=8, label="Replay origin (ȧ=0)"),
                        L2D([0],[0], color=rp_cols["forward"], lw=1.3, dashes=(6,3), label="Replay: forward"),
                        L2D([0],[0], color=rp_cols["reverse"], lw=1.3, dashes=(4,4), label="Replay: reverse"),
                        L2D([0],[0], color=rp_cols["noisy"], lw=1, dashes=(2,3), label="Replay: noisy"),
                        L2D([0],[0], color=rp_cols["preplay"], lw=2, dashes=(8,3), label="Preplay: novel candidate"),
                    ]
                    ax.legend(handles=leg_el, loc="lower left", fontsize=6, framealpha=0.92, ncol=2, handlelength=2.2)

            caption = (
                "Figure 6. Offline hippocampal replay and generative preplay. Each panel shows the agent's online "
                "trajectory (solid coloured line) followed by offline replay and preplay paths generated during the "
                "suspended-blanket regime (ȧ = 0, sensory input disengaged). The diamond marker indicates the "
                "replay origin. Replay paths (purple = forward, orange = reverse, gray = noisy-recombined) re-traverse "
                "stored belief trajectories using the agent's transition model P(s_t|s_{t-1}; θ_updated). Preplay paths "
                "(green, thicker) simulate novel candidate routes."
            )
            fig.text(0.05, 0.01, caption, fontsize=7, color="#374151", style="italic", wrap=True,
                     bbox=dict(facecolor="#f9fafb", edgecolor="#d1d5db", lw=0.5, boxstyle="round,pad=0.3"))

            fig.tight_layout(rect=[0, 0.06, 1, 0.93])
            p6 = os.path.join(save_dir, "fig6_offline_replay.png")
            fig.savefig(p6, dpi=300, bbox_inches="tight"); plt.close(fig); saved.append(p6)

    return saved

# ═══════════════════════════════════════════════════════════════════════════════
# §15  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # [B2] Defaults now match manuscript §3.6 (50 runs x 500 steps).
    # The previous default (100 x 500) was inconsistent with the paper.
    save_dir = "."
    n_steps = 500
    n_runs = 50
    cond_filter = None
    replay = False
    fast_mode = "--fast" in sys.argv
    use_ai = "--active-inference" in sys.argv and not fast_mode
    quantile_mode = "--quantile-mode" in sys.argv

    # [B4] Allow the policy horizon to be overridden from the command line.
    # The default (5) matches the actual implementation; setting
    # --policy-horizon 1 reproduces the literal "one-step lookahead" from
    # earlier drafts of the manuscript.
    policy_horizon = 5

    for i, a in enumerate(sys.argv):
        if a == "--steps" and i+1 < len(sys.argv): n_steps = int(sys.argv[i+1])
        if a == "--runs" and i+1 < len(sys.argv): n_runs = int(sys.argv[i+1])
        if a == "--condition" and i+1 < len(sys.argv): cond_filter = sys.argv[i+1]
        if a == "--save-dir" and i+1 < len(sys.argv): save_dir = sys.argv[i+1]
        if a == "--policy-horizon" and i+1 < len(sys.argv): policy_horizon = int(sys.argv[i+1])
        if a == "--replay": replay = True

    # [B3] --quantile-mode overrides defaults to match the §3.7 (Fig. 8)
    # trajectory-quantile sample size, unless the user has supplied
    # explicit --runs / --steps already.
    if quantile_mode:
        if "--runs" not in sys.argv: n_runs = 30
        if "--steps" not in sys.argv: n_steps = 400

    os.makedirs(save_dir, exist_ok=True)

    has_explicit_cmd = any(a in sys.argv for a in
                           ["--batch","--batch-all","--compare-all","--quantile-mode",
                            "--reproduce-paper"])

    # ── [C3] v11: --reproduce-paper now actually works ────────────────────
    # Regenerates every protocol cited in the manuscript with the correct
    # per-figure sample size, plus the cliff-free supplementary sensitivity.
    if "--reproduce-paper" in sys.argv:
        protocols = [
            # (sub-folder name,             n_runs, n_steps, enable_replay,  description)
            ("demo_fig5_fig6",                  20,    150,  False,  "Figs 5, 6 (PE dynamics & uncertainty)"),
            ("section_3_6_fig7",                50,    500,  False,  "Fig 7 (cross-condition bars, §3.6)"),
            ("section_3_7_fig8",                30,    400,  False,  "Fig 8 (best/median/worst trajectories, §3.7)"),
            ("section_3_7_fig9",                30,    400,  True,   "Fig 9 (offline replay/preplay, §3.7)"),
        ]
        print(f"\n  GPC v11 — --reproduce-paper: regenerating all paper figures")
        print(f"  Active inference: {use_ai} (horizon={policy_horizon if use_ai else '-'})")
        print(f"  Output root: {save_dir}\n")
        for sub, runs, steps, with_replay, desc in protocols:
            sub_dir = os.path.join(save_dir, sub)
            os.makedirs(sub_dir, exist_ok=True)
            print(f"  >>> {sub}: {desc}  ({runs} runs × {steps} steps, replay={with_replay})")
            df, summary, ar = run_batch(
                list(CONDITION_PRESETS.keys()),
                runs, steps,
                csv_path=os.path.join(sub_dir, f"gpc_v11_{sub}.csv"),
                enable_replay=with_replay,
                use_active_inference=use_ai,
                policy_horizon=policy_horizon)
            print_summary(summary, runs, steps, with_replay)
            generate_plots(summary, ar, runs, steps, with_replay, sub_dir)
        # Cliff-free sensitivity for Supplementary Fig. S1
        cf_dir = os.path.join(save_dir, "supp_fig_S1_cliff_free")
        os.makedirs(cf_dir, exist_ok=True)
        print(f"\n  >>> supp_fig_S1_cliff_free: cliff-free sensitivity (50 runs × 500 steps, no obstacles)")
        df, summary, ar = run_batch(
            list(CONDITION_PRESETS.keys()),
            n_runs=50, n_steps=500, cliffs=[],
            csv_path=os.path.join(cf_dir, "gpc_v11_cliff_free.csv"),
            enable_replay=False,
            use_active_inference=use_ai,
            policy_horizon=policy_horizon)
        print_summary(summary, 50, 500, False)
        generate_plots(summary, ar, 50, 500, False, cf_dir)
        print(f"\n  All paper-cited results regenerated under {save_dir}")
        print(f"  Sub-folders:")
        for sub, _, _, _, _ in protocols:
            print(f"    {os.path.join(save_dir, sub)}/")
        print(f"    {cf_dir}/")
        return
    # ──────────────────────────────────────────────────────────────────────

    if "--batch" in sys.argv or "--batch-all" in sys.argv or quantile_mode:
        conds = list(CONDITION_PRESETS.keys()) if ("--batch-all" in sys.argv or quantile_mode) else \
                ([c.strip() for c in cond_filter.split(",")] if cond_filter else
                 ["healthy","bodily","obe","disorientation"])
        print(f"\n  GPC v11 -- {n_runs} runs x {n_steps} steps | Replay: {replay} | "
              f"ActiveInference: {use_ai} (horizon={policy_horizon if use_ai else '-'})")
        csv = os.path.join(save_dir, "gpc_v11_results.csv")
        df, summary, ar = run_batch(conds, n_runs, n_steps, csv_path=csv,
                                    enable_replay=replay,
                                    use_active_inference=use_ai,
                                    policy_horizon=policy_horizon)
        print_summary(summary, n_runs, n_steps, replay)
        saved = generate_plots(summary, ar, n_runs, n_steps, replay, save_dir)
        for f in saved: print(f"  Plot: {f}")
        if HAS_PANDAS and isinstance(df, pd.DataFrame):
            print(f"  CSV: {csv} ({df.shape})")
    elif "--compare-all" in sys.argv:
        conds = list(CONDITION_PRESETS.keys())
        df, summary, ar = run_batch(conds, 1, n_steps,
                                    start_config=StartConfig(random_starts=False),
                                    enable_replay=replay,
                                    use_active_inference=use_ai,
                                    policy_horizon=policy_horizon)
        print_summary(summary, 1, n_steps, replay)
        generate_plots(summary, ar, 1, n_steps, replay, save_dir)
    else:
        print("\n" + "="*80)
        print("  GPC v11 -- Generative Predictive Coding of Self-Location")
        print("  Erdeniz & Yildirim")
        print("="*80)
        if not has_explicit_cmd:
            print("\n  No command specified -- running quick demo (4 conditions x 20 runs x 150 steps)")
            print("  Reproduce all paper figures: python GPC_v11.py --reproduce-paper")
            print("  Paper §3.6 (50x500):  python GPC_v11.py --batch-all")
            print("  Paper §3.7 (30x400):  python GPC_v11.py --quantile-mode")
            print("  With replay:          python GPC_v11.py --batch-all --replay")
            print("  With active inf.:     python GPC_v11.py --batch-all --active-inference")
            print("  Custom:               python GPC_v11.py --batch --condition healthy,obe --runs 30")
            print("  All options:          --steps N  --save-dir DIR  --policy-horizon H")
            print()
        # [C4] v11: comment corrected. Demo size matches the Fig. 5 / Fig. 6
        # captions ("across 150 simulation steps") and the actual demo_steps
        # value below. The previous comment said "n = 20 x 300" which was
        # wrong on both axes.
        demo_conds = ["healthy","bodily","obe","disorientation"]
        demo_runs = 20
        demo_steps = 150
        csv = os.path.join(save_dir, "gpc_v11_demo.csv")
        df, summary, ar = run_batch(demo_conds, demo_runs, demo_steps, csv_path=csv,
                                    enable_replay=True,
                                    use_active_inference=use_ai,
                                    policy_horizon=policy_horizon)
        print_summary(summary, demo_runs, demo_steps, True)
        saved = generate_plots(summary, ar, demo_runs, demo_steps, True, save_dir)
        for f in saved:
            print(f"  Plot saved: {f}")
        if HAS_PANDAS and isinstance(df, pd.DataFrame):
            print(f"  CSV saved: {csv}")
        print(f"\n  Done! Check {save_dir} for output files.")

if __name__ == "__main__":
    main()
