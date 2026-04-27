# Generative Predictive Coding of Self-Location (GPC) — Reference Simulation

This repository contains the reference Python simulation accompanying:

> **Yıldırım, A. & Erdeniz, B.** *Generative Predictive Coding of Self-Location
> in Space: A Computational Model of Hippocampal–Entorhinal and Temporoparietal
> Junction Functional Integration.* (Manuscript submitted for publication, 2026.)

The code (`GPC_v11.py`) is a direct numerical implementation of the formal
generative model in §2 of the paper. It instantiates a precision-weighted
Bayesian agent that jointly maintains an allocentric (hippocampal–entorhinal)
position estimate and an egocentric (temporoparietal-junction-centred)
multisensory estimate, integrates them through a retrosplenial-style
coordinate transform `T_θ`, and updates beliefs by gradient descent on
variational free energy. The same architecture is exercised under four
parameter regimes (healthy, full-body illusion, out-of-body experience,
disorientation), corresponding to the operating regimes described in §3 of
the paper.

The repository is the proof-of-principle simulation for the claims in §3
("Simulation Implementation and Results") and the regime-level dissociations
discussed in §4 ("Discussion").

---

## Quick start

The single command below regenerates **every figure cited in the manuscript**
(Figs 5–9 in §3, plus the cliff-free Supplementary sensitivity check) under the
parameter and sample-size protocols listed in Supplementary Material S2:

```bash
python GPC_v11.py --reproduce-paper --active-inference
```

Each protocol is written into its own sub-folder, with a CSV of per-trial
metrics and a set of publication-quality (300 dpi) figures. Re-running with
the default seed (`base_seed = 42`) reproduces the numbers reported in §3.6.

---

## Software requirements

Tested with Python 3.10 on Linux, macOS, and Windows.

| Package      | Version         |
|--------------|-----------------|
| `numpy`      | ≥ 1.24, < 3.0   |
| `pandas`     | ≥ 2.0,  < 3.0   |
| `matplotlib` | ≥ 3.7,  < 4.0   |

```bash
git clone https://github.com/aybo/gpc-self-location.git
cd gpc-self-location

python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

pip install -r requirements.txt
```

If `pandas` or `matplotlib` are missing, the script still runs but skips CSV
export and figure generation respectively, with a notice on stdout.

---

## What the simulation reproduces

`--reproduce-paper` runs the following protocols. Sample sizes and step counts
are specified per figure caption in the manuscript and re-stated here for
convenience.

| Manuscript figure                                        | Protocol             | Runs × steps | Replay | Output sub-folder         |
|----------------------------------------------------------|----------------------|--------------|--------|---------------------------|
| Fig. 5 — prediction-error dynamics                       | demo (§3.4)          | 20 × 150     |   no   | `demo_fig5_fig6/`         |
| Fig. 6 — posterior uncertainty & information gain        | demo (§3.5)          | 20 × 150     |   no   | `demo_fig5_fig6/`         |
| Fig. 7 — cross-condition quantitative summary            | full batch (§3.6)    | 50 × 500     |   no   | `section_3_6_fig7/`       |
| Fig. 8 — best/median/worst trajectories per regime       | quantile mode (§3.7) | 30 × 400     |   no   | `section_3_7_fig8/`       |
| Fig. 9 — offline hippocampal replay & generative preplay | replay (§3.7)        | 30 × 400     |  yes   | `section_3_7_fig9/`       |
| Supplementary — cliff-free sensitivity                   | clean arena          | 50 × 500     |   no   | `supp_fig_S1_cliff_free/` |

Each sub-folder contains:

- `gpc_v11_<sub>.csv` — per-trial metrics (final localisation error, mean
  `|ε_o|`, mean `|ε_s|`, free energy `F`, posterior uncertainty `tr(Σ_s)`,
  information gain, distance travelled, success rate, effective precisions,
  active-inference flag).
- Up to six condition-comparison figures: prediction-error dynamics; posterior
  uncertainty + information gain; cross-regime bar charts; best trajectory
  per regime; best/median/worst trajectories per regime; offline replay/preplay
  (only with `--replay`).

---

## Operating regimes

The four regimes share the same inferential machinery; they differ only in
the precisions `π_ego`, `π_allo` and the transform impairment `t_impair`,
following §3 of the manuscript and the formal precision thresholds in
Fig. 4 (`π_min = 0.05`, `π_conflict = 3.0`).

| Regime              | Key parameters                                         | Targeted phenomenon                                  |
|---------------------|--------------------------------------------------------|------------------------------------------------------|
| `healthy`           | `π_ego ≈ π_allo ≈ 1`, intact `T_θ`, dynamic precision  | balanced, efficient goal-reaching                    |
| `bodily`            | `π_ego = 6 ≫ π_allo = 0.05`                            | sensory-dominant, jittery belief (full-body illusion)|
| `obe`               | `π_ego = 0.02 < π_min`, `π_allo = 6` dominates         | mislocalised, allocentric drift (out-of-body)        |
| `disorientation`    | `t_impair = 0.85`, precisions intact                   | unreliable retrosplenial transform                   |

The §3.6 quantitative ordering of impairment reproduced under
`--reproduce-paper --active-inference` (50 runs × 500 steps, seed = 42).
The quantities are reported per regime exactly as in the manuscript text;
the full set of metrics for every regime is in the corresponding CSV.

| Regime               | Quantities reported in paper §3.6                                                  |
|----------------------|------------------------------------------------------------------------------------|
| Healthy              | mean `\|ε_o\|` = 0.060, `F` = 0.047, path = 16.2 a.u., success = 100 %              |
| Full-Bodily Illusion | final localisation error = 0.93, `F` = 9.84, path = 31.7 a.u., max `\|ε_s\|` = 4.15 |
| Out-of-Body          | `F` = 0.017, success = 94 %; elevated posterior uncertainty                        |
| Disorientation       | mean `\|ε_o\|` = 0.72 (highest), `F` = 12.67 (highest), path = 30.1 a.u.            |

The OBE regime produces the theoretically diagnostic signature predicted in
§3.5 and §4: low free energy combined with elevated localisation error and a
non-monotonic posterior uncertainty profile (the system converges on an
internally coherent but spatially displaced posterior).

---

## Theory ↔ implementation correspondence

The script is a literal numerical instantiation of the equations in §2 of the
manuscript. The most important correspondences are:

| Manuscript                                                                    | Implementation                                                                                                                  |
|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Allocentric prior `x_allo ∈ ℝ²`                                              | `mu_allo` in `SimState`                                                                                                          |
| Heading `θ`, jointly estimated with position                                  | `mu_theta`; full state `μ_s = (μ_pos,x, μ_pos,y, μ_θ)`                                                                           |
| Egocentric likelihood `P(o\|x_ego)`, per-landmark                             | `mu_ego ∈ ℝ^(N×2)` (distance, bearing) for `N = 4` landmarks                                                                     |
| RSC transform `T_θ(x) = R(θ)(x_target − x_allo) + b`                         | `rsc_transform()`, with learnable bias `b`                                                                                       |
| Sensory PE `ε_o = o − g(μ_x)` (Eq. 15)                                        | `eps_o`                                                                                                                          |
| State PE `ε_s = μ_ego − T_θ(μ_allo)` (Eq. 16)                                 | `eps_s`                                                                                                                          |
| Belief update (Eq. 18) `μ_x ← μ_x + κ Jᵀ Π_o ε_o − κ Π_prior(μ − μ_prior)`   | accuracy + complexity gradient in `sim_step()`; `κ = DT = 0.1`                                                                   |
| Posterior covariance `Σ_post = (Σ_prior⁻¹ + Jᵀ Π_o J)⁻¹`                      | Laplace / Kalman information form in `sim_step()`                                                                                |
| Free energy `F = accuracy + complexity` (Eq. 14)                              | `acc_o + comp` per step                                                                                                          |
| Active inference `P(π) ∝ exp(−γ G(π))` (Eqs. 21–23)                           | `evaluate_policy()` + `active_inference_policy()` over 5 angular biases `{±0.95, ±0.45, 0}` with horizon `H = 5`, `γ_disc = 0.96`|
| Offline generative simulation `x̃_{1:T} ~ P(x_{1:T} \| θ_updated)`             | `generate_replay_path()` (forward / reverse / noisy-recombined) and `generate_preplay_path()`                                    |

The state-level prediction error in the simulation is computed as a
per-landmark residual `ε_s ∈ ℝ^(2N_L)` with `N_L = 4`, equivalent to the
ℝ² positional residual of §2.1.8 projected through `∂T_θ / ∂μ_allo` under the
deterministic limit `Σ_T → 0`. Heading `θ` is estimated jointly with position
because `∂T_θ / ∂θ ≠ 0`. These choices are documented in §3.2 of the paper.

The variable named `cliff` in the source corresponds to the **obstacle**
regions described in the manuscript (an artefact of an earlier draft;
behaviour is unchanged).

---

## Command-line interface

```bash
# Regenerate all paper figures (recommended)
python GPC_v11.py --reproduce-paper --active-inference

# Manual protocols
python GPC_v11.py --batch-all                       # 50 runs × 500 steps  (paper §3.6)
python GPC_v11.py --batch-all --replay              # + offline replay/preplay
python GPC_v11.py --batch-all --active-inference    # active-inference policy
python GPC_v11.py --quantile-mode                   # 30 runs × 400 steps  (paper §3.7)
python GPC_v11.py --compare-all                     # 1 deterministic run per regime

# Subset runs
python GPC_v11.py --batch --condition healthy,obe --runs 30
```

| Flag                  | Effect                                                                  |
|-----------------------|-------------------------------------------------------------------------|
| `--reproduce-paper`   | Regenerate every protocol cited in the manuscript                       |
| `--batch-all`         | Run all four regimes (default 50 × 500, paper §3.6)                     |
| `--quantile-mode`     | Default 30 × 400 for §3.7 trajectory quantiles                          |
| `--batch`             | Run regimes listed via `--condition`                                    |
| `--compare-all`       | One deterministic run per regime, fixed start                           |
| `--condition K1,K2`   | Comma-separated regime keys (`healthy`,`bodily`,`obe`,`disorientation`) |
| `--runs N`            | Trials per regime                                                       |
| `--steps N`           | Simulation steps per trial                                              |
| `--replay`            | Enable offline replay + generative preplay + parameter consolidation    |
| `--active-inference`  | Use the EFE policy for action selection (paper §2.1.7)                  |
| `--policy-horizon H`  | EFE rollout horizon (default 5)                                         |
| `--save-dir DIR`      | Output directory root                                                   |

---

## Reproducibility

- Every trial uses a deterministic seed derived from the trial index and a
  base seed (`base_seed = 42`), as stated in the paper's Reproducibility note
  and Supplementary Material S2.
- Numerical integration uses a fixed step `DT = 0.1`, matched to `κ = dt` in
  Eq. 18 of the paper.
- Posterior covariance is symmetrised and eigenvalue-clipped at `1e-7` jitter
  on every update to keep `Σ_s` symmetric positive-definite (Laplace
  approximation requirement).
- Arena geometry, landmark layout, obstacle regions, start region, and goal
  are declared as module-level constants near the top of `GPC_v11.py` and can
  be edited directly to test other geometries.
- All synthetic data underlying Figs. 5–9 are regenerated by
  `--reproduce-paper`; no static CSVs are committed to the repository.

---

## Repository contents

| File                                   | Purpose                                                        |
|----------------------------------------|----------------------------------------------------------------|
| [`GPC_v11.py`](GPC_v11.py)             | Self-contained simulation: model, batch runner, figure scripts |
| [`requirements.txt`](requirements.txt) | Python dependency pins                                         |
| [`CITATION.cff`](CITATION.cff)         | Machine-readable citation metadata                             |
| [`LICENSE`](LICENSE)                   | MIT licence                                                    |
| [`.gitignore`](.gitignore)             | Standard Python + simulation-output ignores                    |

---

## Citation

If you use this code in academic work, please cite the paper:

```bibtex
@article{yildirim_erdeniz_gpc_2026,
  title   = {Generative Predictive Coding of Self-Location in Space:
             A Computational Model of Hippocampal--Entorhinal and Temporoparietal
             Junction Functional Integration},
  author  = {Y{\i}ld{\i}r{\i}m, Aybars and Erdeniz, Burak},
  note    = {Manuscript submitted for publication},
  year    = {2026}
}
```

A `CITATION.cff` file is provided for GitHub's *Cite this repository* button;
its `journal`, `volume`, `doi`, and `year` fields will be updated upon
publication.

---

## Authors

- **Aybars Yıldırım** — Experimental Psychology Graduate Program, Izmir
  University of Economics, Izmir, Turkey.
  ORCID: [0000-0002-5918-6386](https://orcid.org/0000-0002-5918-6386).
- **Burak Erdeniz** — Department of Psychology and Complex Systems Application
  and Research Center, Izmir University of Economics, Izmir, Turkey.
  ORCID: [0000-0001-5517-5022](https://orcid.org/0000-0001-5517-5022).

For correspondence regarding the paper, please contact the authors at the
address listed in the manuscript.

---

## Licence

The code in this repository is released under the **MIT Licence**
(see [`LICENSE`](LICENSE)), as stated in the *Code Availability* section of
the manuscript.
