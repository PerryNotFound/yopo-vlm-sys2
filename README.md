# yopo-vlm-sys2

[English](README.md) | [中文](README_zh.md)

A **System-2 semantic safety layer** for YOPO-Rally. YOPO (System-1) plans from depth
images at 10 Hz and is blind to *semantic* hazards that depth cannot see — water, sand,
mud. This module adds low-frequency scene understanding with Qwen3-VL-4B and injects it
into the planner via **Semantic Veto on trajectory candidates**.

## Architecture

```
              ┌──────────────────────────────────────────────────┐
 RGB image ──►│ color segmentation (fast, per-frame)              │
 /odom ──────►│   + VLM binary verification (slow, semantic)      │
 /visualize/*►│        ↓                                          │
              │ danger zones → path-zone matching → /vlm_veto     │
              │ all vetoed → RECOVERY → /vlm_goal fan redirection │
              └──────────────────────────────────────────────────┘
                                  ↓
  yopo_node: vetoed candidates get cost +inf → argmin picks only
  among safe candidates (geometric obstacles like trees are still
  handled by YOPO's learned depth cost — the two systems intersect)
```

Design principles: async decoupling (never blocks yopo 10 Hz / mpc 20 Hz), soft timeout
fallback (pure YOPO resumes 2 s after the last veto update), and no single point of
failure (a failed VLM inference silently skips that round).

## Experiments (M0–M4)

| # | Name | Method | Code | Status |
|---|---|---|---|---|
| **M0** | Baseline | Pure YOPO + MPC, no semantic awareness | `diablo_algorithm.launch.py` only | ✅ |
| **M1** | Frame-level Veto | Per-frame binary VLM veto; `/cmd_vel` cut off when unsafe | `eval/m1_safety.py` + `eval/m1_arbitrator.py` | ✅ |
| **M2** | Candidate Semantic Veto | Veto candidate trajectories crossing dangerous zones | `eval/m2_veto.py` | ✅ |
| **M3** | Positive Fusion | Semantic safety score fused into the cost function (soft counterpart of M2) | `eval/m3_fusion.py` | ✅ |
| **M4** | Veto + Recovery | Fallback when all candidates are vetoed (fan redirection / back-up / replan) | RECOVERY seed in `eval/m2_veto.py` | 🔶 |

Perception ablation within M2 (`MODE` switch at the top of `m2_veto.py`):

| Mode | Description | Role in paper |
|---|---|---|
| `color` | Pure HSV color segmentation | Baseline: cannot tell water from blue objects |
| `vlm_zone` | VLM-only zone classification | Baseline: 4B zero-shot spatial grounding is unreliable |
| `hybrid` | Color localization + VLM binary semantic verification | Proposed |

## Layout

```
yopo-vlm-sys2/
├── eval/               # closed-loop experiment nodes (M0–M4) + offline evaluation
├── data/               # semantic dataset build (900 frames; ground/water/sand/mud)
├── finetune/           # QLoRA fine-tuning (Qwen3-VL-4B; train/val JSONL + script)
├── deploy/             # deployment tools (single-image smoke test)
├── archive/            # superseded prototypes with failure notes (archive/README.md)
└── docs/               # handbook and change logs (Chinese)
```

## Run

See `docs/闭环评测工作手册.md` (Chinese) for the full step-by-step guide. In short:
ROS-TCP-Endpoint → Unity EvaluationScene (Play + Connect) → YOPO+MPC launch →
VLM node (`python eval/m2_veto.py`) → set goal via RViz "2D Goal Pose".

> After modifying `yopo_node` (the `/vlm_veto` interface), rebuild with
> `colcon build --packages-select diablo_ros_wrapper` (never `--symlink-install`).

## Metrics (for the paper)

Record `/vlm_state`, `/vlm_veto`, `/odom` via `ros2 bag` and compute: success rate,
VLM intervention count, false-intervention rate, minimum distance to hazard,
time-to-goal, and controlled-stop (STUCK) count.

## Models

- Base: `Qwen3-VL-4B-Instruct` (4-bit NF4; ~0.2–0.3 s per inference)
- Fine-tune: QLoRA (r=16) on 900 frames. The current adapter (trained on flat-color
  patches) underperforms the base model on val (53.9% vs 82.6%), so closed-loop
  experiments use the base model; a texture-augmented data v2 is on the roadmap.
