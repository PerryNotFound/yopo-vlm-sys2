#!/usr/bin/env python3
"""CSRA paper-style figures v2: system architecture + roadmap."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

C_S1, C_VLM, C_SRA, C_DEC, C_SIM = "#dbeafe", "#fef3c7", "#dcfce7", "#fee2e2", "#f1f5f9"
C_EDGE, C_ACC = "#334155", "#7c3aed"
FIG = "/home/admin0/YOPO-Rally-Whole/yopo-vlm-sys2/docs/figures"

def box(ax, x, y, w, h, title, lines, fc, title_fc=None, fontsize=9, title_size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=C_EDGE, lw=1.2, zorder=2))
    ty = y + h - 0.24
    ax.text(x + w/2, ty, title, ha="center", va="center", fontsize=title_size,
            fontweight="bold", color=title_fc or C_EDGE, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w/2, ty - 0.32 - i*0.26, ln, ha="center", va="center",
                fontsize=fontsize, color="#1e293b", zorder=3)

def arrow(ax, x1, y1, x2, y2, label=None, color=C_EDGE, lw=1.6, dashed=False,
          loffset=(0, 0.2), fs=8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, color=color, lw=lw, zorder=4,
                                 linestyle="--" if dashed else "-"))
    if label:
        ax.text((x1+x2)/2 + loffset[0], (y1+y2)/2 + loffset[1], label,
                ha="center", va="center", fontsize=fs, color=color, fontweight="bold",
                zorder=5, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

# ============================================================ Fig 1: system
fig, ax = plt.subplots(figsize=(13.5, 8.4))
ax.set_xlim(0, 13.5); ax.set_ylim(0, 8.4); ax.axis("off")

# top guarantee banners (full width, clear of boxes)
ax.text(6.75, 8.1, "Layer-1 (known hazards):  P(accept hazard) ≤ (α + 1/(n_H+1)) · p_H      [Prop. 1b]",
        ha="center", fontsize=10, color="#b91c1c", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff7ed", ec="#b91c1c", lw=1.1))
ax.text(6.75, 7.5, "Layer-2 (unseen hazards):  open-vocab flags → β spike exposes auditor failure      [Prop. 2]",
        ha="center", fontsize=10, color=C_ACC, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#faf5ff", ec=C_ACC, lw=1.1))

# boxes
box(ax, 0.15, 2.9, 2.1, 2.2, "Unity Sim\n(YOPO-Sim)", ["RGB 10 Hz", "Depth 160×32", "Odom / GT"], C_SIM, title_size=9.5)
box(ax, 2.9, 4.3, 2.9, 2.4, "System-1: YOPO (frozen)",
    ["depth + v + goal  →  30 Hz", "5 candidate splines {τᵢ}", "geometric cost {cᵢ}"], C_S1)
box(ax, 2.9, 0.75, 2.9, 2.5, "Async VLM auditor",
    ["Qwen3-VL-4B, 1–3 s", "Q1: water/mud/sand 8×8 grid", "Q2: any unseen hazard?", "injectable delay"], C_VLM)
box(ax, 6.5, 0.75, 2.9, 2.5, "Spatial anchoring (SRA)",
    ["mask → ground projection", "enclosing disks (over-cover)", "center, radius, r, t_expire", "world frame, not camera"], C_SRA)
box(ax, 6.5, 4.3, 2.9, 2.4, "Replay calibration (offline)",
    ["log ALL candidates (R, c)", "replay argmin for any ε", "ε(α): P(err|haz) ≤ α + 1/(n_H+1)"], "#ede9fe", title_fc=C_ACC)
box(ax, 10.3, 2.6, 3.0, 2.9, "Decision layer (30 Hz)",
    ["R(τᵢ): front-segment only", "ε from replay calibration", "∃ R(τᵢ) ≤ ε → argmin c", "else → calibrated retreat"], C_DEC)
box(ax, 10.3, 0.55, 2.6, 1.1, "MPC tracking", ["YOPO-Rally stack"], C_SIM, title_size=9.5)

# arrows
arrow(ax, 2.25, 4.6, 2.9, 5.5, "depth/odom", loffset=(0.05, 0.15))
arrow(ax, 2.25, 3.4, 2.9, 2.0, "RGB", loffset=(0.1, -0.25))
arrow(ax, 5.8, 5.5, 10.3, 4.6, "{τᵢ, cᵢ}  30 Hz", color="#1d4ed8", loffset=(0, 0.22))
arrow(ax, 5.8, 2.0, 6.5, 2.0, "grid + scores", color="#b45309", loffset=(0, 0.22))
arrow(ax, 9.4, 2.0, 10.75, 2.6, "SRAs", color="#15803d", loffset=(0.3, 0.1))
arrow(ax, 7.95, 4.3, 7.95, 3.25, "", color=C_ACC, dashed=True)
ax.text(8.12, 3.7, "calibrate the deployed policy,\nnot the perception", fontsize=7.5, color=C_ACC, style="italic")
arrow(ax, 9.4, 5.5, 10.3, 4.7, "ε", color=C_ACC, loffset=(0, 0.2))
arrow(ax, 11.8, 2.6, 11.8, 1.65, "τ*", color="#b91c1c", loffset=(0.25, 0))
# cmd_vel return: route below all boxes
ax.plot([10.3, 1.2], [0.25, 0.25], color=C_EDGE, lw=1.6, ls="--", zorder=4)
arrow(ax, 10.3, 0.85, 10.3, 0.25, "", color=C_EDGE, dashed=True)
arrow(ax, 1.2, 0.25, 1.2, 2.9, "", color=C_EDGE, dashed=True)
ax.text(6.0, 0.42, "cmd_vel", ha="center", fontsize=8, color=C_EDGE, fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

fig.suptitle("CSRA — Calibrated Semantic Risk Anchoring  (system architecture)",
             fontsize=13, fontweight="bold", y=0.99)
fig.savefig(f"{FIG}/csra_system.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================================================ Fig 2: roadmap
fig, ax = plt.subplots(figsize=(16, 5.6))
ax.set_xlim(-0.4, 17.6); ax.set_ylim(0, 5.6); ax.axis("off")

weeks = [
    ("W0",    "Assets check\nwet asphalt?\nshadows?",            "#94a3b8", False, ""),
    ("W1",    "Pipeline frozen\n8×8 grid + open\n-vocab prompt", "#60a5fa", True,  "fail → M6a\nseg-driven"),
    ("W2",    "Detection≥0.85\nIoU≥0.4\n(CI lower bound)",       "#f59e0b", True,  "fail → M6a"),
    ("W3",    "M8 vacuous\nreference",                            "#94a3b8", False, ""),
    ("W4",    "Replay calib.\ntwo-sided\ntest",                   "#f59e0b", True,  "fail → check\nseparability"),
    ("W5",    "Delay × advice\ncorrect vs\ncorrupted",            "#ef4444", True,  "fail → Plan B\n(RA-L only)"),
    ("W6",    "Closed-loop\ncoverage\nΔ_cov",                     "#f59e0b", True,  "fail → empirical\nbound, report"),
    ("W7-8",  "Phase-1 matrix\nS1/S2/S4/S6\n+S7 unseen",          "#60a5fa", False, ""),
    ("W9",    "α sweep\n+ main figure",                           "#60a5fa", False, ""),
    ("W10",   "M6a seg.\nbaseline",                               "#94a3b8", False, ""),
    ("W11",   "Real robot\n1 scene",                              "#94a3b8", False, ""),
    ("W12-14","Writing",                                          "#a78bfa", False, ""),
    ("W15",   "Review +\nsubmit",                                 "#22c55e", False, ""),
]
x0, dx, y = 0.45, 1.32, 2.75
ax.plot([x0-0.3, x0+dx*(len(weeks)-1)+0.5], [y, y], color=C_EDGE, lw=2, zorder=1)
for i, (wk, txt, color, gate, fb) in enumerate(weeks):
    x = x0 + dx*i
    ax.add_patch(FancyBboxPatch((x-0.58, y-1.0), 1.16, 1.9,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 fc="white", ec=color, lw=2.4 if gate else 1.2, zorder=2))
    ax.text(x, y+1.22, wk, ha="center", fontsize=10, fontweight="bold", color=color)
    ax.text(x, y-0.05, txt, ha="center", va="center", fontsize=7.6, color="#1e293b", zorder=3)
    if gate:
        ax.text(x, y-1.25, "◆ GATE", ha="center", fontsize=8, color=color, fontweight="bold")
    if fb:
        ax.text(x, y-1.75, fb, ha="center", fontsize=6.8, color="#64748b", style="italic")

xd = x0 + dx*(len(weeks)-1) + 0.15
ax.annotate("IROS 2027 DDL ~Mar 1", xy=(xd, y+1.0), xytext=(xd-0.4, y+2.3),
            ha="center", fontsize=10, fontweight="bold", color="#b91c1c",
            arrowprops=dict(arrowstyle="-|>", color="#b91c1c", lw=1.8))

ax.text(0.3, 5.25, "CSRA Roadmap — Plan A (15 weeks, IROS 2027)", fontsize=14, fontweight="bold")
ax.text(0.3, 4.72, "Two-sided gates: safety side (bound holds) AND liveness side (β ≤ β_max, completion ≥ 0.7×M0)",
        fontsize=9.5, color="#475569")
ax.text(0.3, 0.35, "Cut list (Plan A): M7, M5-uncertainty head, S5-3s if needed, real-robot 2nd scene      |      S7 unseen-hazard scene: zero-shot detection vs M6a (why-VLM evidence)",
        fontsize=8.5, color="#64748b")

fig.savefig(f"{FIG}/csra_roadmap.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("done")
