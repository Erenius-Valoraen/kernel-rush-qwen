"""Generate the README benchmark charts from the official Kernel Rush results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

os.makedirs("assets", exist_ok=True)
plt.rcParams.update({
    "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 140,
})
INK, MUTED, HILITE, REF = "#1f2933", "#9aa5b1", "#2f855a", "#cbd2d9"


def label_bars(ax, bars, vals, fmt="{:.0f}", dy=0.01, color=INK, fw="bold"):
    top = max(vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy * top,
                fmt.format(v), ha="center", va="bottom", fontsize=11, color=color, fontweight=fw)


# ---- 1) progression across optimizations ---------------------------------
steps = ["Native\nTransformers", "+ Triton\nkernels", "+ FP8", "+ Draft\nhead",
         "+ Cross-workload\ncache", "+ Tree\nspeculation"]
scores = [400, 1013, 1064, 1280, 1432, 1471]
fig, ax = plt.subplots(figsize=(9, 4.6))
colors = [MUTED] * (len(scores) - 1) + [HILITE]
bars = ax.bar(range(len(steps)), scores, color=colors, width=0.68)
label_bars(ax, bars, scores)
ax.set_xticks(range(len(steps)))
ax.set_xticklabels(steps, fontsize=10.5)
ax.set_ylabel("Decode throughput  (tokens/sec, higher = better)")
ax.set_title("How the engine got faster, one idea at a time",
             fontsize=14, fontweight="bold", color=INK, pad=12)
ax.set_ylim(0, 1650)
ax.margins(x=0.02)
fig.text(0.5, -0.02, "Qwen3-4B decode on one H100 · geometric mean across the hidden workloads · exact greedy output preserved",
         ha="center", fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig("assets/progression.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- 2) vs native HuggingFace Transformers on the public workloads --------
workloads = ["batch 1\n512 in, 32 out", "batch 4\n2048 in, 32 out", "batch 16\n512 in, 128 out"]
ref = [42, 131, 599]
ours = [302, 558, 4181]
speedup = [o / r for o, r in zip(ours, ref)]
x = range(len(workloads))
fig, ax = plt.subplots(figsize=(9, 4.6))
w = 0.38
b1 = ax.bar([i - w / 2 for i in x], ref, w, label="Native HF Transformers", color=REF)
b2 = ax.bar([i + w / 2 for i in x], ours, w, label="This engine", color=HILITE)
label_bars(ax, b1, ref)
label_bars(ax, b2, ours)
for i, s in enumerate(speedup):
    ax.text(i + w / 2, ours[i] + 0.06 * max(ours), f"{s:.1f}x", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=HILITE)
ax.set_xticks(list(x)); ax.set_xticklabels(workloads, fontsize=10.5)
ax.set_ylabel("Decode throughput  (tokens/sec)")
ax.set_title("Same model, same output — 4–7x faster than stock Transformers",
             fontsize=14, fontweight="bold", color=INK, pad=12)
ax.set_ylim(0, 4700)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("assets/vs_transformers.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- 3) final leaderboard --------------------------------------------------
teams = ["dryfter", "krish", "Krxfty", "KAWK", "mc", "Segfault (ours)"]
vals = [1152, 1156, 1230, 1232, 1291, 1471]
fig, ax = plt.subplots(figsize=(9, 4.0))
colors = [MUTED] * (len(teams) - 1) + [HILITE]
bars = ax.barh(range(len(teams)), vals, color=colors, height=0.68)
for b, v in zip(bars, vals):
    ax.text(b.get_width() - 15, b.get_y() + b.get_height() / 2, f"{v}", ha="right", va="center",
            fontsize=11, fontweight="bold", color="white")
ax.set_yticks(range(len(teams))); ax.set_yticklabels(teams, fontsize=11)
ax.set_xlabel("Score  (tokens/sec)")
ax.set_title("Final leaderboard — Dryft Kernel Rush @ Hack the North",
             fontsize=14, fontweight="bold", color=INK, pad=12)
ax.set_xlim(0, 1600)
fig.tight_layout()
fig.savefig("assets/leaderboard.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

print("wrote assets/progression.png, assets/vs_transformers.png, assets/leaderboard.png")
