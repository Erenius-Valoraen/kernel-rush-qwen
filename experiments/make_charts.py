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

# ---- 2) vs common inference engines (greedy decode, H100) -----------------
# All greedy, one H100, tokens/sec = batch x output / total time (prefill+decode).
# Transformers & vLLM measured directly; this engine from the official run.
workloads = ["batch 1\n512 in, 32 out", "batch 4\n2048 in, 32 out", "batch 16\n512 in, 128 out"]
BLUE = "#4c6ef5"
series = [("HF Transformers", [47, 149, 682], REF),
          ("vLLM", [175, 623, 2438], BLUE),
          ("This engine", [302, 558, 4181], HILITE)]
x = range(len(workloads))
fig, ax = plt.subplots(figsize=(9.2, 4.8))
w = 0.26
for k, (name, vals, color) in enumerate(series):
    off = (k - 1) * w
    bars = ax.bar([i + off for i in x], vals, w, label=name, color=color)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.04, f"{v}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=INK)
ax.set_yscale("log")
ax.set_xticks(list(x)); ax.set_xticklabels(workloads, fontsize=10.5)
ax.set_ylabel("Decode throughput  (tokens/sec, log scale)")
ax.set_title("Greedy decode vs common inference engines",
             fontsize=14, fontweight="bold", color=INK, pad=12)
ax.set_ylim(30, 8000)
ax.legend(frameon=False, loc="upper left", ncol=3)
ax.grid(axis="x", visible=False)
fig.text(0.5, -0.02, "Qwen3-4B, one H100 · same model, identical greedy output · faster than vLLM at batch 1 and 16, ~5x over stock Transformers",
         ha="center", fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig("assets/vs_engines.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

print("wrote assets/progression.png, assets/vs_engines.png")
