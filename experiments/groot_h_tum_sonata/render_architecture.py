"""Render the GR00T-H-N1.7 inference architecture AS RUN on TUM SonATA Franka.
Mirrors NVIDIA's System-2 (VLM) / System-1 (Diffusion Transformer) figure but with
the concrete components and shapes verified from checkpoint-15000/config.json.

Usage (needs matplotlib — run inside the groot-h-dev container or any matplotlib env):
    python render_architecture.py [output.png]
Default output: docs/assets/architecture.png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_BG = "#ffffff"
fig, ax = plt.subplots(figsize=(18, 9.5), dpi=120)
fig.patch.set_facecolor(FIG_BG)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, fc, ec, lw=1.5, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={r*100}",
                                fc=fc, ec=ec, lw=lw, mutation_scale=1))
def txt(x, y, s, fs=12, c="black", w="normal", ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va=va, style=style)
def arrow(x1, y1, x2, y2, c="#444", lw=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                 lw=lw, color=c, shrinkA=0, shrinkB=0))
def tokens(x, y0, n, c, w=2.0, h=2.4, gap=0.7):
    for i in range(n):
        ax.add_patch(plt.Rectangle((x, y0 + i*(h+gap)), w, h, fc=c, ec="white", lw=1))

# ── Title ──────────────────────────────────────────────────────────────────
txt(50, 97, "GR00T-H-N1.7  ·  TUM SonATA Franka  —  inference architecture (as run)",
    fs=17, w="bold")
txt(50, 93.2, "verified from checkpoint-15000/config.json", fs=10.5, c="#666", style="italic")

PURPLE, BLUE, RED, ORANGE, CREAM = "#b39ddb", "#1565c0", "#e57373", "#fb8c00", "#ffe0b2"

# ── INPUTS (left) ────────────────────────────────────────────────────────────
# Image observation: 3 cameras
txt(9, 88, "Image Observation", fs=12.5, w="bold")
cam_labels = ["Third-person", "Wrist", "Ultrasound"]
for i, lbl in enumerate(cam_labels):
    cx = 2 + i*5.2
    ax.add_patch(plt.Rectangle((cx, 79.5), 4.6, 6, fc="#263238", ec="#90a4ae"))
    txt(cx+2.3, 82.5, lbl, fs=7.0, c="white")
txt(9, 77.6, "3 RGB views · 256×256", fs=8.5, c="#555")

# Language
txt(9, 71.5, "Language Instruction", fs=12.5, w="bold")
box(1.5, 64.5, 15, 5.2, "#eceff1", "#90a4ae", lw=1)
txt(9, 67.1, '"Sweep the probe across\nthe abdomen, covering\nthe aorta and IVC."', fs=8.3, style="italic")

# Robot state
txt(9, 58, "Robot State", fs=12.5, w="bold")
box(1.5, 47, 15, 9, "#fde8e8", "#e57373", lw=1.2)
txt(9, 53.4, "joint angles  (7-D)", fs=8.6)
txt(9, 51.0, "force / torque  (6-D)", fs=8.6)
txt(9, 48.6, "EEF pose ref  (6-D, pass-through)", fs=7.8, c="#b71c1c")

# Encode / tokenize arrows
arrow(17, 82.5, 24, 82.5); txt(20.5, 84, "Encode", fs=9.5)
arrow(17, 67, 24, 67);     txt(20.5, 68.4, "Tokenize", fs=9.5)
arrow(17, 51.5, 53, 51.5); txt(22, 53, "Encode", fs=9.5)   # state goes straight to System 1

# ── Token stacks ─────────────────────────────────────────────────────────────
tokens(24.5, 78, 5, PURPLE); txt(26, 76.0, "Image\ntokens", fs=8.5, c="#5e35b1")
tokens(24.5, 63.5, 3, BLUE); txt(26, 61.8, "Text\ntokens", fs=8.5, c="#1565c0")

# ── SYSTEM 2: VLM ────────────────────────────────────────────────────────────
box(31, 58, 19, 33, "#f5f5f5", "#9e9e9e", lw=2, r=0.03)
txt(40.5, 84, "Vision-Language\nModel", fs=15, w="bold")
txt(40.5, 75.5, "System 2", fs=14, w="bold", c="#d81b60")
txt(40.5, 70, "Cosmos-Reason2-2B", fs=11, w="bold", c="#222")
txt(40.5, 67, "(Qwen3-VL architecture)", fs=9, c="#666", style="italic")
txt(40.5, 63.5, "features from layer 16\nembedding dim 2048", fs=8.6, c="#555")

# VLM output tokens
tokens(51.5, 70, 4, PURPLE)
tokens(51.5, 62, 3, BLUE)

# ── SYSTEM 1: Diffusion Transformer ──────────────────────────────────────────
box(56, 20, 24, 71, "#f5f5f5", "#9e9e9e", lw=2, r=0.02)
txt(68, 84, "Diffusion\nTransformer", fs=15, w="bold")
txt(68, 75.5, "System 1", fs=14, w="bold", c="#0288d1")
txt(68, 70.5, "flow-matching DiT", fs=10.5, w="bold", c="#222")
txt(68, 67.5, "32 layers · 32 heads", fs=8.8, c="#555")
txt(68, 64.8, "4 denoising steps", fs=8.8, c="#555")

# state token (red) entering System 1
tokens(53.5, 49.5, 2, RED, h=2.4)

# action tokens entering (cream) + inside (cream/orange) → denoising
txt(50, 33, "Action\ntokens", fs=8.8, w="bold", ha="center")
tokens(54, 22, 6, CREAM)             # noisy action tokens in
tokens(62, 22, 6, CREAM)
tokens(70, 22, 6, ORANGE)            # denoised
arrow(62, 18.5, 76, 18.5, c="#777", lw=1.8); txt(69, 16.7, "Denoising", fs=9.5, c="#555")

# ── OUTPUT ───────────────────────────────────────────────────────────────────
tokens(81.5, 22, 6, ORANGE)
arrow(80.3, 35, 86, 35)
txt(92, 74, "Motor Action", fs=13, w="bold")
box(85.5, 47, 13, 24, "#eceff1", "#607d8b", lw=1.5)
txt(92, 66, "Franka Panda\n+ US probe", fs=11, w="bold")
txt(92, 60.5, "EEF pose action chunk", fs=9.2, c="#222")
txt(92, 57.6, "50 steps × 6-D", fs=9.2, c="#222")
txt(92, 54.6, "REL_XYZ_ROT6D →\nabsolute (x,y,z, roll,pitch,yaw)", fs=8.0, c="#555")
txt(92, 50.0, "targets t+1 … t+50", fs=8.2, c="#b71c1c")

# bottom note
txt(50, 7, "Inference: re-plan every 16 steps (demo) — chunk[k] is the forecast for row anchor+1+k; "
           "first action executed, rest cached.  Action head samples from Gaussian noise (stochastic).",
    fs=9, c="#555", style="italic")

plt.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "docs/assets/architecture.png"
plt.savefig(out, facecolor=FIG_BG, bbox_inches="tight")
print("saved", out)
