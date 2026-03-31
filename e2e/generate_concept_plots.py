"""Generate conceptual diagrams for all operator READMEs.

Pure synthetic data — no model weights or activation dumps needed.
Run from project root:
    python -m e2e.generate_concept_plots
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from e2e.plot_utils import setup_style, save_fig

OPERATORS_DIR = os.path.join(os.path.dirname(__file__), "..", "operators")


def _op_dir(name: str) -> str:
    return os.path.join(OPERATORS_DIR, name)


# =========================================================================
# 01 — Linear: 矩阵乘法彩色网格
# =========================================================================
def plot_01_linear():
    """Visualize y = xW^T as colored grid multiplication."""
    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5),
                             gridspec_kw={"width_ratios": [3, 4, 1, 4]})

    x = np.array([[1, 2, 3, 0.5],
                   [0, -1, 2, 1],
                   [-1, 0, 1, 3]], dtype=np.float32)
    W = np.array([[1, 0, -1, 2],
                  [0.5, 1, 0, -0.5],
                  [2, -1, 1, 0],
                  [0, 0.5, -0.5, 1]], dtype=np.float32)
    y = x @ W.T  # (3,4)

    for ax, mat, title in [(axes[0], x, r"$\mathbf{x}$ (3×4)"),
                            (axes[1], W.T, r"$\mathbf{W}^T$ (4×4)"),
                            (axes[3], y, r"$\mathbf{y}=\mathbf{x}\mathbf{W}^T$ (3×4)")]:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # Multiply sign
    axes[2].text(0.5, 0.5, "@", fontsize=24, ha="center", va="center",
                 transform=axes[2].transAxes)
    axes[2].axis("off")

    fig.colorbar(im, ax=axes, shrink=0.6, label="Value")
    fig.suptitle("Linear Transform: y = x @ W.T", fontsize=13, y=1.02)
    save_fig(fig, os.path.join(_op_dir("01_linear"), "impl.py"), "error_dist")


# =========================================================================
# 02 — Softmax: Temperature scaling
# =========================================================================
def plot_02_softmax():
    """Show how temperature sharpens/flattens the distribution."""
    setup_style()
    logits = np.array([2.0, 1.0, 0.5, -0.5, -1.0, 3.0, 0.1, -0.3])
    labels = [f"{v:+.1f}" for v in logits]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
    temps = [0.25, 1.0, 2.0, 10.0]

    for ax, T in zip(axes, temps):
        scaled = logits / T
        e = np.exp(scaled - np.max(scaled))
        probs = e / e.sum()
        colors = plt.cm.Blues(probs / probs.max() * 0.8 + 0.2)
        ax.bar(range(len(logits)), probs, color=colors, edgecolor="steelblue", linewidth=0.5)
        ax.set_title(f"T = {T}", fontsize=12)
        ax.set_xticks(range(len(logits)))
        ax.set_xticklabels(labels, fontsize=8, rotation=45)
        ax.set_ylim(0, 1.05)
        if T == 0.25:
            ax.set_ylabel("Probability")
        # Annotate max
        idx = np.argmax(probs)
        ax.text(idx, probs[idx] + 0.02, f"{probs[idx]:.2f}", ha="center", fontsize=8, color="red")

    fig.suptitle("Softmax Temperature: Low T → Sharp, High T → Flat", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(_op_dir("02_softmax"), "impl.py"), "temperature")


# =========================================================================
# 03 — LayerNorm: Per-token normalization
# =========================================================================
def plot_03_layer_norm():
    """Show how LayerNorm normalizes each token independently."""
    setup_style()
    rng = np.random.default_rng(42)

    # 4 tokens with different mean/variance
    raw = np.array([rng.normal(5, 2, 64),
                    rng.normal(-3, 0.5, 64),
                    rng.normal(0, 4, 64),
                    rng.normal(10, 1, 64)])

    # Apply LayerNorm
    mean = raw.mean(axis=-1, keepdims=True)
    var = raw.var(axis=-1, keepdims=True)
    normed = (raw - mean) / np.sqrt(var + 1e-6)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    token_labels = ["Token A (μ=5)", "Token B (μ=-3)", "Token C (μ=0)", "Token D (μ=10)"]

    for i in range(4):
        ax1.hist(raw[i], bins=30, alpha=0.6, color=colors[i], label=token_labels[i], density=True)
    ax1.set_title("Before LayerNorm")
    ax1.set_xlabel("Feature value")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=8)

    for i in range(4):
        ax2.hist(normed[i], bins=30, alpha=0.6, color=colors[i], label=token_labels[i], density=True)
    ax2.set_title("After LayerNorm")
    ax2.set_xlabel("Feature value")
    ax2.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8)

    fig.suptitle("LayerNorm: Each Token Independently → μ=0, σ=1", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(_op_dir("03_layer_norm"), "impl.py"), "before_after")


# =========================================================================
# 04 — RMSNorm: Compare with LayerNorm
# =========================================================================
def plot_04_rms_norm():
    """Show RMSNorm vs LayerNorm: RMS keeps mean offset."""
    setup_style()
    rng = np.random.default_rng(42)
    x = rng.normal(5, 2, 256).astype(np.float32)

    # LayerNorm
    ln = (x - x.mean()) / np.sqrt(x.var() + 1e-6)
    # RMSNorm
    rms = x / np.sqrt(np.mean(x**2) + 1e-6)

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.linspace(-4, 6, 80)
    ax.hist(x, bins=bins, alpha=0.4, color="#e74c3c", label=f"Input (μ={x.mean():.1f})", density=True)
    ax.hist(ln, bins=bins, alpha=0.5, color="#3498db", label=f"LayerNorm (μ={ln.mean():.2f})", density=True)
    ax.hist(rms, bins=bins, alpha=0.5, color="#2ecc71", label=f"RMSNorm (μ={rms.mean():.2f})", density=True)
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_title("RMSNorm vs LayerNorm: RMS Preserves Relative Scale Without Centering", fontsize=12)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    save_fig(fig, os.path.join(_op_dir("04_rms_norm"), "impl.py"), "before_after")


# =========================================================================
# 05 — Conv3d Patch Embed: Patch extraction
# =========================================================================
def plot_05_conv3d():
    """Show how an image is sliced into non-overlapping patches."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    # Left: image grid with patch boundaries
    H, W = 6, 6
    patch_h, patch_w = 2, 2
    img = np.random.default_rng(0).random((H, W))
    ax1.imshow(img, cmap="viridis", aspect="equal")
    # Draw patch grid
    for i in range(0, H + 1, patch_h):
        ax1.axhline(i - 0.5, color="red", linewidth=2)
    for j in range(0, W + 1, patch_w):
        ax1.axvline(j - 0.5, color="red", linewidth=2)
    # Number each patch
    idx = 0
    for i in range(0, H, patch_h):
        for j in range(0, W, patch_w):
            ax1.text(j + 0.5, i + 0.5, str(idx), color="white", fontsize=14,
                     ha="center", va="center", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="red", alpha=0.7))
            idx += 1
    ax1.set_title(f"Image {H}×{W} → {idx} Patches ({patch_h}×{patch_w})")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Right: flattened patches → embedding vectors
    n_patches = (H // patch_h) * (W // patch_w)
    d_embed = 8
    embeddings = np.random.default_rng(1).standard_normal((n_patches, d_embed))
    im = ax2.imshow(embeddings, cmap="RdBu_r", aspect="auto")
    ax2.set_xlabel(f"Embedding dim (d={d_embed})")
    ax2.set_ylabel("Patch index")
    ax2.set_yticks(range(n_patches))
    ax2.set_title(f"Patch Embeddings: {n_patches} × d")
    fig.colorbar(im, ax=ax2, shrink=0.8)

    fig.suptitle("Conv3d Patch Embed: Image → Non-overlapping Patches → Vectors", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(_op_dir("05_conv3d_patch_embed"), "impl.py"), "error_dist")


# =========================================================================
# 06 — RoPE: 2D rotation visualization
# =========================================================================
def plot_06_rope():
    """Show how RoPE rotates vector pairs at different frequencies."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: 2D rotation of a vector at different positions
    head_dim = 64
    base = 10000.0
    max_pos = 50
    inv_freq = base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    positions = np.arange(max_pos, dtype=np.float64)
    angles = np.outer(positions, inv_freq)

    # Heatmap of cos(θ)
    im = ax1.imshow(np.cos(angles[:, :16]), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax1.set_xlabel("Frequency pair index (low → high freq)")
    ax1.set_ylabel("Position")
    ax1.set_title("cos(mθ): Rotation Angles")
    fig.colorbar(im, ax=ax1, shrink=0.8)

    # Right: show actual 2D rotation of a vector
    v = np.array([1.0, 0.0])
    positions_show = [0, 5, 10, 20, 40]
    freq_idx = 0  # highest frequency pair
    colors = plt.cm.viridis(np.linspace(0, 1, len(positions_show)))

    for pos, c in zip(positions_show, colors):
        theta = pos * inv_freq[freq_idx]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        v_rot = np.array([v[0] * cos_t - v[1] * sin_t,
                          v[1] * cos_t + v[0] * sin_t])
        ax2.annotate("", xy=v_rot, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=c, lw=2))
        ax2.plot(*v_rot, "o", color=c, markersize=6)
        ax2.text(v_rot[0] * 1.12, v_rot[1] * 1.12, f"pos={pos}",
                 fontsize=8, color=c, ha="center")

    circle = plt.Circle((0, 0), 1, fill=False, color="gray", linestyle="--", alpha=0.4)
    ax2.add_patch(circle)
    ax2.set_xlim(-1.4, 1.4)
    ax2.set_ylim(-1.4, 1.4)
    ax2.set_aspect("equal")
    ax2.set_title("Dim Pair 0: Vector Rotation by Position")
    ax2.set_xlabel("Dim 0")
    ax2.set_ylabel("Dim 1")

    fig.suptitle("RoPE: Position-Dependent Rotation in 2D Sub-spaces", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(_op_dir("06_rotary_pos_embed"), "impl.py"), "rotation_pattern")


# =========================================================================
# 07 — Attention: Small worked example
# =========================================================================
def plot_07_attention():
    """3-token attention example: scores → softmax → weighted sum."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # 3 tokens: "猫" "坐" "垫"
    tokens = ["Cat", "Sat", "Mat"]
    # Manually designed scores to show meaningful attention
    raw_scores = np.array([[5.0, 1.0, 0.5],
                           [0.8, 4.0, 2.0],
                           [0.3, 1.5, 3.5]])

    # Scaled scores
    d_k = 64
    scores = raw_scores / np.sqrt(d_k)
    # Softmax
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = e / e.sum(axis=-1, keepdims=True)

    # Left: raw attention scores
    im1 = ax1.imshow(scores, cmap="YlOrRd", aspect="equal")
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f"{scores[i,j]:.2f}", ha="center", va="center", fontsize=11)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(tokens)
    ax1.set_yticklabels(tokens)
    ax1.set_xlabel("Key")
    ax1.set_ylabel("Query")
    ax1.set_title(r"$\mathbf{Q}\mathbf{K}^T / \sqrt{d_k}$")

    # Right: attention weights after softmax
    im2 = ax2.imshow(weights, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{weights[i,j]:.2f}", ha="center", va="center", fontsize=11,
                     color="white" if weights[i, j] > 0.5 else "black")
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(tokens)
    ax2.set_yticklabels(tokens)
    ax2.set_xlabel("Key")
    ax2.set_ylabel("Query")
    ax2.set_title("Attention Weights (softmax)")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Scaled Dot-Product Attention: Scores → Softmax → Weights", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(_op_dir("07_attention"), "impl.py"), "attn_weights")


# =========================================================================
# 08, 09, 10 — Activation functions: curves + derivatives
# =========================================================================
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _quick_gelu(x):
    return x * _sigmoid(1.702 * x)


def _gelu_exact(x):
    """GELU exact form using numpy (math.erf is available in stdlib)."""
    import math
    return x * 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2)))


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def _numerical_derivative(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)


def plot_08_quickgelu():
    setup_style()
    x = np.linspace(-5, 5, 500)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(x, _quick_gelu(x), label="QuickGELU: x·σ(1.702x)", linewidth=2.5, color="#e74c3c")
    ax1.plot(x, np.maximum(x, 0), "--", label="ReLU", color="#95a5a6")
    ax1.plot(x, _silu(x), ":", label="SiLU: x·σ(x)", color="#3498db")
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.axvline(0, color="k", linewidth=0.5)
    ax1.set_title("Activation Functions")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.legend()

    ax2.plot(x, _numerical_derivative(_quick_gelu, x), label="QuickGELU'", linewidth=2.5, color="#e74c3c")
    ax2.plot(x, np.where(x > 0, 1, 0).astype(float), "--", label="ReLU'", color="#95a5a6")
    ax2.plot(x, _numerical_derivative(_silu, x), ":", label="SiLU'", color="#3498db")
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.axvline(0, color="k", linewidth=0.5)
    ax2.set_title("Derivatives")
    ax2.set_xlabel("x")
    ax2.set_ylabel("f'(x)")
    ax2.legend()

    fig.suptitle("QuickGELU: Smooth Approximation to GELU via Sigmoid", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(_op_dir("08_quickgelu"), "impl.py"), "activation_curves")


def plot_09_gelu():
    setup_style()
    x = np.linspace(-5, 5, 500)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(x, _gelu_exact(x), label="GELU (exact, erf)", linewidth=2.5, color="#2ecc71")
    ax1.plot(x, _gelu_tanh(x), "--", label="GELU (tanh approx)", linewidth=2, color="#e67e22")
    ax1.plot(x, np.maximum(x, 0), ":", label="ReLU", color="#95a5a6")
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.axvline(0, color="k", linewidth=0.5)
    ax1.set_title("GELU: x · Φ(x)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.legend()
    # Highlight the non-monotonic dip
    ax1.annotate("Non-monotonic\ndip ≈ -0.17",
                 xy=(-0.75, _gelu_exact(-0.75)), xytext=(-3.5, 0.6),
                 arrowprops=dict(arrowstyle="->", color="#e74c3c"),
                 fontsize=9, color="#e74c3c")

    diff = _gelu_exact(x) - _gelu_tanh(x)
    ax2.plot(x, diff, color="#e74c3c", linewidth=2)
    ax2.fill_between(x, diff, alpha=0.15, color="#e74c3c")
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_title(f"Exact − Tanh Approximation (max |err| = {np.max(np.abs(diff)):.1e})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Error")

    fig.suptitle("GELU: Gaussian Error Linear Unit", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(_op_dir("09_gelu"), "impl.py"), "activation_curves")


def plot_10_silu():
    setup_style()
    x = np.linspace(-5, 5, 500)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    silu_y = _silu(x)
    ax1.plot(x, silu_y, label="SiLU: x·σ(x)", linewidth=2.5, color="#3498db")
    ax1.plot(x, np.maximum(x, 0), "--", label="ReLU", color="#95a5a6")
    ax1.plot(x, _gelu_exact(x), ":", label="GELU", color="#2ecc71")
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.axvline(0, color="k", linewidth=0.5)
    ax1.set_title("SiLU (Swish): Self-Gated Activation")
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.legend()

    # Show the self-gating: x * gate
    ax2.plot(x, x, "--", label="x (signal)", color="#95a5a6", alpha=0.7)
    ax2.plot(x, _sigmoid(x), "--", label="σ(x) (gate)", color="#e67e22")
    ax2.plot(x, silu_y, label="x · σ(x) (output)", linewidth=2.5, color="#3498db")
    ax2.fill_between(x, 0, silu_y, alpha=0.1, color="#3498db")
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.axvline(0, color="k", linewidth=0.5)
    ax2.set_title("Self-Gating: Signal × Gate")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Value")
    ax2.legend()

    fig.suptitle("SiLU (Swish): x · σ(x) — The Gate Suppresses Negative, Passes Positive",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(_op_dir("10_silu"), "impl.py"), "activation_curves")


# =========================================================================
# 11 — Vision MLP: Bottleneck architecture
# =========================================================================
def plot_11_vision_mlp():
    """Show expand → activate → compress bottleneck."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    # Draw three layers as rectangles with widths proportional to dimension
    dims = [1280, 5120, 1280]
    labels = ["Input\nd=1280", "Hidden\nd=5120\n(QuickGELU)", "Output\nd=1280"]
    colors = ["#3498db", "#e74c3c", "#3498db"]
    max_h = 3.5
    heights = [d / max(dims) * max_h for d in dims]
    x_positions = [0, 3, 6]

    for xp, h, label, c in zip(x_positions, heights, labels, colors):
        rect = plt.Rectangle((xp, (max_h - h) / 2), 1.5, h,
                              facecolor=c, alpha=0.3, edgecolor=c, linewidth=2)
        ax.add_patch(rect)
        ax.text(xp + 0.75, max_h / 2, label, ha="center", va="center", fontsize=10)

    # Arrows between layers
    for i in range(2):
        ax.annotate("", xy=(x_positions[i+1], max_h/2), xytext=(x_positions[i]+1.5, max_h/2),
                     arrowprops=dict(arrowstyle="->", color="black", lw=2))
        op = ["FC1\n(×4)", "FC2\n(÷4)"][i]
        mid_x = (x_positions[i] + 1.5 + x_positions[i+1]) / 2
        ax.text(mid_x, max_h/2 + 0.3, op, ha="center", va="bottom", fontsize=9)

    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, max_h + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Vision MLP: Expand (×4) → QuickGELU → Compress (÷4)", fontsize=13, pad=15)
    save_fig(fig, os.path.join(_op_dir("11_vision_mlp"), "impl.py"), "activation_dist")


# =========================================================================
# 12 — Gated MLP: Gating mechanism
# =========================================================================
def plot_12_gated_mlp():
    """Show gate × up parallel paths in SwiGLU."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: architecture diagram
    ax1.axis("off")
    # Input
    ax1.text(0.5, 0.05, "x (d=1536)", ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#3498db", alpha=0.3))
    # Two paths
    ax1.annotate("", xy=(0.25, 0.25), xytext=(0.5, 0.12),
                 arrowprops=dict(arrowstyle="->", color="black"))
    ax1.annotate("", xy=(0.75, 0.25), xytext=(0.5, 0.12),
                 arrowprops=dict(arrowstyle="->", color="black"))
    ax1.text(0.25, 0.32, "Gate Path\nW_gate → SiLU", ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#e74c3c", alpha=0.3))
    ax1.text(0.75, 0.32, "Up Path\nW_up (linear)", ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#2ecc71", alpha=0.3))
    # Element-wise multiply
    ax1.annotate("", xy=(0.5, 0.5), xytext=(0.25, 0.4),
                 arrowprops=dict(arrowstyle="->", color="#e74c3c"))
    ax1.annotate("", xy=(0.5, 0.5), xytext=(0.75, 0.4),
                 arrowprops=dict(arrowstyle="->", color="#2ecc71"))
    ax1.text(0.5, 0.55, "⊙ (element-wise)", ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f39c12", alpha=0.3))
    # Down projection
    ax1.annotate("", xy=(0.5, 0.72), xytext=(0.5, 0.62),
                 arrowprops=dict(arrowstyle="->", color="black"))
    ax1.text(0.5, 0.78, "W_down → output (d=1536)", ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#3498db", alpha=0.3))
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.02, 0.9)
    ax1.set_title("SwiGLU Architecture", fontsize=12)

    # Right: show gating effect on a synthetic signal
    x = np.linspace(-3, 3, 200)
    gate = _silu(x)
    up = np.sin(2 * x)  # some arbitrary signal
    gated = gate * up

    ax2.plot(x, up, "--", label="Up signal", color="#2ecc71", alpha=0.7)
    ax2.plot(x, gate, "--", label="SiLU gate", color="#e74c3c", alpha=0.7)
    ax2.plot(x, gated, label="Gate ⊙ Up", linewidth=2.5, color="#f39c12")
    ax2.fill_between(x, 0, gated, alpha=0.1, color="#f39c12")
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.axvline(0, color="k", linewidth=0.5)
    ax2.set_title("Gating: Gate Selectively Amplifies/Suppresses Signal")
    ax2.set_xlabel("Hidden dimension value")
    ax2.legend()

    fig.suptitle("Gated MLP (SwiGLU): Two Parallel Paths with Element-wise Gating", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(_op_dir("12_gated_mlp"), "impl.py"), "gate_dist")


# =========================================================================
# 14 — Token Embedding: Lookup + cosine similarity
# =========================================================================
def plot_14_embedding():
    """Show embedding lookup and cosine similarity between related words."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5),
                                    gridspec_kw={"width_ratios": [1, 1]})

    # Left: lookup table visualization
    V, d = 8, 6
    rng = np.random.default_rng(42)
    E = rng.standard_normal((V, d)).astype(np.float32)
    words = ["the", "cat", "dog", "sat", "on", "mat", "pet", "fur"]

    im = ax1.imshow(E, cmap="RdBu_r", aspect="auto")
    ax1.set_yticks(range(V))
    ax1.set_yticklabels([f"{i}: {w}" for i, w in enumerate(words)], fontsize=9)
    ax1.set_xlabel(f"Embedding dim (d={d})")
    ax1.set_title("Embedding Table E[token_id] → vector")
    # Highlight one row
    ax1.add_patch(plt.Rectangle((-0.5, 0.5), d, 1, fill=False, edgecolor="red", linewidth=3))
    ax1.text(d + 0.3, 1, '← E[1]="cat"', fontsize=9, color="red", va="center")

    # Right: cosine similarity heatmap
    # Make cat/dog/pet/fur similar, the/on similar
    E_designed = rng.standard_normal((V, d))
    # Push related words closer
    animal_base = rng.standard_normal(d)
    for idx in [1, 2, 6, 7]:  # cat, dog, pet, fur
        E_designed[idx] = animal_base + rng.standard_normal(d) * 0.3
    func_base = rng.standard_normal(d)
    for idx in [0, 4]:  # the, on
        E_designed[idx] = func_base + rng.standard_normal(d) * 0.3

    norms = np.linalg.norm(E_designed, axis=1, keepdims=True)
    E_normed = E_designed / norms
    cos_sim = E_normed @ E_normed.T

    im2 = ax2.imshow(cos_sim, cmap="RdYlGn", vmin=-1, vmax=1, aspect="equal")
    ax2.set_xticks(range(V))
    ax2.set_yticks(range(V))
    ax2.set_xticklabels(words, fontsize=8, rotation=45)
    ax2.set_yticklabels(words, fontsize=8)
    ax2.set_title("Cosine Similarity: Related Words Cluster")
    fig.colorbar(im2, ax=ax2, shrink=0.8)
    for i in range(V):
        for j in range(V):
            ax2.text(j, i, f"{cos_sim[i,j]:.1f}", ha="center", va="center", fontsize=7)

    fig.suptitle("Token Embedding: ID → Vector Lookup, Semantic Similarity Emerges", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(_op_dir("14_token_embedding"), "impl.py"), "embed_norm")


# =========================================================================
# 15 — Patch Merger: Spatial 2×2 merge
# =========================================================================
def plot_15_patch_merger():
    """Show 2×2 spatial merge: 4 adjacent patches → 1 merged vector."""
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4),
                              gridspec_kw={"width_ratios": [1, 0.3, 1.2]})

    # Left: 6×6 patch grid (before merge)
    H, W = 6, 6
    grid = np.arange(H * W).reshape(H, W)
    ax1 = axes[0]
    ax1.imshow(np.ones((H, W, 3)) * 0.9, aspect="equal")
    for i in range(H):
        for j in range(W):
            # Color 2×2 groups
            group = (i // 2) * (W // 2) + (j // 2)
            colors = plt.cm.Set3(group / 9)
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=colors, alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(j, i, str(grid[i, j]), ha="center", va="center", fontsize=8)
    # Draw 2×2 group borders
    for i in range(0, H + 1, 2):
        ax1.axhline(i - 0.5, color="black", linewidth=2)
    for j in range(0, W + 1, 2):
        ax1.axvline(j - 0.5, color="black", linewidth=2)
    ax1.set_title(f"{H*W} patches (d=1280)", fontsize=11)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Middle: arrow
    axes[1].text(0.5, 0.5, "2×2\nmerge\n→", fontsize=14, ha="center", va="center",
                 transform=axes[1].transAxes)
    axes[1].axis("off")

    # Right: merged result
    mH, mW = H // 2, W // 2
    ax2 = axes[2]
    merged = np.arange(mH * mW).reshape(mH, mW)
    ax2.imshow(np.ones((mH, mW, 3)) * 0.9, aspect="equal")
    for i in range(mH):
        for j in range(mW):
            group = i * mW + j
            colors = plt.cm.Set3(group / 9)
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=colors, alpha=0.7)
            ax2.add_patch(rect)
            orig = [int(grid[i*2, j*2]), int(grid[i*2, j*2+1]),
                    int(grid[i*2+1, j*2]), int(grid[i*2+1, j*2+1])]
            ax2.text(j, i, f"{orig[0]},{orig[1]}\n{orig[2]},{orig[3]}", ha="center", va="center", fontsize=8)

    for i in range(mH + 1):
        ax2.axhline(i - 0.5, color="black", linewidth=2)
    for j in range(mW + 1):
        ax2.axvline(j - 0.5, color="black", linewidth=2)
    ax2.set_title(f"{mH*mW} patches (d=5120)", fontsize=11)
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.suptitle("Patch Merger: 2×2 Adjacent Patches Concatenated → MLP Projects to Output Dim",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(_op_dir("15_patch_merger"), "impl.py"), "error_dist")


# =========================================================================
# 16 — Vision Block: Pre-Norm Transformer block data flow
# =========================================================================
def plot_16_vision_block():
    """Pre-Norm residual block: x → LN → Attn → +x → LN → MLP → +x."""
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")

    # Boxes and arrows for data flow
    steps = [
        (0.02, "x\n(n, 1280)", "#3498db"),
        (0.14, "LN1", "#f39c12"),
        (0.26, "Attention\n(MHA)", "#e74c3c"),
        (0.40, "+ x\n(residual)", "#2ecc71"),
        (0.52, "LN2", "#f39c12"),
        (0.64, "MLP\n(expand→act→compress)", "#e74c3c"),
        (0.78, "+ x'\n(residual)", "#2ecc71"),
        (0.92, "output\n(n, 1280)", "#3498db"),
    ]

    y = 0.5
    box_w = 0.10
    for xp, label, color in steps:
        ax.text(xp + box_w/2, y, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.25,
                          edgecolor=color, linewidth=1.5),
                transform=ax.transAxes)

    # Arrows
    for i in range(len(steps) - 1):
        x1 = steps[i][0] + box_w + 0.01
        x2 = steps[i+1][0] - 0.01
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                     xycoords="axes fraction", textcoords="axes fraction")

    # Residual skip arrows (curved)
    # Skip 1: x → + x
    ax.annotate("", xy=(0.40 + box_w/2, 0.22), xytext=(0.02 + box_w/2, 0.22),
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=2,
                                connectionstyle="arc3,rad=-0.3"),
                xycoords="axes fraction", textcoords="axes fraction")
    # Skip 2: +x → +x'
    ax.annotate("", xy=(0.78 + box_w/2, 0.22), xytext=(0.40 + box_w/2, 0.22),
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=2,
                                connectionstyle="arc3,rad=-0.3"),
                xycoords="axes fraction", textcoords="axes fraction")

    ax.set_title("Vision Transformer Block: Pre-Norm with Two Residual Connections", fontsize=13, pad=15)
    save_fig(fig, os.path.join(_op_dir("16_vision_block"), "impl.py"), "component_error")


# =========================================================================
# 17 — Decoder Layer: Similar but with RMSNorm + GQA + SwiGLU
# =========================================================================
def plot_17_decoder_layer():
    """Decoder layer data flow with RMSNorm, GQA, SwiGLU labels."""
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")

    steps = [
        (0.02, "x\n(B,T,1536)", "#3498db"),
        (0.14, "RMSNorm1", "#f39c12"),
        (0.26, "GQA\n(12Q/2KV)", "#e74c3c"),
        (0.40, "+ x\n(residual)", "#2ecc71"),
        (0.52, "RMSNorm2", "#f39c12"),
        (0.64, "SwiGLU\n(gate⊙up→down)", "#9b59b6"),
        (0.78, "+ x'\n(residual)", "#2ecc71"),
        (0.92, "output\n(B,T,1536)", "#3498db"),
    ]

    y = 0.5
    box_w = 0.10
    for xp, label, color in steps:
        ax.text(xp + box_w/2, y, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.25,
                          edgecolor=color, linewidth=1.5),
                transform=ax.transAxes)

    for i in range(len(steps) - 1):
        x1 = steps[i][0] + box_w + 0.01
        x2 = steps[i+1][0] - 0.01
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                     xycoords="axes fraction", textcoords="axes fraction")

    ax.annotate("", xy=(0.40 + box_w/2, 0.22), xytext=(0.02 + box_w/2, 0.22),
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=2,
                                connectionstyle="arc3,rad=-0.3"),
                xycoords="axes fraction", textcoords="axes fraction")
    ax.annotate("", xy=(0.78 + box_w/2, 0.22), xytext=(0.40 + box_w/2, 0.22),
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=2,
                                connectionstyle="arc3,rad=-0.3"),
                xycoords="axes fraction", textcoords="axes fraction")

    ax.set_title("Text Decoder Layer: RMSNorm + GQA + SwiGLU with Pre-Norm Residuals", fontsize=13, pad=15)
    save_fig(fig, os.path.join(_op_dir("17_decoder_layer"), "impl.py"), "component_error")


# =========================================================================
# 18 — LM Head: logits → softmax → token
# =========================================================================
def plot_18_lm_head():
    """Show hidden → RMSNorm → Linear → logits → argmax → token."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: logits distribution for last token
    rng = np.random.default_rng(42)
    logits = rng.standard_normal(200)  # simplified vocab
    # Make a few tokens stand out
    logits[42] = 8.5
    logits[7] = 6.2
    logits[100] = 5.8

    ax1.bar(range(len(logits)), logits, color="steelblue", alpha=0.6, width=1.0)
    ax1.bar(42, logits[42], color="#e74c3c", width=1.0)
    ax1.text(42, logits[42] + 0.3, "argmax\n(token 42)", ha="center", fontsize=9, color="#e74c3c")
    ax1.set_xlabel("Token ID (vocabulary)")
    ax1.set_ylabel("Logit value")
    ax1.set_title("Raw Logits (151,936 dims in real model)")

    # Right: top-10 after softmax
    top_k = 10
    top_idx = np.argsort(logits)[-top_k:][::-1]
    top_logits = logits[top_idx]
    e = np.exp(top_logits - top_logits.max())
    probs = e / e.sum()

    colors = ["#e74c3c"] + ["#3498db"] * (top_k - 1)
    ax2.barh(range(top_k - 1, -1, -1), probs, color=colors, edgecolor="white")
    ax2.set_yticks(range(top_k - 1, -1, -1))
    ax2.set_yticklabels([f"Token {i}" for i in top_idx], fontsize=9)
    for i, (p, idx) in enumerate(zip(probs, top_idx)):
        ax2.text(p + 0.01, top_k - 1 - i, f"{p:.1%}", va="center", fontsize=9)
    ax2.set_xlabel("Probability (softmax)")
    ax2.set_title("Top-10 Predictions")

    fig.suptitle("LM Head: Hidden State → Logits → Softmax → Next Token", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(_op_dir("18_lm_head"), "impl.py"), "top_logits")


# =========================================================================
# Main
# =========================================================================
ALL_PLOTS = [
    ("01_linear", plot_01_linear),
    ("02_softmax", plot_02_softmax),
    ("03_layer_norm", plot_03_layer_norm),
    ("04_rms_norm", plot_04_rms_norm),
    ("05_conv3d_patch_embed", plot_05_conv3d),
    ("06_rotary_pos_embed", plot_06_rope),
    ("07_attention", plot_07_attention),
    ("08_quickgelu", plot_08_quickgelu),
    ("09_gelu", plot_09_gelu),
    ("10_silu", plot_10_silu),
    ("11_vision_mlp", plot_11_vision_mlp),
    ("12_gated_mlp", plot_12_gated_mlp),
    ("14_token_embedding", plot_14_embedding),
    ("15_patch_merger", plot_15_patch_merger),
    ("16_vision_block", plot_16_vision_block),
    ("17_decoder_layer", plot_17_decoder_layer),
    ("18_lm_head", plot_18_lm_head),
]


if __name__ == "__main__":
    print(f"Generating {len(ALL_PLOTS)} concept diagrams...\n")
    for name, func in ALL_PLOTS:
        try:
            func()
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
    print("\nDone.")
