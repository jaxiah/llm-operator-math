"""Validation utilities for comparing numpy outputs against PyTorch ground truth.

Usage:
    from e2e.validate import validate, load_activation

    expected = load_activation("activations", "model.visual.blocks.0.norm1_output")
    actual = my_numpy_layernorm(x, weight, bias)
    validate("vision_block_0_norm1", actual, expected)
"""

import os
import numpy as np


def validate(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    """Compare two arrays and print a detailed report.

    Args:
        name: Human-readable name for this comparison.
        actual: The numpy-computed result.
        expected: The PyTorch ground truth (loaded from .npy).
        atol: Absolute tolerance for np.allclose.
        rtol: Relative tolerance for np.allclose.

    Returns:
        True if the arrays match within tolerance, False otherwise.
    """
    if actual.shape != expected.shape:
        print(f"[FAIL] {name}")
        print(f"       Shape mismatch: actual={actual.shape}, expected={expected.shape}")
        return False

    max_abs_err = np.max(np.abs(actual - expected))
    mean_abs_err = np.mean(np.abs(actual - expected))
    passed = np.allclose(actual, expected, atol=atol, rtol=rtol)

    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    print(f"       shape={actual.shape}  dtype={actual.dtype}")
    print(f"       max_abs_err={max_abs_err:.6e}  mean_abs_err={mean_abs_err:.6e}")
    print(f"       atol={atol}  rtol={rtol}")

    if not passed:
        # Show where the largest differences are
        diff = np.abs(actual - expected)
        worst_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"       worst_at={worst_idx}  actual={actual[worst_idx]:.6e}  expected={expected[worst_idx]:.6e}")

    return passed


def load_activation(dump_dir: str, name: str) -> np.ndarray:
    """Load a dumped activation tensor from disk.

    Args:
        dump_dir: Path to the directory containing .npy files.
        name: The activation name (without .npy extension).

    Returns:
        The loaded numpy array (always float32).
    """
    path = os.path.join(dump_dir, f"{name}.npy")
    arr = np.load(path)
    return arr.astype(np.float32) if arr.dtype != np.float32 else arr


def list_activations(dump_dir: str) -> list[str]:
    """List all available activation names in a dump directory."""
    names = []
    for f in sorted(os.listdir(dump_dir)):
        if f.endswith(".npy"):
            names.append(f[:-4])  # strip .npy
    return names


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m e2e.validate <dump_dir>")
        print("Lists all available activations in the dump directory.")
        sys.exit(1)

    dump_dir = sys.argv[1]
    activations = list_activations(dump_dir)
    print(f"Found {len(activations)} activations in {dump_dir}:")
    for name in activations:
        arr = np.load(os.path.join(dump_dir, f"{name}.npy"))
        print(f"  {name:60s} shape={str(arr.shape):20s} dtype={arr.dtype}")
