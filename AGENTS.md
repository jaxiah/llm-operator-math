# LLM Operator Math — Agent Workflow Guide

This document defines the structured workflow for decomposing any LLM/VLM into fine-grained operators, producing long-form math explainer READMEs and pure NumPy implementations validated against PyTorch ground truth.

## Goal

For every operator inside a neural network model, produce two artifacts:

1. **README.md** — a long-form, standalone mathematical blog post (400–1200 lines)
2. **impl.py** — a pure NumPy implementation validated against PyTorch ground truth

The workflow is structured in phases. Start from whichever phase matches the current project state.

## Single-Issue Discipline (CRITICAL)

**Each invocation processes exactly ONE issue (= one operator or one operator group).**

This is a hard constraint for context management:
- Phase 0A produces one PRD (bounded scope)
- Phase 0B processes one issue at a time (bounded context)
- Phases 2+3 process one issue at a time: write its impl.py, README, and concept plot

**Never batch multiple operators into a single invocation.** If asked to "do all operators", do the first one, commit, then tell the user to invoke again for the next.

Before starting work on Phases 2-3, identify which ISSUE you are working on. Read it from `BACKLOG/` and confirm with the user if ambiguous.

---

# PHASE 0: PRD & ISSUE PLANNING

> Skip this phase if `BACKLOG/PRD-*.md` and `BACKLOG/ISSUE-*.md` already exist.

## 0A — Write the PRD

If no PRD exists, interview the user to create one.

**Process:**

1. Ask the user for: target model (HuggingFace ID), scope (full pipeline or specific operators), audience level, language preference
2. Explore the model's architecture (read config.json from HuggingFace, examine model source code)
3. Interview the user about: which operators to cover, depth of math explanation, validation strategy
4. Write the PRD to `BACKLOG/PRD-NNN-short-slug.md`

**PRD Template:**

```markdown
## Problem Statement

[What the user wants to learn, from their perspective]

## Solution

[Approach: run model once → dump activations → implement each operator in NumPy → write math explainers]

## User Stories

1. As a learner, I want to see the complete list of operators in inference order...
2. As a learner, I want each operator explained from first principles with MathJax...
   [Extensive list covering: operator list, math explanations, numerical examples, runnable code, validation, modularity]

## Implementation Decisions

- Directory structure: `operators/NN_name/{README.md, impl.py}`
- Activation dumping: PyTorch forward hooks → .npy files
- Validation: `np.allclose` with appropriate tolerances
- Weight loading: safetensors via huggingface_hub
  [Model-specific dimensions and architecture details]

## Testing Decisions

- Each operator validated independently against dumped activations
- E2E runner validates all operators in sequence
- Repeated layers: only validate layer 0

## Out of Scope

[Training, backpropagation, quantization, etc.]
```

**Filename convention**: `PRD-NNN-short-slug.md` where NNN is the next global sequence number.
To find NNN: glob `BACKLOG/*.md`, extract the highest 3-digit prefix across all files, increment by 1.

## 0B — Break PRD into Issues

Break the PRD into **vertical slice** issues — each issue is a thin end-to-end slice, NOT a horizontal layer.

**Process:**

1. Read the PRD
2. Draft slices — typically one issue per operator or operator group:
   - Issue for scaffolding (e2e utilities, dump script, overview)
   - One issue per operator (or group of related operators like "activation functions")
   - Issue for E2E validation runner
3. Present breakdown to user for approval:
   - Title, Type (HITL/AFK), Blocked-by, User stories covered
4. Create `BACKLOG/ISSUE-NNN-slug.md` files in dependency order

**Issue Template:**

```markdown
## Parent PRD

[PRD-NNN-slug.md](PRD-NNN-slug.md)

## What to build

[Concise description of this vertical slice]

## Acceptance criteria

- [ ] Criterion 1
- [ ] Criterion 2

## Blocked by

- [ISSUE-NNN-slug.md](ISSUE-NNN-slug.md) or "None"

## User stories addressed

- User story N
```

---

# PHASE 1: PROJECT SCAFFOLDING

> Skip if `e2e/` directory already exists with dump/validate scripts.

## 1.1 Directory Structure

```
project-root/
  BACKLOG/                     # PRD and ISSUE files
  e2e/
    dump_activations.py        # PyTorch hook-based activation dumper
    validate.py                # Shared validation utilities
    run_numpy_e2e.py           # E2E runner
  operators/
    00_overview/
      README.md                # Complete operator list & architecture
    NN_operator_name/
      README.md                # Long-form math explainer
      impl.py                  # Pure NumPy implementation + validation
  activations/                 # Dumped .npy files (gitignored)
  pyproject.toml
  .gitignore                   # Must ignore: activations/, .venv/, __pycache__/, .claude/
```

## 1.2 Activation Dumping (`e2e/dump_activations.py`)

Register `forward_hook` on every module of interest. For repeated layers, only dump layer 0.

```python
def make_hook(name, output_dir):
    def hook_fn(module, input, output):
        inp = input[0] if isinstance(input, tuple) else input
        np.save(f"{output_dir}/{name}_input.npy", inp.detach().float().cpu().numpy())
        out = output[0] if isinstance(output, tuple) else output
        np.save(f"{output_dir}/{name}_output.npy", out.detach().float().cpu().numpy())
    return hook_fn
```

Naming: `module__path__with__double_underscores_{input|output}.npy`

## 1.3 Validation Utilities (`e2e/validate.py`)

```python
def validate(name, actual, expected, atol=1e-5, rtol=1e-5) -> bool:
    # Shape check → np.allclose → print detailed report (max/mean error)
    # On failure: show worst-case index and values

def load_activation(dump_dir, name) -> np.ndarray:
    # Load .npy, ensure float32
```

**Tolerance guidelines:**

| Operation type                      | atol         | Notes                |
| ----------------------------------- | ------------ | -------------------- |
| Element-wise (activations, norms)   | 1e-5         | High precision       |
| Matrix multiply (linear, attention) | 1e-4         | Float32 accumulation |
| Multi-step compositions (blocks)    | 1e-3 to 0.01 | Error compounds      |

## 1.4 Weight Loading Pattern

```python
from safetensors import safe_open
from huggingface_hub import hf_hub_download, HfApi

def load_safetensors_weights(model_id, keys):
    # Iterate safetensor shards, load requested keys
    # ALWAYS .float().numpy() to convert bf16→float32
```

---

# PHASE 2: OPERATOR IMPLEMENTATION (`impl.py`)

Each `impl.py` follows this strict structure:

```python
"""NN — 算子名称 (English Name): 核心公式

用纯 NumPy 实现 <算子名称>，并用 <模型名> 的真实权重和激活值验证。
"""
import numpy as np
from e2e.validate import validate, load_activation

DUMP_DIR = "activations"
MODEL_ID = "<huggingface-model-id>"

# ---------------------------------------------------------------------------
# 核心算子 (Core Operator)
# ---------------------------------------------------------------------------
def operator_name(x: np.ndarray, ...) -> np.ndarray:
    """One-line math description."""
    # Pure numpy, no torch/scipy
    # Each line maps to a step in the README's derivation
    ...

# ---------------------------------------------------------------------------
# 权重加载工具 (if operator has learnable parameters)
# ---------------------------------------------------------------------------
def load_weights(...): ...

# ---------------------------------------------------------------------------
# 验证 (Validation)
# ---------------------------------------------------------------------------
def validate_case_1() -> bool:
    """Validate against specific model layer."""
    print("\n=== Title (input_shape) -> (output_shape) ===")
    x = load_activation(DUMP_DIR, "<input_name>")
    expected = load_activation(DUMP_DIR, "<output_name>")
    actual = operator_name(x, ...)
    return validate("<test_name>", actual, expected, atol=..., rtol=...)

if __name__ == "__main__":
    results = [validate_case_1(), ...]
    print(f"\n{'='*60}")
    print(f"验证: {sum(results)}/{len(results)} 通过")
    if not all(results):
        raise SystemExit(1)
```

**Rules:**

- Pure NumPy only in core function — no PyTorch, no scipy
- No broadcasting tricks that obscure the math — prefer explicit reshape/transpose
- Variable names match README notation ($W_Q$ → `W_Q`)
- Two validation cases minimum when operator appears in both vision and text paths
- Document weight tying (e.g., `lm_head.weight == embed_tokens.weight`)

---

# PHASE 3: MATH README AUTHORING + CONCEPT PLOT (THE CORE DELIVERABLE)

The README is the **primary artifact** — a standalone, publishable blog post that fully explains the operator.

## Concept Plot Workflow (Integrated into README Writing)

While writing the README, **plan visualizations as part of the writing process** — not as an afterthought. When a diagram would explain the concept better than text:

1. **Decide inline**: "this concept needs a figure" (e.g., a function curve, a tensor shape diagram, an attention heatmap)
2. **Write the image reference immediately**: `![Descriptive Alt Text](./filename.png)` — this is both placeholder and final syntax
3. **Continue writing the README** to completion

After the README is done:

4. **Write the plotting function** in `e2e/generate_concept_plots.py` — append a `plot_NN_operator()` function that generates the PNG with the exact filename referenced in the README
5. **Run the function** to generate the PNG
6. **Read the PNG** to verify rendering (no text overflow, legible labels, correct content)
7. If there are issues, fix and regenerate

**Plot design principles:**
- **Synthetic data only** — concept plots must not depend on model weights or activation dumps
- Focus on **conceptual understanding**, not validation results
- Good plots: function curves + derivatives, small worked-example heatmaps, tensor shape diagrams, architecture data-flow diagrams, gating mechanism visualizations
- Use `e2e/plot_utils.py` for shared matplotlib setup
- One plot per operator is usually enough — don't force a plot where text suffices

## Target Audience

An engineering graduate who:

- Knows linear algebra (matrix multiplication, eigenvalues, vector spaces)
- Knows basic calculus (derivatives, chain rule)
- Knows basic probability (mean, variance)
- Does NOT know deep learning, attention, or neural networks

## Language Rules

- **Primary**: Chinese (Simplified)
- **Keep English for**: technical terms, variable names, function names, paper titles
- **MathJax**: `$...$` inline, `$$...$$` display
- **Code blocks**: Python/NumPy snippets bridging math → implementation

## MANDATORY 10-Section Structure

Every README MUST include ALL of the following:

### Section 1: Opening Hook & Motivation (为什么要关心这个算子?)

- Real-world analogy making the concept tangible
- Why this operator exists — what problem does it solve?
- Where it sits in the inference pipeline
- Give a reason to care before any math appears

### Section 2: Prerequisites (前置知识)

- List and briefly review needed math concepts
- Quick refreshers — don't assume perfect recall
- Forward links: "we'll use this in Section 4..."

### Section 3: Core Mathematical Definition (核心数学定义)

- Formal definition with full notation
- Every symbol defined explicitly on first use
- Display math for main equations

### Section 4: Step-by-Step Derivation (逐步推导)

- One operation at a time, show intermediate results
- Never skip steps
- Explain **why** each step is taken: "注意这里...", "之所以要这样做, 是因为..."

### Section 5: Numerical Example (数值算例)

- Complete small example with concrete numbers
- Dimensions small enough to trace by hand (e.g., 2×3)
- Show every intermediate computation

### Section 6: Geometric / Visual Intuition (几何直觉)

- What does the operation DO to data geometrically?
- Analogies: rotations, projections, scaling, gating
- Project high-dim to 2D/3D for intuition

### Section 7: Design Motivation & History (设计动机与历史)

- Who invented it, which paper, what it replaced
- Why this variant over alternatives
- Tradeoffs gained/lost

### Section 8: Model-Specific Details (在本模型中的具体应用)

- Exact dimensions, parameter counts, config values
- Which layers/modules use this operator
- Model-specific quirks

### Section 9: NumPy Implementation Walkthrough (NumPy 实现详解)

- Complete core function from impl.py
- Annotate each line: which equation does it implement?
- Explicit shape transformations: "输入 shape 为 (B, T, D), reshape 为..."

### Section 10: Common Pitfalls & Summary (常见陷阱与小结)

- 3-5 common mistakes (numerical stability, shape errors, broadcasting)
- Summary table or bullet list of key takeaways
- Optional further reading

## Writing Style Rules

1. **娓娓道来**: Write as if explaining to a curious colleague over coffee. Conversational transitions, not textbook formality.

2. **Progressive complexity**: Start simplest possible, add layers. Never front-load complexity.

3. **Bridge every abstraction**: Abstract concept → immediately follow with concrete example or analogy.

4. **Explicit dimension tracking**: Every tensor shape change → state before/after. `# (B, T, D) -> (B, T, H, d_k)`

5. **Never say "obviously" or "trivially"**: If it were obvious, the reader wouldn't need the document.

6. **Length targets**:
   - Core operators (attention, RoPE, conv3d): 800–1200 lines
   - Simple operators (activation functions, norms): 400–700 lines
   - Composite operators (full blocks, MLP): 600–900 lines

7. Use `---` between major sections for visual breathing room.

8. Numbered headings: `## 1. ...`, `### 1.1 ...`

---

# PHASE 4: E2E VALIDATION RUNNER

Create `e2e/run_numpy_e2e.py` that:

1. Imports each operator's validation functions
2. Runs them in inference execution order
3. Collects pass/fail results
4. Prints summary table

```
============================================================
E2E Validation Summary: 20/20 PASS
============================================================
[PASS] 01_linear: vision_fc1_linear
[PASS] 01_linear: text_gate_proj_linear
...
```

---

# EXECUTION STRATEGY

## Single-Issue Execution Flow (Per Invocation)

Each invocation for a specific operator follows this sequence:

1. Read the ISSUE from `BACKLOG/`
2. Write `impl.py` (Phase 2) — core operator + validation
3. Write `README.md` (Phase 3) — with `![](./name.png)` inline where needed
4. Append plot function to `e2e/generate_concept_plots.py` and run it
5. Read the generated PNG to verify rendering quality
6. Commit all artifacts for this operator

## Quality Checklist (before commit for each operator)

- [ ] `impl.py` runs individually: `python -m operators.NN_name.impl`
- [ ] README has all 10 sections
- [ ] MathJax renders (no broken `$...$` pairs)
- [ ] Dimensions consistent across README and impl.py
- [ ] Model-specific numbers accurate
- [ ] Variable names in code match README notation
- [ ] Concept plot PNG exists and renders correctly
- [ ] README image reference `![...](./name.png)` matches actual PNG filename

## Phase Detection

When starting, check project state:

- No `BACKLOG/` → start from Phase 0A
- `BACKLOG/PRD-*.md` but no issues → start from Phase 0B
- `BACKLOG/ISSUE-*.md` but no `e2e/` → start from Phase 1
- `e2e/` exists but missing `operators/` → start from Phase 2
- `operators/*/impl.py` exist but READMEs are short/missing → start from Phase 3
- Everything exists → Phase 4 (validate & polish)
