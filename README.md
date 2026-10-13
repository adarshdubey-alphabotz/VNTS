# VNTS: VYOMA Neural Training Systems

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](pyproject.toml)

VNTS is an independent open-source research initiative investigating whether neural network training can be restructured to eliminate reliance on specialized GPU accelerators, reduce total memory footprint, and lower training compute.

Current deep learning architectures rely on reverse-mode automatic differentiation (backpropagation). While effective, backpropagation imposes global backward locking: intermediate activations across all layers must be stored in memory until the backward pass completes ($O(L \cdot B \cdot S \cdot d)$ memory). This forces practitioners onto expensive high-bandwidth GPU clusters.

VNTS approaches training from first principles, testing alternative credit assignment mechanisms, decoupled forward plasticity, and closed-form representation solves under controlled, resource-bounded conditions.

---

## Benchmark Results

Evaluated on 4 vCPUs (AMD EPYC) with a strict <2.0 GiB RAM budget across three controlled non-linear benchmarks:
1. **B1: Interleaved Manifold Spiral Classification** (4 classes, non-linear coordinate warping)
2. **B2: Hierarchical Non-Linear Regression** (16 inputs, multi-frequency cross-terms)
3. **B3: High-Dimensional Correlated Pattern Discrimination** (64 features, 5 classes)

All runs used identical MLP topologies (`[128, 128, 64]`), batch size 128, 30 epochs, and seed 42.

| Method | Paradigm | B1 Test Acc | B2 Test $R^2$ | B3 Test Acc | FLOPs vs Backprop | Peak Process RSS | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Backprop (AdamW)** | Conventional Autograd | 50.0% | 0.6454 | 100.0% | 1.00x | 302.6 MB | Baseline |
| **Backprop (SGD+Mom)** | Conventional Autograd | 46.9% | 0.6559 | 100.0% | 1.00x | 303.4 MB | Baseline |
| **Direct Feedback Alignment (DFA)** | Random Feedback Projection | 30.4% | -0.3177 (Diverged) | 26.3% | 0.68x | 305.6 MB | Failed to generalize |
| **SPSA / MeZO** | Isotropic Zeroth-Order | 32.6% | 0.0699 | 99.9% | 1.00x | 307.3 MB | High variance |
| **Structured Forward-Grad (L-SFG)** | Low-Rank Subspace Perturbation | 31.1% | 0.0083 | 23.0% | 1.00x | 307.4 MB | Falsified |
| **Closed-Form Latent (CLA)** | Analytic Ridge Solves | 25.3% | 0.2313 | 19.8% | 0.85x | 312.2 MB | Falsified for deep stacks |
| **Forward Residual Plasticity (FRP)** | **Decoupled Local Forward** | **46.0%** | **0.6391** | **100.0%** | **0.69x (-31% FLOPs)** | **307.3 MB** | **Validated Candidate** |

### Key Takeaways

- **FRP matches Backprop without a backward graph:** FRP achieves $R^2 = 0.6391$ (vs $0.6454$ for AdamW) and 100% classification accuracy on B3 while decoupling layers completely.
- **31% FLOP reduction:** Because FRP does not compute activation gradients through intermediate weight transposes ($\delta_l = W^T \delta$), it requires ~31% fewer multiply-accumulate operations per token than backpropagation.
- **Flat depth memory scaling:** Activations $a_{l-1}$ are deallocated as soon as layer $l$ finishes its forward step. Activation memory scales as $O(B \cdot d)$, completely invariant to network depth $L$.

---

## Core Mechanism: Forward Residual Plasticity (FRP)

In standard backpropagation, layer $l$ is blocked until downstream layers finish:
$$\nabla_{W_l} \mathcal{L} = \delta_l a_{l-1}^T, \quad \text{where } \delta_l = (W_{l+1}^T \delta_{l+1}) \odot \sigma'(z_l)$$

In FRP, each layer $l$ maintains a lightweight local readout projection $H_l: \mathbb{R}^{d_l} \to \mathbb{R}^{d_{\text{out}}}$ and updates its weights immediately during the forward pass using a composite local loss:
$$\mathcal{L}_l = \mathcal{L}_{\text{task}}(H_l(a_l), y) + \alpha_{\text{rep}} \mathcal{L}_{\text{cov}}(a_l)$$

Where $\mathcal{L}_{\text{cov}}$ is an inter-channel decorrelation penalty that prevents representation collapse:
$$\mathcal{L}_{\text{cov}}(a) = \frac{1}{d} \sum_{i \ne j} \text{Cov}(a)_{ij}^2 + \frac{1}{d} \sum_{i} \max(0, 1 - \sqrt{\text{Var}(a)_i + \epsilon})$$

Once $a_l$ is computed, $a_{l-1}$ is detached and discarded. The computation graph never crosses layer boundaries.

```
Standard Backprop:
  Input -> [Layer 1] -> [Layer 2] -> [Layer 3] -> Loss
              ^             ^             ^         |
              |_____________|_____________|_________|  (Global backward graph locked in memory)

Forward Residual Plasticity (FRP):
  Input -> [Layer 1] -> a_1 (detached) -> [Layer 2] -> a_2 (detached) -> [Layer 3] -> Output
              |                               |                              |
            Head 1 -> Local Loss            Head 2 -> Local Loss           Head 3 -> Task Loss
```

---

## Project Structure

```
VNTS/
├── docs/
│   ├── 01_ENVIRONMENT_AUDIT.md              # Infrastructure and memory audit
│   ├── 02_THEORETICAL_RECONSTRUCTION.md     # First-principles cost breakdown
│   ├── 03_LITERATURE_AND_OPEN_QUESTIONS.md  # Alternative paradigms & breakdown points
│   ├── 04_RESEARCH_HYPOTHESES.md            # Hypotheses formulation & falsification criteria
│   ├── 05_EXPERIMENTAL_PROTOCOL.md          # Benchmark harness design & controls
│   └── 06_EMPIRICAL_FINDINGS_AND_ROADMAP.md # Detailed experimental analysis & findings
├── src/
│   └── vnts/
│       ├── __init__.py
│       ├── benchmarks.py                    # Deterministic benchmark datasets
│       ├── models.py                        # Clean modular MLP implementations
│       ├── metrics.py                       # RSS memory profiling & FLOP counter
│       ├── trainers.py                      # Backprop, DFA, FRP, SPSA, L-SFG, CLA
│       └── run_experiments.py               # Master CLI experiment runner
├── results/
│   └── experimental_results.json            # Machine-readable benchmark run logs
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Installation

```bash
git clone https://github.com/utkarshdubey2008/VNTS.git
cd VNTS

# Set up isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Run Benchmarks

Run the complete comparative suite across all baselines and candidates:

```bash
python -m vnts.run_experiments --epochs 30 --batch-size 128
```

Results will be written to `results/experimental_results.json`.

---

## Research Roadmap

1. **Decoupled Attention for Transformers:** Adapt FRP to causal self-attention blocks, testing whether intermediate transformer blocks can train without depth-wise backprop graphs.
2. **Fused C++/SIMD Kernel:** Implement FRP layer updates in a fused CPU kernel (AVX2/AVX-512) to execute activation calculation, local projection, and weight updates in a single memory pass.
3. **Quantized Local Updates:** Test INT8/FP8 robustness under forward-local plasticity compared to standard backpropagation.

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
