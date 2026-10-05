# Benchmarks

## Modules benchmark

`benchmark_modules.py` measures the performance of the forward and backward passes of the warp-nn modules
(activations, layers and operators) of the working tree (**current** implementation), against:

- their **PyTorch** counterparts, and
- a **baseline** warp-nn implementation, exported from a git reference (by default, the `develop` branch).

Each module is benchmarked for the batch sizes (number of environments) most commonly used in Isaac Lab tasks:
`1` (single environment), `50` (play), `4096` (training default), `8192` (large-scale training) and `32768`
(PPO mini-batch: 4096 environments × 16 rollouts / 2 mini-batches).
The other input sizes are representative of Isaac Lab policies (see `_specs.py`).

### Usage

From the repository root (with the `tests` extra installed, which provides PyTorch):

```bash
# all the modules, eager execution, Markdown report on stdout
python -m benchmarks.benchmark_modules

# some modules (case-insensitive, shell-style patterns), CUDA graphs, all the report formats
python -m benchmarks.benchmark_modules --modules Linear "Conv*" LSTM --cuda-graph \
    --format markdown json csv html --output-dir benchmark-results

# list the modules, and all the options
python -m benchmarks.benchmark_modules --list
python -m benchmarks.benchmark_modules --help
```

The baseline git reference must exist locally (`git fetch origin develop:develop` in CI, or use `--baseline-ref
origin/develop`). Use `--no-baseline` to compare against PyTorch only.

### Report

Each table row is a module and each sub-row is a batch size. The columns are the speedups of the current warp-nn
implementation:

| Column | Speedup = reference time / warp-nn time |
|:---|:---|
| Forward vs PyTorch | PyTorch forward pass is the baseline (1x) |
| Backward vs PyTorch | PyTorch backward pass is the baseline (1x) |
| Forward vs develop | baseline warp-nn forward pass is the baseline (1x) |
| Backward vs develop | baseline warp-nn backward pass is the baseline (1x) |

`> 1x` means that the current warp-nn implementation is faster. A `~` marks differences that are not significant
(the confidence interval includes 1x, the difference is below `--min-effect` (2% by default), or the instance sets
disagree). `n/a` marks
unavailable comparisons (e.g. a module that does not exist in the baseline, see the errors section), and `—` the
backward pass of non-differentiable modules.
In the Markdown report (which has no colors), the speedups are prefixed with 🟢 (significantly faster),
🔴 (significantly slower) or ⚪ (no significant difference).

The `json` and `csv` reports contain the raw statistics (median and quartiles of the per-iteration times, number of
samples and iterations per sample, speedups and their confidence intervals). The `html` report is a standalone page
(color-coded cells, with the absolute times and confidence interval in their tooltips). All the reports start with
the specifications of the GPU and CPU, since the results depend on both.

### Continuous integration

The `benchmark-modules` job of `.gitlab-ci.yml` benchmarks every merge request pipeline against the `develop` branch
(`BENCHMARK_BASELINE_REF` variable), and exposes the reports (HTML, Markdown, JSON and CSV) in the merge request
without blocking the merge: the "Modules benchmark" link (exposed artifact) opens the job artifacts.

The job runs (on a GPU runner) only for merge requests that change the library (`warp_nn`), the benchmarks, the
project configuration or the CI configuration.

### Methodology

What is measured:

- **Forward pass**: inference, without recording the operations for differentiation (no `wp.Tape` for warp-nn,
  `torch.no_grad()` for PyTorch).
- **Backward pass**: backpropagation of the gradients of all the outputs to the inputs and parameters, through the
  operations recorded by a single forward pass (`wp.Tape.backward(grads=...)` and
  `torch.autograd.backward(..., retain_graph=True)`).
- **Eager** mode (default) measures the end-to-end throughput, including the host overhead (Python, kernel launches)
  that is not hidden by the device execution, which dominates for small batch sizes. **CUDA graph** mode
  (`--cuda-graph`) replays captured graphs, measuring (mostly) the device execution.
- The benchmark allows TF32 math in PyTorch's matrix multiplications and convolutions (cuBLAS and cuDNN) by default,
  while warp-nn runs in full single precision. Use `--disable-tf32` to run PyTorch in full single precision too.
- The warp-nn backward pass includes copying the incoming gradients into the outputs' gradients on every call, as
  required by `wp.Tape.backward(grads=...)` (the adjoint kernels zero the outputs' gradients), while PyTorch uses
  them directly. This copy is a noticeable part of the backward pass of cheap modules for small batch sizes.
- `BatchNorm` (in training mode) does not support a batch size of 1, neither in warp-nn nor in PyTorch, so it is
  reported as `n/a`.

How it is measured, to produce results that can be used to make optimization decisions:

- **Warmup**: every pass is executed once when it is built (compiling/loading the warp-nn kernels, allocating the
  cached arrays, initializing PyTorch libraries) and before capturing CUDA graphs. Each implementation is then warmed
  up again before every measurement, and the device is kept busy for a while before benchmarking (and after slow
  builds, e.g. kernel compilation) so that its clocks ramp up.
- **Samples**: each timing sample runs a batch of iterations, calibrated so that it lasts at least
  `--min-sample-time` (20 ms by default), between device synchronizations. An untimed iteration precedes each
  sample. The garbage collector is disabled while timing.
- **Interleaving**: the samples of the implementations (PyTorch, current and baseline) are interleaved, one sample of
  each per round, cycling through all their orderings. Slow drifts of the device and host state (clocks, temperature,
  other processes) then affect all the implementations alike, and none of them is favored by the state left by the
  previous one.
- **Instance sets**: several instances of each implementation are built (`--instance-sets`, 2 by default), in
  alternating orders (PyTorch first, then the warp-nn implementations in swapped orders: current and baseline, then
  baseline and current), and each set of instances is measured with its share of the samples and of the time budget
  (and the same number of rounds). The memory placement of an instance, which depends on the allocation history,
  can make it a few percent slower than an identical one: alternating orders and averaging over the instances reduce
  this bias. For decisions on small differences, use more instance sets (e.g. `--instance-sets 4`, which also makes
  the speedup robust to an outlier instance). If the other instances cannot be built (e.g. out of memory), only the
  first set is measured.
- **Statistics**: the speedup of each instance set is the median of its per-round (paired) time ratios, and the
  reported speedup is the median of those (in log space; the geometric mean of two sets). Its confidence interval is
  estimated by bootstrapping the rounds within each instance set, and extended to cover the speedups of all the sets.
  A difference is only significant if all the instance sets agree on it.
  The (percentile bootstrap) confidence interval is slightly optimistic for few samples,
  and many cells are compared in a full run: expect a few spurious significant cells, which the minimum effect
  (`--min-effect`) mostly filters out.
- **Robust mode** (`--robust`): the performance of some kernels (e.g. memory-bound ones, for large batch sizes)
  depends on the memory placement of their arrays, sometimes by tens of percent, so a single instance can be
  "lucky" or "unlucky". With a fixed build order, such effects can show up as small (a few percent) but reproducible
  differences between identical implementations. The robust mode randomizes the placement of every instance (with
  randomly sized padding blocks allocated before building it) and uses 6 instance sets (unless `--instance-sets` is
  set), so that placement effects average out, and can only be reported as significant if all the sets agree.
  Each instance set measures at least a complete cycle of the orderings of the implementations (6 rounds for 3
  implementations), so 6 instance sets measure at least 36 rounds, regardless of `--samples` and `--max-time`.
  It is slower (about 2-3x), so use it before optimization decisions on large-batch, memory-bound modules
  (e.g. `--robust --modules LayerNorm "*Norm"`). The per-instance-set speedups (in the HTML tooltips and in the JSON
  report) show how sensitive a module is to its placement.

Tips:

- Close other GPU and CPU intensive applications (the host overhead is sensitive to CPU frequency scaling), and
  use a machine with stable clocks (e.g. a plugged-in laptop, or locked GPU clocks) for the most reliable results.
- To decide on small differences, increase the sample time and/or the number of samples (e.g.
  `--min-sample-time 0.04 --samples 60`) for the modules of interest. Running with the baseline equal to the
  current implementation (A/A test) shows the noise floor of the machine.
