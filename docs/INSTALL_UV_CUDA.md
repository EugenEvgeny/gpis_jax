# Installation with uv: CPU, CUDA 12, CUDA 13

`GP` and `GPIS` use **exactly the same JAX source code** on CPU and CUDA.
Selecting a CUDA extra installs the appropriate NVIDIA JAX backend; the
runtime remains platform-independent. The system must have a compatible
**NVIDIA driver**. CUDA extras do not ship NVIDIA drivers.

Run the following in the extracted project directory (alongside `pyproject.toml`).

## CPU (default)

```bash
uv sync --extra dev
uv run gp-jax-device --backend cpu --smoke-test
uv run python examples/demo.py
uv run pytest -q tests/test_gp.py
```

`uv sync` alone installs the package and its core dependencies; `--extra dev`
adds `pytest`. Both default to JAX CPU support; if the environment already
contains a supported GPU plugin, auto device selection may still use it.
You can force CPU execution using `JAX_PLATFORMS=cpu` **before launching Python**:

```bash
JAX_PLATFORMS=cpu uv run gp-jax-device --backend cpu --smoke-test
```

## NVIDIA GPU: CUDA 13 (recommended for recent GPUs and drivers)

```bash
uv sync --extra cuda13 --extra dev
uv run --extra cuda13 gp-jax-device --backend cuda --require-gpu --smoke-test
uv run --extra cuda13 python examples/demo.py
```

Requires Linux x86-64 / aarch64, an NVIDIA driver **>= 580**, and an NVIDIA GPU
with compute capability **>= 7.5**. Uses the CUDA/cuDNN distributions provided
by the JAX pip extra; no separately installed CUDA toolkit is required.

## NVIDIA GPU: CUDA 12 (older driver/hardware)

```bash
uv sync --extra cuda12 --extra dev
uv run --extra cuda12 gp-jax-device --backend cuda --require-gpu --smoke-test
uv run --extra cuda12 python examples/demo.py
```

Requires Linux x86-64 / aarch64, an NVIDIA driver **>= 525**, and compute
capability **>= 5.2**. See the current [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for compatibility details. CUDA 12 support will eventually be deprecated.

## HPC / system-provided CUDA (advanced)

When CUDA and cuDNN are provided by your cluster rather than pip wheels, use
`uv sync --extra cuda13-local` or `uv sync --extra cuda12-local` (only one CUDA
extra at a time). Verify the compatible CUDA/cuDNN versions and set the
library/toolkit search paths according to the official JAX installation guide.
The bundled pip-based extras are preferred for most installations; in particular
an unrelated `LD_LIBRARY_PATH` can override those packaged CUDA libraries.

## Running on a specific GPU

On a CUDA-enabled machine, JAX normally chooses the available GPU as its
highest-priority backend. To require CUDA rather than silently falling back
to CPU, set the JAX platform **before starting the interpreter**:

```bash
JAX_PLATFORMS=cuda uv run --extra cuda13 python examples/demo.py
```

Alternatively, choose a device programmatically (make training arrays **inside**
the context to prevent existing CPU-committed arrays from anchoring computation
on the wrong device):

```python
import jax
import jax.numpy as jnp
from gaussian_process import get_device, GP
from kernels import SEKernel

cuda_device = get_device("cuda")  # errors if no usable GPU
with jax.default_device(cuda_device):
    X = jnp.linspace(-1, 1, 32)[:, None]
    y = jnp.sin(X[:, 0])
    model = GP(SEKernel(dim=1), X, y, noise=0.01)
    # Reusable JIT-compiled numerical posterior (recommended on GPU):
    predictor = model.compile_predict()
    mean, variance = predictor(jnp.array([[0.2]]))
    print(mean.devices(), variance.devices())
```

`compile_predict()` is shared by `GP` and `GPIS`. It snapshots the current
conditioned state to keep Python cache checks out of JIT tracing. **Recreate the
compiled predictor after refitting the model, changing geometric priors, or
updating kernel parameters.** Use `model.predict(...)` for dynamically updated
models that must automatically detect and refresh state.

To choose a specific NVIDIA GPU, set `CUDA_VISIBLE_DEVICES=0` before starting
Python; device IDs become relative to the visible set. `get_device('cuda', index=0)`
selects from the JAX-visible GPU devices. For multi-process GPU sharing,
consult the JAX platform/runtime documentation.

### Diagnostics

```bash
nvidia-smi
uv run --extra cuda13 gp-jax-device --backend cuda --require-gpu --smoke-test
```

The checker prints the chosen JAX device, compiles GP and GPIS inference, and
verifies that its output arrays actually reside on that device. A CPU-only
installation will **fail** a `--require-gpu` check, not report a false success.

- Linux is the supported native CUDA wheel platform; native Windows and macOS
  cannot run the NVIDIA JAX CUDA backend. Windows WSL2 may work experimentally.
- CUDA 12 and CUDA 13 extras are **mutually exclusive**. `uv --all-extras`
  intentionally fails because it would request incompatible runtimes.
- Set `JAX_ENABLE_X64=1` for higher-precision numerical checks, at the cost of
  GPU memory and sometimes throughput. The test suite requests float64 for
  strict comparisons.
- Exact GP conditioning remains O(N³) compute and O(N²) memory, regardless of
  CPU vs GPU. Increase the batch size cautiously for large query sets.
- If the GPU check fails, check driver version, compute capability, GPU
  visibility and `LD_LIBRARY_PATH` conflicts. Do not disable CUDA version
  checks merely to suppress an incompatibility error.

## Building distributable packages

```bash
uv build
# creates dist/*.whl and dist/*.tar.gz
```

The wheel contains the `gaussian_process`, `kernels`, and `geometric_priors`
packages plus the `GP.py`/`GPIS.py` compatibility import modules. Optional
CUDA dependencies are declared in package metadata, so a consumer can also
install the published wheel with `uv pip install 'jax-gp-gpis-refactor[cuda13]'`.

The installation commands were checked on a CPU-only machine. **An actual GPU
run requires an NVIDIA host; it cannot be validated on CPU-only hardware.**
