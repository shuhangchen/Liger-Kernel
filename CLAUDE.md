# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Liger Kernel is a collection of efficient Triton kernels for LLM training that increases multi-GPU training throughput by 20% and reduces memory usage by 60%. It provides optimized implementations of neural network layers (RMSNorm, RoPE, SwiGLU, CrossEntropy, etc.) with full Hugging Face compatibility.

## Development Commands

### Essential Commands
```bash
# Install for development
pip install -e ".[dev]"

# Run tests and linting (use these before committing)
make test              # Correctness tests
make checkstyle        # Ruff linting and formatting  
make test-convergence  # Convergence tests (fp32 and bf16)
make all              # All above tests

# Benchmarking
make run-benchmarks           # Run all benchmarks
make run-benchmarks OVERWRITE=1  # Overwrite existing data
```

### Single Test Execution
```bash
# Run specific test file
pytest test/transformers/test_rms_norm.py -v

# Run single test function
pytest test/transformers/test_rms_norm.py::test_rms_norm_correctness -v

# Run convergence tests for specific precision
pytest test/convergence/fp32/test_mini_models.py -v
```

## Architecture

### Core Components
- **`src/liger_kernel/ops/`**: Triton kernel implementations (low-level)
- **`src/liger_kernel/transformers/`**: PyTorch nn.Module wrappers (high-level API)
- **`src/liger_kernel/chunked_loss/`**: Memory-efficient post-training losses (DPO, ORPO, etc.)
- **`src/liger_kernel/transformers/model/`**: Model-specific monkey patches

### Key Patterns
1. **Triton Operations**: Low-level kernels in `ops/` implement core functionality
2. **PyTorch Wrappers**: `transformers/` provides nn.Module interfaces that call ops
3. **Monkey Patching**: Model support via strategic replacement of specific layers
4. **Chunked Processing**: Loss functions use chunking for memory efficiency

### Adding New Kernels
1. Implement Triton kernel in `src/liger_kernel/ops/new_op.py`
2. Create PyTorch wrapper in `src/liger_kernel/transformers/new_op.py`
3. Export in `src/liger_kernel/transformers/__init__.py`
4. Add tests in `test/transformers/test_new_op.py`
5. Add convergence test in `test/convergence/`
6. Create benchmark in `benchmark/scripts/benchmark_new_op.py`

### Adding Model Support
1. Identify target layers to replace with Liger kernels
2. Add monkey-patching logic in `transformers/monkey_patch.py`
3. Add model-specific implementation in `transformers/model/new_model.py`
4. Export monkey patch function in `transformers/__init__.py`

## Testing Requirements

### Before Committing
Always run: `make test`, `make checkstyle`, `make test-convergence`

### Test Types
- **Correctness**: `test/transformers/` - Verify kernel outputs match PyTorch
- **Convergence**: `test/convergence/` - Full model training matches reference
- **Chunked Loss**: `test/chunked_loss/` - Memory-efficient loss implementations

### Hardware Considerations
- Tests automatically detect CUDA/ROCm/Intel XPU
- Convergence tests run in both fp32 and bf16
- Specify GPU type when reporting issues

## Dependencies and Setup

### Core Requirements
- `torch>=2.1.2`
- `triton>=2.3.1` (CUDA) or `>=3.0.0` (ROCm)
- Platform detection is automatic

### Development Dependencies
Installed via `pip install -e ".[dev]"`: transformers, pytest, matplotlib, mkdocs

## Code Quality

- **Linter**: Ruff (configured in pyproject.toml)
- **Formatting**: Auto-applied via `make checkstyle`
- **Testing**: pytest with automatic CUDA cache clearing