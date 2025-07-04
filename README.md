# Inference Backend Benchmark

A benchmarking tool for comparing inference backends (vLLM, SGLang, TGI) using the same model and configuration.

## Quick Start

### 1. Configure Benchmarks

Edit `bench-config.yaml` to set your model and benchmark parameters:

```yaml
model: &model Qwen/Qwen3-0.6B

benchmark: 
  dataset-name: "random"
  request-rate: 1
  random-input-len: 1024
  random-output-len: 128
  num-prompts: 100

backends:
  vllm:
    image: vllm/vllm-openai:latest
    port: 8000
    args: [--model, *model, --dtype, bfloat16]
  # ... other backends
```

### 2. Run Benchmarks

```bash
# Run all backends
python backend_launcher.py

# Skip specific backends
python backend_launcher.py --disable sglang,tgi

# Show container logs during benchmarking
python backend_launcher.py --show-logs
```

### 3. View Results

```bash
# List available models
python summarize_benchmarks.py --list-models

# Summarize results for a specific model (markdown output)
python summarize_benchmarks.py Qwen-Qwen3-0.6B

# Save markdown report to file
python summarize_benchmarks.py Qwen-Qwen3-0.6B -o report.md

# View results for a specific run
python summarize_benchmarks.py Qwen-Qwen3-0.6B --run-id 20241201-120000-abc12345
```

## Results

Results are stored in `./results/[model]/[backend]/[run_id]/` with comprehensive metadata including backend arguments and benchmark parameters.

The summary report generates markdown-formatted output with tables and rankings that can be saved to file or viewed directly in the terminal.
