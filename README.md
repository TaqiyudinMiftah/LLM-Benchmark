# LLM Benchmark

A repository for benchmarking Large Language Models (LLMs) using the [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) evaluation framework.

## Overview

This project evaluates the **Bakti-8B-Base** model from [aitfindonesia](https://huggingface.co/aitfindonesia/Bakti-8B-Base) on standard NLP benchmarks including:

- **MMLU** (Massive Multitask Language Understanding) - Tests knowledge across 57 subjects
- **HellaSwag** - Tests commonsense reasoning

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Sanity_check_run.ipynb` | Quick sanity check with limited samples (10 samples) |
| `Bakti-8B-Base-MMLU,Hellaswag.ipynb` | Full evaluation on single GPU |
| `Accelerator-Bakti-8B-Base-MMLU,Hellaswag.ipynb` | Multi-GPU accelerated evaluation using Accelerate |

## Requirements

- Python 3.8+
- CUDA-compatible GPU
- Dependencies:
  ```
  lm-eval
  transformers>=4.44.0
  accelerate>=0.34.0
  bitsandbytes
  sentencepiece
  datasets>=2.14.0
  ```

## Installation

```bash
pip install -q lm-eval
pip install -q "transformers>=4.44.0" "accelerate>=0.34.0" "bitsandbytes" "sentencepiece"
pip install -q "datasets>=2.14.0"
```

## Usage

### Single GPU Evaluation

```bash
lm-eval \
  --model hf \
  --model_args "pretrained=aitfindonesia/Bakti-8B-Base,trust_remote_code=True,load_in_4bit=True,bnb_4bit_quant_type=nf4,bnb_4bit_compute_dtype=bfloat16" \
  --tasks mmlu,hellaswag \
  --num_fewshot 5 \
  --device cuda:0 \
  --batch_size auto \
  --output_path hasil_evaluasi_bakti_8b.json
```

### Multi-GPU Evaluation (with Accelerate)

```bash
accelerate launch --multi_gpu --num_processes 2 -m lm_eval \
  --model hf \
  --model_args pretrained=aitfindonesia/Bakti-8B-Base,trust_remote_code=True,load_in_4bit=True,bnb_4bit_quant_type=nf4,bnb_4bit_compute_dtype=bfloat16 \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path ./hasil_evaluasi_bakti_8b
```

## Model Configuration

The evaluation uses 4-bit quantization for memory efficiency:

| Parameter | Value |
|-----------|-------|
| Quantization | 4-bit (NF4) |
| Compute dtype | bfloat16 |
| Trust remote code | True |

## Parameters

| Parameter | Description |
|-----------|-------------|
| `--tasks` | Benchmark tasks to run (mmlu, hellaswag) |
| `--num_fewshot` | Number of few-shot examples (0 or 5) |
| `--batch_size` | Batch size for evaluation (auto for automatic) |
| `--limit` | Limit number of samples (for testing) |
| `--output_path` | Path to save evaluation results |

## Output

Results are saved in JSON format containing:
- Task-specific accuracy scores
- Model configuration details
- Evaluation metadata

## License

This project is for educational and research purposes.

## Acknowledgments

- [EleutherAI](https://github.com/EleutherAI) for the lm-evaluation-harness
- [aitfindonesia](https://huggingface.co/aitfindonesia) for the Bakti-8B-Base model
