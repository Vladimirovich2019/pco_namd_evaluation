# PCO-nAMD Evaluation Code

Code for zero-shot MLLM evaluation on the PCO-nAMD dataset: MNV subtype classification and anti-VEGF treatment response prediction.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/config.json` with your API endpoint and key:

```json
{
    "GPT": {
        "model_name": "gpt-5",
        "base_url": "https://your-proxy-url/v1/chat/completions",
        "key": "sk-your-api-key"
    },
    ...
}
```

The code uses an OpenAI-compatible API interface. You can configure any model provider that supports this format.

## Dataset

Download the PCO-nAMD dataset from figshare. The expected directory structure:

```
data/
├── PCO-nAMD_MNV.xlsx          # MNV metadata (327 patients)
├── PCO-nAMD_Response.xlsx     # Response metadata (101 patients)
├── PCO-nAMD_MNV/              # MNV images
│   ├── 001/
│   │   ├── cfp.jpg
│   │   └── oct.jpg
│   └── ...
└── PCO-nAMD_Response/        # Response images
    ├── 004/
    │   ├── cfp.jpg
    │   ├── oct.jpg
    │   └── post-oct.jpg
    └── ...
```

The code loads data from `./data/` by default (configurable in `utils/__init__.py` via `DATA_DIR`).

## Usage

### Run a single experiment

```bash
python run.py -m GPT --task mnv --run 1 --fresh
```

**Arguments:**
- `-m`: Model key (GPT, Gemini, Qwen) — must match `config/config.json`
- `--task`: Task name (see below)
- `--run`: Run number (for repeated experiments, appends `_run{N}` to filename)
- `--fresh`: Ignore existing results and start fresh
- `--eval`: Evaluate existing results without running inference

### Evaluate results

```bash
python run.py -m GPT --task mnv --run 1 --eval
```

### Available tasks

| Task | Description | Paper setting |
|------|-------------|---------------|
| `mnv` | MNV classification (CFP+OCT) | Figure 5a |
| `mnv_cfp_only` | MNV classification (CFP only) | Figure 5a |
| `mnv_oct_only` | MNV classification (OCT only) | Figure 5a |
| `response_direct` | Response prediction (images only, no annotations) | Figure 5b, Images Only |
| `given_biomarker_noimg` | Response prediction (annotations only, no images) | Figure 5b, w/o Images |
| `given_biomarker` | Response prediction (images + annotations) | Figure 5b, Combination |

### Aggregate multiple runs

```bash
python agg_runs.py 3 GPT mnv mnv_cfp_only response_direct given_biomarker
```

### Reproduce paper metrics

```bash
python scripts/reproduce_mnv_metrics.py
python scripts/reproduce_binary_metrics.py
```

## Output

Results are saved to `results/` as:
- `{task_prefix}_detailed_{model}_tokens16384_run{N}.json` — full per-case predictions
- `{task_prefix}_summary_{model}_tokens16384_run{N}.csv` — summary table
