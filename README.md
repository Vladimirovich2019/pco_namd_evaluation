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

Download the PCO-nAMD dataset from [Figshare](https://figshare.com/articles/dataset/PCO-nAMD_Paired_CFP_and_OCT_Dataset_for_Neovascular_Age-Related_Macular_Degeneration/32873501). The expected directory structure:

```
data/
├── PCO-nAMD_MNV.xlsx          # MNV metadata (327 patients)
├── PCO-nAMD_Response.xlsx     # Response metadata (101 patients)
├── PCO-nAMD_MNV/              # MNV images
│   ├── 001/
│   │   ├── cfp.jpg            # color fundus photograph (JPEG)
│   │   └── oct.png            # OCT B-scan (pre-treatment, lossless PNG)
│   └── ...
└── PCO-nAMD_Response/        # Response images
    ├── 004/
    │   ├── cfp.jpg            # color fundus photograph (JPEG)
    │   ├── oct.png            # OCT B-scan (pre-treatment, lossless PNG)
    │   └── post-oct.png       # OCT B-scan (post-treatment, lossless PNG)
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
| `mnv` | MNV classification (CFP+OCT) | Figure 4 |
| `mnv_cfp_only` | MNV classification (CFP only) | Figure 4 |
| `mnv_oct_only` | MNV classification (OCT only) | Figure 4 |
| `response_direct` | Response prediction (images only, no annotations) | Figure 4, Images Only |
| `given_biomarker_noimg` | Response prediction (annotations only, no images) | Figure 4, w/o Images |
| `given_biomarker` | Response prediction (images + annotations) | Figure 4, Combination |

### Reproduce paper metrics

After the prediction runs have been written to `results/`, reproduce every
reported metric with:

```bash
python scripts/reproduce_metrics.py
```

The script reads the `*_detailed_*.json` files under `results/` and prints, for
each model and task:

- **Four-class tasks** (`mnv`, `mnv_cfp_only`, `mnv_oct_only`, and the three
  response settings): balanced accuracy, macro-F1, per-class recall and the
  confusion matrix.
- **Binary response** (Good/Partial vs Poor/Non-response): balanced accuracy
  and macro-F1.

An alternative results directory can be passed as the first argument:

```bash
python scripts/reproduce_metrics.py /path/to/results
```

Label mapping and metric definitions are imported from `run.py` and `utils/`,
so these figures always agree with the in-run evaluation (`--eval`).

Cases whose prediction is missing or could not be recognized are excluded before
metric calculation; the script prints how many cases were excluded per experiment.
See `EXCLUDED_CASES.md` for the counts in the reported results.

### Evaluate a single run

```bash
python run.py -m GPT --task mnv --run 1 --eval
```

## Metric calculation

Only cases that have both a ground-truth label and a prediction that resolves to
one of the four class labels enter the metric calculation. Predictions that are
**missing** (no usable answer was returned) or **unrecognized** (an answer was
returned, but it does not resolve to a class) are excluded, so all reported
balanced accuracy, macro-F1 and per-class recall values share the same
denominator. `EXCLUDED_CASES.md` lists how many cases this affected.

Balanced accuracy is the mean of the per-class recalls and macro-F1 the mean of
the per-class F1 scores. The same two metrics are used for the four-class tasks
and for the binary response task (Good/Partial vs Poor/Non-response).

## Output

Results are saved to `results/` as:
- `{task_prefix}_detailed_{model}_tokens16384_run{N}.json` — full per-case predictions
- `{task_prefix}_summary_{model}_tokens16384_run{N}.csv` — summary table
