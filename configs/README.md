# FaStfact Configuration Templates

This directory contains example configuration files for customized use cases. These templates can be used as starting points for your evaluations.

## Usage

### Using Config Files (Future Enhancement)
```bash
# This feature will be added in a future version
python -m pipeline.FaStfact --config configs/default.json --input_file data.jsonl
```


## Parameter Explanations

### Model Configuration
- `model_name_extraction`: LLM for claim extraction (gpt-4o, gpt-4o-mini)
- `model_name_verification`: LLM for claim verification (gpt-4o, gpt-4o-mini)

### Evaluation Parameters
- `label_n`: Number of verification labels (2, 3, or 5)
- `pre_veri_label_m`: Number of pre-verification labels (3, 5, or 6)
- `search_res_num`: Number of search results per claim
- `verify_res_num`: Number of verification passages per claim

### Performance Parameters
- `response_worker_num`: Parallel workers for response processing
- `search_worker_num`: Parallel workers for web search (max 40 for Jina API)

### Quality Control
- `token_prob_threshold`: Token probability threshold for filtering (0-1)
- `uncertainty_threshold`: Uncertainty threshold for filtering (optional)


