############################### DEFAULT CONFIGURATION ###############################

# Default configuration for FaStfact evaluation
from dataclasses import dataclass

@dataclass
class FaStfactConfig:
    # Directories
    data_dir: str = "./data"
    output_dir: str = "./data"
    cache_dir: str = "./data/cache"

    # Models
    model_name_extraction: str = "gpt-4o-mini"
    model_name_verification: str = "gpt-4o-mini"

    # Claim extraction & verification parameters
    label_n: int = 5
    pre_veri_label_m: int = 6
    stride: int = 0
    verify_res_num: int = 10
    search_res_num: int = 5
    token_prob_threshold: float = 0.995
    uncertainty_threshold: float = None
    do_not_pre_verify: bool = False

    # Parallelization
    response_worker_num: int = 32
    search_worker_num: int = 3

    # Flags
    use_external_extraction_model: bool = False
    use_external_verification_model: bool = False
    use_base_extraction_model: bool = False
    use_base_verification_model: bool = False
    ignore_cache: bool = False

############################ MODEL PRICING (for cost calculation) ############################
# ratio: 10 ** (-6)
model_pricing = {
    "gpt-4o": {
        "input": 2.50,
        "cached_input": 1.25,
        "output": 10.00,
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "cached_input": 0.075,
        "output": 0.60,
    },
    "gpt-4o-mini-fine-tuning": {
        "input": 0.30,
        "cached_input": 0.15,
        "output": 1.20,
        "training": 3.00,
    },
    "gpt-4o-fine-tuning": {
        "input": 3.75,
        "cached_input": 1.875,
        "output": 15.00,
        "training": 25.00,
    },
    "gpt-4": {
        "input": 30.00,
        "output": 60.00,
    },
    "gpt-4-32k": {
        "input": 60.00,
        "output": 120.00,
    },
    "gpt-4-turbo": {
        "input": 10.00,
        "output": 30.00,
    },
    "gpt-3.5-turbo": {
        "input": 0.50,
        "output": 1.50,
    },
    "gpt-3.5-turbo-instruct": {
        "input": 1.50,
        "output": 2.00,
    },
    "gpt-3.5-turbo-16k-0613": {
        "input": 3.00,
        "output": 4.00,
    },
    "openai-o1": {
        "input": 15.00,
        "cached_input": 7.50,
        "output": 60.00,
    },
    "openai-o3-mini": {
        "input": 1.10,
        "cached_input": 0.55,
        "output": 4.40,
    },
    "davinci-002": {
        "input": 2.00,
        "output": 2.00,
    },
    "babbage-002": {
        "input": 0.40,
        "output": 0.40,
    },
}

# Example usage (optional):
# To access the price of gpt-4o input:
# gpt4o_input_price = model_pricing["gpt-4o"]["input"]
# print(f"gpt-4o input price: {gpt4o_input_price}")