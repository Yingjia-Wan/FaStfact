import csv
from statistics import mean
from typing import Dict, List, Union, Optional
from collections import defaultdict
from .config import model_pricing
import json
import math
import os

class F1_score():
    def __init__(self, bench_model_response_stats_dict,
                  bench_model_token_stats_dict,
                  result_dir, 
                  model_name_extraction, model_name_verification,
                  total_time, abstain_count, total_responses,
                  ):
        '''
        Initialize F1_score calculator for FaStFact evaluation metrics.
        
        Args:
            bench_model_response_stats_dict (dict): Nested dictionary containing response statistics
                organized by benchmark and model. Structure: {bench: {model: [response_stats]}}
            bench_model_token_stats_dict (dict): Nested dictionary containing token usage statistics
                organized by benchmark and model. Structure: {bench: {model: [token_stats]}}
            result_dir (str): Path to output CSV file for results
            model_name_extraction (str): Name of the model used for claim extraction
            model_name_verification (str): Name of the model used for claim verification
            total_time (float): Total processing time in seconds
            abstain_count (int): Number of responses that abstained from evaluation
            total_responses (int): Total number of responses processed (sample size)
        '''
        
        # initialize
        self.bench_model_response_stats_dict = bench_model_response_stats_dict
        self.bench_model_token_stats_dict = bench_model_token_stats_dict
        self.result_dir = result_dir
        self.model_name_extraction, self.model_name_verification = model_name_extraction, model_name_verification
        self.total_time, self.abstain_count, self.sample_size = total_time, abstain_count, total_responses
        self.FaStfact_gamma = 0.1
        

        # add up token stats for all benchs and models
        self.total_token_stats = self.total_token_stats_and_cost(bench_model_token_stats_dict, model_name_extraction, model_name_verification)

        # get K from response stats for all benchs and models (average them in write_avg_numbers)
        self.bench_K_dict = self.get_K_stats(bench_model_response_stats_dict)

    def total_token_stats_and_cost(self, bench_model_token_stats_dict, model_name_extraction, model_name_verification):
        '''
        Calculate total tokens per phase (extract/search/verify) and overall totals,
        organized by benchmark and model. Then calculate cost based on model pricing.
        
        Args:
            bench_model_token_stats_dict (dict): Nested dictionary containing token statistics
                organized by benchmark and model. Structure: {bench: {model: [token_stats]}}
            model_name_extraction (str): Name of the model used for claim extraction
            model_name_verification (str): Name of the model used for claim verification
            
        Returns:
            dict: Aggregated token statistics and costs organized by benchmark and model.
                Structure: {
                    "benchmark_name": {
                        "model_name": {
                            "extract": {"prompt": int, "response": int, "cost": float},
                            "search": int,
                            "verify": {"prompt": int, "response": int, "cost": float},
                            "total_API": {"prompt": int, "response": int, "cost": float}
                        }
                    }
                }
                
        Note:
            - Token counts are aggregated across all samples for each benchmark-model combination
            - Costs are calculated using model pricing from config and converted to dollars (divided by 1M)
            - Search tokens are only associated with search API.
            - total_API combines extract and verify phases only
        '''
        total_token_dict = defaultdict(lambda: defaultdict(lambda: {
            "extract": {"prompt": 0, "response": 0, "cost": 0},
            "search": 0,
            "verify": {"prompt": 0, "response": 0, "cost": 0},
            "total_API": {"prompt": 0, "response": 0, "cost": 0},
        }))

        for bench, model_token_stats in bench_model_token_stats_dict.items():
            for model_name, token_stats_list in model_token_stats.items():
                for token_stats in token_stats_list:
                    # Extract phase
                    extract = token_stats.get("extract", {})
                    total_token_dict[bench][model_name]["extract"]["prompt"] += extract.get("prompt_tok_cnt", 0)
                    total_token_dict[bench][model_name]["extract"]["response"] += extract.get("response_tok_cnt", 0)
                    
                    # Search phase
                    search_tokens = token_stats.get("search", 0)
                    total_token_dict[bench][model_name]["search"] += search_tokens
                    
                    # Verify phase
                    verify = token_stats.get("verify", {})
                    total_token_dict[bench][model_name]["verify"]["prompt"] += verify.get("prompt_tok_cnt", 0)
                    total_token_dict[bench][model_name]["verify"]["response"] += verify.get("response_tok_cnt", 0)
                
                # add to total_API
                total_token_dict[bench][model_name]["total_API"]["prompt"] = total_token_dict[bench][model_name]["extract"]["prompt"] + total_token_dict[bench][model_name]["verify"]["prompt"]
                total_token_dict[bench][model_name]["total_API"]["response"] = total_token_dict[bench][model_name]["extract"]["response"] + total_token_dict[bench][model_name]["verify"]["response"]
                # convert total tokens to cost
                extract_price, verify_price = model_pricing[model_name_extraction], model_pricing[model_name_verification]
                extract_cost = extract_price['input'] * total_token_dict[bench][model_name]["extract"]["prompt"] + extract_price['output'] * total_token_dict[bench][model_name]["extract"]["response"]
                verify_cost = verify_price['input'] * total_token_dict[bench][model_name]["verify"]["prompt"] + verify_price['output'] * total_token_dict[bench][model_name]["verify"]["response"]
                total_token_dict[bench][model_name]["extract"]["cost"], total_token_dict[bench][model_name]["verify"]["cost"] = extract_cost / 1000000, verify_cost / 1000000
                total_token_dict[bench][model_name]["total_API"]["cost"] = (extract_cost + verify_cost) / 1000000
        
        print("\nFinal Token Stats and Costs:")
        print(json.dumps(total_token_dict, indent=2, default=lambda x: dict(x)))
        # Convert defaultdict to regular dict
        return {
            bench: {
                model: dict(stats) for model, stats in models.items()
            }
            for bench, models in total_token_dict.items()
        }

    def get_K_stats(self, bench_model_response_stats_dict):
        '''
        Calculate K statistics (median and max claim counts) for each benchmark.
        
        Args:
            bench_model_response_stats_dict (dict): Nested dictionary containing response statistics
                organized by benchmark and model. Structure: {bench: {model: [response_stats]}}
                
        Returns:
            dict: K statistics for each benchmark. Structure: {
                "benchmark_name": {
                    "K_median": int,
                    "K_max": int
                }
            }
            
        Note:
            - K_median: Median number of total claims across all samples in the benchmark
            - K_max: Maximum number of total claims across all samples in the benchmark
            - These values are used for calculating recall metrics in FaStFact evaluation
        '''
        bench_K_dict = defaultdict(lambda: defaultdict(int))
        for bench, model_response_stats in bench_model_response_stats_dict.items():
            claim_num_lst = []
            for model_name, response_stats_list in model_response_stats.items():
                for response_stats in response_stats_list:
                    claim_num_lst.append(response_stats["total"]["total_claims"])
            claim_num_lst.sort()
            K_median = claim_num_lst[len(claim_num_lst)//2]
            K_max = claim_num_lst[-1]
            
            bench_K_dict[bench]["K_median"] = K_median
            bench_K_dict[bench]["K_max"] = K_max
        return bench_K_dict

    def _get_F1(self, precision_lst, recall_med_list, recall_max_list):
        f1_med_list = []
        f1_max_list = []
        for p, r_med, r_max in zip(precision_lst, recall_med_list, recall_max_list):
            if p == 0:  # Handle case where support_claims = 0
                f1_med_list.append(0)
            else:
                f1_med_list.append(self._safe_division(2 * p * r_med, p + r_med))
                f1_max_list.append(self._safe_division(2 * p * r_max, p + r_max))
        return f1_med_list, f1_max_list
    
    
    def _get_FaStfact_score(self, precision_lst: List[float], support_claims_lst: List[int], K: float):
        """
        FaStfact Score = “Informativeness Penalty rate” * Precision 
                    = (2 / (1 + alpha * exp(|S-K|))) * P
        alpha: a parameter to control smoothness / penalty rate
        Returns:
            - score_lst: List of FaStfact Scores using K (either K_median or K_max)
        """
        score_lst = []
        for P, S in zip(precision_lst, support_claims_lst):
            if P is None:
                score_lst.append(None)
                continue
            penalty = 2 / (1 + math.exp(self.FaStfact_gamma * abs(S - K)))
            score = round(penalty * P, 3)
            score_lst.append(score)
        return score_lst
    
    def _safe_mean(self, values):
        return round(mean(values), 3) if values else None

    def _safe_division(self, numerator, denominator):
        if denominator is None or denominator == 0:
            return None
        return round(numerator / denominator, 3)

    
    def write_avg_numbers(self) -> None:
        '''
        - Calculate the avg numbers for each model and benchmark.
        - Write to CSV file.
        - Metrics:
            - P (factscore) = S/K
            - SAFE F1@K = 2PR/P+R, R = min(S/K, 1)
            - FaStfact F1@K = 2PR/P+R, R = 2 / (1 + exp(|S-K|))
            - FaStfact Score = “Informativeness Penalty rate” * Precision P = (2 / (1 + exp(|S-K|))) * P
        '''
        results_json = []
        with open(self.result_dir, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            header = [
                "sampled_model",
                "model_name_extraction", "model_name_verification", 
                "benchmark", "sample_size", "abstain_rate", 
                "input_tkn", "resp_tkn", "search_tkn", 
                "time(s)", "cost($)",
                "length", "num_sentences", "num_claims",
                "total:support_claims", "total:unsupport_claims", "total:irrelevant_claims",
                "pre_verified(confident)", "pre_support", "pre_unsupport",
                "pre_verified(confident)%", "pre_support/pre_unsupport", "pre_support%", "pre_unsupport%",
                "verified", "verified_support", "verified_unsupport", "verified_inconclusive", "verified_controversial_claims", "verified_contradicted", "verified_unverifiable",
                "P(factscore)", 
                "K_med", "K_max",
                "R_med", "R_max", 
                "SAFE_F1_med", "SAFE_F1_max",
                "FaStfact_F1_med", "FaStfact_F1_max",
                "FaStfact_score_med", "FaStfact_score_max",
            ]
            writer.writerow(header)

            abstain_rate = self._safe_division(self.abstain_count, self.sample_size)

            for bench, model_response_stats_dict in self.bench_model_response_stats_dict.items():
                K_median = self.bench_K_dict[bench]['K_median']
                K_max = self.bench_K_dict[bench]['K_max']

                for model_name, response_stats_list in model_response_stats_dict.items():
                    total_prompt_tok_cnt = self.total_token_stats[bench][model_name]["total_API"]["prompt"]
                    total_resp_tok_cnt = self.total_token_stats[bench][model_name]["total_API"]["response"]
                    total_search_tok_cnt = self.total_token_stats[bench][model_name]["search"]
                    total_cost = self.total_token_stats[bench][model_name]["total_API"]["cost"]

                    pre_verified_claims = [x["preverification"].get("directly_preverified_claims", 0) for x in response_stats_list]
                    pre_supported_claims = [x["preverification"].get("pre_supported_claims", 0) for x in response_stats_list]
                    pre_unsupported_claims = [x["preverification"].get("pre_unsupported_claims", 0) for x in response_stats_list]

                    verified_claims = [x["verification"].get("total_verified_claims", 0) for x in response_stats_list]
                    verified_supported_claims = [x["verification"].get("verified_supported_claims", 0) for x in response_stats_list]
                    verified_unsupported_claims = [x["verification"].get("verified_unsupported_claims", 0) for x in response_stats_list]
                    verified_inconclusive_claims = [x["verification"].get("verified_inconclusive_claims", 0) for x in response_stats_list]
                    verified_controversial_claims = [x["verification"].get("verified_controversial_claims", 0) for x in response_stats_list]
                    verified_contradicted_claims = [x["verification"].get("verified_contradicted_claims", 0) for x in response_stats_list]
                    verified_unverifiable_claims = [x["verification"].get("verified_unverifiable_claims", 0) for x in response_stats_list]

                    num_sentences = [x.get("num_sentences", 0) for x in response_stats_list]
                    length = [x.get("length", 0) for x in response_stats_list]
                    total_claims = [x["total"].get("total_claims", 0) for x in response_stats_list]
                    total_supported_claims = [x["total"].get("total_supported_claims", 0) for x in response_stats_list]
                    total_unsupported_claims = [x["total"].get("total_unsupported_claims", 0) for x in response_stats_list]
                    total_irrelevant_claims = [x["total"].get("total_irrelevant_claims", 0) for x in response_stats_list]

                    P_list = [x.get("P", None) for x in response_stats_list]
                    SAFE_R_med_list = [min(self._safe_division(support, K_median), 1) if self._safe_division(support, K_median) is not None else None for support in total_supported_claims]
                    SAFE_R_max_list = [min(self._safe_division(support, K_max), 1) if self._safe_division(support, K_max) is not None else None for support in total_supported_claims]
                    FaStfact_R_med_list = [2 / (1 + math.exp(self.FaStfact_gamma * abs(support - K_median))) for support in total_supported_claims]
                    FaStfact_R_max_list = [2 / (1 + math.exp(self.FaStfact_gamma * abs(support - K_max))) for support in total_supported_claims]

                    metrics = {
                        "P_list": P_list,
                        "SAFE_R_med_list": SAFE_R_med_list,
                        "SAFE_R_max_list": SAFE_R_max_list,
                        "FaStfact_R_med_list": FaStfact_R_med_list,
                        "FaStfact_R_max_list": FaStfact_R_max_list
                    }
                    for name, lst in metrics.items():
                        metrics[name][:] = [v for v in lst if v is not None]

                    P_list = metrics["P_list"]
                    SAFE_R_med_list = metrics["SAFE_R_med_list"]
                    SAFE_R_max_list = metrics["SAFE_R_max_list"]
                    FaStfact_R_med_list = metrics["FaStfact_R_med_list"]
                    FaStfact_R_max_list = metrics["FaStfact_R_max_list"]

                    # SAFE F1@K, FaStfact F1@K, FaStfact Score
                    SAFE_f1_med_list, SAFE_f1_max_list = self._get_F1(P_list, SAFE_R_med_list, SAFE_R_max_list)
                    FaStfact_f1_med_list, FaStfact_f1_max_list = self._get_F1(P_list, FaStfact_R_med_list, FaStfact_R_max_list)
                    FaStfact_score_med_list = self._get_FaStfact_score(P_list, total_supported_claims, K_median)
                    FaStfact_score_max_list = self._get_FaStfact_score(P_list, total_supported_claims, K_max)

                    row = [
                        model_name,  # sampled_model for factuality evaluation
                        self.model_name_extraction, self.model_name_verification,
                        bench,
                        self.sample_size,
                        round(abstain_rate, 4),

                        total_prompt_tok_cnt,
                        total_resp_tok_cnt,
                        total_search_tok_cnt,
                        self.total_time,
                        round(total_cost, 4),

                        self._safe_mean(length),
                        self._safe_mean(num_sentences),
                        self._safe_mean(total_claims),
                        self._safe_mean(total_supported_claims),
                        self._safe_mean(total_unsupported_claims),
                        self._safe_mean(total_irrelevant_claims),

                        self._safe_mean(pre_verified_claims),
                        self._safe_mean(pre_supported_claims),
                        self._safe_mean(pre_unsupported_claims),

                        self._safe_division(sum(pre_verified_claims), sum(total_claims)),
                        self._safe_division(sum(pre_supported_claims), sum(pre_unsupported_claims)),
                        self._safe_division(sum(pre_supported_claims), sum(total_claims)),
                        self._safe_division(sum(pre_unsupported_claims), sum(total_claims)),

                        sum(verified_claims),
                        sum(verified_supported_claims),
                        sum(verified_unsupported_claims),
                        sum(verified_inconclusive_claims),
                        sum(verified_controversial_claims),
                        sum(verified_contradicted_claims),
                        sum(verified_unverifiable_claims),

                        self._safe_mean(P_list),
                        K_median,
                        K_max,
                        self._safe_mean(SAFE_R_med_list),
                        self._safe_mean(SAFE_R_max_list),
                        self._safe_mean(SAFE_f1_med_list),
                        self._safe_mean(SAFE_f1_max_list),
                        self._safe_mean(FaStfact_f1_med_list),
                        self._safe_mean(FaStfact_f1_max_list),
                        self._safe_mean(FaStfact_score_med_list),
                        self._safe_mean(FaStfact_score_max_list),
                    ]

                    # save and print
                    writer.writerow(row)
                    print(f"\n[Results for {bench}-{model_name}]")
                    print("-" * 50)
                    print(f"{'Metric':<25} | {'Median':<15} | {'Max':<15}")
                    print("-" * 50)
                    print(f"{'Precision':<25} | {self._safe_mean(P_list):<15.4f} | {'-':<15}")
                    print(f"{'K':<25} | {K_median:<15.4f} | {K_max:<15.4f}")
                    print(f"{'Longfact F1@k':<25} | {self._safe_mean(SAFE_f1_med_list):<15.4f} | {self._safe_mean(SAFE_f1_max_list):<15.4f}")
                    print(f"{'FaStfact F1@k':<25} | {self._safe_mean(FaStfact_f1_med_list):<15.4f} | {self._safe_mean(FaStfact_f1_max_list):<15.4f}")
                    print(f"{'FaStfact Score':<25} | {self._safe_mean(FaStfact_score_med_list):<15.4f} | {self._safe_mean(FaStfact_score_max_list):<15.4f}")
                    print("-" * 50)
                    
                    # Convert to dict for JSON
                    result_dict = dict(zip(header, row))
                    results_json.append(result_dict)

        # Write JSON file
        json_path = self.result_dir.replace(".csv", ".json")
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(results_json, jf, indent=2, ensure_ascii=False)


