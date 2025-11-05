"""
This is the main script to evaluate factual score about the model responses.
It is comprised of three stages: 
    (1) claim extraction and preverification, 
    (2) evidence search, and 
    (3) claim verification.
The (1) and (3) stage calls an LLM, and (2) uses a search API and SERP scraping API.
"""
# TODO: clean code structure; move arg to config file

import os
import json
import argparse
from collections import defaultdict
import spacy
from tqdm import tqdm
import time

from pipeline import utils, score_metrics
from .claim_extractor import ClaimExtractor
from .web_search_API import WebSearchAPI
from .retriever import BM25_retriever
from .claim_verifier import ClaimVerifier
from .config import FaStfactConfig


from concurrent.futures import ThreadPoolExecutor, as_completed

class FaStfact(object):
    def __init__(self,
                 model_name_extraction='gpt-4-0125-preview',
                 model_name_verification='gpt-4o',
                 use_external_extraction_model=False,
                 use_external_verification_model=False,
                 use_base_extraction_model=False,
                 use_base_verification_model=False,
                 data_dir='./data',
                 cache_dir='./data/cache',
                 output_dir='./data',
                 label_n=4,
                 pre_veri_label_m=6,
                 stride=0,
                 verify_res_num=5,
                 search_res_num=10,
                 do_not_pre_verify=False,
                 uncertainty_threshold=None,
                 token_prob_threshold=None,
                 ignore_cache=False,
                 search_worker_num = 1,
                 response_worker_num = 1,
                 ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.spacy_nlp = spacy.load('en_core_web_sm')
        self.system_message_extraction = "You are a helpful assistant who can extract verifiable atomic claims from a piece of text and evaluate the factual correctness of each claim. Each atomic claim should be verifiable against reliable external world knowledge (e.g., via Wikipedia)"

        self.model_name_extraction = model_name_extraction
        self.model_name_verification = model_name_verification
        self.claim_extractor = ClaimExtractor(model_name_extraction, pre_veri_label_m,
                                              stride = stride,
                                              cache_dir=self.cache_dir,
                                              use_external_model=use_external_extraction_model, 
                                              use_base_model=use_base_extraction_model,
                                              do_not_pre_verify = do_not_pre_verify,
                                              uncertainty_threshold=uncertainty_threshold,
                                              token_prob_threshold=token_prob_threshold,
                                              ignore_cache=ignore_cache
                                              )
        self.fetch_search = WebSearchAPI(ignore_cache=ignore_cache)
        self.claim_verifier = ClaimVerifier(model_name=model_name_verification, label_n=label_n,
                                            cache_dir=self.cache_dir,
                                            use_external_model=use_external_verification_model, 
                                            use_base_model=use_base_verification_model,
                                            ignore_cache=ignore_cache
                                            )
        self.label_n = label_n
        self.pre_veri_label_m = pre_veri_label_m
        self.search_res_num = search_res_num
        self.verify_res_num = verify_res_num
        self.do_not_pre_verify = do_not_pre_verify
        self.uncertainty_threshold = uncertainty_threshold
        self.token_prob_threshold = token_prob_threshold
        self.search_worker_num = search_worker_num
        self.response_worker_num = response_worker_num
        self.stride = stride

    def _create_model_output_dir(self, input_file_name):
        """Helper method to create the output directory path."""
        model_extract = self.model_name_extraction.split("/")[-1]
        model_verify = self.model_name_verification.split("/")[-1]
        base_dir = os.path.join(
            self.output_dir,
            input_file_name,
            f"stride={self.stride}_m={self.pre_veri_label_m}_n={self.label_n}",
        )
        # Conditionally append suffixes
        if self.do_not_pre_verify:
            base_dir += "_noPreverify"
        if self.uncertainty_threshold is not None:
            base_dir += f"_uncertainty={str(self.uncertainty_threshold).replace('.', 'p')}"
        if self.token_prob_threshold is not None:
            base_dir += f"_logprob={str(self.token_prob_threshold).replace('.', 'p')}"

        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def get_fsfact_score(self, data, input_file_name):
        """
        Main method to orchestrate the three subprocesses.
        """
        start_time = time.time()
        model_output_dir = self._create_model_output_dir(input_file_name)
            
        # Stage 1: Extract claims
        claims_output_path = os.path.join(model_output_dir, "claims.jsonl")
        extracted_claims, abstain_count = self.extract_claims(data, claims_output_path)
        extraction_timepoint = time.time()
        print(f"extraction time: {extraction_timepoint - start_time}")

        # Stage 2: Search evidence
        evidence_output_path = os.path.join(model_output_dir, "retrieved_evidence.jsonl")
        searched_claims = self.search_evidence(extracted_claims, evidence_output_path)
        search_timepoint = time.time()
        print(f"search time: {search_timepoint - extraction_timepoint}")

        # Stage 3: Verify claims
        verification_output_path = os.path.join(model_output_dir, f'verification_n={self.label_n}.jsonl')
        self.verify_claims(searched_claims, verification_output_path)
        end_time = time.time()
        print(f"verification time: {end_time - search_timepoint}")

        # Step 4: calculate avg F1@K
        total_time = end_time - start_time
        fsfact_score_path = os.path.join(model_output_dir, f'fsfact_score_n={self.label_n}.csv')
        self.calculate_avg_score(verification_output_path, fsfact_score_path, 
                                self.model_name_extraction, self.model_name_verification, 
                                total_time, abstain_count, len(data))

    ######################################## extract and pre-verify claims ########################################
    def extract_claims(self, data, claims_output_path):
        """
        Extract claims from the model responses.
        Saved to claims.jsonl, where each line consists of the extracted claims from a QA response.
        """

        # Load saved items in the existing output file
        extracted_data = set()
        abstain_count = 0
        extracted_claims = []

        if os.path.exists(claims_output_path):
            with open(claims_output_path, "r") as f:
                existing_extracted_claims = [json.loads(line) for line in f if line.strip()]
            for item in existing_extracted_claims:
                if "prelabel_stats" in item and "extract" in item.get("token_stats", {}):
                    key = (item['response'].strip(), item['prompt_source'], item['model'])
                    extracted_data.add(key)
                    extracted_claims.append(item)  # Keep valid cached entries
            print(f"Loaded {len(extracted_data)} samples with existing extracted claims from {claims_output_path}")

        # Extract claims
        skip_extract_count = 0
        with open(claims_output_path, "w") as f:

            # Write preserved cached entries
            active_item = []
            for item in extracted_claims:
                f.write(json.dumps(item) + "\n")
            
            for dict_item in tqdm(data, desc='Claim Extraction Progress: '): # per question
                # check existing items
                response, prompt_source, model = dict_item["response"], dict_item["prompt_source"], dict_item["model"]
                key = (response.strip(), prompt_source, model)

                # skip the item if existing
                if key in extracted_data:
                    skip_extract_count += 1
                    continue

                # save abstained responses without calling extractor (write, append, track)
                if utils.is_abstain_response(response):
                    question = dict_item["question"] if "question" in dict_item else ''
                    claim_dict = {"question": question.strip(),
                                   "response": response.strip(),
                                   "abstained": True,
                                   "prompt_source": prompt_source,
                                   "model": model, 
                                   'claims_lst': [],
                                    "token_stats": {
                                            "extract": {
                                                    "prompt_tok_cnt": 0,
                                                    "response_tok_cnt": 0,
                                                }
                                            }
                                   }
                    f.write(json.dumps(claim_dict) + "\n")
                    extracted_claims.append(claim_dict) # also save abstained sample
                    extracted_data.add(key)
                    abstain_count += 1
                    continue

                active_item.append(dict_item)

            with ThreadPoolExecutor(max_workers=self.response_worker_num) as executor:
                futures = []
                for dict_item in active_item:
                    futures.append(
                        executor.submit(
                            self._extract_claims_single,
                            dict_item,
                        )
                    )
                for future in futures:
                    claim_dict = future.result()
                    extracted_claims.append(claim_dict)
                    f.write(json.dumps(claim_dict) + '\n')
                    f.flush()

        return extracted_claims, abstain_count
    
    def _extract_claims_single(self, sample):
        response, prompt_source, model = sample["response"], sample["prompt_source"], sample["model"]
        question = sample["question"] if "question" in sample and sample["question"] else ''
        question_id = sample["question_id"] if "question_id" in sample and sample["question_id"] else None

        (
            prelabel_stats,
            preverification_claims_lst, # NOTE: we filtered out the repeated identical claims
            prompt_tok_cnt, 
            response_tok_cnt
        ) = self.claim_extractor.run(question, response)

        claim_dict = {
            "question": question.strip(),
            "question_id": question_id if question_id else None,
            "prompt_source": prompt_source,
            "response": response.strip(),
            "model": model,
            "abstained": False,
            "claims_lst": preverification_claims_lst,
            "prelabel_stats": prelabel_stats,
            "token_stats": {
                "extract": {
                        "prompt_tok_cnt": prompt_tok_cnt,
                        "response_tok_cnt": response_tok_cnt,
                    }
                }
            }

        return claim_dict

    ######################################## search evidence ########################################
    def search_evidence(self, extracted_claims, evidence_output_path):
        """
        - Search webpages to support or refute the extracted claims.
        - Retrieve webpages and extract the relevant evidence regarding the claims.
        - Update the token stats
        - Update the "claims_lst" item withthe newly searched evidence. Save to the evidence_output_path.

        INPUT: extracted_claims: jsonl where each line represents represents a response dict with the above items in claim_dict.
        RETURN: searched_claims: updated jsonl where lines with 'unsure' prelabels are added two extra items:

            search_results: 
            [{'title': ..., 'description': ..., 'text':..., 'usage': {"token": x}}, {...}, ...]

            retrieved_search_results:
            [{'title': str, 'description': str, 'retrieved_text': str}, {...}, ...]

        """
        # load already searched items
        searched_data = set()
        searched_claims_lst = []

        if os.path.exists(evidence_output_path):
            with open(evidence_output_path, "r") as f:
                existing_searched_evidences = [json.loads(line) for line in f]
            for dict_item in existing_searched_evidences:
                # Create a tuple key to identify already searched items
                if "search" in dict_item.get("token_stats", {}):
                    response, prompt_source, model = dict_item["response"], dict_item["prompt_source"], dict_item["model"]
                    key = (response.strip(), prompt_source, model)
                    searched_data.add(key)
                    searched_claims_lst.append(dict_item)
            print(f"Loaded {len(searched_data)} existing searched items from {evidence_output_path}")
        
        # search
        skip_search_count = 0
        active_item = []
        with open(evidence_output_path, "w") as f:

            # Write preserved cached entries
            for item in searched_claims_lst:
                f.write(json.dumps(item) + "\n")

            # gather dict_items to search
            for dict_item in tqdm(extracted_claims, desc='Search Progress: '): # per question response
                response, prompt_source, model = dict_item["response"], dict_item["prompt_source"], dict_item["model"]
                key = (response.strip(), prompt_source, model)

                # Skip if this item has already been searched
                if key in searched_data:
                    skip_search_count += 1
                    continue

                # if abstained responses or no unsure claims to verify, save without searching (write, append, track)
                claims_to_verify = [claim for claim in dict_item['claims_lst'] if claim.get('pre-label') == 'unsure']
                if dict_item['abstained'] or claims_to_verify == []:
                    dict_item["token_stats"]['search'] = 0
                    f.write(json.dumps(dict_item) + "\n")
                    searched_claims_lst.append(dict_item)
                    searched_data.add(key)
                    continue
                
                # gather dict_items to search
                active_item.append(dict_item)

            # search by calling search API and web crawler
            with ThreadPoolExecutor(max_workers=self.search_worker_num) as executor:
                futures = []
                for dict_item in active_item: # per response line
                    futures.append(
                        executor.submit(
                            self._search_evidence_single,
                            dict_item
                        )
                    )

                for future in futures:
                    searched_claims = future.result()
                    searched_claims_lst.append(searched_claims)
                    f.write(json.dumps(searched_claims) + '\n')
                    f.flush()

        print(f"Skipped {skip_search_count} samples that already has existing evidences!")
        print(f"Knowledge searching is done! saved to {evidence_output_path}")
        return searched_claims_lst

    def _search_evidence_single(self, sample): # search per response

        claims_lst = sample["claims_lst"]

        # search evidence (searcher's get_content will only update unsure claims with search evidence)
        if claims_lst:
            searched_claims_lst, search_tokens = self.fetch_search.get_content(claims_lst, self.search_res_num)
            sample["token_stats"]['search'] = search_tokens
        else:
            print('no unsure/verifiable claims to search for the response.')
            # directly return the sample if no claims to search
            sample["token_stats"]['search'] = 0
            return sample
        
        # retrieve evidence for verification; update searched_claims_lst
        retriever = BM25_retriever(target_chunk_size=200, 
                                    overlap=30, 
                                    n=self.verify_res_num) #TODO: move to config
        for claim in searched_claims_lst:
            claim_text = claim['claim']
            claim_evidence_list = claim.get('search_results', [])
            # if not unsure claims or claim has no search results, skip retrieval
            if not claim_evidence_list or claim_evidence_list == []:
                continue
            # retrieve and add a new item 'retrieved_search_results' to the current claim dict
            claim['retrieved_search_results'] = retriever.retrieve(claim_text, claim_evidence_list)
        
        # update sample with retrieved and searched claims lst
        sample["claims_lst"] = searched_claims_lst

        return sample

    ######################################## verify claims ##########################################
    def verify_claims(self, searched_claims, verification_output_path):
        """
        Verify the claims against the searched evidence.
        INPUT: searched_claims: jsonl of all responses where each line represents a response dict with the following items:
                {
                "question": str,
                "prompt_source": str,
                "response": str,
                "model": str,
                "abstained": bool,
                (opt)"prelabel_stats": prelabel_stats,
                (opt)"claims_lst" (ordered): [
                                        {
                                        "claim_id": int, "claim": str, "pre-label": str,
                                        (opt)"search_results": [{'title': ..., 'description': ..., 'url': ..., 'content': ..., 'usage': {"token": x}}, {...}, ...],
                                        (opt)"retrieved_search_results": [{'title': str, 'description': str, 'retrieved_text': str}, {...}, ...],
                                        },
                                    ],
                (opt)"token_stats": {
                                    "extract": {
                                            "prompt_tok_cnt": prompt_tok_cnt,
                                            "response_tok_cnt": response_tok_cnt,},
                                    "search": search_tokens,
                                    },
                }
        RETURN: None
        SAVED: verified_claims: updated jsonl where each line represents a response dict with two updated items:
                1. updated (opt) "claims_lst" (ordered): [
                                                            {
                                                            ...,
                                                            (added, opt) "verification_result": {'verification_context': str, 'verification_label': str}
                                                            }
                2. added "response_stats": {
                        "sentences": 1, 
                        "verified_claims": 4, "verified_supported_claims": 4, "verified_unsupported_claims": 0, "verified_inconclusive_claims": 0, 
                        "pre_supported_claims": 9, "pre_unsupported_claims": 4, "pre_verified_claims": 13, 
                        "all_claims": 17, "all_support_claims": 13, 
                        "P": 0.7647058823529411}}
                3. updated 'token_stats': {
                                            "extract": {
                                                        "prompt_tok_cnt": prompt_tok_cnt,
                                                        "response_tok_cnt": response_tok_cnt,
                                                        },
                                            "search": search_tokens,
                                            "verify": {
                                                        "prompt_tok_cnt": prompt_tok_cnt,
                                                        "response_tok_cnt": response_tok_cnt,
                                                        },
                                            }
        """
        verified_data = set()
        verified_claims = []

        # load already verified items
        if os.path.exists(verification_output_path):
            with open(verification_output_path, "r") as f:
                existing_verified_items = [json.loads(line) for line in f]
            for dict_item in existing_verified_items:
                # Create a tuple key to identify already verified items
                if dict_item.get("response_stats") is not None and "verify" in dict_item.get("token_stats", {}):
                    response, prompt_source, model = dict_item["response"].strip(), dict_item["prompt_source"], dict_item["model"]
                    key = (response, prompt_source, model)
                    verified_data.add(key)
                    verified_claims.append(dict_item)
            print(f"Loaded {len(verified_data)} existing verified items")

        # verify
        skip_verify_count = 0
        active_items = []

        # check and rewrite existing/abstained verified items first
        with open(verification_output_path, "w") as f:
            # Write existing verified entries
            for item in verified_claims:
                f.write(json.dumps(item) + "\n")

            for dict_item in tqdm(searched_claims, desc='Claim Verification Progress: '): # per response
                response, prompt_source, model_name = dict_item["response"], dict_item["prompt_source"], dict_item["model"]
                key = (response.strip(), prompt_source, model_name)

                # skip if already verified
                if key in verified_data:
                    skip_verify_count += 1
                    continue

                # skip if abstained response (write, mark as verified)
                if dict_item['abstained']:
                    f.write(json.dumps(dict_item) + "\n")
                    verified_data.add(key)
                    print("Skipping abstained/no claims for verification. Excluded from response_stats results.")
                    continue

                active_items.append(dict_item)
                
            print(f"Skipped {skip_verify_count} samples that already has existing verification!")
            
        # Process new unverified and non-abstained items with append mode
        with open(verification_output_path, "a") as f:
            with ThreadPoolExecutor(max_workers=self.response_worker_num) as executor:
                futures = []
                for dict_item in active_items:
                    futures.append(
                        executor.submit(
                            self._verify_claims_single,
                            dict_item
                        )
                    )

                for future in as_completed(futures):
                    result = future.result()
                    f.write(json.dumps(result) + '\n')
                    f.flush()

            print(f"Claim verification is saved to {verification_output_path}")

    def _verify_claims_single(self, sample): # per response

        # update "claims_lst" with the verification results
        verified_claims_lst, prompt_tok_cnt, response_tok_cnt = self.claim_verifier.verifying_claim(
            sample["claims_lst"], 
            verify_res_num=self.verify_res_num
            )
        sample["claims_lst"] = verified_claims_lst

        # update "token_stats" and "response_stats"
        sample = utils.update_verification_stats(sample, prompt_tok_cnt, response_tok_cnt)

        return sample
    
    def calculate_avg_score(self, verification_output_path, fsfact_score_path, 
                           model_name_extraction, model_name_verification, 
                           total_time, abstain_count, total_responses):
        """
        Calculate and save the avg scores based on the saved verification results:
        Include multiple score metrics:
            P, R, K
            SAFE_F1@K
            FaStfact_F1@K
            FaStfact_Score
        """
        print(f"Calculating and saving FaStfact scores to {fsfact_score_path}")
        
        # group responses based on model and bench (prompt_source)
        bench_model_response_stats_dict = defaultdict(lambda: defaultdict(list))
        bench_model_token_stats_dict = defaultdict(lambda: defaultdict(list))
        with open(verification_output_path, "r") as f:
            for line in f:
                dict_item = json.loads(line)
                # only calculate non-abstained response
                if not dict_item.get('abstained', False):
                    prompt_source = dict_item["prompt_source"]
                    model_name = dict_item["model"]
                    bench_model_response_stats_dict[prompt_source][model_name].append(dict_item["response_stats"])
                    bench_model_token_stats_dict[prompt_source][model_name].append(dict_item["token_stats"])

        
        # initialize score_metric
        score_metric = score_metrics.F1_score(
                                                bench_model_response_stats_dict,
                                                bench_model_token_stats_dict,
                                                fsfact_score_path, 
                                                model_name_extraction, model_name_verification,
                                                total_time, abstain_count, total_responses,
                                                )
        # calculate factuality score, saved to .csv and json
        score_metric.write_avg_numbers()

        print(f"Time used: {total_time}.")
        print(f"Abstained response rate: {abstain_count}/{total_responses}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="FaStfact: Factual Score evaluation for LLM responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m pipeline.FaStfact --input_file data.jsonl
  
  # High-performance setup
  python -m pipeline.FaStfact --input_file data.jsonl --response_worker_num 32 --search_worker_num 6
  
  # Cost-effective setup
  python -m pipeline.FaStfact --input_file data.jsonl --model_name_extraction gpt-4o-mini --model_name_verification gpt-4o-mini
        """
    )
    
    # Input/Output Configuration
    io_group = parser.add_argument_group('Input/Output Configuration')
    io_group.add_argument("--data_dir", type=str, default='./data', 
                         help='Directory containing input data files')
    io_group.add_argument("--input_file", type=str, required=True,
                         help='Input JSONL file containing model responses to evaluate')
    io_group.add_argument("--output_dir", type=str, default='./data',
                         help='Directory to save evaluation results')
    io_group.add_argument("--cache_dir", type=str, default='./data/cache',
                         help='Directory to store cached results for faster re-runs')
    
    # Model Configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--model_name_extraction", type=str, default="gpt-4o-mini",
                            help='LLM model for claim extraction (e.g., gpt-4o, gpt-4o-mini)')
    model_group.add_argument("--model_name_verification", type=str, default="gpt-4o-mini",
                            help='LLM model for claim verification (e.g., gpt-4o, gpt-4o-mini)')
    model_group.add_argument("--use_external_extraction_model", action='store_true',
                            help='Use external API for extraction model')
    model_group.add_argument("--use_external_verification_model", action='store_true',
                            help='Use external API for verification model')
    model_group.add_argument("--use_base_extraction_model", action='store_true',
                            help='Use base model for extraction (no fine-tuning)')
    model_group.add_argument("--use_base_verification_model", action='store_true',
                            help='Use base model for verification (no fine-tuning)')
    
    # Evaluation & Performance Configuration
    eval_group = parser.add_argument_group('Evaluation & Performance Configuration')
    eval_group.add_argument("--label_n", type=int, default=5, choices=[2, 3, 5],
                           help='Number of verification labels (2=supported/unsupported, 3=adds inconclusive, 5=adds uncertain/irrelevant)')
    eval_group.add_argument("--pre_veri_label_m", type=int, default=6, choices=[3, 5, 6],
                           help='Number of pre-verification labels for claim filtering')
    eval_group.add_argument("--stride", type=int, default=0,
                           help='Stride for text chunking (0=use whole response, >0=chunk with overlap)')
    eval_group.add_argument("--search_res_num", type=int, default=5,
                           help='Number of search results to retrieve per claim')
    eval_group.add_argument("--verify_res_num", type=int, default=10,
                           help='Number of retrieved passages used for verification')
    eval_group.add_argument("--do_not_pre_verify", action='store_true',
                           help='Skip pre-verification step (process all claims)')
    eval_group.add_argument("--uncertainty_threshold", type=float, default=None,
                              help='Uncertainty threshold for claim filtering (0 to max entropy log(m))')
    eval_group.add_argument("--token_prob_threshold", type=float, default=0.995,
                              help='Token probability threshold for claim filtering (0 to 1)')
    eval_group.add_argument("--response_worker_num", type=int, default=32,
                           help='Number of parallel workers for response processing')
    eval_group.add_argument("--search_worker_num", type=int, default=3,
                           help='Number of parallel workers for web search (max 40 for Jina API rate limits)')
    
    # System Configuration
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument("--ignore_cache", action='store_true',
                             help='Ignore cached results and recompute everything')
    system_group.add_argument("--print_config", action='store_true',
                             help='Print effective configuration and exit')
    system_group.add_argument("--config", type=str, default=None,
                             help='Path to config file (JSON/YAML) - CLI args override config values')
    args = parser.parse_args()

    # Handle print_config option
    if args.print_config:
        print("FaStfact Configuration:")
        print("=" * 50)
        for group in parser._action_groups:
            if group.title and not group.title.startswith('positional'):
                print(f"\n{group.title}:")
                print("-" * len(group.title))
                for action in group._group_actions:
                    if not action.option_strings:
                        continue
                    value = getattr(args, action.dest)
                    if value is not None:
                        print(f"  {action.option_strings[0]}: {value}")
        print("\n" + "=" * 50)
        exit(0)

    # innitiate
    fsfact = FaStfact(model_name_extraction=args.model_name_extraction,
                    model_name_verification=args.model_name_verification,
                    use_external_extraction_model=args.use_external_extraction_model,
                    use_external_verification_model=args.use_external_verification_model,
                    use_base_extraction_model=args.use_base_extraction_model,
                    use_base_verification_model=args.use_base_verification_model,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    cache_dir=args.cache_dir,
                    label_n=args.label_n,
                    pre_veri_label_m = args.pre_veri_label_m,
                    stride = args.stride,
                    search_res_num=args.search_res_num,
                    verify_res_num=args.verify_res_num,
                    do_not_pre_verify=args.do_not_pre_verify,
                    uncertainty_threshold=args.uncertainty_threshold,
                    token_prob_threshold=args.token_prob_threshold,
                    ignore_cache=args.ignore_cache,
                    search_worker_num = args.search_worker_num,  # RPM is 40 for Jina API. https://jina.ai/reader/#rate-limit
                    response_worker_num = args.response_worker_num,
                    )

    input_file_name = "".join(args.input_file.split('.')[:-1]) # strip off file suffix .jsonl
    input_path = os.path.join(args.data_dir, args.input_file)
    print(f"Processing input file: {input_path}")

    # read jsonl in lists of json
    with open(input_path, "r") as f:
        data = [json.loads(x) for x in f.readlines() if x.strip()]

    # eval pipeline
    fsfact.get_fsfact_score(data, input_file_name)
