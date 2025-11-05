from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pdb
import json
import re
from .response_API import GetResponse
from .utils import get_device_map


class ClaimVerifier():
    def __init__(self, model_name, label_n=5, cache_dir="./data/cache/",
                 use_external_model=False, use_base_model=False,
                 ignore_cache=False):
        self.demos_path = f"./prompt/verification/jina_search/few_shot_examples_{label_n}labels.jsonl"
        self.model = None
        self.model_name = model_name
        self.label_n = label_n
        self.ignore_cache = ignore_cache
        if os.path.isdir(model_name) or use_external_model:
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
                device_map = "sequential" # Unsloth does not support multi GPU settings currently.
            )
            FastLanguageModel.for_inference(self.model)
            self.tokenizer.padding_side = "left"
            self.alpaca_prompt = open("./prompt/verification/verification_alpaca_template.txt", "r").read()
            self.instruction = open("./prompt/verification/verification_instruction_binary_no_demo.txt", "r").read()
        elif use_base_model:
            # Load base LLM like meta-llama/Meta-Llama-3-8B-Instruct
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            device_map = get_device_map()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map = device_map # Unsloth does not support multi GPU settings currently.
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.alpaca_prompt = open("./prompt/verification/verification_alpaca_template.txt", "r").read()
            self.instruction = open("./prompt/verification/verification_instruction_binary_no_demo.txt", "r").read()
        else:
            # Use gpt or claude
            cache_dir = os.path.join(cache_dir, model_name)
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, "claim_verification_cache.json")
            self.get_model_response = GetResponse(cache_file=self.cache_file,
                                                  model_name=model_name,
                                                  max_tokens=1000,
                                                  temperature=0,
                                                  ignore_cache=self.ignore_cache)
            self.system_message = "You are a helpful assistant who can verify the truthfulness of a claim against reliable external world knowledge."
            self.prompt_initial_temp = self.get_initial_prompt_template()

    def get_instruction_template(self):
        # Determine the base path based on the model name
        base_path = "./prompt/verification/"

        # model
        if "claude" in self.model_name:
            base_path += "verification_instruction_claude_"
        else:
            base_path += "jina_search/verification_instruction_"

        # label_n
        suffix = f"{self.label_n}labels_reasoning.txt"

        # Construct the full path and read the template
        prompt_temp_path = base_path + suffix
        return open(prompt_temp_path, "r").read()

    def get_initial_prompt_template(self):
        prompt_temp = self.get_instruction_template()
        with open(self.demos_path, "r") as f:
            example_data = [json.loads(line) for line in f if line.strip()]
        element_lst = []
        for dict_item in example_data:
            claim = dict_item["claim"]
            search_result_str = dict_item["search_result"]
            reasoning_for_label = dict_item["reasoning_for_label"]
            human_label = dict_item["human_label"]

            if "claude" in self.model_name:
                element_lst.extend([search_result_str, claim, human_label])
            else:
                # openai
                element_lst.extend([claim, search_result_str, reasoning_for_label, human_label])

        prompt_few_shot = prompt_temp.format(*element_lst)

        if "claude" in self.model_name:
            self.your_task = "Your task:\n\n{search_results}\n\nClaim: {claim}\n\nTask: Given the search results above, is the claim supported or unsupported? Mark your decision with ### signs.\n\nYour decision:"
        else:
            # openai
            self.your_task = "Your task:\n\nClaim: {claim}\nSearched Results: {search_results}\nReasoning: \nDecision:"

        """
        choose system message
        """
        if self.label_n == 2: # support / unsupport
            self.system_message = "You are a helpful assistant who can judge whether a claim is supported by the searched evidence or not."
        elif self.label_n == 3: # support / refute / inconclusive
            self.system_message = "You are a helpful assistant who can judge whether a claim is supported or refuted by the searched evidence, or whether there is not enough information in the searched evidence to make a judgement."
        elif self.label_n == 5: # support / refute / not enough evidence / conflicting evidence / unverifiable
            self.system_message = "You are a helpful assistant who can judge whether a claim is supported or refuted by the searched evidence, or whether there is conflicting or not enough information in the searched evidence to make a judgement."
        return prompt_few_shot
    
    def _verify_single_claim(self, claim, verify_res_num=10, cost_estimate_only=False):
        '''
        For each claim dict with prelabel 'unsure' in claims_lst for a response, verify it using search evidence of the claim.
        Add new item to the claim dict:
            (added, opt) "verification_result": {'verification_context': str, 'verification_label': str}
        Args:
            claim (dict): Claim dictionary with optional search results.
            verify_res_num (int, optional): Max number of search results to use. Defaults to 10.
            cost_estimate_only (bool, optional): If True, only estimates token cost. Defaults to False.

        Returns:
            tuple: (updated_claim, prompt_tok_cnt, response_tok_cnt)
        '''
        prompt_tok_cnt, response_tok_cnt = 0, 0
        
        # format evidence context for prompting the verifier
        search_res_str = ""
        search_cnt = 1
        if claim.get("retrieved_search_results", []) != []:
            for search_dict in claim["retrieved_search_results"][:verify_res_num]:
                # if no web-crawled content available, fall back to description snippet
                content = search_dict['retrieved_text'].strip() or search_dict['description'].strip()
                search_res_str += (
                    f"Evidence {search_cnt}\n"
                    f"Source Title: {search_dict['title'].strip()}\n"
                    f"Content: {content}\n\n"
                )
                search_cnt += 1
        claim_text = claim["claim"]

        if self.model:
            usr_input = f"Claim: {claim_text.strip()}\n\n{search_res_str.strip()}"
            formatted_input = self.alpaca_prompt.format(self.instruction, usr_input)

            inputs = self.tokenizer(formatted_input, return_tensors="pt").to("cuda")
            output = self.model.generate(
                **inputs,
                max_new_tokens=500,
                use_cache=True,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ],
                pad_token_id=self.tokenizer.eos_token_id,
            )
            response = self.tokenizer.batch_decode(output)
            clean_output = (
                " ".join(response)
                .split("<|end_header_id|>\n\n")[-1]
                .replace("<|eot_id|>", "")
                .replace("<|endoftext|>", "")
                .strip()
            )

        else: # GPT
            prompt_tail = self.your_task.format(
                claim=claim_text, search_results=search_res_str.strip()
            )
            prompt = f"{self.prompt_initial_temp}\n\n{prompt_tail}"
            prompt = prompt.replace("<|endoftext|>", "").strip()

            response, logprobs, p_toks, r_toks = self.get_model_response.get_response(
                self.system_message,
                prompt,
                cost_estimate_only=cost_estimate_only,
                uncertainty_threshold=None,
                token_prob_threshold=None,
            )
            # TODO: also use logprobs threshold for verification.
            
            prompt_tok_cnt += p_toks
            response_tok_cnt += r_toks

            # parse response (allow a list of label headers to ensure label extraction)
            headers = ["Decision:", "Final label:", "Label:", "Verification:", "Conclusion:", "Verdict:"]
            for header in headers:
                reasoning_part, _, decision_part = response.partition(header)
                if _:
                    break
            
            # extract reasoning part
            clean_reasoning = reasoning_part.replace("Reasoning:", "").strip() or response.strip()

            # Extract label
            label_match = re.search(r'###\s*([^#]+?)\s*###', decision_part or response)
            if label_match:
                clean_label = label_match.group(1).strip().lower().replace('.', '')
            else:
                print(f"Warning: Verification Label not found in the verifier response: \n{response}")
                clean_label = "invalid_label"

        claim["verification_result"] = {
            "verification_context": search_res_str,
            "reasoning_for_verification_label": clean_reasoning,
            "verification_label": clean_label,
        }
        return claim, prompt_tok_cnt, response_tok_cnt


    def verifying_claim(self, searched_claims, verify_res_num=10, cost_estimate_only=False):
        """
        Verify claims using the search results.
        Args:
            searched_claims: (list[dict]) (searched claims for a response, in order): 
                                    [
                                        {
                                        "claim_id": int, "claim": str, "pre-label": str,
                                        (opt)"search_results": [{'title': ..., 'description': ..., 'url': ..., 'content': ..., 'usage': {"token": x}}, {...}, ...],
                                        (opt)"retrieved_search_results": [{'title': str, 'description': str, 'retrieved_text': str}, {...}, ...],
                                        },
                                        ...
                                    ],
        RETURN:
            tuple: (verified_claims_lst: a new list by adding new item to searched_claims: [
                                                            {
                                                            ...,
                                                            (added, opt) "verification_result": {'verification_context': str, 'verification_label': str},
                                                            }
                                                            ...
                                                        ]
                    , prompt_tok_cnt, response_tok_cnt)
        """
        total_prompt_tok_cnt = 0
        total_response_tok_cnt = 0
        verified_claims = {}  # Use dict to track by claim_id
        num_verifier_workers = max(1, len(searched_claims)) if searched_claims else 1

        with ThreadPoolExecutor(max_workers=num_verifier_workers) as executor:
            # Create futures with claim_id tracking
            future_to_id = {}
            for claim in searched_claims:
                if claim.get('pre-label') == "unsure":
                    future = executor.submit(
                        self._verify_single_claim,
                        claim.copy(),
                        verify_res_num,
                        cost_estimate_only,
                    )
                    future_to_id[future] = claim['claim_id']

            # Process results as they complete
            for future in as_completed(future_to_id.keys()):
                try:
                    verified_claim, prompt_tok_num, response_tok_num = future.result()
                    verified_claims[future_to_id[future]] = verified_claim
                    total_prompt_tok_cnt += prompt_tok_num
                    total_response_tok_cnt += response_tok_num
                except Exception as e:
                    print(f"Verification failed in ClaimVerifier for claim: {e}")
                    
        # Preserve original order using list comprehension
        final_claims = [
            verified_claims.get(claim['claim_id'], claim)
            for claim in searched_claims
        ]

        return final_claims, total_prompt_tok_cnt, total_response_tok_cnt