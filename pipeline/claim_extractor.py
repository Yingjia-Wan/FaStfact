from concurrent.futures import ThreadPoolExecutor
import os
import regex
import pdb
import json
import spacy
from tqdm import tqdm
from .response_API import GetResponse
from .utils import *
import re


class ClaimExtractor():
    def __init__(self, model_name, label_m, stride=None, cache_dir="./data/cache/", 
                 use_external_model=False, use_base_model=False, do_not_pre_verify=False, uncertainty_threshold=None, token_prob_threshold=None, 
                 ignore_cache=False):
        '''
        Initialize ClaimExtractor for extracting and pre-verifying claims from model responses.
        
        Args:
            model_name (str): Name or path of the model to use for claim extraction.
                Can be a local model path, HuggingFace model name, or API model name.
            label_m (int): Number of few-shot examples to use in the prompt template.
            stride (int, optional): Number of sentences per chunk for processing long responses. If 0, processes the entire response as one chunk.
            cache_dir (str): Directory to store cached responses.
            use_external_model (bool): Whether to load a local model using Unsloth.
            use_base_model (bool): Whether to load a base HuggingFace model directly.
            do_not_pre_verify (bool): Whether to skip pre-verification step in extraction.
            uncertainty_threshold (float, optional): Threshold for uncertainty-based filtering.
            token_prob_threshold (float, optional): Threshold for token probability-based confidence.
            ignore_cache (bool): Whether to ignore existing cache and make new API calls.
        '''
        self.model = None
        self.model_name = model_name
        self.label_m = label_m
        self.stride = stride
        self.do_not_pre_verify = do_not_pre_verify
        self.uncertainty_threshold = uncertainty_threshold
        self.token_prob_threshold=token_prob_threshold

        if os.path.isdir(model_name) or use_external_model:
            from unsloth import FastLanguageModel
            # doc: https://github.com/unslothai/unsloth/blob/e32fc240884435527660bb79a5664a94e27a7576/unsloth/models/loader.py#L70
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=1000,
                dtype=None,
                load_in_4bit=True,
                device_map = "sequential" # Unsloth does not support multi GPU settings currently.
            )
            FastLanguageModel.for_inference(self.model)
            self.model = self.model.to("cuda")
            self.alpaca_prompt = open("./prompt/extraction/extraction_alpaca_template.txt", "r").read()

        elif use_base_model:
            # Load base LLM like meta-llama/Meta-Llama-3-8B-Instruct
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            device_map = get_device_map()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map = device_map
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.alpaca_prompt = open("./prompt/extraction/extraction_alpaca_template.txt", "r").read()

        else:
            # Use gpt or claude
            cache_dir = os.path.join(cache_dir, model_name)
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, f"claim_extraction_cache.json")
            self.get_model_response = GetResponse(cache_file=self.cache_file,
                                                  model_name=model_name,
                                                  max_tokens=1000,
                                                  temperature=0,
                                                  ignore_cache=ignore_cache)
            self.system_message = "You are a helpful assistant who can extract verifiable atomic claims from a piece of text. Each atomic fact should be verifiable against reliable external world knowledge (e.g., via Wikipedia)"

        self.spacy_nlp = spacy.load('en_core_web_sm')

    def run(self, question, response):
        """
        Given a model output (optionally qa output to a question)
        - split the response into sentences using spaCy
        - snippet = question (context1)<SOS>chunk<EOS>(context2)
        - extract facts from each snippet and preverify
        """  
        preverification_stats, prelabel_claims_lst, prompt_tok_cnt, response_tok_cnt = self.extract_using_chunk(response, question, cost_estimate_only=False)
        return preverification_stats, prelabel_claims_lst, prompt_tok_cnt, response_tok_cnt

    def _process_chunk(self, chunk, question, cost_estimate_only):
        '''
        Process a single chunk of text to extract and filter claims.
        
        Args:
            chunk (str): Text chunk to process for claim extraction
            question (str): Original question context (can be empty for non-QA tasks)
            cost_estimate_only (bool): Whether to only estimate costs without API calls
            
        Returns:
            tuple: (snippet, filtered_claims, prompt_tok_num, response_tok_num)
                - snippet (str): Formatted input sent to the model
                - filtered_claims (list): List of extracted claims with pre-labels and metadata
                - prompt_tok_num (int): Number of tokens in the prompt
                - response_tok_num (int): Number of tokens in the model response
                
        Process:
            1. Format the chunk with question context (QA or non-QA format)
            2. Extract claims using fact_extractor
            3. Filter out empty claims and comments
            4. Assign unique claim IDs and clean claim text
            5. Return processed claims with token counts
        '''
        if question == '':
            print("Processing Non-QA generation...")
            snippet = f"Generation: <SOS>{chunk.strip()}<EOS>".strip()
            extracted_claims, prompt_tok_num, response_tok_num = self.fact_extractor(
                snippet, "", qa_input=False, cost_estimate_only=cost_estimate_only
            )
        else:
            snippet = f"Question: {question.strip()}\nResponse: <SOS>{chunk.strip()}<EOS>".strip()
            extracted_claims, prompt_tok_num, response_tok_num = self.fact_extractor(
                snippet, "", qa_input=True, cost_estimate_only=cost_estimate_only
            )

        # Filter claims while maintaining structure
        filtered_claims = [
                            {   "claim_id": get_claim_id(claim_dict["claim"].strip(), index=idx),
                                **{k: v.strip() if k == "claim" else v for k, v in claim_dict.items()},
                            }
                            for idx, claim_dict in enumerate(extracted_claims)
                            if claim_dict["claim"].strip() and not claim_dict["claim"].startswith("Note:")
                        ]

        return (
            snippet,
            filtered_claims,
            prompt_tok_num,
            response_tok_num,
        )

    def extract_using_chunk(self, response, question, cost_estimate_only):
        '''
        Extract and pre-verify claims from a model response by parallel processing it in chunks.
        
        Args:
            response (str): The model's response text to extract claims from
            question (str): The original question that prompted the response.
                Can be empty string for non-QA generation tasks.
            cost_estimate_only (bool): Whether to only estimate costs without making API calls
            
        Returns:
            tuple: (preverification_stats, extracted_claims_lst, prompt_tok_cnt, response_tok_cnt)
                - preverification_stats (dict): Statistics of pre-verification labels.
                    Format: {'supported': int, 'unsupported': int, 'irrelevant': int, 'unsure': int}
                - extracted_claims_lst (list): List of extracted claims with metadata.
                    Each claim dict contains: {"claim_id": int, "claim": str, "pre-label": str, "prelabel_confidence": float}
                - prompt_tok_cnt (int): Total prompt tokens used across all chunks
                - response_tok_cnt (int): Total response tokens generated across all chunks
                
        Process:
            1. Decompose response into chunks based on stride parameter
            2. Process each chunk in parallel using ThreadPoolExecutor
            3. Extract claims and pre-verify them using fact_extractor
            4. Aggregate results and calculate statistics
        '''

        chunks = self.get_chunk(response, self.stride)
        
        prompt_tok_cnt = 0
        response_tok_cnt = 0
        snippet_lst = []
        preverification_stats = {'supported': 0, 'unsupported': 0, 'irrelevant': 0, 'unsure': 0}
        extracted_claims_lst = []
        
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(
                    self._process_chunk,
                    chunk,
                    question,
                    cost_estimate_only,
                )
                for chunk in chunks
            ]

            for future in futures:
                (
                    snippet,
                    extracted_claims,
                    chunk_prompt_tok,
                    chunk_response_tok,
                ) = future.result()

                snippet_lst.append(snippet)
                prompt_tok_cnt += chunk_prompt_tok
                response_tok_cnt += chunk_response_tok
                
                # Aggregate all claims
                extracted_claims_lst.extend(extracted_claims)
                
                # Calculate pre-label stats distribution
                for label in preverification_stats:
                    preverification_stats[label] += sum(1 for c in extracted_claims if c["pre-label"] == label)


        return (
            preverification_stats,
            extracted_claims_lst,
            prompt_tok_cnt,
            response_tok_cnt,
        )

    def get_sentence(self, text):
        # use spaCy to split the text into sentences
        return [x.text.strip() for x in self.spacy_nlp(text).sents]
    
    def get_chunk(self, text, stride):
        '''
        Split text into chunks based on sentence count for processing.
        
        Args:
            text (str): Input text to be chunked
            stride (int): Number of sentences per chunk.
                - 0: Process entire text as single chunk
                - >0: Number of sentences per chunk with sliding window
                
        Returns:
            list: List of text chunks, each containing the specified number of sentences
            
        Note:
            - Uses spaCy for sentence segmentation
            - Empty text returns empty list
            - Chunks are created by joining sentences with spaces
            - Overlapping context can be added in future versions
        '''
        if stride == 0:
            return [text] # feed the extractor the whole response 
            #TODO: handle the edge case if response exceeds the context length

        chunks = []
        sentences = self.get_sentence(text)
        if not sentences:
            return []
        for i in range(0, len(sentences), stride):
            # TODO: add overlapping context and SOS/EOS here instead of in prompt
            chunk = sentences[i:i + stride]
            chunks.append(' '.join(chunk))
        return chunks

    def get_prompt_template(self, qa_input):
        if self.do_not_pre_verify:
            if qa_input:
                prompt_template = open(f"./prompt/extraction/qa_template_noPreverify.txt", "r").read()
            else:
                # TODO
                print("Prompt template for non-qa not-pre-verify is not in the folder. We will extend the evaluation scope to nonQA soon.")
        else:
            if qa_input:
                prompt_template = open(f"./prompt/extraction/m={self.label_m}_qa_template.txt", "r").read()
            else:
                # TODO
                print("Prompt template for non-qa pre-verify is not in the folder. We will extend the evaluation scope to nonQA soon.")

        return prompt_template

    def fact_extractor(self, snippet, sentence, qa_input=False, cost_estimate_only=False):
        """
        Extract factual claims from a text snippet using either a local model 
        or an API-based model. Returns structured claims with preliminary labels 
        (supported / unsupported / irrelevant / unsure) and optional confidence scores.

        Args:
            snippet (str): Text snippet to analyze.
            sentence (str): Sentence context (unused placeholder).
            qa_input (bool, optional): Use QA-style prompt for API models. Defaults to False.
            cost_estimate_only (bool, optional): If True, only estimates token cost. Defaults to False.

        Returns:
            tuple: (claims, prompt_tok_cnt, response_tok_cnt)
                - claims (list[dict]): Each item contains:
                    {"claim": str, "pre-label": str, "prelabel_confidence": float or None}
                - prompt_tok_cnt (int): Number of prompt tokens used.
                - response_tok_cnt (int): Number of response tokens used.
        """
        if self.model:
            formatted_input = self.alpaca_prompt.format(snippet, "")
            inputs = self.tokenizer(formatted_input, return_tensors="pt").to("cuda")

            outputs = self.model.generate(**inputs, max_new_tokens=1000, use_cache=True)
            output_str = ' '.join(self.tokenizer.batch_decode(outputs))
            
            clean_output = output_str.split("### Response:")[-1].strip().replace("</s>", "")

            if not clean_output or "no verifiable claim" in clean_output.lower():
                return [], 0, 0

            claims = [x.strip() for x in clean_output.split("\n")]
            
            extract_claims_lst = []

            for claim in claims:
                cleaned_claim = re.sub(r'###[^#]+###', '', claim).strip()

                if claim.endswith("###TRUE###"):
                    extracted_claim_with_prelabel = {"claim": cleaned_claim, "pre-label": "supported"}
                elif claim.endswith("###FALSE###"):
                    extracted_claim_with_prelabel = {"claim": cleaned_claim, "pre-label": "unsupported"}
                elif claim.endswith("###IRRELEVANT###"):
                    extracted_claim_with_prelabel = {"claim": cleaned_claim, "pre-label": "irrelevant"}
                else:
                    extracted_claim_with_prelabel = {"claim": cleaned_claim, "pre-label": "unsure"}

                # add the claim with label to the extract_claims_lst
                extract_claims_lst.append(extracted_claim_with_prelabel)

            return extract_claims_lst, 0, 0

        else:
            ### Prompting base approach via API call
            prompt_template = self.get_prompt_template(qa_input)
            prompt_text = prompt_template.format(snippet=snippet)

            # get response
            response, logprobs, prompt_tok_cnt, response_tok_cnt = self.get_model_response.get_response(self.system_message,
                                                                                            prompt_text,
                                                                                            cost_estimate_only,
                                                                                            self.uncertainty_threshold,
                                                                                            self.token_prob_threshold)

            if not response or "no verifiable claim" in response.lower():
                return [], prompt_tok_cnt, response_tok_cnt
            else:
                extract_claims_lst = []

                # judge whether to enable logprobs checking
                if logprobs is not None:
                    # decompose the response, logprobs into (clean) claims with labels, claim_logprobs
                    claims, claims_logprobs = decompose_claims_and_logprobs(response, logprobs)
                    use_logprobs = True if len(claims) == len(claims_logprobs) and self.token_prob_threshold is not None else False
                    #TODO: report error using try exception
                else:
                    use_logprobs = False

                # no logprobs confidence checking
                if not use_logprobs:
                    # decompose response into clean claims with labels
                    claims = clean_claim_format(response)

                    # group based on labels
                    for claim in claims:
                        cleaned_claim = re.sub(r'###[^#]+###', '', claim).strip()

                        if claim.endswith("###TRUE###"):
                            extracted_claim_with_prelabel = {"claim": cleaned_claim, "pre-label": "supported", "prelabel_confidence": None}
                        elif claim.endswith("###FALSE###"):
                            extracted_claim_with_prelabel = {"claim": cleaned_claim, "pre-label": "unsupported", "prelabel_confidence": None}
                        elif claim.endswith("###IRRELEVANT###"):
                            extracted_claim_with_prelabel = {"claim": cleaned_claim, "pre-label": "irrelevant", "prelabel_confidence": None}
                        else:
                            extracted_claim_with_prelabel = {"claim": cleaned_claim, "pre-label": "unsure", "prelabel_confidence": None}

                        # add the claim with label to the extract_claims_lst
                        extract_claims_lst.append(extracted_claim_with_prelabel)

                # confidence checking
                else: 
                    for claim, claim_logprobs in zip(claims, claims_logprobs):
                        # Collect all_claims
                        cleaned_claim = re.sub(r'###[^#]+###', '', claim).strip()
                        is_confident = True

                        # if response logpobs available, check confidence
                        if claim_logprobs is not None:
                            # Calculate target probability for TRUE/FALSE; return None for other labels
                            target_prob = calculate_uncertainty(claim, claim_logprobs, self.model_name)
                            # print('target_prob: ', target_prob)

                            # if label not in [TRUE/FALSE] => target_prob is None; automatically unsure/irrelevant
                            if target_prob is None:
                                if claim.endswith("###IRRELEVANT###"):
                                    extracted_claim_with_prelabel = {"claim": cleaned_claim, "pre-label": "irrelevant", "prelabel_confidence": None}
                                else:
                                    # Add to unsure_claims: (LIKELY TRUE, LIKELY FALSE, UNSURE)
                                    extracted_claim_with_prelabel = {"claim": cleaned_claim, "pre-label": "unsure", "prelabel_confidence": None}
                                continue
                            # if label in [TRUE/FALSE] => check confidence vs threshold
                            else:
                                # print("target_prob < token_prob_threshold: ", target_prob < self.token_prob_threshold)
                                if target_prob < self.token_prob_threshold:
                                    is_confident = False

                        # Initialize claim dict with prelabel_confidence (only for labels in [TRUE, FALSE])
                        extracted_claim_with_prelabel = {"claim": cleaned_claim, "prelabel_confidence": round(target_prob, 6) if claim_logprobs is not None else None}

                        # checked logprob-based confidence -> lower than threshold
                        if is_confident:
                            if claim.endswith("###TRUE###"):
                                extracted_claim_with_prelabel.update({"pre-label": "supported"})
                            elif claim.endswith("###FALSE###"):
                                extracted_claim_with_prelabel.update({"pre-label": "unsupported"})
                            elif claim.endswith("###IRRELEVANT###"):
                                extracted_claim_with_prelabel.update({"pre-label": "irrelevant"})
                        else:
                            extracted_claim_with_prelabel.update({"pre-label": "unsure"})
                        
                        # add the claim with label to the extract_claims_lst
                        extract_claims_lst.append(extracted_claim_with_prelabel)

                return extract_claims_lst, prompt_tok_cnt, response_tok_cnt
