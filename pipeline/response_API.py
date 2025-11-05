import os
import json
import tiktoken
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import logging
import threading
import fcntl

class GetResponse():
    def __init__(self, cache_file, model_name="gpt-4-0125-preview", max_tokens=1000, temperature=0, ignore_cache=False):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_file = cache_file
        self.ignore_cache = ignore_cache
        self.lock = threading.Lock()  # For thread-safe cache operations
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

        # get model keys
        load_dotenv()
        if "gpt" in model_name:
            self.key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_BASE_URL")
            if not self.key:
                raise ValueError("API key not found in environment variables")
            self.client = OpenAI(api_key=self.key)
            self.seed = 1130
        elif "claude" in model_name:
            self.key = os.getenv("CLAUDE_API_KEY")
            self.client = Anthropic(api_key=self.key)

        # cache related
        if self.ignore_cache:
            self.cache_dict = {}
        else:
            self.cache_dict = self.load_cache()
        self.add_n = 0
        self.save_interval = 1
        self.print_interval = 20

    # Returns the response from the model given a system message and a prompt text.
    def get_response(self, system_message, prompt_text, cost_estimate_only=False, uncertainty_threshold=None, token_prob_threshold=None):
        prompt_tokens = len(self.tokenizer.encode(prompt_text))
        if cost_estimate_only:
            response_tokens = 0
            return None, None, prompt_tokens, response_tokens

        # check if prompt is in cache; if so, return from cache
        if not self.ignore_cache:
            cache_key = prompt_text.strip()
            if cache_key in self.cache_dict:
                cached_response, cached_logprobs_data = self.cache_dict[cache_key]
                return cached_response, cached_logprobs_data, 0, 0

        if "gpt" in self.model_name:
            message = [{"role": "system", "content": system_message},
                        {"role": "user", "content": prompt_text}]
            response_params = {
                                "model": self.model_name,
                                "messages": message,
                                "max_completion_tokens": self.max_tokens,
                                "temperature": self.temperature,
                                "seed": self.seed,
                            }
            
            # if no logprob_threshold is set: skip using the param logprobs because some models do not have the param
            if uncertainty_threshold is not None or token_prob_threshold is not None:
                response_params["logprobs"] = True
            try:
                response = self.client.chat.completions.create(**response_params)
                response_content = response.choices[0].message.content.strip()

                if "logprobs" in response_params:
                    # Convert logprobs to a serializable format (dict) to later save in cache
                    logprobs_data = {
                        'tokens': [token.token for token in response.choices[0].logprobs.content],
                        'logprob': [token.logprob for token in response.choices[0].logprobs.content],
                    } if response.choices[0].logprobs else None
                else:
                    logprobs_data = None
            except Exception as e:
                logging.error(f"Unexpected error: {e}. Skipping this sample.")
                return "", None, 0, 0

        elif "claude" in self.model_name:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            response_content = response.content[0].text.strip()
            logprobs_data = None

        if not self.ignore_cache:
            # update cache
            self.cache_dict[cache_key] = (response_content.strip(), logprobs_data)
            self.add_n += 1

            # save cache every save_interval times
            if self.add_n % self.save_interval == 0:
                with self.lock:
                    self.save_cache()

            if self.add_n % self.print_interval == 0:
                print(f"Saving # {self.add_n} cache to {self.cache_file}...")

        response_tokens = len(self.tokenizer.encode(response_content))
        return response_content, logprobs_data, prompt_tokens, response_tokens

    # Returns the number of tokens in a text string.
    def tok_count(self, text: str) -> int:
        num_tokens = len(self.tokenizer.encode(text))
        return num_tokens

    def save_cache(self):
        """
        Save cache atomically with proper error handling and file locking.
        """
        try:
            temp_path = self.cache_file + '.tmp'
            current_cache = self.load_cache()
            self.cache_dict.update(current_cache)
            
            # Write to temporary file first
            with open(temp_path, 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # File lock (Unix)
                json.dump(self.cache_dict, f, indent=4, ensure_ascii=False)
                f.flush()  # Ensure all data is written
                os.fsync(f.fileno())  # Force write to disk
            
            # Atomic rename (this is the only operation that can't be interrupted)
            os.replace(temp_path, self.cache_file)
                
        except (IOError, OSError, json.JSONDecodeError) as e:
            logging.error(f"Failed to save cache: {str(e)}")
            # Clean up temporary file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        except Exception as e:
            logging.error(f"Unexpected error saving response cache: {str(e)}")

    def load_cache(self):
        """
        Load cache with robust error handling and file locking.
        """
        cache = {}
        if not os.path.exists(self.cache_file):
            return cache
            
        try:
            with open(self.cache_file, 'r') as f:
                fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock for reading
                try:
                    cache = json.load(f)
                except json.JSONDecodeError as e:
                    logging.error(f"Cache file corrupted, starting fresh: {str(e)}")
                    return {}
        except (IOError, OSError) as e:
            logging.error(f"Error reading cache file: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error loading cache: {str(e)}")
            
        return cache
