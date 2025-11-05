'''
Using s.jina.ai for web search. See jina reader website and repo for more information.

Parameters in the script (better to move):
- max_workers
- max_retries
- timeout

error info: https://jina.ai/serve/concepts/orchestration/handle-exceptions/
retry mechanism:
1. retriable errors: (HTTP Status Codes:)
    524 (Timeout): The server took too long to respond. This might happen if the service is under heavy load or experiencing a temporary issue.
    503 (Service Unavailable): The server is temporarily unable to handle the request. This could be due to maintenance or overload.
    429 (Too Many Requests): The server is rate-limiting you. Retrying after some time (respecting the Retry-After header, if present) is usually recommended.
    422(FailedNavigation): Some webpages are unaccessible by Jina. "AssertionFailureError","status":42206,"message":"Failed to goto xxx: TimeoutError: Navigation timeout of 30000 ms.
2. non-retriable error:
    those on the client or sever side which a retry cannot fix. e.g., :
    400 (Bad Request): The request is malformed (e.g., invalid URL or parameters). Fix the query or input.
    401 (Unauthorized): The API key or credentials are missing or invalid. Check authentication details.
    403 (Forbidden): Access is denied, possibly due to lack of permissions or exceeding quotas. Verify your access rights or contact support.
    404 (Not Found): The requested resource doesn't exist. Verify the endpoint or query.
    500 (Internal Server Error): Indicates a problem with the server itself. While this might resolve with a retry, it often requires contacting support or waiting for a server fix.
'''
# TODO: clean code; cache handling; move parameters to be configurable

import os
import json
import requests
from dotenv import load_dotenv
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
import time


class WebSearchAPI():
    def __init__(self, ignore_cache=False):

        load_dotenv()
        self.jina_key = os.getenv("JINA_KEY")
        self.serper_key = os.getenv("SERPER_KEY")
        
        # define URL and headers for the API request
        self.url_prefix = 'https://s.jina.ai/'
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.jina_key}',
            'X-Token-Budget': '20000', # the total token budget for all search results of one query
            'X-Remove-Selector': 'header, .class, #id', # remove page headers, etc in SERP
            'X-Retain-Images': 'none', # remove all images in SERP
            'X-Return-Format': 'text', # return only text with markdowns and htmls removed (the downside is the output is relatively unorganized and LLM-unfriendly.)
            'X-Timeout': '150',
            'X-With-Generated-Alt': 'true'
        }
        
        # set parameters for retries and multiprocessing
        self.max_retries = 10
        self.max_workers = 5 # TODO: please also set search max_workers in main script argument: search_worker_num.
        self.lock = threading.Lock()  # For thread-safe cache operations

        # innitialize cache related variables
        self.ignore_cache = ignore_cache
        self.cache_file = "data/cache/web_search_cache.json"
        self.cache_dict = self.load_cache()
        self.add_n = 0
        self.save_interval = 10

    def encode_url(self, query, search_res_num):
        '''
        Encode query and construct Jina API URL.
        
        Args:
            query (str): Search query to encode
            search_res_num (int): Number of results to request
            
        Returns:
            str: Complete API URL with encoded parameters
        '''
        url_prefix = 'https://s.jina.ai/'
        encoded_query = quote_plus(query)
        url = url_prefix + f'?q={encoded_query}&num={search_res_num}'
        return url

    def get_content(self, claim_lst, search_res_num):
        '''
        Search for evidence for 'unsure' claims and return updated claims with search results.
        
        Args:
            claim_lst (list): List of claims with pre-labels {'unsure', 'supported', 'unsupported', 'irrelevant'}
            search_res_num (int): Number of search results to retrieve per claim
            
        Returns:
            tuple: (updated_claims, total_search_tokens)
                - updated_claims: Original claims with 'search_results' added for 'unsure' claims
                - total_search_tokens: Total tokens consumed across all searches
        '''
        futures = []
        search_tokens = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks only for "unsure" claims
            for claim in claim_lst:
                if claim['pre-label'] == 'unsure':
                    future = executor.submit(
                        self.get_search_res, 
                        claim['claim'], 
                        search_res_num
                    )
                    # Store the future and its corresponding claim, so that we can update the claim later
                    futures.append((future, claim))

        # Process results and update claims (for 'unsure' claims)
        for future, claim in futures:
            try:
                content_res = future.result()
                # Add search results directly to the claim
                searched_results_per_claim = content_res.get('data', [])
                claim['search_results'] = searched_results_per_claim

                for result in searched_results_per_claim:
                    search_tokens += result.get('usage', {}).get('tokens', 0)
                
            except Exception as e:
                print(f"Error processing claim '{claim['claim_id']}': {e}")
                claim['search_results'] = []  # Ensure key exists even on failure

        # Return the modified list with all claims
        return claim_lst, search_tokens


    def get_search_res(self, query, search_res_num):
        '''
        Get search results from Jina API for a single claim.
        
        Args:
            query (str): Claim string to search for
            search_res_num (int): Number of search results to return
            
        Returns:
            dict: Search results in format {'data': [{'title': str, 'description': str, 'url': str, 'content': str, 'usage': {'tokens': int}}]}
        '''
        # Thread-safe cache check
        if not self.ignore_cache:
            cache_key = query.strip()
            with self.lock:
                if cache_key in self.cache_dict:
                    return self.cache_dict[cache_key]

        # search in max_retries
        retries = 0
        while retries < self.max_retries:
            try:
                # Encode and send the request
                encoded_url = self.encode_url(query, search_res_num)
                response = requests.get(encoded_url, headers=self.headers, timeout=180)

                # Success
                if response.status_code == 200:
                    content_json = response.json()

                    if not self.ignore_cache:
                        with self.lock:
                            # Update cache
                            self.cache_dict[cache_key] = content_json
                            self.add_n += 1
                            # Save cache periodically
                            if self.add_n % self.save_interval == 0:
                                self.save_cache()
                    
                    return content_json
                
                # Error
                elif response.status_code in [503, 524]:
                    sleep_time = random.uniform(1, 3) * (retries + 1)  # Linear backoff
                    print(f"Retryable error {response.status_code} for '{query}'. Sleeping for {sleep_time:.1f} seconds... (Attempt {retries + 1}/{self.max_retries})")
                    time.sleep(sleep_time)
                    retries += 1
                    continue
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    retries += 1
                    print(f"Rate limit hit for '{query}'. Retrying after {retry_after} seconds...")
                    # sleep and retry, until reach max_retries
                    time.sleep(retry_after)
                    continue
                else:
                    print(f"Non-retryable error for query '{query}': {response.status_code}")
                    if response.status_code == 422:
                        print(f"The site page is likely unaccesible by Jina's web crawler.\
                                If the error persists for many queries, double check your search API request code and timeout configuration.")
                    # TODO: optional config to fall back to serper search if jina error
                    break
            except requests.exceptions.Timeout:
                print(f"Timeout error for query '{query}'. Retrying...")
                retries += 1
                sleep_time = (2 ** retries) * random.uniform(0.5, 1.5)
                time.sleep(min(sleep_time, 60))
            except requests.exceptions.RequestException as e:
                print(f"Request exception for query '{query}': {e}")
                break
                
            retries += 1
            # backoff_time = random.uniform(0, min(2 ** retries, 30))
            # time.sleep(backoff_time)

        # Log and cache failure
        print(f"Max retries exceeded for '{query}'. Returning empty result. Please consider decreasing search_worker_num.")
        with self.lock:
            self.cache_dict[cache_key] = {'data': []}
        return {'data': []}

    def save_cache(self):
        '''
        Save current cache to file, merging with any updates from parallel processes.
        '''
        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        cache = self.load_cache().items()
        for k, v in cache:
            self.cache_dict[k] = v

        with open(self.cache_file, "w") as f:
            json.dump(self.cache_dict, f, indent=4)

    def load_cache(self):
        '''
        Load cache from file or return empty dict if file doesn't exist.
        
        Returns:
            dict: Cached search results
        '''
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                cache = json.load(f)
        else:
            cache = {}
        return cache

