from collections import defaultdict
import re
import numpy as np
import hashlib
import nltk

############################################## Response Text Formatting ##############################################
def clean_claim_label(claim, suffixes):
    """
    Clean the claim by removing any of the specified suffixes if the claim ends with one of them.
    
    Args:
        claim (str): The claim to clean.
        suffixes (list): A list of suffixes to check against.
        
    Returns:
        str or None: The cleaned claim if it matches a suffix, otherwise None.
    """
    for suffix in suffixes:
        if claim.endswith(suffix):
            return claim.replace(suffix, "").strip()
    return None

def is_abstain_response(response):
    abstain_responses = [
        "i am sorry",
        "i'm sorry",
        "i apologize",
        "i'm afraid",
        "i am afraid",
        "my apologies",
        "sorry, but i can't",
        "sorry, but i cannot",
        "sorry, but i can not",
        "sorry, i can't",
        "sorry, i can not",
        "sorry, i cannot",
        "sorry, i do not",
        "sorry, i don't",
        "sorry, i am not",
        "sorry, i'm not",
        "i am unable to",
        "i'm unable to",
        "i am not able to",
        "i'm not able to"
    ]
    normalized_response = response.lower()
    # if model abstains from answering or generate empty response:
    if normalized_response.strip() == '' or any(phrase in normalized_response for phrase in abstain_responses):
        return True
    return False


def clean_claim_format(response):
    '''
    remove noisy formats in response to filter out claims; remove duplicate claims
    '''
    # Remove "Facts:" or "Claims:" line
    response = re.sub(r'^Facts:|^Claims:', '', response, flags=re.IGNORECASE | re.MULTILINE).strip()
    
    # Remove itemized list markers and split into individual claims
    claims = [x.strip().replace("- ", "") for x in response.split("\n")]
    
    # Remove numbers at the beginning of each claim
    claims = [re.sub(r"^\d+\.?\s", "", x) for x in claims]
    
    # Filter out any empty strings
    claims = [claim for claim in claims if claim]
    
    # Remove <SOS> and <EOS> if they exist
    claims = [re.sub(r'<SOS>|<EOS>', '', claim).strip() for claim in claims if claim]
    
    # Remove duplicate claims while preserving order
    seen = set()
    unique_claims = []
    for claim in claims:
        if claim not in seen:
            seen.add(claim)
            unique_claims.append(claim)
    return unique_claims

def decompose_claims_and_logprobs(response, logprobs):
    """
    Decompose the response into claims (using clean_claim_format) and group logprob tokens into a list of dictionaries.
    Each dictionary maps token strings (as they appear in that claim) to a list of their logprobs.
    A claim is assumed to end when a token containing the label marker "###" is encountered.
    """
    # Get the list of claims
    claims = clean_claim_format(response)
    
    tokens = logprobs["tokens"]
    lps = logprobs["logprob"]
    
    claims_logprobs = []      # List to store logprobs per claim (as dictionaries)
    current_claim_tokens = [] # Tokens for the current claim
    current_claim_logprobs = [] # Corresponding logprobs
    
    for token, lp in zip(tokens, lps):
        current_claim_tokens.append(token)
        current_claim_logprobs.append(lp)
        
        # Check if this token signals the end of a claim (contains "###")
        if "###\n" in token:
            claim_dict = {}
            # For each token in the current claim, append its logprob in a list
            for t, p in zip(current_claim_tokens, current_claim_logprobs):
                claim_dict.setdefault(t, []).append(p)
            claims_logprobs.append(claim_dict)
            
            # Reset for the next claim
            current_claim_tokens = []
            current_claim_logprobs = []
    
    # If any tokens remain that didn't end with a label token, add them as a claim. (the last claim does not end with ###\n but ###)
    if current_claim_tokens:
        claim_dict = {}
        for t, p in zip(current_claim_tokens, current_claim_logprobs):
            claim_dict.setdefault(t, []).append(p)
        claims_logprobs.append(claim_dict)
    
    return claims, claims_logprobs

def check_token_logprobs(claim, logprobs_data):
    '''
    returns the logprob for the label token.
    if logprobs_data has not been collected or if no T/F label is found, return 0 -> it always passes the threshold.
    '''
    # if no logprobs data from the model:
    if logprobs_data is None:
        return 0
        
    # Extract the token to check (either "TRUE" or "FALSE")
    match = re.search(r'###(TRUE|FALSE)###', claim)
    if not match:
        return 0  # Return 0 if no match is found
    target_token = match.group(1)
    
    # find the logprob of the target token
    for token, logprob in zip(logprobs_data['tokens'], logprobs_data['logprob']):
        if token == target_token:
            return logprob
    return 0  # Return 0 if token not found

def calculate_uncertainty(claim, claim_logprobs, model):
    '''
    a claim_logprobs is a dict of {token: a list of logprobs for the token}, as there can be duplicates tokens in a claim; 
    we get the last logprob as that is for the label token.
    For labels:
    ['TRUE', 'FALSE'] -> return target_logprob
    ['UNSURE', 'IRRELEVANT', 'LIKELY TRUE', 'LIKELY FALSE'] -> return None
    others -> return None
    '''
    if claim_logprobs is None:
        print('no logprobs from the model.')
        return None

    # Extract the label from the claim
    label_match = re.search(r'###([^#]+)###', claim)
    if not label_match:
        return None  # No label found
    label = label_match.group(1).strip().upper()
    
    # Check if the label is one of the possible labels
    if label not in ["TRUE", "FALSE"]:
        return None
    
    # sanity check
    if label not in claim_logprobs:
        print('label not in logprobs[tokens]')
        return None
    
    # Get the log probability of the first token
    target_logprob = claim_logprobs[label][-1]
    target_prob = np.exp(target_logprob)  # Convert log probability to probability

    # print('claim:', claim, ' | ', 'extracted label:', label, ' | ', 'model prob:', target_prob)
    return target_prob

############################################## Stats Saving ##############################################
def update_verification_stats(dict_item, prompt_tok_cnt, response_tok_cnt):
    """
    Updates the response_stats and token_stats for a given dictionary item after verification.
    Args:
        dict_item (dict): The dictionary containing response data (question, response, claims_lst, etc).
        prompt_tok_cnt (int): Token count for the verification prompt.
        response_tok_cnt (int): Token count for the verification response.
    Returns:
        dict: The updated dictionary item with response_stats and token_stats.
    """
    # Update token_stats with verification token counts
    token_stats = dict_item.get("token_stats", {})
    token_stats["verify"] = {
        "prompt_tok_cnt": prompt_tok_cnt,
        "response_tok_cnt": response_tok_cnt
    }
    dict_item["token_stats"] = token_stats

    # Initialize response_stats if not present
    response_stats = dict_item.get("response_stats", defaultdict(lambda: defaultdict(int)))

    # response stats: number of sentences and length of the response
    response_text = dict_item.get("response", "")
    sentences = nltk.sent_tokenize(response_text)
    response_stats["num_sentences"] = len(sentences)
    response_stats["length"] = len(response_text.split())

    # claim-level:
    claims_lst = dict_item.get("claims_lst", [])

    # preverification stats (prelabels distribution)
    response_stats["preverification"]["directly_preverified_claims"] = sum([1 for claim in claims_lst if claim['pre-label'] != 'unsure'])
    pre_supported = sum([1 for claim in claims_lst if claim.get("pre-label", "") == "supported"])
    pre_unsupported = sum([1 for claim in claims_lst if claim.get("pre-label", "") == "unsupported"])
    pre_irrelevant = sum([1 for claim in claims_lst if claim.get("pre-label", "") == "irrelevant"])
    pre_unsure = sum([1 for claim in claims_lst if claim.get("pre-label", "") == "unsure"])
    response_stats["preverification"]["pre_supported_claims"] = pre_supported
    response_stats["preverification"]["pre_unsupported_claims"] = pre_unsupported
    response_stats["preverification"]["pre_irrelevant_claims"] = pre_irrelevant
    response_stats["preverification"]["pre_unsure_claims"] = pre_unsure

    # verification stats (verification labels distribution)
    verified_claims = [claim for claim in claims_lst if claim.get("verification_result") is not None]
    assert pre_unsure == len(verified_claims)

    label_counts = {
        "supported": 0,
        "refuted": 0,
        "conflicting evidence": 0,
        "unverifiable": 0,
        "not enough evidence": 0
    }

    for claim in claims_lst:
        if verification_result := claim.get("verification_result"):
            label = verification_result["verification_label"].lower().strip()
            label_counts[label] = label_counts.get(label, 0) + 1

    verified_unsupported = (
        label_counts["refuted"] + 
        label_counts["conflicting evidence"] + 
        label_counts["unverifiable"] +
        label_counts["not enough evidence"]
    )

    response_stats["verification"].update({
        "total_verified_claims": len(verified_claims),
        "verified_supported_claims": label_counts["supported"],
        "verified_contradicted_claims": label_counts["refuted"],
        "verified_controversial_claims": label_counts["conflicting evidence"],
        "verified_inconclusive_claims": label_counts["not enough evidence"],
        "verified_unverifiable_claims": label_counts["unverifiable"],
        "verified_unsupported_claims": verified_unsupported
    })

    # total claims stats
    response_stats["total"]["total_claims"] = len(claims_lst)
    response_stats["total"]["total_supported_claims"] = pre_supported + label_counts["supported"]
    response_stats["total"]["total_unsupported_claims"] = pre_unsupported + verified_unsupported
    response_stats["total"]["total_irrelevant_claims"] = pre_irrelevant

    # Calculate precision P
    response_stats["P"] = response_stats["total"]["total_supported_claims"] / len(claims_lst) if len(claims_lst) != 0 else None

    # update dict_item with updated response_stats
    dict_item["response_stats"] = response_stats

    return dict_item



############################################## Miscellaneous ##############################################

def get_device_map():
    import torch
    """Determine the device map based on available GPUs or fallback to CPU."""
    if torch.cuda.is_available():
        return "auto"
    else:
        return {"": "cpu"}

def get_claim_id(claim_text, index=None):
    """
    Generates a unique claim ID that shows both position in the list
    and a content fingerprint for verification
    
    Args:
        claim_text: The text content of the claim
        index: Optional positional index in the claims list
    
    Returns:
        str: ID in format "claim-[index]-[hash]" or "claim-[hash]"
    """
    content_hash = hashlib.md5(claim_text.encode()).hexdigest()[:6]
    
    if index is not None:
        return f"{content_hash}-{index}"
    else:
        return f"{content_hash}"
