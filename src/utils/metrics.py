import re, csv, json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from sklearn.decomposition import PCA
import numpy as np

def load_ngrams_tsv(filepath: str, skip_header=True) -> set:
    """
    Load distorted-language n-grams from a TSV file with columns:
      categories | markers | variants
    
    - markers column: base n-gram
    - variants column: JSON list of variant forms (optional)
    
    Returns a set of lowercased n-grams (base + all variants).
    """
    ngrams = set()
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        if skip_header:
            next(reader, None)  # skip header row
        for row in reader:
            if len(row) < 2:
                continue
            # column 1: base marker
            base = row[1].strip().lower()
            if base:
                ngrams.add(base)
            # column 2: variants (may be empty or JSON list)
            if len(row) > 2 and row[2].strip():
                variants_str = row[2].strip()
                try:
                    variants = json.loads(variants_str)
                    if isinstance(variants, list):
                        for v in variants:
                            clean = v.strip().lower()
                            if clean:
                                ngrams.add(clean)
                except json.JSONDecodeError:
                    # if not valid JSON, treat as plain text (single variant)
                    clean = variants_str.lower()
                    if clean:
                        ngrams.add(clean)
    return ngrams

def contains_ngram(text: str, ngrams: set) -> bool:
    """Check if any n-gram from ngrams appears in text (case-insensitive, word boundaries)."""
    text_low = text.lower()
    for ng in ngrams:
        # use word-boundary regex so "cat" doesn't match inside "catch"
        if re.search(r'\b' + re.escape(ng) + r'\b', text_low):
            return True
    return False

def analyze_distorted_language(network, ngrams_file: str, ngrams = None, n: int = 5, skip_header= True):
    """
    For each agent in the network, count distorted-language n-grams in:
      - the first N tweets
      - the last N tweets
    Prints a summary and returns results as a dict.
    
    Args:
        network: The network object (must have .all_agents attribute).
        ngrams_file (str): Path to the TSV file with distorted-language n-grams.
        n (int): Number of tweets from the start/end to analyze.
    
    Returns:
        dict: {agent_id: {"first_n": count, "last_n": count, "total_tweets": int}}
    """
    if ngrams is None:
        ngrams = load_ngrams_tsv(ngrams_file, skip_header=skip_header)
    # print(ngrams)
    print(f"Loaded {len(ngrams)} distorted-language n-grams from {ngrams_file}")
    highest_frac = 0

    results = {}
    for agent in network.all_agents:
        history = getattr(agent, "tweethistory", [])
 
        # now only consider actual tweets
        history = [t for t in history if t!= "NO_TWEET"]
        first_tweets = history[:n]
        last_tweets = history[-n:] if len(history) >= n else history

        first_tweets = [t for t in first_tweets if t != "NO_TWEET"]
        last_tweets = [t for t in last_tweets if t != "NO_TWEET"]
        
        first_count = sum(1 for tweet in first_tweets if contains_ngram(tweet, ngrams))
        last_count = sum(1 for tweet in last_tweets if contains_ngram(tweet, ngrams))
        
        results[agent.ID] = {
            "first_n": first_count,
            "last_n": last_count,
            "Length_last_tweets": len(last_tweets),
            "Length_first_tweets": len(first_tweets),
            "total_tweets": len(history),
            "frac_distorted_first": first_count / len(first_tweets) if len(first_tweets)>0 else 0,
            "frac_distorted_last": last_count / len(last_tweets) if len(last_tweets)>0 else 0,
        }
        highest_frac = max(highest_frac, results[agent.ID]["frac_distorted_last"])
        highest_frac = max(highest_frac, results[agent.ID]["frac_distorted_first"])
    return results, highest_frac


# TF-IDF computation
def compute_tf_idf(all_tweets):
    '''Compute TF-IDF for a list of tweets.
    Args:
        all_tweets (List(str)): List of tweet texts.
    Returns:
        vocab (np.array): Vocabulary array. 
        vectorizer: Fitted TfidfVectorizer object.
    '''

    # remove english common words
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

    # fit the model
    vectorizer.fit(all_tweets)

    # get vocabulary
    vocab = np.array(vectorizer.get_feature_names_out())
    return  vocab, vectorizer

def retrieve_tf_idf(networks, num_steps= 30, shift=5, n_grams= None):
    '''Retrieve TF-IDF data from the network's agents' tweet histories.
    Args:
        network: The network object.
        num_steps (int): Number of steps used in TF-IDF retrieval.
        shift (int): Shift between windows.
    Returns:
        global_tf_idf (np.array): TF-IDF matrix for all windows.
        vocab (np.array): Vocabulary array.
        vectorizer: Fitted TfidfVectorizer object.
    '''
    tf_idf_all = []
    docs_per_network = []
  
    for i,  network in enumerate(networks):
        # calculate number of windows
        num_windows = max(1, (network.iterations - num_steps) // shift + 1)
        tf_idf_lists = [[] for _ in range(num_windows)]
        for agent in network.all_agents:

            # iterate over windows
            for w in range(num_windows):
                # extend with tweets in this window
                tf_idf_lists[w].extend(agent.tweethistory[(w* shift):(w*shift + num_steps)])

            # extend with remaining tweets
            if (num_windows - 1) * shift + num_steps < network.iterations:
                if len(tf_idf_lists) < num_windows + 1:
                    tf_idf_lists.append([])
                tf_idf_lists[-1].extend(agent.tweethistory[((num_windows-1) * shift + num_steps):])
            
            # also keep a vocab of all tweets
            tf_idf_all.extend(agent.tweethistory)
            if n_grams is not None and i == 0:
                tf_idf_all.extend(n_grams)
    
        cleaned_tf_idf = []
        w = 0
        while w < len(tf_idf_lists):
            tf_idf_lists[w] = [t for t in tf_idf_lists[w] if t != "NO_TWEET"]
            tf_idf_lists[w] = " ".join(tf_idf_lists[w])
            # remove empty tweet lists
            if tf_idf_lists[w] == "":
                print("WARNING: Empty tweet list for window ", w)
                tf_idf_lists.pop(w)
            else:
                cleaned_tf_idf.append(tf_idf_lists[w])
                w+=1

        docs_per_network.append(cleaned_tf_idf)
    
    tf_idf_all = [t for t in tf_idf_all if t!= "NO_TWEET"]

    if len(tf_idf_all) == 0:
        raise ValueError("No valid tweets found in the network for TF-IDF computation.")
    vocab, vectorizer = compute_tf_idf(tf_idf_all)

    global_tf_idf = [vectorizer.transform(doc).toarray() for doc in docs_per_network]

    # global_tf_idf = global_tf_idf.toarray()
    return global_tf_idf, vocab, vectorizer

def reduce_dimensionality(tf_idf_matrices, n_components=2):
    '''Reduce dimensionality of TF-IDF matrix using PCA.
    Args:
        tf_idf_matrices (np.array): TF-IDF matrix.
        n_components (int): Number of PCA components.
    Returns:
        reduced_runs (np.array): PCA-reduced data.
    '''
    assert n_components <= tf_idf_matrices[0].shape[1], "n_components must be <= number of features"
    tf_idf_stacked = np.vstack(tf_idf_matrices)
    pca = PCA(n_components=n_components)
    pca.fit(tf_idf_stacked)
    reduced_runs = [pca.transform(tf_idf_matrix) for tf_idf_matrix in tf_idf_matrices]
    return reduced_runs