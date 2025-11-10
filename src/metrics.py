import re, csv, json
from collections import defaultdict

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

def analyze_distorted_language(network, ngrams_file: str, n: int = 5, column_idx: int = 0):
    """
    For each agent in the network, count distorted-language n-grams in:
      - the first N tweets
      - the last N tweets
    Prints a summary and returns results as a dict.
    
    Args:
        network: The network object (must have .all_agents attribute).
        ngrams_file (str): Path to the TSV file with distorted-language n-grams.
        n (int): Number of tweets from the start/end to analyze.
        column_idx (int): Column index in TSV containing n-grams (default: 0).
    
    Returns:
        dict: {agent_id: {"first_n": count, "last_n": count, "total_tweets": int}}
    """
    ngrams = load_ngrams_tsv(ngrams_file, column_idx=column_idx)
    print(f"Loaded {len(ngrams)} distorted-language n-grams from {ngrams_file}")
    highest_frac = 0

    results = {}
    for agent in network.all_agents:
        history = getattr(agent, "tweethistory", [])
        # filter out None / "NO_TWEET" entries
        
        history = [t for t in history if t]
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