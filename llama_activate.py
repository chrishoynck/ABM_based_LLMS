
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed, pipeline
import os, torch
import sys
import inspect
from src.classes.agent import Agent
import src.metrics as metrics
from src.classes.network import RandomNetwork
import src.visualization as vis

######################################################################
### Llama 2 Setup
######################################################################


print(torch.cuda.is_available())
llama_model= "meta-llama/Llama-3.2-1B-Instruct"
# when setting possible enironment variables in the future
MODEL_ID = os.environ.get("LLAMA_ID", llama_model)
CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", None)
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# set seeds for reproducibility
SEED = 1234
os.environ["PYTHONHASHSEED"] = str(SEED)   # best set before Python starts                # if you still use np.random.*
set_seed(SEED)                              # seeds Python, NumPy, Torch (HF helper)


# setyp initial llm
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, use_fast=True)

# Ensure a pad token exists (prevents fallback messages)
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

# Pipeline configuration
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    tokenizer=tokenizer,
    # local_files_only = True,
    torch_dtype=DTYPE,
    device_map="auto",                        # shard/assign automatically
    trust_remote_code=True,
    max_new_tokens=256,
    return_full_text=False,                        
)



if __name__ == "__main__":
    
    print(type(pipe.model))
    print("GENERATOR: ", inspect.signature(pipe.model.generate))
    
    # Create a random network and run some rounds
    r_network = RandomNetwork(p=0.5, num_agents=30, mean=0, starting_distribution=0.5, seed=42)
    for r in range(100):
        r_network.update_round(tokenizer, pipe)
    tweet_history = [(a.ID, a.tweethistory) for a in r_network.all_agents]

    # print tweet histories
    for agent_id, hist in tweet_history:
        print(f"Agent {agent_id}: {hist}")
        print("\n")

    n=10
    distorted_language = metrics.analyze_distorted_language(
        r_network,
        ngrams_file="data/distorted_language_ngrams.tsv",
        skip_header=True,
        n=n,
        column_idx=0,
    )

    # Print the results
    for agent_id, met in distorted_language.items():
        print(f"Agent {agent_id}:")
        print(f"  First {n} tweets: {met['first_n']} distorted")
        print(f"  Last {n} tweets: {met['last_n']} distorted")
        print(f"  Total tweets: {met['total_tweets']}")
        print("  Fraction distorted in first tweets: {:.2f}".format(met['frac_distorted_first']))
        print("  Fraction distorted in last tweets: {:.2f}".format(met['frac_distorted_last']))
        print("\n")

    # wat = vis.print_network(r_network)

