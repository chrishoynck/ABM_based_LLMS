
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed, pipeline
import os, torch
import sys, argparse
import inspect
from src.classes.agent import Agent
import src.metrics as metrics
from src.classes.network import RandomNetwork, ScaleFreeNetwork
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

def build_network(args):
    if args.net == "sf":
        return ScaleFreeNetwork(
            m=args.m,
            num_agents=args.num_agents,
            mean=args.mean,
            starting_distribution=args.starting_distribution,
            seed=args.seed,
        )
    else:
        return RandomNetwork(
            p=args.p,
            k=args.k,
            num_agents=args.num_agents,
            mean=args.mean,
            starting_distribution=args.starting_distribution,
            seed=args.seed,
        )

def generate_parser():
    "parse all given arguments"
    parser = argparse.ArgumentParser(description="Run LLM agent simulation.")
    parser.add_argument("net", nargs="?", choices=["sf", "r"], default="sf", help="Network type: sf=ScaleFree, r=Random")
    parser.add_argument("--rounds", type=int, default=10, help="Number of update rounds")
    parser.add_argument("--num_agents", type=int, default=10, help="Total number of agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mean", type=float, default=0.0, help="Mean (passed to network)")
    parser.add_argument("--starting_distribution", type=float, default=0.5, help="Fraction of D vs H agents")
    # Scale-free specific
    parser.add_argument("--m", type=int, default=2, help="Edges per new node (scale-free)")
    # Random network specific
    parser.add_argument("--p", type=float, default=0.5, help="Edge probability (random network)")
    parser.add_argument("--k", type=int, default=0, help="Regular degree (Watts–Strogatz if >0)")

    return parser.parse_args()

if __name__ == "__main__":
    
    print(type(pipe.model))
    print("GENERATOR: ", inspect.signature(pipe.model.generate))

    # parse arguments and build network
    args = generate_parser()
    network = build_network(args)
    

    # update network for a given amount of rounds (agents send out tweets)
    for r in range(args.rounds):
        network.update_round(tokenizer, pipe)
        if r%10 == 0:
            print(f"finished round {r}")
    tweet_history = [(a.ID, a.tweethistory) for a in network.all_agents]

    # print tweet histories
    for agent_id, hist in tweet_history:
        print(f"Agent {agent_id}: {hist}")
        print("\n")

    n=10
    distorted_language, highest_frac = metrics.analyze_distorted_language(
        network,
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

    wat = vis.print_network(network)

