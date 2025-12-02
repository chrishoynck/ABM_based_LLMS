
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed, pipeline
import os, torch
import sys, argparse
import inspect
# from src.classes.agent import Agent
import utils.metrics as metrics
from classes.network import RandomNetwork, ScaleFreeNetwork
import utils.load_personas as lp
import utils.visualization as vis
import utils.reading_in as ri

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

def build_network(args, personas, depressed_personas=None):
    if args.net == "sf":
        return ScaleFreeNetwork(
            m=args.m,
            num_agents=args.num_agents,
            starting_distribution=args.starting_distribution,
            seed=args.seed,
            personas=personas,
            depressed_personas=depressed_personas,
            
        )
    else:
        return RandomNetwork(
            p=args.p,
            k=args.k,
            num_agents=args.num_agents,
            starting_distribution=args.starting_distribution,
            seed=args.seed,
            personas=personas,
            depressed_personas=depressed_personas,
        )

def generate_parser():
    "parse all given arguments"
    parser = argparse.ArgumentParser(description="Run LLM agent simulation.")
    parser.add_argument("net", nargs="?", choices=["sf", "r"], default="sf", help="Network type: sf=ScaleFree, r=Random")
    parser.add_argument("--rounds", type=int, default=10, help="Number of update rounds")
    parser.add_argument("--num_agents", type=int, default=10, help="Total number of agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--starting_distribution", type=float, default=0.5, help="Fraction of D vs H agents")
    # Scale-free specific
    parser.add_argument("--m", type=int, default=2, help="Edges per new node (scale-free)")
    # Random network specific
    parser.add_argument("--p", type=float, default=0.5, help="Edge probability (random network)")
    parser.add_argument("--k", type=int, default=0, help="Regular degree (Watts–Strogatz if >0)")
    parser.add_argument("--depressed", action="store_true", help="Include depressed personas")
    parser.add_argument("--enforce_ngrams", action="store_true", help="Enforce distorted-language n-grams in tweets")
    parser.add_argument("--use_saved_network", action="store_true", help="Use saved network properties to reload network")

    return parser.parse_args()

def update_network(network, fracs_dist_step = [], running_fracs = [], rounds=1, seed=42, enforce_ngrams = False):
    """Update the network for one round and return the mean fraction of distorted tweets."""

    # only enforce n-grams if specified
    if enforce_ngrams:
        distorted_tweets = lp.load_distorted_tweets("data/distorted_tweets.csv", numtweets=1000, seed=seed)
    else:
        distorted_tweets = []
    
    n_grams = metrics.load_ngrams_tsv("data/distorted_language_ngrams.tsv")
    
    for _ in range(rounds):
        mean_running_frac, frac_distorted_this_step = network.update_round(tokenizer, pipe, n_grams=n_grams, distorted_tweets=distorted_tweets)
        print(f"Round {network.iterations}: Mean running fraction of distorted agents: {mean_running_frac:.4f}, Fraction distorted this step: {frac_distorted_this_step:.4f} ")
        running_fracs.append(mean_running_frac)
        fracs_dist_step.append(frac_distorted_this_step)
        if network.iterations % 10 == 0:
            print(f"finished round {network.iterations}")
    return running_fracs, network, fracs_dist_step

def run_simulation(
    net="sf",
    rounds=10,
    num_agents=10,
    seed=42,
    starting_distribution=0.5,
    m=2,
    p=0.5,
    k=0, 
    depressed = False, 
    enforce_ngrams = False):

    """Run the simulation and return the network + tweet history."""
    set_seed(SEED)     
    print(type(pipe.model))
    print("GENERATOR: ", inspect.signature(pipe.model.generate))

    # build an argparse-like namespace
    args = argparse.Namespace(
        net=net,
        rounds=rounds,
        num_agents=num_agents,
        seed=seed,
        starting_distribution=starting_distribution,
        m=m,
        p=p,
        k=k,
        depressed=depressed,
        enforce_ngrams=enforce_ngrams,
    )
    personas = None
    # load personas
    if False:
        personas = lp.load_personas_from_file("data/personas_10k.csv", args.num_agents, seed=args.seed)

    # only load depressed personas if specified
    if depressed:
        depressed_personas = lp.load_depressed_personas("data/depressed.csv", personass_to_load=1, seed=args.seed)
    else:
        depressed_personas = None

    # build network
    network = build_network(args, personas=personas, depressed_personas=depressed_personas)

    # run updates
    running_fracs, network, fracs_dist_step = update_network(network, fracs_dist_step=[], running_fracs=[], rounds=rounds, seed=seed, enforce_ngrams=enforce_ngrams)
    tweet_history = [(a.ID, a.tweethistory) for a in network.all_agents]

    return network, tweet_history, running_fracs, fracs_dist_step

def retrieve_existing_net(args):
    '''Retrieve file path for existing network based on arguments.
    Args:
        args: Argument namespace containing network parameters.
    Returns:
        file_path (str): Path to the saved network file.
    '''
    if args.enforce_ngrams:
        state = "enforced_ngrams"
    elif args.depressed:
        state = "depressed"
    else:
        state = "basis"
    
    if args.net == "sf":
        what_network = "sf"
        parameter = f'{args.m}'
    else:
        parameter = f'{str(args.p).replace(".", "_")}'
        what_network = "rand"
    
    file_path = f"data/networks/{state}/{what_network}/{parameter}/num_agents{args.num_agents}_{args.rounds}_net_{args.seed}.txt"
    return file_path

if __name__ == "__main__":
    # keep CLI behavior
    args = generate_parser()
    # states = ["basis", "depressed", "enforced_ngrams"]
    # states = ["basis"]
    states = ["depressed"]
    # states = ["enforced_ngrams"]
    if args.use_saved_network:
        file_paths = []
        networks = []
        for state in states:
            args.enforce_ngrams = (state == "enforced_ngrams")
            args.depressed = (state == "depressed")
            file_path = retrieve_existing_net(args)
            file_paths.append(file_path)
            # reload network from saved properties
            network, running_fracs, fracs_dist_step= ri.generate_network(file_path, pipe, starting_distribution=args.starting_distribution)
            networks.append(network)
    else:
        network, tweet_history, running_fracs, fracs_dist_step = run_simulation(
            net=args.net,
            rounds=args.rounds,
            num_agents=args.num_agents,
            seed=args.seed,
            starting_distribution=args.starting_distribution,
            m=args.m,
            p=args.p,
            k=args.k,
            depressed=args.depressed,
            enforce_ngrams=args.enforce_ngrams,
        )
        # return output file the network is printed to
        file_output_path = ri.read_out_network_properties(network, args.seed, fracs_dist_step, running_fracs, enforce_ngrams=args.enforce_ngrams, depressed=args.depressed)
        print(f"Network properties saved to {file_output_path}")

        # reload network from saved properties
        network, running_fracs, fracs_dist_step = ri.generate_network(file_output_path, pipe, starting_distribution=0.5)
        running_fracs, network, fracs_dist_step = update_network(network, fracs_dist_step=fracs_dist_step, running_fracs=running_fracs, rounds=args.rounds, seed=args.seed)
        tweet_history = [(a.ID, a.tweethistory) for a in network.all_agents]

        file_output_path = ri.read_out_network_properties(network, args.seed, fracs_dist_step, running_fracs, enforce_ngrams=args.enforce_ngrams, depressed=args.depressed)
        print(f"Network properties saved to {file_output_path}")
        networks = [network]

    # network.
    # print tweet histories (optional)
    # for agent_id, hist in tweet_history:
    #     print(f"Tweet history for Agent {agent_id}:")
    #     for tweet in hist:
    #         print(f"{tweet}")
    
    # vis.distorted_info(network.cds_info)

    # set directory parameters for plotting
    pmtje = 0.0
    mmtje = 0
    if args.net == "sf":
        mmtje = network.m
    elif args.net == "r":
        pmtje = args.p
    
    # plot TF-IDF PCA
    n_grams = metrics.load_ngrams_tsv("data/distorted_language_ngrams.tsv")
    global_tf_idf, _, _ = metrics.retrieve_tf_idf(networks, num_steps=100, shift=5, n_grams=n_grams)
    pca_runs = metrics.reduce_dimensionality(global_tf_idf, n_components=2)
    vis.plot_tf_idf_PCA(pca_runs, states, num_steps=100, shift=5)
    print("Plotted TF-IDF PCA:", len(global_tf_idf), "entries.")

    # vis.plot_running_fracs(running_fracs, mmtje, pmtje, enforced_ngrams=args.enforce_ngrams, depressed=args.depressed, type_nn=args.net)
    # vis.plot_distorted_fracs(fracs_dist_step, mmtje, pmtje, enforced_ngrams=args.enforce_ngrams, depressed=args.depressed, type_nn=args.net)
    # vis.print_network(network)

    


