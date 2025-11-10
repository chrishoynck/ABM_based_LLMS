
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed, pipeline
import os, torch
import sys
import inspect
from src.classes.agent import Agent
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
    r_network = RandomNetwork(p=0.5, num_agents=10, mean=0, starting_distribution=0.5, seed=42)
    for r in range(10):
        r_network.update_round(tokenizer, pipe)
    tweet_history = [(a.ID, a.tweethistory) for a in r_network.all_agents]

    # print tweet histories
    for agent_id, hist in tweet_history:
        print(f"Agent {agent_id}: {hist}")
        print("\n")
    
    
    # wat = vis.print_network(r_network)

