
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
# GEN = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
# interactive part
SYSTEM_PROMPT = "You are a concise, helpful assistant."
SEED = 1234
os.environ["PYTHONHASHSEED"] = str(SEED)   # best set before Python starts                # if you still use np.random.*
set_seed(SEED)                              # seeds Python, NumPy, Torch (HF helper)


# setyp initial llm
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, use_fast=True)
# Ensure a pad token exists (prevents fallback messages)
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    tokenizer=tokenizer,
    # local_files_only = True,
    torch_dtype=DTYPE,
    device_map="auto",                        # shard/assign automatically
    trust_remote_code=True,
    max_new_tokens=256,
    # generator=GEN,
    # temperature=0.0,
    return_full_text=False,                        
)
GEN = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(SEED)


def chat_turn(history, user_msg: str) -> (str, list):
    # build messages with history + new user turn
    msgs = history + [{"role": "user", "content": user_msg}]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    out = pipe(
        prompt,
        do_sample=False,          # deterministic; set True + temperature/top_p to sample
        generator= GEN,
        max_new_tokens=256,
        return_full_text=False,   # only return the completion (safer than slicing)
    )[0]["generated_text"].strip()
    # append assistant reply to history
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": out})
    return out, history

def repl(history=None):
    ''' 
    Conversational function with this llama model, 
    History of llama model is updated with each message.
    when using /reset: history is forgotten
    when using /exit: conversation is stopped. 
    '''
    print("Chat ready. Type your message. Commands: /reset, /exit")
    while True:
        try:
            user = input("\nYou: ").strip()
            if not user:
                continue
            if user.lower() in {"/exit", "/quit"}:
                print("Bye!")
                break
            if user.lower() == "/reset":
                history[:] = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("(history reset)")
                continue
            reply, history = chat_turn(history, user)
            print(f"Assistant: {reply}")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break


def act_llama2(text: str, prev=None) -> str:
    """Raw continuation (no chat template)."""
    out = pipe(text, generator=GEN, do_sample=True)[0]["generated_text"]
    return out[len(text):].strip()

def chat_once(user_msg: str) -> str:
    messages = [{"role": "user", "content": user_msg}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = pipe(prompt, do_sample=False)[0]["generated_text"]
    return out[len(prompt):].strip()

if __name__ == "__main__":
    # history = [{"role": "system", "content": SYSTEM_PROMPT}]
    # print("starting converstations")
    # print("Raw continuation:", act_llama2("how high are trees?"))
    # print("Chat answer:", chat_once("How are you?"))
    # repl(history)

    # Make two agents A,B who follow each other
    # A = Agent(0, identity="A")
    # B = Agent(1, identity="B")
    # C = Agent(2, identity="C")S
    # A.add_edge(B); B.add_edge(A)
    # A.add_edge(C); C.add_edge(A)
    # B.add_edge(C); C.add_edge(B)

    # # Seed: activate A to kick things off
    # A.activation_state = True
    # B.activation_state = False  
    # C.activation_state = False

    # for r in range(20):
    #     for a in (A, B, C):
    #         a.step_llm_tweet(tokenizer, pipe, round_idx=r)
    #     for a in (A, B, C):
    #         a.commit()
    #     print(f"Round {r}: A_active={A.activation_state}, A_last={A.last_tweet} | "
    #         f"B_active={B.activation_state}, B_last={B.last_tweet} | "
    #         f"C_active={C.activation_state}, C_last={C.last_tweet}")
    # print("A tweet history:", A.tweethistory)
    # print("B tweet history:", B.tweethistory)
    # print("C tweet history:", C.tweethistory)

    print(type(pipe.model))
    print(inspect.signature(pipe.model.generate))
    

    r_network = RandomNetwork(p=0.5, num_agents=10, mean=0, starting_distribution=0.5, seed=42)
    for r in range(10):
        r_network.update_round(tokenizer, pipe)
    tweet_history = [a.tweethistory for a in r_network.all_agents]
    for hist in tweet_history:
        print(hist)
    wat = vis.print_network(r_network)
    print(wat)

