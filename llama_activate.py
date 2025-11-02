
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline
import os, torch
import sys
from src.classes.agent import Agent


######################################################################
### Llama 2 Setup
######################################################################


print(torch.cuda.is_available())
llama_model= "meta-llama/Llama-3.2-1B-Instruct"

# when setting possible enironment variables in the future
MODEL_ID = os.environ.get("LLAMA_ID", llama_model)
CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", None)
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# interactive part
SYSTEM_PROMPT = "You are a concise, helpful assistant."


# setyp initial llm
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, use_fast=True)
# Ensure a pad token exists (prevents fallback messages)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    tokenizer=tokenizer,
    # local_files_only = True,
    torch_dtype=DTYPE,
    device_map="auto",                        # shard/assign automatically
    trust_remote_code=True,
    # model_kwargs={
    #     # enable if available; otherwise omit
    #     "attn_implementation": "flash_attention_2",
    # },
    max_new_tokens=256,
    # temperature=0.0,
    return_full_text=False,                        
)


def chat_turn(history, user_msg: str) -> (str, list):
    # build messages with history + new user turn
    msgs = history + [{"role": "user", "content": user_msg}]
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    out = pipe(
        prompt,
        do_sample=False,          # deterministic; set True + temperature/top_p to sample
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
    out = pipe(text, do_sample=False)[0]["generated_text"]
    return out[len(text):].strip()

def chat_once(user_msg: str) -> str:
    messages = [{"role": "user", "content": user_msg}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = pipe(prompt, do_sample=False)[0]["generated_text"]
    # Strip the prompt part
    return out[len(prompt):].strip()

if __name__ == "__main__":
    # history = [{"role": "system", "content": SYSTEM_PROMPT}]
    # print("starting converstations")
    # print("Raw continuation:", act_llama2("how high are trees?"))
    # print("Chat answer:", chat_once("How are you?"))
    # repl(history)

    # Make two agents A,B who follow each other
    A = Agent(0, identity="H")
    B = Agent(1, identity="D")
    C = Agent(2, identity="N")
    A.add_edge(B); B.add_edge(A)
    A.add_edge(C); C.add_edge(A)
    B.add_edge(C); C.add_edge(B)

    # Seed: activate A to kick things off
    A.activation_state = True
    B.activation_state = False  
    C.activation_state = False

    for r in range(20):
        for a in (A, B, C):
            a.step_llm_tweet(tokenizer, pipe, round_idx=r)
        for a in (A, B, C):
            a.commit()
        print(f"Round {r}: A_active={A.activation_state}, A_last={A.last_tweet} | "
            f"B_active={B.activation_state}, B_last={B.last_tweet} | "
            f"C_active={C.activation_state}, C_last={C.last_tweet}")
    print("A tweet history:", A.tweethistory)
    print("B tweet history:", B.tweethistory)
    print("C tweet history:", C.tweethistory)
