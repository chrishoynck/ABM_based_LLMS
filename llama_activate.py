
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline
import os, torch
import sys


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
history = [{"role": "system", "content": SYSTEM_PROMPT}]

# setyp initial llm
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, use_fast=True)

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
)

# --- minimal interactive chat REPL (uses chat template + history) ---



def chat_turn(user_msg: str) -> str:
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
    return out

def repl():
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
            reply = chat_turn(user)
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
    print("starting converstations")
    print("Raw continuation:", act_llama2("how high are trees?"))
    print("Chat answer:", chat_once("How are you?"))
    repl()

