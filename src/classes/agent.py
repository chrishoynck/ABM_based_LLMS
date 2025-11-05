import numpy as np, torch

class Agent:
    """
    A agent in the network, with a unique ID and a response threshold.
    The response threshold is a random number between 0 and 1, which is used to determine whether the agent will respond to a piece of news.
    The agent can be in one of two states: activated or not activated.
    The agent can also be a sampler, which means that it will always respond to a piece of news, regardless of the response threshold.
    """
    def __init__(self, ID, identity, rng=None):
        """
        Initialize the agent.

        Args:
            ID (int): The unique ID of the agent.
            identity (str): The identity of the agent (either "L" or "R").
            rng (np.random.Generator, optional): The random number generator to use. Defaults to None.
        
        Attributes:
            response_threshold (float): The response threshold of the agent.
            activation_state (bool): Whether the agent is activated or not.
            agent_connections (set): The set of agents that the agent is connected to.
        """
        self.ID = ID
        self.identity: str = identity
        self.agent_connections = set()
        self.activation_state = False
        self.response_threshold = rng.random() if rng else np.random.random()

        # Additional attributes for LLM interaction
        self.rng = rng if rng else np.random.default_rng()
        self._base_seed = int(self.rng.integers(0, 2**31 - 1))
        self._torch_gen = None  # will be created on first use
        self.tweethistory = []
        self.last_tweet: str | None = None
        self._next_activation_state = False 

    # attempt to seed the whole thing -> did not work
    def generate_with_gen_chat(self, prompt, pipe, temperature=0.8, top_p=0.95, max_new_tokens=256):
        tok = pipe.tokenizer

        # transform prompts into tensors
        inputs = tok(prompt, return_tensors="pt")

        # moves them to the correct device
        inputs = {k: v.to(pipe.model.device) for k, v in inputs.items()}

        out_ids = pipe.model.generate(
            **inputs,
            do_sample=True,
            seeded=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            generation_config=None,  # ensure no conflicting config object
        )

        # only grab newly generated tokens
        new_tokens = out_ids[0, inputs["input_ids"].shape[1]:]

        # decode and return the generated text
        return tok.decode(new_tokens, skip_special_tokens=True).strip()
    
    # helper function to seed but did not work. 
    def _ensure_torch_gen(self, llm_pipe):
        """Create (or reuse) a persistent per-agent torch.Generator on the right device."""
        dev = llm_pipe.model.device
        if self._torch_gen is None or str(self._torch_gen.device) != str(dev):
            self._torch_gen = torch.Generator(device=dev).manual_seed(self._base_seed)
        return self._torch_gen
    
    def build_tweet_prompt(self, tokenizer, identity, round_idx, neighbor_pairs, max_chars=240, force_active=False):
        # neighbor_pairs: list of (neighbor_id, last_text)

        # own history block
        own_block = "" 
        # if len(self.tweethistory) == 0:
        #     own_block = "(no own previous tweets)"
        # else:
        #     recent = list(reversed(self.tweethistory[-5:]))  # newest first
        #     own_block = "\n".join(f"- {t[:max_chars]}" for t in recent)
        
        neighbor_block = "(no neighbor tweets)" if len(neighbor_pairs) == 0 else "\n".join(
            f"- Agent {nid}: {txt[:240]}" for nid, txt in neighbor_pairs
        )

        if not force_active:
            sup_text = "Decide whether to post a short new tweet (<= " f"{max_chars} chars)."
        else:
            sup_text = "You must post a short new tweet (<= " f"{max_chars} chars)."

        system = (
            f"You are a social media user {identity}.\n"
           "You are given your recent tweets (in case you've already tweeted) and a short list of neighbor tweets (in case they have tweeted) (you can decide to randomly post).\n"
            f"{sup_text}\n"
            "Only post your new tweet.\n"
            "REPLY FORMAT (exactly):\n"
            "If you want to tweet, reply with: TWEET: <your tweet text>\n"
            "If you don't want to tweet, reply with: NO_TWEET\n"
            "Do not add anything else, do not explain.\n\n"
        )
        user = (
            f"Identity: {identity}\n"
            f"Round: {round_idx}\n"
            f"Your recent tweets:\n{own_block}\n\n"
            f"Neighbor tweets:\n{neighbor_block}"
        )

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # One message per neighbor (clear separation)
        for nid, txt in neighbor_pairs:
            messages.append({"role": "user", "content": f"Neighbor {nid} tweeted:\n{txt}"})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    

    def step_llm_tweet(self, tokenizer, llm_pipe, round_idx:int, max_chars = 240, force_active=False):
        """
        Use the LLM to decide whether to tweet or not.

        Args:
            llm_pipe: The LLM pipeline to use.
            round_idx (int): The current round index.
            max_chars (int, optional): The maximum number of characters for the tweet. Defaults to 240.
        Returns:
            bool: Whether the agent decided to tweet or not.
        """
        neighbor_msgs = []

        

        activated, activated_neighbors = self.respond()
        # if force_active:
        #     activated = True
        if True:
            for n in activated_neighbors:
                if n.activation_state and n.last_tweet:
                    neighbor_msgs.append((n.ID, n.last_tweet))
            neighbor_msgs = self.rng.permutation(neighbor_msgs)[:5]  # limit to first 5 neighbors
        
        prompt = self.build_tweet_prompt(
            tokenizer, self.ID, round_idx, neighbor_msgs, max_chars=max_chars, force_active=force_active
        )
        # #gen = self._ensure_torch_gen(llm_pipe)
        # if gen is None:
        #     raise ValueError("Torch generator not initialized properly.")
        
        out = llm_pipe(
            prompt,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=256,
            # generator=gen,
        )[0]["generated_text"].strip() 

        do_tweet, tweet = self.parse_tweet_decision(out)
        if do_tweet:
            tweet = tweet.strip()
            if len(tweet) > max_chars:
                tweet = tweet[:max_chars]
            self.last_tweet = tweet
            self.tweethistory.append(tweet)
            self._next_activation_state = True
        else:
            self.last_tweet = None
            self._next_activation_state = False

    # Finalize the activation state for this step
    def commit(self):
        self.activation_state = self._next_activation_state

    def reset_activation_state(self):
        self.activation_state = False

    def parse_tweet_decision(self, text: str):
        t = text.strip()
        low = t.lower()

        # all text that is not generated in proper format is treated as no tweet
        if "no_tweet" in low:
            return False, ""
        if "tweet:" not in low:
            return False, ""
        # prefer the explicit "TWEET:" pattern
        idx = low.find("tweet:")
        if idx != -1:
            return True, t[idx:].strip()
        # fallback: if any non-empty content, treat as tweet
        return (len(t) > 0), t
        
    def respond(self) -> (bool, set):
        """
        respond to the news intensity and returns False if the activation state did not change, True otherwise.

        Args:
            intensity (float): The intensity of the news.
            analyze (bool): Whether to analyze the cascade or not.
        
        Returns:
            bool: True if the activation state changed, False otherwise.
            set: The set of agents that should be activated
        """
        neighbors_activated = 0
        actually_activated = []
        new_activation_state = False


        if len(self.agent_connections) > 0: 
            actually_activated = [agent for agent in self.agent_connections if agent.activation_state] 
            neighbors_activated = len(actually_activated)
            fraction_activated = neighbors_activated/len(self.agent_connections)
        else:
            fraction_activated = 0
        new_activation_state = fraction_activated > self.response_threshold
        if new_activation_state:
            return True, set(actually_activated)

        return False, set()

    def add_edge(self, agent):
        """
        Add an edge to the agent.

        Args:
            agent (agent): The agent to add as an edge.
        """
        self.agent_connections.add(agent)

    def remove_edge(self, agent):
        """
        Remove an edge from the agent.

        Args:
            agent (agent): The Agent to remove as an edge.
        """
        self.agent_connections.discard(agent)
    
    def reset_agent(self):
        """
        Reset the agent to its initial state.
        """
        self.activation_state = False
        self.last_tweet = None
        self.tweethistory = []


    def __hash__(self):
        """
        Hash the agent by its ID and identity.
        Needed for the set data structure.

        Returns:
            int: The hash of the agent.
        """
        return hash((self.ID, self.identity)) 

    def __eq__(self, other):
        """
        Check if the agent is equal to another agent.
        """
        return isinstance(other, Agent) and self.ID == other.ID and self.identity == other.identity

