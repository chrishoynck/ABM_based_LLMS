import numpy as np

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
        self.tweethistory = []
        self.last_tweet: str | None = None
        self._next_activation_state = False 
    
    def build_tweet_prompt(self, tokenizer, identity, round_idx, neighbor_pairs, max_chars=240):
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

        system = (
           "You are given your recent tweets (in case you've already tweeted) and a short list of neighbor tweets (in case they have tweeted) (you can decide to randomly post).\n"
            "Decide whether to post a short new tweet (<= " f"{max_chars} chars).\n"
            "Only post your new tweet, don't be repetitive.\n"
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

    def step_llm_tweet(self, tokenizer, llm_pipe, round_idx:int, max_chars = 240):
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
        if True:
            for n in activated_neighbors:
                if n.activation_state and n.last_tweet:
                    neighbor_msgs.append((n.ID, n.last_tweet))
            neighbor_msgs = self.rng.permutation(neighbor_msgs)[:5]  # limit to first 5 neighbors

        prompt = self.build_tweet_prompt(
            tokenizer, self.identity, round_idx, neighbor_msgs, max_chars=max_chars
        )

        out = llm_pipe(prompt)[0]["generated_text"]
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
        if "no_tweet" in low and "tweet:" not in low:
            return False, ""
        # prefer the explicit "TWEET:" pattern
        idx = low.find("tweet:")
        if idx != -1:
            return True, t[idx:].strip()
        # fallback: if any non-empty content, treat as tweet
        return (len(t) > 0), t
        
    def respond(self, use_llm=False, llm_pipe=None) -> (bool, set):
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
        pass


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

