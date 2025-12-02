import numpy as np
import torch
from classes.agent import Agent
from scipy.spatial.distance import cdist
# from scipy import stats
# from powerlaw import Fit
import bisect

class _Network:
    """
    This function is the parent class for the RandomNetwork and ScaleFreeNetwork classes.
    A network of agents, with a specified number of agents and a correlation between the two media hubs.
    The network can be initialized as a random network or a scale-free network.
    The network can be updated by responding to news intensities and adjusting the network accordingly.    
    """

    def __init__(self, num_agents=200, starting_distribution=0.5, directed=False, seed=None, personas = None):
        """
        Initialize the network with a specified number of agents, mean, correlation, starting distribution, update fraction, and seed.

        Args:
            num_agents (int): The number of agents in the network.
            starting_distribution (float): The starting distribution of the agents.
            seed (int): The seed for the random number generator.

        Attributes:
            iterations (int): The number of iterations the network has been updated.
            activated (set): The set of activated agents.
            rng (np.random.Generator): The random number generator.
            alterations (int): The number of alterations made to the network in each round.
            new_edge (list): The list of new edges added to the network.
            removed_edge (list): The list of edges removed from the network.
            connections (set): The set of connections between agents.
            all_agents (list): The list of all agents in the network.
        """
        self.iterations = 0
        self.activated = set()
        self.directed = directed

        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self._torch_gen = None

        self.new_edge = []
        self.removed_edge = []

        personas = self.rng.permutation(personas) if personas is not None else [None]*num_agents

        # create agents
        self.agentsD = [Agent(i, rng=np.random.default_rng(seed + i), persona=personas[i]) for i in range(int(num_agents * starting_distribution))]
        self.agentsH = [Agent(i + len(self.agentsD), rng=np.random.default_rng(seed + i + len(self.agentsD)), persona = personas[i + len(self.agentsD)]) for i in range(int(num_agents * (1 - starting_distribution)))]
        self.connections = set()
        self.all_agents = self.agentsD + self.agentsH

        self.cds_info = []
        self.agent_w_highest_deg = self.all_agents[0] # placeholder
      
    def clean_network(self):
        """
        Clean the network by unactivating all agents.    
        """
        self.activated = set()

    def add_connection(self, agent1, agent2):
        """
        Add an undirected connection between two agents (if not already present).

        Args:
            agent1 (Agent): The first agent to connect.
            agent2 (Agent): The second agent to connect.
        """

        if agent1 != agent2: 
            agent1.add_edge(agent2)
            if not self.directed:
                agent2.add_edge(agent1)
            self.connections.add((agent1, agent2))
            if not self.directed:
                self.connections.add((agent2, agent1))

    def remove_connection(self, agent1, agent2):
        """
        Remove the connection between two agents if it exists.

        Args:
            agent1 (Agent): The first agent to disconnect.
            agent2 (Agent): The second agent to disconnect.
        """
        if agent1 != agent2:
            agent1.remove_edge(agent2)
            if not self.directed:
                agent2.remove_edge(agent1)
            self.connections.remove((agent1, agent2))
            if not self.directed:
                self.connections.remove((agent2, agent1))

    def inference_w_batches(self,pipe, prompts, batch_size, **gen_kwargs):
        """
        Run LLM pipeline on a list of prompts in mini-batches.
        Returns a flat list of outputs, same order as prompts.
        """
        all_outputs = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_outputs = pipe(batch_prompts, **gen_kwargs)
            all_outputs.extend(batch_outputs)
        return all_outputs
    
    def update_round(self, tokenizer, pipe, update_fraction=0.5, n_grams=[], distorted_tweets=[]):
        """
        Update the network for one round by responding to news intensities and adjusting the network accordingly.
        """
        self.iterations +=1
        prompts = []
        batch_size = 8
        agents_w_prompt = []

        # force tweets for first round
        if self.iterations == 1: # or len(self.activated) == 0:
            for agent in self.all_agents:
                # set default
                agent.send_tweet(max_chars=240, raw_tweet= "NO_TWEET")
            for agent in self.rng.choice(self.all_agents, int(len(self.all_agents) * update_fraction), replace=False):
                prompt = agent.step_llm_tweet(tokenizer, round_idx=self.iterations, force_active=True)
                prompts.append(prompt)
                agents_w_prompt.append(agent)

        # normal inferencing
        else:
            # randomize order of agent updates
            permuted = self.rng.permutation(self.all_agents)
            for agent in permuted:
                prompt = agent.step_llm_tweet(tokenizer, round_idx=self.iterations, force_active=False)
                prompts.append(prompt)
                agents_w_prompt.append(agent)
            assert len(agents_w_prompt) == len(self.all_agents), f"Agents w promt {len(agents_w_prompt)}, Agents: {len(self.all_agents)} All agents should have a prompt"
        
        #seed this thing 
        if self._torch_gen is None:
            dev = pipe.model.device
            self._torch_gen = torch.Generator(device=dev).manual_seed(self.seed)

        # generate outputs in parallel
        if prompts:
            gen_kwargs = dict(
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=256,
            kwargs={"generator": self._torch_gen},
            )
            out = self.inference_w_batches(pipe, prompts, batch_size=batch_size, **gen_kwargs)
        
        # agents send out their tweets
        for agent, tweet in zip(agents_w_prompt, out):
            if len(distorted_tweets)!=0 and agent == self.agent_w_highest_deg:
                    if self.iterations == 1:
                        print ("Agent with highest degree is tweeting distorted tweet")
                    agent.send_tweet(max_chars =240, raw_tweet = distorted_tweets[self.iterations % len(distorted_tweets)])
            else:
                agent.send_tweet(max_chars =240, raw_tweet = tweet[0]["generated_text"].strip())
        
        # self.cds_info = []
        # state update after all agents have decided
        distorted_this_step = 0
        distorted_fracs = []
        num_active_agents = 0
        for agent in self.all_agents:
            agent.commit(n_grams=n_grams)
            self.cds_info.append((agent.frac_distorted_neigh,  agent.activation_state))
            if agent.activation_state: 
                num_active_agents+=1

            # this agent always sends out distorted tweets
            if agent != self.agent_w_highest_deg or len(distorted_tweets) ==0:

                if len(agent.distorted_tweets) > 0: 

                    # compute distorted_tweetst this step
                    if agent.activation_state and agent.distorted_tweets[-1]:
                        assert agent.last_tweet != "NO_TWEET", "activated agent should have tweeted"
                        distorted_this_step +=1

                    # running window over last 5 tweets
                    distorted_fracs.append(np.sum(agent.distorted_tweets)/len(agent.distorted_tweets))
                    assert distorted_fracs[-1] <= 1, "error in distorted frac calculation"
                else:
                    distorted_fracs.append(0)
        
        if len(distorted_fracs) == 0:
            print("no distorted fracs recorded, returning 0")
            return 0, 0
        
        # prevent devision by 0
        if num_active_agents == 0: 
            dist_this_step_norm = 0
        else: 
            dist_this_step_norm =  distorted_this_step / num_active_agents
        return np.mean(distorted_fracs), dist_this_step_norm

        
class RandomNetwork(_Network):
    """
    This class represents a random network of agents.
    It inherits from the _Network class and initializes the network by connecting all agents with a probability `p`.
    """

    def __init__(self, p=0.1, k=0, depressed_personas=None, **kwargs):
        """
        Initialize the network by connecting all agents with a probability `p`.
        If `p` is very low, the network will resemble a regular network with fixed degree `k`.
        If `p` is high, it will resemble an Erdős–Rényi random network.

        Args:
            p (float): The probability of connecting two agents.
            k (int): The degree of the network.
        """
        super().__init__(**kwargs)
        self.p = p
        self.k = k

        self.initialize_network(depressed_personas=depressed_personas)

    def initialize_network(self, depressed_personas=None):
        """
        Initialize the network
        """
        if self.k >0:
            print(f"A Wattz-Strogatz network is initialized with beta value {self.p} and regular network degree {self.k}, and correlation {self.correlation}")
            # If degree `k` is provided, ensure each agent has exactly `k` connections.
            # This creates a regular network first, and then we adjust using `p`.
            for agent1 in self.all_agents:
                available_agents = self.all_agents.copy()
                # Create k regular connections for each agent
                # available_agents = list(self.all_agents - {agent1})
                available_agents.remove(agent1)
                for _ in range(self.k):
                    agent2 = self.rng.choice(available_agents)
                    self.add_connection(agent1, agent2)
                    available_agents.remove(agent2)

            # Now use `p` to add random edges between any pair of agents
            for agent1 in self.all_agents:
                for agent2 in self.all_agents:
                    if agent1 != agent2 and (agent2 not in agent1.agent_connections):
                        if self.rng.random() < self.p:
                            self.add_connection(agent1, agent2)
        else:
            print(f'A random network is initialized with p: {self.p} and {len(self.all_agents)} agents')
            # If no degree `k` is provided, fall back to the Erdős–Rényi model
            for agent1 in self.all_agents:
                for agent2 in self.all_agents:
                    if agent1 != agent2 and (agent2 not in agent1.agent_connections):
                        if self.rng.random() < self.p:
                            self.add_connection(agent1, agent2)
        self.agent_w_highest_deg = max(self.all_agents, key=lambda a: len(a.agent_connections))
        if depressed_personas is not None:
            # currently one persona in data 
            self.agent_w_highest_deg.persona = self.rng.choice(depressed_personas)
            print(f"Agent with highest degree is assigned depressed persona: {self.agent_w_highest_deg.persona['name']}, ID: {self.agent_w_highest_deg.ID}")

    def network_adjustment(self, sL, sR):
        """
        Adjust the network by breaking ties and adding new connections.

        Args:
            sL: Normalized significance for the left media hub.
            sR: Normalized significance for the right media hub.
        """
        self.new_edge = []
        self.removed_edge = []

        if len(self.activated)>0:
            # Select an active agent involved in the cascade
            # sort for reproducability purposes
            active_agent = self.rng.choice(list(sorted(self.activated, key=lambda x: x.ID)))

            if ((active_agent.identity == 'H' and sL <= active_agent.response_threshold) or
                (active_agent.identity == 'D' and sR <= active_agent.response_threshold)):

                # Break a tie with an active neighbor (use set for efficiency)
                active_neighbors = [n for n in active_agent.agent_connections if n.activation_state]
                number_of_connections = len(self.connections)

                # If active neighbors exist, remove an edge
                if len(active_neighbors) > 0:
                    
                    self.alterations+=1
                    
                    # remove edge, sort active neighbors for reproducability
                    break_agent = self.rng.choice(sorted(active_neighbors, key=lambda x: x.ID))
                    self.remove_connection(active_agent, break_agent)
                    self.removed_edge.extend([active_agent.ID, break_agent.ID])

                    # only if an edge is removed, add an extra edge.
                    # agent1 = self.rng.choice(list(self.all_agents))
                    agent1 = self.rng.choice(self.all_agents)
                    cant_be_picked = agent1.agent_connections.copy()
                    cant_be_picked.add(agent1)
                    # agent2 = self.rng.choice(List(self.all_agents - cant_be_picked))

                    filtered_agents = [agent for agent in self.all_agents if agent not in cant_be_picked]
                    agent2 = self.rng.choice(filtered_agents)
                    self.new_edge.extend([agent1.ID, agent2.ID])

                    # add edge
                    self.add_connection(agent1, agent2)

                assert number_of_connections == len(self.connections), "invalid operation took place, new number of edges is different than old"


class ScaleFreeNetwork(_Network):
    """
    This class represents a scale-free network of agents.
    It inherits from the _Network class and initializes the network by connecting agents in a scale-free manner.
    """
    def __init__(self, m=2, plot=False, depressed_personas=None, **kwargs):
        """
        Initialize the network by connecting agents in a scale-free manner.
        The network is initialized with `m` connections for each new agent.

        Args:
            m (int): The number of connections for each new agent.
            plot (bool): Boolean flag to indicate whether to plot the degree distribution.
        """
        super().__init__(**kwargs)
        self.m = m
        self.plot = plot
        self.degree_distribution = {} 
        self.total_degree = 0
        self.cumulative_degree_list = []

        self.initialize_network(depressed_personas=depressed_personas)

    def initialize_network(self, depressed_personas=None):
        """
        1) Select m initial agents, fully connect them (seed network).
        2) For each remaining agent, connect it to m existing agents with probability
        = (agent_degree / total_degree) using _pick_agent_by_degree_global().
        3) Assertions ensure total_degree > 0 for valid probability-based sampling.
        MAYBE REQUIRES WORK: some checks to ensure no agents gets stuck with degree < m
        """
        # Basic checks
        n = len(self.all_agents)
        assert self.m > 0, "m must be positive."
        assert self.m < n, "Number of connections 'm' must be less than number of agents."

        # Initialize degree_distribution to 0 for all agents
        for agent in self.all_agents:
            self.degree_distribution[agent] = 0

        # Step 1: Pick m initial agents and fully connect them
        #m0_agents = self.rng.choice(self.all_agents, self.m, replace=False)  # Use self.rng.choice for reproducibility

        m1 = int(self.m/2)
        if self.m %2 == 0:
            m2 = int(self.m/2)
        else:
            m2 = int(self.m/2) + 1

        # balanced out hubs
        m0_agents = np.concatenate([self.rng.choice(self.agentsD, m1, replace=False), self.rng.choice(self.agentsH, m2, replace=False)])

        if self.m > 1:  # Fully connect seed agents only if m > 1
            for i in range(len(m0_agents)):
                for j in range(i + 1, len(m0_agents)):
                    self.add_connection(m0_agents[i], m0_agents[j])

                    # if directed, enforce bidirection
                    if self.directed:
                        self.add_connection(m0_agents[j], m0_agents[i])

        else:  # Handle the case for m=1
            # If m=1, connect the seed agent to another random agent
            random_agent = self.rng.choice([agent for agent in self.all_agents if agent not in m0_agents])
            self.add_connection(m0_agents[0], random_agent)

        # Ensure total_degree is initialized properly
        assert self.total_degree > 0, "Seed network must have edges, so total_degree > 0."

        # Step 2: For the remaining agents, attach each with m edges via scale-free selection
        remaining_agents = [agent for agent in self.all_agents if agent not in m0_agents]
        for new_agent in remaining_agents:
            assert self.total_degree > 0, "Cannot do preferential attachment if total_degree = 0."

            # Use a set to track which agents have already been chosen
            chosen = set()
            forbidden = {new_agent}  # Prevent self-loops

            while len(chosen) < self.m:
                candidate = self._pick_agent_by_degree_global(forbidden=forbidden, max_tries=500)
                assert candidate is not None, "Initialization failed as no candidate for connection is found"
                chosen.add(candidate)
                forbidden.add(candidate)  # Ensure unique connections

            # Add edges to the chosen agents
            for target_agent in chosen:
                self.add_connection(new_agent, target_agent)
        
        assert all(self.degree_distribution[agent] >= self.m for agent in remaining_agents), (
            f"Some later added agents have degree less than m={self.m}. Check initialization logic."
        )

        self.agent_w_highest_deg = max(self.all_agents, key=lambda a: len(a.agent_connections))
        if depressed_personas is not None:
            # currently one persona in data 
            self.agent_w_highest_deg.persona = self.rng.choice(depressed_personas)
            print(f"Agent with highest degree is assigned depressed persona: {self.agent_w_highest_deg.persona['name']}, ID: {self.agent_w_highest_deg.ID}")
        # # Step 4: Verify the scale-free properties
        # self.verify_scale_free_distribution(self.plot)


    def _pick_agent_by_degree_global(self, forbidden=set(), max_tries=100):
        """
        Pick an agent (not in 'forbidden') by sampling from self.cumulative_degree_list.
        Returns the chosen agent or None if we fail after max_tries.
        """
        assert len(self.cumulative_degree_list) == len(self.degree_distribution), (
            "Cumulative degree list and degree distribution lengths do not match."
        )
        assert self.total_degree > 0, "Total degree must be positive for preferential sampling."

        for _ in range(max_tries):
            target_sum = self.rng.random() * self.total_degree
    
            # Use binary search to find the index of the selected agent
            idx = bisect.bisect_left(self.cumulative_degree_list, target_sum)
            if idx >= len(self.all_agents):
                idx = len(self.all_agents) - 1  # Safeguard against index overflow

            candidate = self.all_agents[idx]
            
            # Check if the candidate is not in the forbidden set
            if candidate not in forbidden:
                return candidate

        # If we fail after max_tries, return None
        assert candidate is None, "Failed to pick a agent after max_tries."
        return None

    def add_connection(self, agent1, agent2):
        """
        Add an undirected connection between two agents, updating:
            - self.connections
            - self.degree_distribution
            - self.total_degree
            - self.cumulative_degree_list
        """
        if agent1 != agent2 and (agent1, agent2) not in self.connections:
            agent1.add_edge(agent2)
            self.connections.add((agent1, agent2))
            self.degree_distribution[agent1] = self.degree_distribution.get(agent1, 0) + 1
            self.total_degree += 1 

            if not self.directed:
                agent2.add_edge(agent1)
                self.connections.add((agent2, agent1))
                # Update degree distribution
                self.degree_distribution[agent2] = self.degree_distribution.get(agent2, 0) + 1
                self.total_degree += 1  # 2 'ends' of edges

            # Rebuild the cumulative sums for probability sampling
            self._rebuild_cumulative_list()

    def remove_connection(self, agent1, agent2):
        """
        Remove an undirected connection between two agents (if it exists), updating:
            - self.connections
            - self.degree_distribution
            - self.total_degree
            - self.cumulative_degree_list
        """
        if agent1 != agent2 and (agent1, agent2) in self.connections:
            agent1.remove_edge(agent2)
            self.connections.remove((agent1, agent2))

            # Update degree distribution
            self.total_degree -= 1
            self.degree_distribution[agent1] -= 1

            if not self.directed:
                agent2.remove_edge(agent1)
                self.connections.remove((agent2, agent1))

                # update degree distribution
                self.total_degree -= 1
                self.degree_distribution[agent2] -= 1

            self._rebuild_cumulative_list()

    def _rebuild_cumulative_list(self):
        """
        Rebuild 'cumulative_degree_list' from 'degree_distribution'.
        cumulative_degree_list[i] = sum of degrees up to the i-th agent in iteration order.
        This is used for efficient probability-based agent selection via bisect.
        """
        self.cumulative_degree_list.clear()
        running_sum = 0
        for deg in self.degree_distribution.values(): # IMPORTANT THIS MAINTANS ORDER
            running_sum += deg
            self.cumulative_degree_list.append(running_sum)
        
    def network_adjustment(self, sL, sR):
        """
        Adjust the network by breaking ties and adding new connections in a scale-free manner.
        """

        # Ensure there are activated agents
        if len(self.activated) == 0:
            return

        # Select a valid active agent with more than m connections
        active_agents_list = list(sorted(self.activated, key=lambda x: x.ID))
        active_agent = self.rng.choice(active_agents_list)
        retries = 100  # Limit retries to avoid infinite loops

        while len(active_agent.agent_connections) <= self.m and retries > 0:
            active_agent = self.rng.choice(active_agents_list)
            retries -= 1

        if retries == 0:
            return

        assert len(active_agent.agent_connections) > self.m, "Selected active agent does not have enough connections."

        # Check if the active agent satisfies the conditions for breaking ties
        if not (
            (active_agent.identity == 'H' and sL <= active_agent.response_threshold)
            or (active_agent.identity == 'D' and sR <= active_agent.response_threshold)
        ):
            return  # Skip adjustment if the active agent does not meet conditions

        # Identify active neighbors
        active_neighbors = [n for n in active_agent.agent_connections if n.activation_state]
        assert len(active_neighbors) > 0, f"Active agent {active_agent} has no active neighbors to break ties with."
        active_neighbors = sorted(active_neighbors, key=lambda x: x.ID)
        for _ in range(100):  
            break_agent = self.rng.choice(active_neighbors)
            if len(break_agent.agent_connections) > self.m:
                self.remove_connection(active_agent, break_agent)
                break
        else:
            return 

        # Assert that the edge was removed successfully
        assert len(active_agent.agent_connections) >= self.m, "Edge removal violated minimum degree constraint."
        assert len(break_agent.agent_connections) >= self.m, "Edge removal violated minimum degree constraint."

        # Add a new edge according to scale-free properties
        agent1 = self.rng.choice(self.all_agents)
        assert agent1 is not None, "Failed to pick a valid agent1 for rewiring."

        forbidden = set(agent1.agent_connections) | {agent1, active_agent}
        agent2 = self._pick_agent_by_degree_global(forbidden=forbidden)
        assert agent2 is not None, "Failed to pick a valid agent2 for rewiring."

        self.add_connection(agent1, agent2)

        self.alterations += 1

        # Ensure network integrity after adjustment
        assert all(len(agent.agent_connections) >= self.m for agent in [active_agent, break_agent, agent1, agent2]), (
            "Network adjustment violated the minimum degree constraint."
        )

class SocialDistanceAttachment(_Network):
    """
    This class represents a social distance attachment network of agents.
    It inherits from the _Network class and initializes the network by connecting agents based on social distance.
    """
    def __init__(self, alpha, m, dist_type= "uniform", **kwargs):
        """
        Initialize the network by connecting agents based on social distance.
        """
        super().__init__(**kwargs)
        # Additional initialization for social distance attachment can be added here
        self.alpha = alpha
        self.m = m
        self.dist_type = dist_type
        self.dist_matrix = None

    def initialize_network(self):
        """
        Initialize the network based on social distance attachment.
        """
        X = self.sample_positions(self.num_agents, self.m, self.dist_type)
        self.dist_matrix = cdist(X, X)   # shape (N, N) 
        np.fill_diagonal(self.dist_matrix, np.inf)  # avoid self-loops


    def sample_positions(N, m, space_type, n_clusters=4):
        if space_type == "uniform":
            # [0, 1]^m
            return np.random.rand(N, m)

        if space_type == "gaussian_clusters":
            # equally sized clusters
            pts_per_cluster = N // n_clusters
            rest = N - pts_per_cluster * n_clusters
            sizes = [pts_per_cluster]*n_clusters
            sizes[0] += rest

            centers = np.random.uniform(-1, 1, size=(n_clusters, m))
            X = []
            for c, size in zip(centers, sizes):
                X.append(c + 0.1 * np.random.randn(size, m))
            return np.vstack(X)

        if space_type == "lognormal":
            return np.random.lognormal(mean=0.0, sigma=1.0, size=(N, m))

        raise ValueError("unknown space_type")

