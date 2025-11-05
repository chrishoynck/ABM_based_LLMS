import numpy as np
from src.classes.agent import Agent
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
# from powerlaw import Fit
import bisect

class _Network:
    """
    This function is the parent class for the RandomNetwork and ScaleFreeNetwork classes.
    A network of agents, with a specified number of agents and a correlation between the two media hubs.
    The network can be initialized as a random network or a scale-free network.
    The network can be updated by responding to news intensities and adjusting the network accordingly.    
    """

    def __init__(self, num_agents=200, mean=0, starting_distribution=0.5, directed=False, seed=None):
        """
        Initialize the network with a specified number of agents, mean, correlation, starting distribution, update fraction, and seed.

        Args:
            num_agents (int): The number of agents in the network.
            mean (float): The mean of the news intensities.
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
        self.mean = mean
        self.activated = set()
        self.directed = directed

        self.rng = np.random.default_rng(seed)

        self.new_edge = []
        self.removed_edge = []

        self.agentsD = [Agent(i, "D", rng=self.rng) for i in range(int(num_agents * starting_distribution))]
        self.agentsH = [Agent(i + len(self.agentsD), "H", rng=self.rng) for i in range(int(num_agents * (1 - starting_distribution)))]
        self.connections = set()
        self.all_agents = self.agentsD + self.agentsH
      
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

    def generate_DSC_significance(self):
        """
        Generate news signifiance for both hubs based on their correlation.
        
        Returns:
            Tuple of news significance for the left and right media hubs.
        """
        covar = [[1, self.correlation ], [self.correlation, 1]]
        stims = self.rng.multivariate_normal(mean = [self.mean, self.mean], cov = covar, size = 1)
        stims_perc = stats.norm.cdf(stims, loc = 0, scale = 1) 
        return stims_perc[0][0], stims_perc[0][1]
    
    

    def update_round(self, tokenizer, pipe, update_fraction=0.5):
        """
        Update the network for one round by responding to news intensities and adjusting the network accordingly.
        """
        self.iterations +=1
        # force tweets for first round
        if self.iterations == 1: # or len(self.activated) == 0:
            
            for agent in self.rng.choice(self.all_agents, int(len(self.all_agents) * update_fraction), replace=False):
                agent.step_llm_tweet(tokenizer, pipe, round_idx=self.iterations, force_active=True)
        else:
            # randomize order of agent updates
            self.rng.shuffle(self.all_agents)
            for agent in self.all_agents:
                agent.step_llm_tweet(tokenizer, pipe, round_idx=self.iterations)
        for agent in self.all_agents:
            agent.commit()
            
        # self.activated = set()
        
class RandomNetwork(_Network):
    """
    This class represents a random network of agents.
    It inherits from the _Network class and initializes the network by connecting all agents with a probability `p`.
    """

    def __init__(self, p=0.1, k=0, **kwargs):
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

        self.initialize_network()

    def initialize_network(self):
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
    def __init__(self, m=2, plot=False, **kwargs):
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

        self.initialize_network()

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
            agent2.add_edge(agent1)
            self.connections.add((agent1, agent2))
            self.connections.add((agent2, agent1))

            # Update degree distribution
            self.degree_distribution[agent1] = self.degree_distribution.get(agent1, 0) + 1
            self.degree_distribution[agent2] = self.degree_distribution.get(agent2, 0) + 1
            self.total_degree += 2  # 2 'ends' of edges

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
            agent2.remove_edge(agent1)
            self.connections.remove((agent1, agent2))
            self.connections.remove((agent2, agent1))

            # Update degree distribution
            self.degree_distribution[agent1] -= 1
            self.degree_distribution[agent2] -= 1
            self.total_degree -= 2

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
        
    def initialize_network(self):
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
        m0_agents = self.rng.choice(self.all_agents, self.m, replace=False)  # Use self.rng.choice for reproducibility

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
        else:  # Handle the case for m=1
            # If m=1, connect the seed agent to another random agent
            random_agent = self.rng.choice([agent for agent in self.all_agents if agent not in m0_agents])
            self.add_connection(m0_agents[0], random_agent)

        # Ensure cumulative degree list is rebuilt after seed network
        self._rebuild_cumulative_list()

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
                chosen.add(candidate)
                forbidden.add(candidate)  # Ensure unique connections

            # Add edges to the chosen agents
            for target_agent in chosen:
                self.add_connection(new_agent, target_agent)

        assert all(degree >= self.m for degree in self.degree_distribution.values()), (
            f"Some agents have degree less than m={self.m}. Check initialization logic."
        )

        # # Step 4: Verify the scale-free properties
        # self.verify_scale_free_distribution(self.plot)



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