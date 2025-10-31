import numpy as np
from src.classes.agent import Agent
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from powerlaw import Fit
import bisect

class _Network:
    """
    This function is the parent class for the RandomNetwork and ScaleFreeNetwork classes.
    A network of agents, with a specified number of agents and a correlation between the two media hubs.
    The network can be initialized as a random network or a scale-free network.
    The network can be updated by responding to news intensities and adjusting the network accordingly.    
    """

    def __init__(self, num_agents=200, mean=0, starting_distribution=0.5, seed=None):
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

        self.rng = np.random.default_rng(seed)

        self.new_edge = []
        self.removed_edge = []

        self.agentsD = [Agent(i, "D", rng=self.rng) for i in range(int(num_agents * starting_distribution))]
        self.agentsH = [Agent(i + len(self.agentsL), "H", rng=self.rng) for i in range(int(num_agents * (1 - starting_distribution)))]
        self.connections = set()
        self.all_agents = self.agentsL + self.agentsR
    
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
            agent2.add_edge(agent1)
            self.connections.add((agent1, agent2))
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
            agent2.remove_edge(agent1)
            self.connections.remove((agent1, agent2))
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

    
    def pick_samplers(self):
        """
        Pick samplers for the Depressed and Happy media hubs.

        Returns:
            Tuple of sets of samplers for the depressed and happy media hubs.
        """
        
        all_samplers_H, all_samplers_D = set(), set()
        # for agent in self.rng.choice(list(self.all_agents), int(len(self.all_agents) * self.update_fraction), replace=False):
        for agent in self.rng.choice(self.all_agents, int(len(self.all_agents) * self.update_fraction), replace=False):
            if agent.identity == 'H':
                all_samplers_H.add(agent)
            elif agent.identity == 'D':
                all_samplers_D.add(agent)
            else:
                raise ValueError("node identity should be assigned")
            # assert agent.sampler_state == False, "at this point all samplers states should be false"
            assert agent.activation_state == False, "at this point all nodes should be inactive"
            agent.sampler_state = True
        return (all_samplers_H, all_samplers_D)

    def update_round(self):
        """
        Update the network for one round by responding to news intensities and adjusting the network accordingly.
        """
        self.iterations +=1
        # sL, sR = self.generate_news_significance()

        allsamplers = self.pick_samplers()
        # Respond to the news intensities, continue this untill steady state is reached
        # self.run_cascade(sL, sR, allsamplers)

        # Network adjustment
        # self.network_adjustment(sL, sR)

        # Reset states for next round
        for node in self.activated:
            node.reset_activation_state()
            
        self.activated = set()
        
class RandomNetwork(_Network):
    """
    This class represents a random network of nodes.
    It inherits from the _Network class and initializes the network by connecting all nodes with a probability `p`.
    """

    def __init__(self, p=0.1, k=0, **kwargs):
        """
        Initialize the network by connecting all nodes with a probability `p`.
        If `p` is very low, the network will resemble a regular network with fixed degree `k`.
        If `p` is high, it will resemble an Erdős–Rényi random network.

        Args:
            p (float): The probability of connecting two nodes.
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
            # If degree `k` is provided, ensure each node has exactly `k` connections.
            # This creates a regular network first, and then we adjust using `p`.
            for node1 in self.all_nodes:
                available_nodes = self.all_nodes.copy()
                # Create k regular connections for each node
                # available_nodes = list(self.all_nodes - {node1})
                available_nodes.remove(node1)
                for _ in range(self.k):
                    node2 = self.rng.choice(available_nodes)
                    self.add_connection(node1, node2)
                    available_nodes.remove(node2)

            # Now use `p` to add random edges between any pair of nodes
            for node1 in self.all_nodes:
                for node2 in self.all_nodes:
                    if node1 != node2 and (node2 not in node1.node_connections):
                        if self.rng.random() < self.p:
                            self.add_connection(node1, node2)
        else:
            print(f'A random network is initialized with p: {self.p} and {len(self.all_nodes)} nodes and correlation {self.correlation}')
            # If no degree `k` is provided, fall back to the Erdős–Rényi model
            for node1 in self.all_nodes:
                for node2 in self.all_nodes:
                    if node1 != node2 and (node2 not in node1.node_connections):
                        if self.rng.random() < self.p:
                            self.add_connection(node1, node2)

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
            # Select an active node involved in the cascade
            # sort for reproducability purposes
            active_node = self.rng.choice(list(sorted(self.activated, key=lambda x: x.ID)))

            if ((active_node.identity == 'L' and sL <= active_node.response_threshold) or
                (active_node.identity == 'R' and sR <= active_node.response_threshold)):
                
                # Break a tie with an active neighbor (use set for efficiency)
                active_neighbors = [n for n in active_node.node_connections if n.activation_state]
                number_of_connections = len(self.connections)

                # If active neighbors exist, remove an edge
                if len(active_neighbors) > 0:
                    
                    self.alterations+=1
                    
                    # remove edge, sort active neighbors for reproducability
                    break_node = self.rng.choice(sorted(active_neighbors, key=lambda x: x.ID))
                    self.remove_connection(active_node, break_node)
                    self.removed_edge.extend([active_node.ID, break_node.ID])
                    
                    # only if an edge is removed, add an extra adge. 
                    # node1 = self.rng.choice(list(self.all_nodes))
                    node1 = self.rng.choice(self.all_nodes)
                    cant_be_picked = node1.node_connections.copy()
                    cant_be_picked.add(node1)
                    # node2 = self.rng.choice(List(self.all_nodes - cant_be_picked))

                    filtered_nodes = [node for node in self.all_nodes if node not in cant_be_picked]
                    node2 = self.rng.choice(filtered_nodes)
                    self.new_edge.extend([node1.ID, node2.ID])

                    # add edge
                    self.add_connection(node1, node2)
                
                assert number_of_connections == len(self.connections), "invalid operation took place, new number of edges is different than old"


class ScaleFreeNetwork(_Network):
    """
    This class represents a scale-free network of nodes.
    It inherits from the _Network class and initializes the network by connecting nodes in a scale-free manner.
    """
    def __init__(self, m=2, plot=False, **kwargs):
        """
        Initialize the network by connecting nodes in a scale-free manner.
        The network is initialized with `m` connections for each new node.

        Args:
            m (int): The number of connections for each new node.
            plot (bool): Boolean flag to indicate whether to plot the degree distribution.
        """
        super().__init__(**kwargs)
        self.m = m
        self.plot = plot
        self.degree_distribution = {} 
        self.total_degree = 0
        self.cumulative_degree_list = []

        self.initialize_network()

    def _pick_node_by_degree_global(self, forbidden=set(), max_tries=100):
        """
        Pick a node (not in 'forbidden') by sampling from self.cumulative_degree_list.
        Returns the chosen node or None if we fail after max_tries.
        """
        assert len(self.cumulative_degree_list) == len(self.degree_distribution), (
            "Cumulative degree list and degree distribution lengths do not match."
        )
        assert self.total_degree > 0, "Total degree must be positive for preferential sampling."

        for _ in range(max_tries):
            target_sum = self.rng.random() * self.total_degree
    
            # Use binary search to find the index of the selected node
            idx = bisect.bisect_left(self.cumulative_degree_list, target_sum)
            if idx >= len(self.all_nodes):
                idx = len(self.all_nodes) - 1  # Safeguard against index overflow

            candidate = self.all_nodes[idx]
            
            # Check if the candidate is not in the forbidden set
            if candidate not in forbidden:
                return candidate

        # If we fail after max_tries, return None
        assert candidate is None, "Failed to pick a node after max_tries."
        return None

    def add_connection(self, node1, node2):
        """
        Add an undirected connection between two nodes, updating:
            - self.connections
            - self.degree_distribution
            - self.total_degree
            - self.cumulative_degree_list
        """
        if node1 != node2 and (node1, node2) not in self.connections:
            node1.add_edge(node2)
            node2.add_edge(node1)
            self.connections.add((node1, node2))
            self.connections.add((node2, node1))

            # Update degree distribution
            self.degree_distribution[node1] = self.degree_distribution.get(node1, 0) + 1
            self.degree_distribution[node2] = self.degree_distribution.get(node2, 0) + 1
            self.total_degree += 2  # 2 'ends' of edges

            # Rebuild the cumulative sums for probability sampling
            self._rebuild_cumulative_list()

    def remove_connection(self, node1, node2):
        """
        Remove an undirected connection between two nodes (if it exists), updating:
            - self.connections
            - self.degree_distribution
            - self.total_degree
            - self.cumulative_degree_list
        """
        if node1 != node2 and (node1, node2) in self.connections:
            node1.remove_edge(node2)
            node2.remove_edge(node1)
            self.connections.remove((node1, node2))
            self.connections.remove((node2, node1))

            # Update degree distribution
            self.degree_distribution[node1] -= 1
            self.degree_distribution[node2] -= 1
            self.total_degree -= 2

            self._rebuild_cumulative_list()

    def _rebuild_cumulative_list(self):
        """
        Rebuild 'cumulative_degree_list' from 'degree_distribution'.
        cumulative_degree_list[i] = sum of degrees up to the i-th node in iteration order.
        This is used for efficient probability-based node selection via bisect.
        """
        self.cumulative_degree_list.clear()
        running_sum = 0
        for deg in self.degree_distribution.values(): # IMPORTANT THIS MAINTANS ORDER
            running_sum += deg
            self.cumulative_degree_list.append(running_sum)
        
    def initialize_network(self):
        """
        1) Select m initial nodes, fully connect them (seed network).
        2) For each remaining node, connect it to m existing nodes with probability 
        = (node_degree / total_degree) using _pick_node_by_degree_global().
        3) Assertions ensure total_degree > 0 for valid probability-based sampling.
        """
        # Basic checks
        n = len(self.all_nodes)
        assert self.m > 0, "m must be positive."
        assert self.m < n, "Number of connections 'm' must be less than number of nodes."

        # self.rng.shuffle(self.all_nodes)
        # identities = ['L'] * len(self.nodesL) + ['R'] * len(self.nodesR)
        # self.rng.shuffle(identities)
        # for i, node in enumerate(self.all_nodes):
        #     node.identity = identities[i]
        
        # Initialize degree_distribution to 0 for all nodes
        for node in self.all_nodes:
            self.degree_distribution[node] = 0

        
        # Step 1: Pick m initial nodes and fully connect them
        m0_nodes = self.rng.choice(self.all_nodes, self.m, replace=False)  # Use self.rng.choice for reproducibility

        m1 = int(self.m/2)
        if self.m %2 == 0:
            m2 = int(self.m/2)
        else:
            m2 = int(self.m/2) + 1

        # balanced out hubs
        m0_nodes = np.concatenate([self.rng.choice(self.nodesR, m1, replace=False), self.rng.choice(self.nodesL, m2, replace=False)])

        if self.m > 1:  # Fully connect seed nodes only if m > 1
            for i in range(len(m0_nodes)):
                for j in range(i + 1, len(m0_nodes)):
                    self.add_connection(m0_nodes[i], m0_nodes[j])
        else:  # Handle the case for m=1
            # If m=1, connect the seed node to another random node
            random_node = self.rng.choice([node for node in self.all_nodes if node not in m0_nodes])
            self.add_connection(m0_nodes[0], random_node)

        # Ensure cumulative degree list is rebuilt after seed network
        self._rebuild_cumulative_list()

        # Ensure total_degree is initialized properly
        assert self.total_degree > 0, "Seed network must have edges, so total_degree > 0."
        
        # can_be_picked = self.all_nodes.copy()
        # node2 = self.rng.choice(List(self.all_nodes - cant_be_picked))

        # Step 2: For the remaining nodes, attach each with m edges via scale-free selection
        remaining_nodes = [node for node in self.all_nodes if node not in m0_nodes]
        for new_node in remaining_nodes:
            assert self.total_degree > 0, "Cannot do preferential attachment if total_degree = 0."

            # Use a set to track which nodes have already been chosen
            chosen = set()
            forbidden = {new_node}  # Prevent self-loops

            while len(chosen) < self.m:
                candidate = self._pick_node_by_degree_global(forbidden=forbidden, max_tries=500)
                chosen.add(candidate)
                forbidden.add(candidate)  # Ensure unique connections

            # Add edges to the chosen nodes
            for target_node in chosen:
                self.add_connection(new_node, target_node)

        assert all(degree >= self.m for degree in self.degree_distribution.values()), (
            f"Some nodes have degree less than m={self.m}. Check initialization logic."
        )

        # Step 4: Verify the scale-free properties
        self.verify_scale_free_distribution(self.plot)

    def verify_scale_free_distribution(self, plot):
        """
        Check if the network exhibits scale-free characteristics
        """
        # Calculate node degrees
        degrees = [len(node.node_connections) for node in self.all_nodes]
        
        # Compute log-log plot for degree distribution
        degree_counts = {}
        for degree in degrees:
            degree_counts[degree] = degree_counts.get(degree, 0) + 1
        
        unique_degrees = list(degree_counts.keys())
        frequencies = list(degree_counts.values())
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.loglog(unique_degrees, frequencies, 'bo')
            plt.title('Degree Distribution (Log-Log Scale)')
            plt.xlabel('Degree')
            plt.ylabel('Frequency')
            plt.show()

        assert all(degree >= self.m for degree in self.degree_distribution.values()), (
        f"Some nodes have degree less than m={self.m}. Check initialization logic."
        )
        
        # Basic scale-free network indicators
        assert max(degrees) > np.mean(degrees) * 2, "Network lacks high-degree nodes"
        assert len([d for d in degrees if d > np.mean(degrees) * 2]) > 0, "No significant hub nodes"
        print("Intializing a scale-free network with m:", self.m)
        fit = Fit(degrees)
        print(f"Power-law fit: alpha={fit.power_law.alpha}, KS={fit.power_law.KS()}")
        assert fit.power_law.KS() < 0.5, f"Power-law fit is not significant; {fit.power_law.KS()}"
        # assert fit.power_law.alpha < 7, f"Power-law exponent is too high; {fit.power_law.alpha}"

    def network_adjustment(self, sL, sR):
        """
        Adjust the network by breaking ties and adding new connections in a scale-free manner.
        """

        # Ensure there are activated nodes
        if len(self.activated) == 0:
            return

        # Select a valid active node with more than m connections
        active_nodes_list = list(sorted(self.activated, key=lambda x: x.ID))
        active_node = self.rng.choice(active_nodes_list)
        retries = 100  # Limit retries to avoid infinite loops
        
        while len(active_node.node_connections) <= self.m and retries > 0:
            active_node = self.rng.choice(active_nodes_list)
            retries -= 1

        if retries == 0:
            return

        assert len(active_node.node_connections) > self.m, "Selected active node does not have enough connections."

        # Check if the active node satisfies the conditions for breaking ties
        if not (
            (active_node.identity == 'L' and sL <= active_node.response_threshold)
            or (active_node.identity == 'R' and sR <= active_node.response_threshold)
        ):
            return  # Skip adjustment if the active node does not meet conditions

        # Identify active neighbors
        active_neighbors = [n for n in active_node.node_connections if n.activation_state]
        assert len(active_neighbors) > 0, f"Active node {active_node} has no active neighbors to break ties with."
        active_neighbors = sorted(active_neighbors, key=lambda x: x.ID)
        for _ in range(100):  
            break_node = self.rng.choice(active_neighbors)
            if len(break_node.node_connections) > self.m:
                self.remove_connection(active_node, break_node)
                break
        else:
            return 

        # Assert that the edge was removed successfully
        assert len(active_node.node_connections) >= self.m, "Edge removal violated minimum degree constraint."
        assert len(break_node.node_connections) >= self.m, "Edge removal violated minimum degree constraint."

        # Add a new edge according to scale-free properties
        node1 = self.rng.choice(self.all_nodes)
        assert node1 is not None, "Failed to pick a valid node1 for rewiring."

        forbidden = set(node1.node_connections) | {node1, active_node}
        node2 = self._pick_node_by_degree_global(forbidden=forbidden)
        assert node2 is not None, "Failed to pick a valid node2 for rewiring."

        self.add_connection(node1, node2)

        self.alterations += 1

        # Ensure network integrity after adjustment
        assert all(len(node.node_connections) >= self.m for node in [active_node, break_node, node1, node2]), (
            "Network adjustment violated the minimum degree constraint."
        )