import numpy as np
import torch
from classes.agent import Agent
from scipy.spatial.distance import cdist
from powerlaw import Fit

import bisect as bs_norm
from scipy.optimize import bisect
import utils.visualization as vis

class _Network:
    """
    This function is the parent class for the RandomNetwork and ScaleFreeNetwork classes.
    A network of agents, with a specified number of agents and a correlation between the two media hubs.
    The network can be initialized as a random network or a scale-free network.
    The network can be updated by responding to news intensities and adjusting the network accordingly.    
    """

    def __init__(self, num_agents=200, 
                 directed=False, 
                 seed=None, 
                 well_being = None, 
                 personas = None, 
                 state="basis"):
        """
        Initialize the network with a specified number of agents, mean, correlation, update fraction, and seed.

        Args:
            num_agents (int): The number of agents in the network.
            seed (int): The seed for the random number generator.

        Attributes:
            iterations (int): The number of iterations the network has been updated.
            activated (set): The set of activated agents.
            rng (np.random.Generator): The random number generator.
            well_being (list): The list of well-being scores for the agents.
            connections (set): The set of connections between agents.
            all_agents (list): The list of all agents in the network.
        """
        self.iterations = 0
        self.activated = set()
        self.directed = directed
        self.state = state

        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self._torch_gen = None

        personas = self.rng.permutation(personas) if personas is not None else [None]*num_agents
        self.well_being = self.rng.permutation(well_being) if well_being is not None else [None]*num_agents

        # create agents
        self.all_agents = [Agent(i, rng=np.random.default_rng(seed + i), persona=personas[i], 
                              well_being=self.well_being[i]) for i in range(int(num_agents))]
        self.connections = set()

        self.cds_info = []
        self.agent_w_highest_deg = self.all_agents[0] # placeholder
      

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
        if len(prompts) == 0:
            return []
        
        if True:
            all_outputs = []
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_outputs = pipe(batch_prompts, **gen_kwargs)
                all_outputs.extend(batch_outputs)
            

        return all_outputs
    
    def update_round(self, tokenizer, pipe, update_fraction=0.5, n_grams=[], distorted_tweets=[], check_point= 100):
        """
        Update the network for one round by responding to news intensities and adjusting the network accordingly.
        """
        self.iterations += 1
        batch_size = 8
        prompts, agents_w_prompt = self._prepare_prompts(tokenizer, update_fraction)
        update_score = False

        if self.iterations % check_point== 0:
            _= self._phq9_questionnaire(tokenizer, pipe)
            update_score = True
            

        # seed this thing
        if True:
            self._ensure_torch_generator(pipe)
        
        # generate outputs in parallel
        out = self._generate_outputs(pipe, prompts, batch_size)

        # agents send out their tweets + state update + stats
        mean_distorted_frac, dist_this_step_norm = self._apply_outputs_and_update_state(
            agents_w_prompt, out, n_grams, distorted_tweets, update_score=update_score
        )

        return mean_distorted_frac, dist_this_step_norm

    def _prepare_prompts(self, tokenizer, update_fraction):
        """
        Prepare prompts and collect agents that will receive prompts for this round.
        """
        prompts = []
        agents_w_prompt = []

        # force tweets for first round
        if self.iterations == 1:  # or len(self.activated) == 0:
            for agent in self.all_agents:
                # set default
                agent.send_tweet(max_chars=240, raw_tweet="NO_TWEET")
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
            assert len(agents_w_prompt) == len(self.all_agents), (
                f"Agents w promt {len(agents_w_prompt)}, Agents: {len(self.all_agents)} "
                f"All agents should have a prompt"
            )

        return prompts, agents_w_prompt
    

    def _phq9_questionnaire(self, tokenizer, pipe):
        """
        Have all agents complete the PHQ-9 questionnaire via LLM and update their well-being scores.
        Args:
            tokenizer: The tokenizer for the LLM.
            pipe: The LLM pipeline for generating responses.
        """
        # prepare prompts for all agents
        prompts = []
        for agent in self.all_agents:
            prompt = agent.phq9_questionnaire_prompt(tokenizer, agent.tweethistory[-20:])
            prompts.append(prompt)
        
        # inference with LLM
        out = self._generate_outputs(pipe, prompts, batch_size=8)
        
        # update well-being scores based on responses
        for agent, answer in zip(self.all_agents, out):
            questionnaire_answers = answer[0]["generated_text"].strip()
            sum_score = agent.parse_phq9_answers(questionnaire_answers)
            agent.update_well_being(sum_score)


    def _ensure_torch_generator(self, pipe):
        """
        Ensure a torch.Generator is initialized for reproducible LLM sampling.
        """
        if self._torch_gen is None:
            dev = pipe.model.device
            self._torch_gen = torch.Generator(device=dev).manual_seed(self.seed)


    def _generate_outputs(self, pipe, prompts, batch_size):
        """
        Run the LLM in batches over the given prompts.
        """
        if not prompts:
            return []
        
        gen_kwargs = dict(
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            max_new_tokens=256,
            kwargs={"generator": self._torch_gen},
        )
        out = self.inference_w_batches(pipe, prompts, batch_size=batch_size, **gen_kwargs)
        return out

    def _apply_outputs_and_update_state(self, agents_w_prompt, out, n_grams, distorted_tweets, update_score):
        """
        Use LLM outputs to update agents' tweets and activation states,
        then compute distorted tweet statistics for this round.
        """
        # agents send out their tweets
        for agent, tweet in zip(agents_w_prompt, out):
            
            if len(distorted_tweets) != 0 and agent == self.agent_w_highest_deg:
                if self.iterations == 1:
                    print("Agent with highest degree is tweeting distorted tweet")
                agent.send_tweet(
                    max_chars=240,
                    raw_tweet=distorted_tweets[self.iterations % len(distorted_tweets)],
                )
            # PREPAREE FOR VLLM
            else:
                raw = tweet[0]["generated_text"].strip()
                agent.send_tweet(
                    max_chars=240,
                    raw_tweet=raw,
                )

        # state update after all agents have decided
        distorted_this_step = 0
        distorted_fracs = []
        num_active_agents = 0
        for agent in self.all_agents:
            distorted = agent.commit(n_grams=n_grams, update_score=update_score)
            self.cds_info.append((agent.frac_distorted_neigh, agent.activation_state, distorted))
            if agent.activation_state:
                num_active_agents += 1

            # this agent always sends out distorted tweets
            if agent != self.agent_w_highest_deg or len(distorted_tweets) == 0:

                if len(agent.distorted_tweets) > 0:

                    # compute distorted_tweets this step
                    if agent.activation_state and agent.distorted_tweets[-1]:
                        assert agent.last_tweet != "NO_TWEET", "activated agent should have tweeted"
                        distorted_this_step += 1

                    # running window over last 5 tweets
                    distorted_fracs.append(np.sum(agent.distorted_tweets) / len(agent.distorted_tweets))
                    assert distorted_fracs[-1] <= 1, "error in distorted frac calculation"
                else:
                    distorted_fracs.append(0)

        if len(distorted_fracs) == 0:
            print("no distorted fracs recorded, returning 0")
            return 0, 0

        # prevent division by 0
        if num_active_agents == 0:
            dist_this_step_norm = 0
        else:
            dist_this_step_norm = distorted_this_step / num_active_agents

        return np.mean(distorted_fracs), dist_this_step_norm


class SocialDistanceAttachment(_Network):
    """
    This class represents a social distance attachment network of agents.
    It inherits from the _Network class and initializes the network by connecting agents based on social distance.
    """
    def __init__(self, 
                 alpha, 
                 dim, 
                 degree, 
                 sdc=False, 
                 plot=False, 
                 depressed_personas=None, 
                 dist_type= "gaussian_clusters", 
                 form_connections=True, 
                 **kwargs):
        """
        Initialize the network by connecting agents based on social distance.
        """
        super().__init__(**kwargs)
        # Additional initialization for social distance attachment can be added here
        self.alpha = alpha
        self.dim = dim
        self.dist_type = dist_type
        self.b = 0.0
        self.agent_positions = None
        self.degree = degree
        self.sdc = sdc
        
        if form_connections:
            self.initialize_network(depressed_personas=depressed_personas, plot=plot)

    def initialize_network(self, depressed_personas=None, plot=False):
        """
        Initialize the network based on social distance attachment.
        """

        # retrieve phq9 scores if available
        phq9_scores = []
        if any(agent.well_being and "phq9_sumscore" in agent.well_being for agent in self.all_agents):
            phq9_scores = [agent.well_being.get("phq9_sumscore", 0) if agent.well_being else 0 for agent in self.all_agents]

         # generate agent positions for distance calculations
        self.agent_positions = self.sample_positions(len(self.all_agents), space_type=self.dist_type, phq9_scores=phq9_scores)
        self.dist_matrix = cdist(self.agent_positions, self.agent_positions)
        # print("Distance matrix:", self.dist_matrix)
        print("N =", len(self.all_agents), "target degree =", self.degree, "max possible =", len(self.all_agents)-1)
        np.fill_diagonal(self.dist_matrix, np.inf)

        # find b parameter for target expected degree
        self.b = self.find_b_for_target_Ek()
        self.generate_connections()

        if depressed_personas is not None:
            # currently one persona in data 
            self.agent_w_highest_deg.persona = self.rng.choice(depressed_personas)
            print(f"Agent with highest degree is assigned depressed persona: {self.agent_w_highest_deg.persona['name']}, ID: {self.agent_w_highest_deg.ID}")

        self.verify_scale_free_distribution(plot)

    def sample_positions(self, N, space_type, n_clusters=4, phq9_scores=[]):
        """ Sample agent positions in a given space type.
        Args:
            N (int): number of agents
            space_type (str): type of space to sample positions from
            n_clusters (int): number of clusters for gaussian_clusters space type
            phq9_scores (list): list of PHQ-9 scores for agents
        Returns:
            positions (np.ndarray): sampled positions of shape (N, dim)
        """

        assert self.dim >= 0, "Dimension must be non-negative"
        positions = None
        
        if self.dim > 0 and len(phq9_scores) > 0:
            use_dim = self.dim - 1
        else:
            use_dim = self.dim
        
        # add additional dimension for phq9 scores if specified and dim > 1
        if use_dim > 0:
            if space_type == "uniform":
                # [0, 1]^m
                positions = self.rng.random((N, use_dim))

            elif space_type == "gaussian_clusters":
                # equally sized clusters
                pts_per_cluster = N // n_clusters
                rest = N - pts_per_cluster * n_clusters
                sizes = [pts_per_cluster]*n_clusters

                # assign remaining points
                sizes[0] += rest

                # generate centers of gaussians
                centers = self.rng.uniform(-1, 1, size=(n_clusters, use_dim))
                agent_points = []
                for c, size in zip(centers, sizes):
                    # generate points around each center
                    agent_points.append(c + 0.1 * self.rng.standard_normal((size, use_dim)))
                positions = np.vstack(agent_points)
            elif space_type == "lognormal":
                positions = self.rng.lognormal(mean=0.0, sigma=1.0, size=(N, use_dim))
            else:
                raise ValueError("unknown space_type")
            
        # add phq9 as an additional dimension
        if len(phq9_scores) > 0:

            # make column array to stack later
            phq9_array = np.array(phq9_scores).reshape(-1, 1)

            # Normalize to [0, 1]
            if phq9_array.max() > phq9_array.min():
                phq9_norm = (phq9_array - phq9_array.min()) / (phq9_array.max() - phq9_array.min())
            else:
                phq9_norm = np.zeros_like(phq9_array)*0.5
            
            phq9_norm = 2*phq9_norm - 1  # scale to [-1, 1]
            if positions is None:
                positions = phq9_norm
            else:
                positions = np.hstack((positions, phq9_norm))
        # print(positions)
        return positions

    @staticmethod
    def expected_degree_for_b(b, D, alpha):
        """helper function to compute expected degree for given b"""
        pij = 1.0 / (1.0 + (D / b)**alpha)

        # no self-loops
        np.fill_diagonal(pij, 0.0)
        N = D.shape[0]
        return pij.sum() / N

    def find_b_for_target_Ek(self, tol=1e-3):
        """
        find b such that expected degree is degree using bisection
        by searching in log space
        """
        # work in log-space: b = exp(z)

        def f(z):
            # searching in log space
            b = np.exp(z)
            return self.expected_degree_for_b(b, self.dist_matrix, self.alpha) - self.degree

        # bracket in log-space
        Dmax = self.dist_matrix[~np.isinf(self.dist_matrix)].max()
        z_low  = np.log(1e-6)
        z_high = np.log(Dmax * 10 + 1e-3)

        # bisect to find b 
        z_star = bisect(f, z_low, z_high, xtol=tol)
        return np.exp(z_star)
    

    def sda_graph(self, N):
        """ Generate a social distance attachment graph with N nodes in dim dimensions.
        Args:
            N (int): number of nodes

        Returns:
            A (np.ndarray): adjacency matrix of the generated graph
            P (np.ndarray): positions of the nodes
        """
        
        
        # generate connection probabilities between nodes
        P = 1.0 / (1.0 + (self.dist_matrix / self.b)**self.alpha)
        np.fill_diagonal(P, 0.0)

        # sample adjacency matrix (undirected)
        rand_probs = self.rng.random((N, N))
        A = (rand_probs < P).astype(int)


        if self.directed:
            return P, A
        
        # generate adjacency matrix (enforce symmetry (undirected))
        A = np.triu(A, 1)
        A = A + A.T

        return P, A
    
    def generate_connections(self):
        ''' Generate connections based on social distance attachment.
        ''' 

        total_degree = 0
        n = len(self.all_agents)
        if self.sdc:
            stud_list = self.generate_stub_list(n, gamma=2.5, degree=self.degree)
            print(f"Generated stub list with mean degree {np.mean(stud_list):.2f}")
            P, _ = self.sda_graph(n)
            adjacency = self.network_powerlaw(P, stud_list)
        else:
            _, adjacency = self.sda_graph(n)
        for i in range(n):
            if self.directed:
                start_val = 0
            else:
                start_val = i + 1
            for j in range(start_val, n):
                if adjacency[i, j] == 1:
                    self.add_connection(self.all_agents[i], self.all_agents[j])
                    total_degree += 1
                    if not self.directed:
                        total_degree += 1
        
        print(f"Social Distance Attachment network initialized with average degree {total_degree / n:.2f}")

    
    def generate_stub_list(self, N, gamma, degree):
        """ Generate a list of stubs for each node based on desired degree.

        Returns:
            stud_list (list): list of remaining stubs for each node
        """

        # generate degrees from powerlaw (scaled later)
        stud_list = self.rng.zipf(gamma, size=N)

        # consider to clip at 0
        stud_list = np.clip(stud_list, a_min=1, a_max=N-1)

        mean_degree = np.mean(stud_list)
        if mean_degree == 0:
            raise ValueError("Mean degree is zero, no connections can be made.")

        # scale to desired degree
        stud_list = (stud_list / mean_degree) * degree
        stud_list = np.round(stud_list).astype(int)

        # ensure sum of degrees is even
        if stud_list.sum() % 2 == 1:
            idx = self.rng.integers(0, N)
            if stud_list[idx] < N - 1:
                stud_list[idx] += 1
            else:
                stud_list[idx] -= 1

        return stud_list
    
    def network_powerlaw(self, P, stud_list):
        """ Generate a scale-free network withg SDC.

        Args:
            P (np.ndarray): connection probability matrix
            stud_list (list): list of remaining stubs for each node
        Returns:
            A (np.ndarray): adjacency matrix of the generated graph
        """
        prob_m = 10**-9
        A = np.zeros_like(P)
        P.clip(min=prob_m, max=1, out=P)

        # don't allow nodes with 0 stubs to connect or self-loops
        P[stud_list <= 0, :] = 0.0
        P[:, stud_list <= 0] = 0.0
        np.fill_diagonal(P, 0.0)

        assert np.sum(stud_list) > 0 , "stub list sum is 0, cannot generate powerlaw network"

        while np.sum(stud_list) > 0:
            
            if P.sum() == 0:
                print("No more possible connections can be made, exiting loop.")
                break

            # select agent 1 based on distance probabilities
            select_probs = P.sum(axis=1)/P.sum()
            agent1_ID = self.rng.choice(len(self.all_agents), p=select_probs)
            possible_conn = P[agent1_ID, :]

            # no possible neighbors for this agent
            if possible_conn.sum() == 0:
                stud_list[agent1_ID] = 0
                P[agent1_ID, :] = 0
                continue
            
            # select agent 2 based on distance probabilities
            select_probs_agent2 = P[agent1_ID, :]/P[agent1_ID, :].sum()
            agent2_ID = self.rng.choice(len(self.all_agents), p=select_probs_agent2)

            # update stub list
            stud_list[agent1_ID] -= 1

            # update matrices
            A[agent1_ID, agent2_ID] = 1
            P[agent1_ID, agent2_ID] = 0
            if stud_list[agent1_ID] == 0:
                P[agent1_ID, :] = 0.0
                P[:, agent1_ID] = 0.0

            # if undirected, update the other direction
            if not self.directed:
                stud_list[agent2_ID] -= 1
                A[agent2_ID, agent1_ID] = 1
                P[agent2_ID, agent1_ID] = 0
                if stud_list[agent2_ID] == 0:
                    P[agent2_ID, :] = 0.0
                    P[:, agent2_ID] = 0.0
        
        print("leftowving stubs after powerlaw network generation:", np.sum(stud_list))
        return A
    
    def verify_scale_free_distribution(self, plot):
        """
        Check if the network exhibits scale-free characteristics
        """
        # Calculate node degrees
        degrees = [len(agent.agent_connections) for agent in self.all_agents]
        
        # Compute log-log plot for degree distribution
        degree_counts = {}
        for degree in degrees:
            degree_counts[degree] = degree_counts.get(degree, 0) + 1
        
        unique_degrees = list(degree_counts.keys())
        frequencies = list(degree_counts.values())
        
        if plot:
            vis.check_degree_distribution(unique_degrees, frequencies)

        fit = Fit(degrees)
        print(f"Power-law fit: alpha={fit.power_law.alpha}, KS={fit.power_law.KS()}")
        if self.sdc:    
            assert fit.power_law.KS() < 0.5, f"Power-law fit is not significant; {fit.power_law.KS()}"
            # assert fit.power_law.alpha < 7, f"Power-law exponent is too high; {fit.power_law.alpha}"

        



        

                