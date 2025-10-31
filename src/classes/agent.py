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

    def reset_activation_state(self):
        self.activation_state = False
        
    def respond(self, intensity=0, analyze=False):
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
