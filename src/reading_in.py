from src.classes.network import RandomNetwork, ScaleFreeNetwork
from src.classes.agent import Agent
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from multiprocessing import Pool
import numpy as np
from collections import defaultdict

PROCESSES = 10

def get_network_properties(network, seed):
    """
    Extracts and returns the properties of a network for analysis or storage.
    Supports RandomNetwork and ScaleFreeNetwork.
    Stores it in a dictionary, values can be accessed with the corresponding keys. 
    Useful for effectively extracting network properties. 

    Args:
        network (object): The network object to extract properties from.
        seed (int): The seed used for network generation.

    Returns:
        dict: A dictionary containing the properties of the network:
        - Number of Agents
        - Number of Edges
        - Correlation
        - Seed
        - Update Fraction
        - Connections
        - Agents
        - P value (for RandomNetwork)
        - Degree (k) (for RandomNetwork)
        - Initial Edges (m) (for ScaleFreeNetwork)
        - Total Degree (for ScaleFreeNetwork)
        - Degree Distribution (for ScaleFreeNetwork)
    """
    corr = network.correlation
    agent_info = []
    connection_IDs = []

    # Collect agent and connection information
    for agent in network.all_agents:
        agent_info.append((agent.ID, agent.identity, agent.response_threshold))
    for conn in network.connections:
        connection_IDs.append((conn[0].ID, conn[1].ID))
    
    # Common properties for all network types
    properties = {
        "Number of Agents": len(network.all_agents),
        "Number of Edges": len(network.connections),
        "Correlation": corr,
        "Seed": seed,
        "Update Fraction": network.update_fraction,
        "Connections": connection_IDs,
        "Agents": agent_info
    }

    # Add properties specific to RandomNetwork
    if isinstance(network, RandomNetwork):
        properties["P value"] = network.p
        properties["Degree (k)"] = network.k

    # Add properties specific to ScaleFreeNetwork
    elif isinstance(network, ScaleFreeNetwork):
        properties["Initial Edges (m)"] = network.m
        properties["Total Degree"] = network.total_degree
        properties["Degree Distribution"] = network.degree_distribution
    else:
        print("Network should be either scale-free or random")

    return properties


def parallel_network_generation(whichrun, num_agents, seed, corr, iterations, update_fraction, starting_distribution, p, m=0, network_type="random"):
    """
    Generates and simulates a network in parallel.

    This function creates a network of a specified type, runs multiple iterations 
    to update its structure, and saves its properties to a file.

    Args:
        whichrun: Index of the current run, used to adjust the seed.
        num_agents: Number of agents in the network.
        seed: Base random seed for initialization.
        corr: Correlation value of the news.
        iterations: Number of iterations to update the network.
        update_fraction: Fraction of agents sampled for news. 
        starting_distribution: fraction left oriented vs right oriented agents
        p : Probability parameter for random network generation.
        m: Parameter for scale-free networks.
        network_type: Type of network ("random" or "scale_free", default: "random").

    Outputs:
        - Saves network properties to a file in a predefined directory.
        - Prints the number of alterations made during simulation.
    """
    seed += whichrun
    # Dynamically select the network class
    if network_type == "random":
        network = RandomNetwork(num_agents=num_agents, mean=0, correlation=corr, update_fraction=update_fraction, starting_distribution=starting_distribution, seed=seed, p=p)
    elif network_type == "scale_free":
        network = ScaleFreeNetwork(num_agents=num_agents, mean=0, correlation=corr, update_fraction=update_fraction, starting_distribution=starting_distribution, seed=seed, m=m)
    else:
        raise ValueError(f"Unsupported network type: {network_type}")

    # Prepare the output directory
    output_folder = f"networks/dummy/{network_type}/{corr}" 
    output_filename = f"network_{whichrun}.txt"  
    output_path = os.path.join(output_folder, output_filename)
    os.makedirs(output_folder, exist_ok=True)
    number_of_alterations = 0

    # Simulate the network over multiple iterations
    for _ in range(iterations):
        network.update_round()
        number_of_alterations += network.alterations
        network.clean_network()
    print(f"Number of alterations for run {whichrun}: {number_of_alterations}")
    
    # Get network properties
    network_properties = get_network_properties(network, seed)

    # Write the properties to the file
    with open(output_path, "w") as file:
        file.write("Network Properties\n")
        file.write("==================\n")
        for key, value in network_properties.items():
            file.write(f"{key}: {value}\n")



def generate_networks(correlations, initial_seeds, num_agents, iterations, how_many, update_fraction, starting_distribution, p, network_sort="random", m=0):
    """
    Generates multiple networks in parallel.

    This function creates networks using different correlation values and seeds,
    running the process in parallel.

    Args:
        correlations (list of float): Correlation values for network generation.
        initial_seeds (list of int): Seeds for random initialization.
        num_agents (int): Number of agents in each network.
        iterations (int): Number of iterations to update the network.
        how_many (int): Number of networks to generate per correlation.
        update_fraction (float): Fraction of agents updated per iteration.
        starting_distribution (str): Initial distribution type.
        p (float): Probability parameter for network formation.
        network_sort (str, optional): Network generation method (default: "random").
        m (int, optional): Extra parameter affecting network structure (default: 0).
    """
    print(f"starting parallel generation of {network_sort} networks ({num_agents} agents)")
    print("-----------------------------------------")
    runs = np.arange(how_many)  # Create a range for the runs
    num_threads = min(how_many, 10)
    
    for j, corr in enumerate(correlations): 
        print(f"Starting correlation {corr}")
        seed = int(initial_seeds[j])
        
        # Partially apply parameters for the worker function
        worker_function = partial(
            parallel_network_generation,
            num_agents=num_agents,
            seed=seed,
            corr=corr,
            iterations=iterations,
            update_fraction=update_fraction,
            starting_distribution=starting_distribution,
            p=p,
            m=m,
            network_type=network_sort,
        )
        
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            list(executor.map(worker_function, runs))


def read_network_properties(file_path):
    """
    Reads network properties from a .txt file and converts them back
    into a dictionary with appropriate datatypes.

    Args:
        file_path (str): Path to the .txt file containing network properties.

    Returns:
        dict: Network properties with restored data types.
    """
    properties = {}

    with open(file_path, "r") as file:
        lines = file.readlines()
    
    for line in lines[2:]:  # Skip the header lines
        key, value = line.strip().split(": ", 1)
        if key == "Number of Agents" or key == "Number of Edges":
            properties[key] = int(value)
        elif key == "Correlation" or key == "P value" or key == "Update fraction":
            properties[key] = float(value)
        elif key == "Seed":
            properties[key] = int(value)
        elif key == "Connections":
            # Parse connections as a list of tuples
            connections = eval(value)  # Use eval to safely parse the list of tuples
            properties[key] = [(int(a), int(b)) for a, b in connections]
        elif key == "Agents":
            # Parse agents as a list of tuples
            agents = eval(value)  # Use eval to safely parse the list of tuples
            properties[key] = [(int(agent_id), identity, float(threshold)) for agent_id, identity, threshold in agents]

        else:
            properties[key] = value
    return properties


def read_and_load_networks(num_runs, num_agents, update_fraction, average_degree, starting_distribution, correlations, whichtype):
    """
    Reads and loads networks from stored files.

    This function loads network structures from saved files, reconstructs them, 
    and returns a dictionary mapping (correlation, run index) to the network states.

    Args:
        num_runs (int): Number of network runs to load per correlation.
        num_agents (int): Number of agents in each network.
        update_fraction (float): Fraction of agents updated per iteration.
        average_degree (float): Average degree of agents in the network.
        starting_distribution (str): Initial distribution type.
        correlations (list of float): Correlation values for network generation.
        whichtype (str): Type of network ("random" or "scale-free").

    Returns:
        dict: A dictionary where keys are (correlation, run index) tuples, 
              and values are (before_network, after_network) pairs.
    """
    p = average_degree/(num_agents-1) 
    networks = defaultdict(tuple)
    for corr in correlations:
        for i in range(num_runs):
            network_properties = read_network_properties(f"networks/{whichtype}/{corr}/network_{i}.txt")
            seedje = network_properties["Seed"]
            search_agents = defaultdict(Agent)

            if whichtype == "random":
                before_network = RandomNetwork(num_agents=num_agents, mean=0, correlation=corr, update_fraction=update_fraction, starting_distribution=starting_distribution, seed=seedje, p=p) 
                after_network = RandomNetwork(num_agents=num_agents, mean=0, correlation=corr, update_fraction=update_fraction, starting_distribution=starting_distribution, seed=seedje, p=p) 
            else: 
                m= int(network_properties["Initial Edges (m)"])
                before_network = ScaleFreeNetwork(num_agents=num_agents, m=m, mean=0, correlation=corr, update_fraction=update_fraction, starting_distribution=starting_distribution, seed=seedje) 
                after_network = ScaleFreeNetwork(num_agents=num_agents, m=m, mean=0, correlation=corr, update_fraction=update_fraction, starting_distribution=starting_distribution, seed=seedje) 
            after_network.connections = set()

            for agent in after_network.all_agents:
                agent.agent_connections = set()
                search_agents[agent.ID] = agent

            for (agent1, agent2) in network_properties["Connections"]:
                search_agents[agent1].agent_connections.add(search_agents[agent2])
                after_network.connections.add((search_agents[agent1], search_agents[agent2]))

            networks[(corr, i)] = (before_network, after_network)

    return networks


def read_and_load_network_sub(sub_id, corr, num_agents, update_fraction, average_degree, starting_distribution, whichtype):
    """
    Reads and reconstructs a single network from a stored file.

    This function loads a specific network instance based on its ID and correlation value,
    then reconstructs its structure and connections.

    Args:
        sub_id (int): The identifier of the network instance to load.
        corr (float): The correlation value associated with the network.
        num_agents (int): Number of agents in the network.
        update_fraction (float): Fraction of agents updated per iteration.
        average_degree (float): Average degree of agents in the network.
        starting_distribution (str): Initial distribution type.
        whichtype (str): Type of network ("random" or "scale-free").

    Returns:
        tuple: A pair (before_network, after_network), where `before_network` represents
               the initial network structure and `after_network` represents the reconstructed
               network with updated connections.
    """
    p = average_degree/(num_agents-1) 

    network_properties = read_network_properties(f"networks/{whichtype}/{corr}/network_{sub_id}.txt")
    seedje = network_properties["Seed"]
    search_agents = defaultdict(Agent)

    if whichtype == "random":
        before_network = RandomNetwork(num_agents=num_agents, mean=0, correlation=corr, update_fraction=update_fraction, starting_distribution=starting_distribution, seed=seedje, p=p) 
        after_network = RandomNetwork(num_agents=num_agents, mean=0, correlation=corr, update_fraction=update_fraction, starting_distribution=starting_distribution, seed=seedje, p=p) 
    else: 
        m= int(network_properties["Initial Edges (m)"])
        before_network = ScaleFreeNetwork(num_agents=num_agents, m=m, mean=0, correlation=corr, update_fraction=update_fraction, starting_distribution=starting_distribution, seed=seedje)
        after_network = ScaleFreeNetwork(num_agents=num_agents, m=m, mean=0, correlation=corr, update_fraction=update_fraction, starting_distribution=starting_distribution, seed=seedje) 
    after_network.connections = set()

    for agent in after_network.all_agents:
        agent.agent_connections = set()
        search_agents[agent.ID] = agent

    for (agent1, agent2) in network_properties["Connections"]:
        search_agents[agent1].agent_connections.add(search_agents[agent2])
        after_network.connections.add((search_agents[agent1], search_agents[agent2]))

    return (before_network, after_network)