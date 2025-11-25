from classes.network import RandomNetwork, ScaleFreeNetwork
import ast, torch
import numpy as np
# import pickle, base64

# # helper to decode saved RNG state
# def _decode_rng_state(encoded: str):
#     raw = base64.b64decode(encoded.encode("ascii"))
#     return pickle.loads(raw)

def read_in_network_properties(file_path):
    properties = {}

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    for line in lines[2:]:  # Skip the header lines
        key, value = line.strip().split(": ", 1)

        if key in ("Number of Agents", "Number of Edges", "Seed", "Iterations", "Agent_w_Highest_Deg"):
            properties[key] = int(value)

        elif key in ( "P value", "Update Fraction"):
            properties[key] = float(value)

        # save distorted fracs as metric
        elif key == "Distorted Frac":
            # value = value.replace("nan", "0")
            distorted_fracs = ast.literal_eval(value)
            properties[key] = [float(f) for f in distorted_fracs]
        elif key == "Connections":
            # Parse connections as a list of tuples (id1, id2)
            connections = ast.literal_eval(value)
            properties[key] = [(int(a), int(b)) for a, b in connections]
        elif key == "CDS Info":
            cds_info = ast.literal_eval(value)
            properties[key] = [(float(frac_neigh), bool(act_agent)) for frac_neigh, act_agent in cds_info]
        elif key in ("Network RNG State", "Torch RNG State"):
            properties[key] = ast.literal_eval(value)
        elif key == "Agents":
            # Parse agents as a list of tuples
            # (agent_id, identity, activation_state, tweethistory, active_tweethistory, distorted_tweethistory, frac_distorted_neigh)
            # agents = ast.literal_eval(value)
            value = value.replace("nan", "None")
            try:
                agents = ast.literal_eval(value)
            except ValueError as e:
                print("Failed to literal_eval Agents value:")
                print(value)
                raise
            parsed_agents = []
            for agent_id, persona, activation_state, tweethistory, active_tweethistory, distorted_tweethistory, frac_distorted_neigh in agents:
                parsed_agents.append(
                    (
                        int(agent_id),
                        persona,
                        activation_state,
                        tweethistory,
                        active_tweethistory,
                        distorted_tweethistory,
                        float(frac_distorted_neigh),
                    )
                )
            properties[key] = parsed_agents

        else:
            properties[key] = value
    return properties

def read_out_network_properties(network, seed, distorted_fracs):
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
        - Seed
        - Connections
        - Agents
        - P value (for RandomNetwork)
        - Degree (k) (for RandomNetwork)
        - Initial Edges (m) (for ScaleFreeNetwork)
        - Total Degree (for ScaleFreeNetwork)
        - Degree Distribution (for ScaleFreeNetwork)
    """
    agent_info = []
    connection_IDs = []

    # Collect agent and connection information
    for agent in network.all_agents:
        agent_info.append((agent.ID, agent.persona, agent.activation_state, 
                           agent.tweethistory, agent.active_tweethistory[-5:],
                           agent.distorted_tweets[-5:], agent.frac_distorted_neigh))
    for conn in network.connections:
        connection_IDs.append((conn[0].ID, conn[1].ID))
    
    # Common properties for all network types
    properties = {
        "Number of Agents": len(network.all_agents),
        "Number of Edges": len(network.connections),
        "Seed": seed,
        "Connections": connection_IDs,
        "Agents": agent_info, 
        "Iterations": network.iterations,
        "Distorted Frac": [float(x) for x in distorted_fracs],
        "CDS Info": network.cds_info,
        "Agent_w_Highest_Deg": network.agent_w_highest_deg.ID,
    }

    # randomness:
    properties["Torch RNG State"] = network._torch_gen.get_state().tolist()
    properties["Network RNG State"] = network.rng.bit_generator.state

    # Add properties specific to RandomNetwork
    if isinstance(network, RandomNetwork):
        network_type = "rand"
        properties["P value"] = network.p
        properties["Degree (k)"] = network.k

    # Add properties specific to ScaleFreeNetwork
    elif isinstance(network, ScaleFreeNetwork):
        network_type = "sf"
        properties["Initial Edges (m)"] = network.m
        properties["Total Degree"] = network.total_degree
    else:
        print("Network should be either scale-free or random")

    file_output_path = f"data/networks/network_properties_{seed}_{network_type}_{len(network.all_agents)}_{network.iterations}.txt"
    with open(file_output_path, "w", encoding="utf-8") as file:
        file.write("Network Properties\n")
        file.write("==================\n")
        for key, value in properties.items():
            file.write(f"{key}: {value}\n")
    return file_output_path


def generate_network(file_path, pipe, starting_distribution=0.5):
    """
    Load a single network from a saved properties file created by get_network_properties.

    This fully reconstructs:
    - topology (connections)
    - per-agent state (persona, activation_state, tweet histories, frac_distorted_neigh)
    - iterations counter

    Args:
        file_path (str): Path to the saved properties file.
        starting_distribution (str): Whatever you passed originally to the constructor.

    Returns:
        network: A reconstructed RandomNetwork or ScaleFreeNetwork instance.
    """
    props = read_in_network_properties(file_path)
    
    # metric
    distorted_fracs = props["Distorted Frac"]

    # network props
    num_agents = props["Number of Agents"]
    seed = props["Seed"]
    iterations = props["Iterations"]

    # ---- 1. Instantiate the right network type (no correlations) ----
    if "P value" in props:
        # RandomNetwork
        p = props["P value"]
        network = RandomNetwork(
            num_agents=num_agents,
            starting_distribution=starting_distribution,
            seed=seed,
            p=p,
        )
    elif "Initial Edges (m)" in props:
        # ScaleFreeNetwork
        m = int(props["Initial Edges (m)"])
        network = ScaleFreeNetwork(
            num_agents=num_agents,
            m=m,
            starting_distribution=starting_distribution,
            seed=seed,
        )
        # If you really care you could also restore these cached values:
        # network.total_degree = props["Total Degree"]
        # network.degree_distribution = props["Degree Distribution"]
    else:
        raise ValueError("Could not infer network type from properties file.")

    # Make sure we continue from the saved iteration count
    network.iterations = iterations
    network.cds_info = props["CDS Info"]
    network.degree_distribution = {}

    # set randomness: 
    network.rng = np.random.default_rng()
    network.rng.bit_generator.state = props["Network RNG State"]

    network._torch_gen = torch.Generator(device=pipe.model.device).manual_seed(seed)
    state_tensor = torch.ByteTensor(props["Torch RNG State"])
    network._torch_gen.set_state(state_tensor)

    # create a map to map index ot id, (currently index is same as id)
    id_to_agent = {agent.ID: agent for agent in network.all_agents}

    # ---- 3. Restore agent-level state ----
    # (agent_id, persona, activation_state,
    #  tweethistory, active_tweethistory, distorted_tweethistory, frac_distorted_neigh)
    for (agent_id, persona, activation_state,
         tweethistory, active_tweethistory,
         distorted_tweethistory, frac_distorted_neigh) in props["Agents"]:

        ag = id_to_agent[agent_id]
        ag.persona = persona
        ag.activation_state = activation_state
        ag.tweethistory = list(tweethistory)
        ag.active_tweethistory = list(active_tweethistory)
        ag.distorted_tweets = list(distorted_tweethistory)
        ag.frac_distorted_neigh = frac_distorted_neigh
        ag.agent_connections = set()  # will be populated below

        # rebuild degree distribution when adding connections
        network.degree_distribution[ag] = 0

    # ---- 4. Restore connections ----
    # Clear whatever constructor created
    network.connections = set()

    # reset connections. 
    for id1, id2 in props["Connections"]:
        a1 = id_to_agent[id1]
        a2 = id_to_agent[id2]
        network.add_connection(a1, a2)

    network.agent_w_highest_deg = id_to_agent[props["Agent_w_Highest_Deg"] ]
    return network, distorted_fracs