import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def print_network(network):
    """
    Print network at one single iteration

    Args:
        network: The network object to visualize.
    """
    color_map = ['lightblue'] * len(network.agentsD) + ['#FF6666'] * len(network.agentsH)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(network.all_agents)))
    graph.clear_edges()
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)
    
    # Set positions and draw the graph
    plt.figure(figsize=(16,8))
    pos = nx.kamada_kawai_layout(graph, scale=0.6)
    nx.draw(
        graph,
        pos,
        node_color=color_map,
        with_labels=True,
        edge_color="lightgray",
        width=0.2,
        node_size=400,
        font_size=10,
    )
    plt.savefig("network_snapshot.png", dpi=300)
    plt.show()
    return 

def distorted_info(cds_info):
    '''
    This function bins fractions of distorted neighbors, and plots the probability corresponding to that to tweet.
    Args:
        cds_info(List(Tuple)): List of tuples with cds_frac 
    '''
    cds_info = np.array(cds_info)
    cds_frac = cds_info[:, 0]
    tweeted = cds_info[:, 1]
    bins = 10
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_idx = np.digitize(cds_frac, bin_edges, right=True) - 1
    # make sure cds_frac == 1 also gets bin
    bin_idx = np.clip(bin_idx, 0, bins - 1)
    tweet_prob = []
    bin_centers = []
    for i in range(bins):
        if i not in bin_idx:
            tweet_prob.append(np.nan)
        else:
            frac_tweeted = np.mean(tweeted[bin_idx==i])
            tweet_prob.append(frac_tweeted)
        bin_centers.append(0.5 * (bin_edges[i] + bin_edges[i+1]))

    width = bin_edges[1] - bin_edges[0] 

    plt.bar(bin_centers, tweet_prob, width=width, align="center", edgecolor="black")
    plt.xlabel("Fraction of neighbors with CDS (prev round)")
    plt.ylabel("P(tweet)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3, axis="y")
    plt.show()

def plot_distorted_fracs(frac_distorted_this_step, m=0, p=0.0, enforced_ngrams=False, depressed=False, type_nn='rand'):
    '''
    This function plots the fraction of distorted tweets per round.
    Args:
        distorted_fracs(List(Float)): List of CDS fractions per round
    '''
    plt.plot(frac_distorted_this_step, marker='o', markersize=2)
    plt.xlabel("Round")
    plt.ylabel("Fraction of distorted tweets")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)

    if enforced_ngrams:
        setting = "enforced_ngrams"
    elif depressed:
        setting = "depressed"
    else:
        setting = "basis"

    if type_nn == 'sf':
        parameter = f'{m}'
    else:
        parameter = f'{str(p).replace(".", "")}'

    path = f"plots/networks/{setting}/{type_nn}/{parameter}"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f"{path}/distorted_step_fracs_3.png", dpi=300)
    plt.show()


def plot_running_fracs(running_fracs, m=0, p=0.0, enforced_ngrams=False, depressed=False, type_nn='rand'):
    '''
    This function plots the running mean fraction of distorted tweets over rounds.
    Args:
        running_fracs(List(Float)): List of running mean fractions over rounds
    '''
    plt.plot(running_fracs, marker='o')
    plt.xlabel("Round")
    plt.ylabel("Mean fraction of distorted active tweets (all agents)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)

    if enforced_ngrams:
        setting = "enforced_ngrams"
    elif depressed:
        setting = "depressed"
    else:
        setting = "basis"

    if type_nn == 'sf':
        parameter = f'{m}'
    else:
        parameter = f'{str(p).replace(".", "")}'

    path = f"plots/networks/{setting}/{type_nn}/{parameter}"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f"{path}/running_fracs_3.png", dpi=300)
    plt.show()

def plot_tf_idf_PCA(reduced_runs, states, num_steps=100, shift=5):
    '''
    This function plots the PCA-reduced TF-IDF data.
    Args:
        reduced_data (np.array): 2D array with reduced TF-IDF data.
        network: The network object.
        num_steps (int): Number of steps used in TF-IDF retrieval.
        shift (int): Shift used in TF-IDF retrieval.
    '''
    plt.figure(figsize=(3, 3))
    plt.title(f'TF-IDF PCA \n (window size={num_steps}, shift={shift})')
    for i, reduced in enumerate(reduced_runs):
        print("shape reduced:", reduced.shape)
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=10, label=states[i])
        plt.plot(reduced[:, 0], reduced[:, 1], alpha=0.4)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"plots/tf_idf_pca_window{num_steps}_shift{shift}_{len(states)}{states[0]}.png", bbox_inches='tight', dpi=300)
    plt.show()


def plot_tf_idf_PCA_runs(mean_traj, std_traj=None, num_steps=100, shift=5, save=False):
    """
    Plot PCA-reduced TF-IDF trajectories.

    Args:
        mean_traj: dict[setting] -> array (T, 2)
        std_traj:  dict[setting] -> array (T, 2), optional
        num_steps: TF-IDF window size
        shift:     TF-IDF shift
        save:      whether to save the figure
    """
    plt.figure(figsize=(3, 3))
    plt.title(f"TF-IDF PCA\n(window={num_steps}, shift={shift})")

    for setting, traj in mean_traj.items():
        traj = np.asarray(traj)  # (T, 2)

        if std_traj is not None and setting in std_traj:
            std = np.asarray(std_traj[setting])           # (T, 2)
            std_norm = np.linalg.norm(std, axis=1)        # (T,)
            # scale marker size a bit with std (optional)
            s = 5 + 20 * (std_norm / (std_norm.max() + 1e-8))
        else:
            s = 10

        plt.scatter(traj[:, 0], traj[:, 1], s=s, alpha=0.7, label=setting)
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.5)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def check_degree_distribution(unique_degrees, frequencies):
    plt.figure(figsize=(10, 6))
    plt.loglog(unique_degrees, frequencies, 'bo')
    plt.title('Degree Distribution (Log-Log Scale)')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()


