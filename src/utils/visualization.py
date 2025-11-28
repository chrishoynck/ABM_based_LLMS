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

    path = f"plots/{setting}/{type_nn}/{parameter}"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f"{path}/distorted_step_fracs.png", dpi=300)
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

    plt.savefig(f"{path}/running_fracs.png", dpi=300)
    plt.show()




