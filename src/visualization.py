import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from collections import defaultdict
from scipy.stats import ttest_ind_from_stats
import networkx as nx
import numpy as np
import powerlaw
from matplotlib.patches import Rectangle
from PIL import Image, ImageSequence
import os


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
    return "ja"

def initial_graph(network, clusters=[]):
    """
    Create the initial graph and assign node colors.

    Args:
        network: The network object.

    Returns:
        A tuple containing the graph and the list of node colors.
    """
    colors = ['lightblue'] * len(network.agentsH) + ['#FF6666'] * len(network.agentsD)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(network.all_agents)))

    return graph, colors


def self_sort(frame, network, graph, colors, pos, pos_target, ax, seed):
    """
    Update the graph for each animation frame.

    Args:
        frame: Current frame number.
        network: The network object.
        graph: The NetworkX graph object.
        colors: List of node colors.
        pos: Current node positions.
        pos_target: Target positions for interpolation.
        ax: Matplotlib axis object.
        seed: Seed for random layout generation.

    Returns:
        List of artists to update.
    """
    print(f"Processing frame {frame}")
    framestep = 200
    limit = 100000
    alterations_per_frame = 1
    frames_per_alteration = 15
    ax.clear()

    # static counter to track sub-frames
    if not hasattr(self_sort, "frame_counter"):
        self_sort.frame_counter = 0

    if self_sort.frame_counter == 0:
        current_alteration = network.give_alterations()

        while network.give_alterations() == current_alteration and network.iterations <= limit:
            network.update_round()

        # stop animation if desired iteration limit is reached
        if network.iterations >= limit:
            self_sort.ani.event_source.stop()
            print("Iteration limit reached")
            return []

    self_sort.frame_counter = (self_sort.frame_counter + 1) % frames_per_alteration

    # clear and rebuild edges in the graph
    graph.clear_edges()
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)

    # re-add the removed edge for visualization
    graph.add_edge(network.removed_edge[0], network.removed_edge[1])

    # calculate averages for right and left nodes
    average_right, average_left = calculate_fraction(network)

    # define kawai layout <:-)
    pos_target.update(nx.kamada_kawai_layout(graph, scale=0.8))

    # smoothly interpolate positions between frames
    alpha = (frame % framestep) / 1000

    # 
    for node in pos:
        pos[node] = pos[node] * (1 - alpha) + pos_target[node] * alpha

    # draw nodes
    nodes = nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors, node_size=400)

    # draw regular edges in gray
    regular_edges = [
        edge for edge in graph.edges if edge != network.removed_edge and edge != network.new_edge
    ]
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=regular_edges, edge_color="gray", width=0.5)

    # draw removed edge in red
    if network.removed_edge in graph.edges:
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=[network.removed_edge], edge_color="red", width=1.5)

    # draw new edge in green
    if network.new_edge in graph.edges:
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=[network.new_edge], edge_color="green", width=1.5)

    # add title
    ax.text(
    0.94, 1, 
    f"ITERATIONS   {network.iterations}", 
    fontsize=16, 
    color="gray", 
    transform=ax.transAxes, 
    ha="left", 
    va="center", 
    fontfamily="Arial"  
    )

    ax.text(
    0.94, 0.96, 
    f"ALTERATIONS  {network.give_alterations()}", 
    fontsize=16, 
    color="gray", 
    transform=ax.transAxes, 
    ha="left", 
    va="center", 
    fontfamily="Arial" 
    )

    # red circle for right average
    ax.add_patch(Circle((0.96, 0.91), 0.02, color="#FF6666", transform=ax.transAxes))
    text_right = ax.text(
        0.99, 0.91, f"           {average_right:.2f}",
        fontsize=16, color="gray", transform=ax.transAxes, ha="left", va="center", fontfamily="Arial"
    )

    # blue circle for left average
    ax.add_patch(Circle((0.96, 0.85), 0.02, color="lightblue", transform=ax.transAxes))
    text_left = ax.text(
        0.99, 0.85, f"           {average_left:.2f}",
        fontsize=16, color="gray", transform=ax.transAxes, ha="left", va="center", fontfamily="Arial"
    )

    # Return all drawn artists
    return [nodes]


def plot_network(network):
    """
    Plot and animate the network.

    Args:
        network: The network object to visualize.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    graph, colors = initial_graph(network)

    for _ in range(len(network.all_nodes) // 2):
        node_a, node_b = np.random.choice(len(network.all_nodes), 2, replace=False)
        graph.add_edge(node_a, node_b)
    
    # pos = nx.spring_layout(graph, k=0.2, iterations=25, seed=seed)
    seed = 33
    pos = nx.kamada_kawai_layout(graph, scale=0.6)
    pos_target = pos.copy()

    for spine in ax.spines.values():  # remove borders
        spine.set_visible(False)

    # leave space for title
    # Create the animation
    ani = FuncAnimation(
        fig, 
        self_sort, 
        frames=1200, 
        interval=200,  
        fargs=(network, graph, colors, pos, pos_target, ax, seed),
        blit = True
    )
    self_sort.ani = ani
    ani.save("animations/network_animation10.gif", fps=15,  writer="ffmpeg")
    plt.show()


def plot_network_clusters(network, cluster):
    """
    Plots a network with clusters using a force-directed layout.

    Args:
        network: The network object to visualize. 
        cluster: Cluster-related data for node coloring and graph structure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    graph, colors = initial_graph(network, cluster)
    seed = 33

    density = nx.density(graph)
    k = 0.1 / (1 + density) 
    pos = nx.spring_layout(graph, k=0.01, iterations=200, seed=seed)
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)
    nx.draw(
        graph, 
        pos,
        ax=ax,
        with_labels=False, 
        node_color=colors, 
        node_size=100, 
        edge_color="grey", 
        width = 0.4
    )
    ax.set_title(f"animating cluster connectivity", fontsize=14)

    plt.show()


def create_distribution(data, num_exp=1):
    """
    This function iterates through a list of cascades and extracts the size of the cascade, 
    the number of times this cascade size occurred and
    the average polarization of this cascade size. 

    Args:
        data: dictionary of the sizes with their cascade list of political orientations of nodes within that cascade.
        num_exp: number of runs done.

    Returns:
        counts: list of frequencies a particular size occured (matched index-wise with size)
        sizes: list of sizes that occurred
        avg_polarizations: average polarizaiton of that particular cascade size (matched index-wise with size)
    """
    sizes = []
    counts = []
    avg_polarizations = []

    # values essentially holds the list of the orientation of nodes (+1 for left, -1 for right) within a cascade
    for size, values in data.items():
        sizes.append(size)  # The cascade size
        counts.append(len(values))  # Number of occurrences 

        # the polarizations are already weighted by its prevelences
        polarization_val = np.mean(np.abs(values))
        
        if np.isnan(polarization_val):
            print(f"Warning: NaN detected in mean absolute values for input {values}")
            polarization_val = 0  # Default to 0 or another fallback value
        

        avg_polarizations.append(polarization_val)
    counts=np.array(counts, dtype=np.float64)

    #normalize by dividng through number of experiments
    counts/=num_exp

    return counts, sizes, avg_polarizations


def plot_cascade_power_law(data, stadium, what_net, largest_size=120, num_exp=30, save=False, correlation=0, averaged=True):
    """
    Zooming in on the tail of the distribution and see if there's a powerlaw that can be detected. Cascade distriubution
    is visualized for cascade sizes of bigger and equal to 2. 
    
    Args:
        data: A dictionary where keys are cascade sizes and values are lists of the polarizations of nodes within a cascade.
        stadium: before/after updating
        largest_size: largest cascade size occurring for a particular correlation value
        num_exp: number of experiments done.
        save: if true, saved to specified file. 
        corr: correlation between news sources. 
        averaged: in case averaged -> calculates the average size per sampled individual.If not, calculates dist of all cascades

    """
    # recalculating the full distribution and selecting the relevant cascade sizes (>=2)
    counts, sizes, _ = create_distribution(data, num_exp)
    sizes = np.array(sizes)
    counts = np.array(counts)
    valid_indices = sizes >=2
    filtered_sizes = sizes[valid_indices]
    filtered_counts = counts[valid_indices]

    expanded_data = np.repeat(filtered_sizes, np.int64(filtered_counts*30)) #bringing back the actual number of occurrences
    fit = powerlaw.Fit(expanded_data, xmin=2)
    print(f"Estimated exponent: {fit.power_law.alpha}")
    
    R, p = fit.distribution_compare('power_law', 'lognormal')
    print(f"Likelihood Ratio (R): {R}, p-value: {p}")

    if p < 0.05:
        print("Power law is significantly better than log-normal.")
    else:
        print("Log-normal might be a better fit.")
    plt.subplots(figsize=(5, 3.5))

    # Plot the fitted power-law
    fit.power_law.plot_pdf(color="r", linestyle="--", label=f"Power-Law Fit (α={fit.power_law.alpha:.2f})")

    # normalizing data to align with powerlaw package
    normalized_counts = filtered_counts / 30  
    plt.scatter(filtered_sizes, normalized_counts, color="grey", edgecolor="black", s=20, linewidth=0.5, label="pdf of size distribution")
    
    # log log scale for powerlaw detection
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()

    # plt.title(f"Powerlaw  {correlation})")
    plt.xlabel("Cascade Size")
    plt.ylabel("Number of Occurrences")
    plt.xlim(0, largest_size+1) 
    plt.ylim(10e-4)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    # save the plot
    if save:
        if not averaged: 
            plt.savefig(f"plots/experiment_results/cascade_distribution/{what_net}/powerlaw_{stadium}_{correlation}.png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"plots/experiment_results/cascade_distribution/{what_net}/powerlaw_{stadium}_{correlation}.png", dpi=300, bbox_inches='tight')

    plt.show()


def plot_cascade_animation(cascades_before, cascades_after, correlations, largest_sizes, num_exp, what_net, save=False, averaged=True):
    """
    Create an animated visualization of cascade distributions over multiple correlation values.

    Args:
        cascades : Dictionary where keys are correlation values and values are the distribution data.
        correlations : List of correlation values to iterate over.
        largest_sizes : Dictionary mapping correlation values to the largest cascade size.
        num_exp : Number of experiments for normalization.
        save : If True, saves the animation as a .gif file.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(7, 5), sharex=True)

    # **Initialize bar objects**
    sizes1 = list(sorted(set(size for data in cascades_before.values() for size in data.keys())))
    sizes2 = list(sorted(set(size for data in cascades_after.values() for size in data.keys())))

    initial_counts1 = np.zeros(len(sizes1))
    initial_counts2 = np.zeros(len(sizes2))

    if averaged:
        wid = 0.05
    else:
        wid = 1

    bars1 = ax1.bar(sizes1, initial_counts1, color="gray", edgecolor="black", linewidth=0.5, width=wid)
    bars2 = ax2.bar(sizes2, initial_counts2, color="gray", edgecolor="black", linewidth=0.5, width=wid)

    # **Create a fixed colorbar (Do NOT recreate it in every frame)**
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=[ax1,ax2])
    cbar.set_label("0: non polarized - 1: fully polarized")

    def update(frame):
        """Update function for animation (updates bars but not colorbar)."""
        corr = correlations[frame]
        data1 = cascades_before[corr]
        largest_size = largest_sizes[corr]

        data2 = cascades_after[corr]

        counts1 = np.zeros(len(sizes1))
        avg_polarizations1 = np.zeros(len(sizes1))

        counts2 = np.zeros(len(sizes2))
        avg_polarizations2 = np.zeros(len(sizes2))

        # Process data for this frame
        for i, size in enumerate(sizes1):
            if size in data1:
                values = data1[size]
                counts1[i] = len(values) / num_exp  # Normalize counts
                avg_polarizations1[i] = np.mean(np.abs(values))  # Average polarization

         # Process data for this frame
        for i, size in enumerate(sizes2):
            if size in data2:
                values = data2[size]
                counts2[i] = len(values) / num_exp  # Normalize counts
                avg_polarizations2[i] = np.mean(np.abs(values))  # Average polarization

        # **Update bar heights and colors instead of recreating bars**
        for bar, count, color_value in zip(bars1, counts1, avg_polarizations1):
            bar.set_height(count)
            bar.set_color(plt.cm.RdYlGn_r(color_value))  # Update color
            bar.set_edgecolor("black")
        
        for bar, count, color_value in zip(bars2, counts2, avg_polarizations2):
            bar.set_height(count)
            bar.set_color(plt.cm.RdYlGn_r(color_value))  # Update color
            bar.set_edgecolor("black")


        ax1.set_title(f"Bofore (Correlation: {corr})")
        ax1.set_xlim(0, largest_size)
        ax1.set_yscale("log")

        ax2.set_title(f"After (Correlation: {corr})")
        ax2.set_xlim(0, largest_size)
        ax2.set_yscale("log")

        if not averaged: 
            ax1.set_ylim(10e-3, 10e4)
            ax2.set_ylim(10e-3, 10e3)
        else: 
            ax1.set_ylim(10e-3, 10e2)
            ax2.set_ylim(10e-3, 10e2)

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(correlations), repeat=False)

    # Save or display animation
    if save:
        ani.save(f"animations/cascade_distribution/dummy/{what_net}/animation_combination.gif", writer="ffmpeg", fps=1.5, dpi=300)
    else:
        plt.show()


def plot_cascade_dist_average(data, stadium, what_net, largest_size=120, num_exp=30, save=False, correlation=0, averaged=True):
    """
    Plot a distribution of cascade sizes with dots representing the number of occurrences
    and dot colors representing the average polarization, to see if a powerlaw can be observed in the tail.
    
    Args:
        data: A dictionary where keys are cascade sizes and values are lists of the polarizations of nodes within a cascade.
        stadium: before/after updating
        largest_size: largest cascade size occurring for a particular correlation value
        num_exp: number of experiments done.
        save: if true, saved to specified file. 
        corr: correlation between news sources. 
        averaged: in case averaged -> calculates the average size per sampled individual.If not, calculates dist of all cascades

    """
    # Create a custom green-to-red colormap
    green_to_red = plt.cm.RdYlGn_r
    counts, sizes, avg_polarizations = create_distribution(data, num_exp)
    
    # Normalize polarization for coloring
    colors = [green_to_red(p) for p in avg_polarizations]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(sizes, counts, color=colors, edgecolor="black", linewidth=0.5, s=20)

    # Add a colorbar for polarization
    sm = plt.cm.ScalarMappable(cmap=green_to_red, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("0:non polarized-1:fully polarized")
    
    # set scales to log log for powerlaw detectio
    ax.set_yscale("log")
    ax.set_xscale("log")

    # Add labels and title
    ax.set_title(f"Cascade Size Distribution with Polarization (correlation: {correlation})")
    ax.set_xlabel("Cascade Size")
    ax.set_ylabel("Number of Occurrences")
    ax.set_xlim(0, largest_size+1) 
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    # save the plot
    if save:
        if not averaged: 
            plt.savefig(f"plots/experiment_results/cascade_distribution/{what_net}/{stadium}_{correlation}.png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"plots/experiment_results/cascade_distribution/{what_net}/averaged_{stadium}_{correlation}.png", dpi=300, bbox_inches='tight')

    plt.show()


def calculate_average_per_gamma(cas, num_runs): 
    """
    Calculate the average cascade size and polarization per correlation value.

    Args:
        cas : Tuple containing cascade distributions for before and after updating.
        num_runs : Number of experiments.
    """
    cascades_before_averaged, cascades_after_averaged = cas
    values_bef, values_af = defaultdict(), defaultdict()
    variance_bef, variance_af = defaultdict(), defaultdict()
    # variance_size_bef, variance_size_af = defaultdict(), defaultdict()

    for key,value in cascades_before_averaged.items():
        counts_bef, sizes_bef, pol_bef = create_distribution(value, num_runs)

        
        # weight sizes by their prevelence
        mean_size_bef = np.average(sizes_bef, weights=counts_bef)

        # the polarization is already weighted by prevelence so is not weighted again
        mean_pol_bef = np.mean(pol_bef) 
        values_bef[key] = (mean_size_bef, mean_pol_bef)    
        size_var_bef = np.average((sizes_bef-mean_size_bef)**2, weights=counts_bef)
        pol_var_bef = np.average((pol_bef-mean_pol_bef)**2, weights=counts_bef)
        variance_bef[key] = (size_var_bef, pol_var_bef)

    for key,value in cascades_after_averaged.items():
        counts_af, sizes_af, pol_af = create_distribution(value, num_runs)
        mean_size_af = np.average(sizes_af, weights=counts_af)
        mean_pol_af = np.mean(pol_af) 
        values_af[key] = (mean_size_af, mean_pol_af)    
        size_var_af = np.average((sizes_af-mean_size_af)**2, weights=counts_af)
        pol_var_af = np.average((pol_af-mean_pol_af)**2, weights=counts_af)
        variance_af[key] = (size_var_af, pol_var_af)

    return values_bef, values_af, variance_bef, variance_af


def calculate_fraction(network):
    """
    Calculate the fraction of right and left connections for each node in the network.

    Args:
        network: The network object.

    Returns: 
        average_right: The average fraction of right connections per right node.
        average_left: The average fraction of left connections per left node.
    """
    fraction_right = 0
    fraction_left = 0

    for i in network.all_nodes:
        identity = i.identity
        right = 0
        left = 0
        for j in i.node_connections:
            if j.identity == "D":
                right += 1

            if j.identity == "H":
                left += 1

        connections = len(i.node_connections)

        if identity == "D" and connections != 0:
            fraction_right += right / connections
        
        if identity == "H" and connections != 0:
            fraction_left += left / connections
        
    average_right = fraction_right / (len(network.all_agents) / 2)
    average_left = fraction_left / (len(network.all_agents) / 2)

    return average_right, average_left

def plot_cascades_gamma(cas, num_runs, what_net):
    """
    This plot calculates the average cascade size and polarization per correlation value for two networks.
    allowing for clear visual comparison

    Args: 
        cas: cascade distributions for both networks (along with polarization)
        num_runs: number of experiments.
        what_net: "random" (comparing before and after for random), "scale_free" (comparing before after scale_free) 
        "both" (comparing after scale free vs after random)
    """
    values_bef, values_af, variance_bef, variance_af = calculate_average_per_gamma(cas, num_runs)
    fig, ax = plt.subplots(figsize=(5,4))

    green_to_red = plt.cm.RdYlGn_r
    # Normalize polarization for coloring
    keys_bef = list(values_bef.keys())  # Gamma values (X-axis)
    sizes_bef = [v[0] for v in values_bef.values()]  # Sizes for Y-axis
    pol_bef = [v[1] for v in values_bef.values()]  # Polarization for colors

    keys_af = list(values_af.keys())  # Gamma values (X-axis)
    sizes_af = [v[0] for v in values_af.values()]  # Sizes for Y-axis
    pol_af = [v[1] for v in values_af.values()]  # Polarization for colors

    #Coloring scheme with polarization
    norm = plt.Normalize(vmin=0, vmax=1)
    colors_bef = [green_to_red(norm(p)) for p in pol_bef]
    colors_af = [green_to_red(norm(p)) for p in pol_af]
    
    var_sizes_bef = np.array([variance_bef[k][0] for k in keys_bef])  # Variance for sizes (Before)
    var_sizes_af = np.array([variance_af[k][0] for k in keys_af])  # Variance for sizes (After)

    #Computing confidence interval with calculated variance
    sem_sizes_bef = np.sqrt(var_sizes_bef) / np.sqrt(num_runs)
    sem_sizes_af = np.sqrt(var_sizes_af) / np.sqrt(num_runs)
    ci_sizes_bef = 1.96 * sem_sizes_bef
    ci_sizes_af = 1.96 * sem_sizes_af

    sm = plt.cm.ScalarMappable(cmap=green_to_red, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("0:non polarized-1:fully polarized")
    
    # make an errorbar to visualize CI
    ax.errorbar(keys_bef, sizes_bef, yerr=ci_sizes_bef, color="black", linewidth = 0.8, capsize=2, label="", linestyle="none", zorder=1)
    ax.errorbar(keys_af, sizes_af, yerr=ci_sizes_af, color="black", linewidth=0.8, capsize=2, label="", linestyle="none", zorder=1)

    # distinct between before after comparison and scale-free vs random comparison
    if what_net != "both":
        label1 = "Before"
        label2 = "After"
        marker1 = "o"
        marker2 = "s"
    else:
        label1 = "Random"
        label2 = "Scale-free"
        marker1 = "v"
        marker2 = "h"

    # plotting for both networks
    for x, y, color in zip(keys_bef, sizes_bef, colors_bef):
        ax.scatter(x, y, color=color, edgecolor="black", linewidths=0.5, marker=marker1, s=50, label="Before" if x == keys_bef[0] else "", zorder=2)

    for x, y, color in zip(keys_af, sizes_af, colors_af):
        ax.scatter(x, y, color=color, edgecolor="black", linewidths=0.5, marker=marker2, s=50, label="After" if x == keys_af[0] else "", zorder=2)
    
    legend_handles = [
    plt.Line2D([], [], marker=marker1, color="w", markerfacecolor="white", markersize=6, markeredgecolor="black", label=label1),
    plt.Line2D([], [], marker=marker2, color="w", markerfacecolor="white", markersize=6, markeredgecolor="black", label=label2),
    ]
    ax.set_xlabel("News Correlation")
    ax.set_ylabel("Average Cascade Size")
    ax.legend(handles=legend_handles)

    # saving plot in destined folder
    plt.savefig(f"plots/experiment_results/cascade_distribution/{what_net}/averaged_over_gammas.png", dpi=300, bbox_inches='tight')
    plt.show()


def test_significance(values_bef, values_af, variance_bef, variance_af, num_runs=30):
    """
    Perform t-tests on cascade sizes and polarization before vs. after for each gamma value.

    Args:
        values_bef : Dictionary mapping gamma values to (mean_size, mean_polarization) for before.
        values_af : Dictionary mapping gamma values to (mean_size, mean_polarization) for after.
        variance_bef : Dictionary mapping gamma values to (variance_size, variance_polarization) for before.
        variance_af : Dictionary mapping gamma values to (variance_size, variance_polarization) for after.
        num_runs : Number of runs per gamma value (default is 30).

    Returns:
        results : Dictionary mapping gamma values to t-test results for size and polarization.
    """
    results = {}

    for gamma in values_bef.keys():
        mean_size_bef, mean_pol_bef = values_bef[gamma]
        mean_size_af, mean_pol_af = values_af[gamma]
        
        var_size_bef, var_pol_bef = variance_bef[gamma]
        var_size_af, var_pol_af = variance_af[gamma]

        # Compute t-test for cascade sizes
        t_size, p_size = ttest_ind_from_stats(mean1=mean_size_bef, std1=np.sqrt(var_size_bef), nobs1=num_runs,
                                              mean2=mean_size_af, std2=np.sqrt(var_size_af), nobs2=num_runs, 
                                              equal_var=False) 

        # Compute t-test for polarization
        t_pol, p_pol = ttest_ind_from_stats(mean1=mean_pol_bef, std1=np.sqrt(var_pol_bef), nobs1=num_runs,
                                            mean2=mean_pol_af, std2=np.sqrt(var_pol_af), nobs2=num_runs, 
                                            equal_var=False)  

        results[gamma] = {
            "t_size": t_size, "p_size": p_size,
            "t_pol": t_pol, "p_pol": p_pol
        }

    return results


def statistics_cascades(cas_sf, cas_rand, num_runs):
    """
    Perform statistical significance tests on cascade sizes and polarization for scale-free and random networks.

    Args:
        cas_sf : Tuple containing cascade distributions for scale-free networks.
        cas_rand : Tuple containing cascade distributions for random networks.
        num_runs : Number of experiments
    """
    cascades_before_averaged_sf, cascades_after_averaged_sf = cas_sf
    cascades_before_averaged_rand, cascades_after_averaged_rand = cas_rand
    which_cas = [(cascades_before_averaged_sf, cascades_after_averaged_sf), (cascades_before_averaged_rand, cascades_after_averaged_rand), (cascades_after_averaged_rand, cascades_after_averaged_sf)]

    for i, what in enumerate(["scale_free", "random", "both"]):
        values_bef, values_af, variance_bef, variance_af = calculate_average_per_gamma(which_cas[i], num_runs)
        results = test_significance(values_bef, values_af, variance_bef, variance_af, num_runs)
        output_file = f"statistics/dummy/cascades/results_bef_af_{what}.txt"

        with open(output_file, "w") as f:
            f.write(f"Statistical significance for {what} network type (cascade experiments)\n")
            for gamma, res in results.items():
                f.write(f"Gamma = {gamma}:\n")
                f.write(f"  Size: t = {res['t_size']:.3f}, p = {res['p_size']:.3g}\n")
                f.write(f"  Polarization: t = {res['t_pol']:.3f}, p = {res['p_pol']:.3g}\n")
                f.write("--------------------------------------------------------\n")






