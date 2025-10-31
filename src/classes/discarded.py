# funcions from Networks, measuring cascades!
def run_cascade(self, sL, sR, all_samplers, analyze=False):
    """
    Continue responding to the news intensities until a steady state is reached (no changes in activation state).
    This is the cascade event.

    Args:
        sL: Normalized significance for the left media hub.
        sR: Normalized significance for the right media hub.
        all_samplers: Tuple of sets of samplers for the left and right media hubs.
        analyze: Boolean flag to indicate whether to analyze the network.
    
    Returns:
        List of activated nodes in each round.
    """
    self.activated=set()
    steady_state_reached = True
    union_to_consider= set()
    all_left, all_right = all_samplers

    # inject news for left oriented nodes
    for nodeL in all_left:
        nodeL.reset_node()
        active_state, to_consider_L = nodeL.respond(sL, analyze=analyze)
        if active_state:
            union_to_consider.update(to_consider_L)
            steady_state_reached = False
            self.activated.add(nodeL)

    # inject news for right oriented nodes
    for nodeR in all_right:
        nodeR.reset_node()
        active_state, to_consider_R = nodeR.respond(sR, analyze=analyze)
        if active_state:
            union_to_consider.update(to_consider_R)
            steady_state_reached = False
            self.activated.add(nodeR)

    while not steady_state_reached:
        steady_state_reached = True
        new_to_consider = set()

        for individual in union_to_consider:
            # omit redundant checks by returning only the neighbors of newly activated nodes. 
            active_state, to_consider = individual.respond(analyze=analyze)

            if active_state:
                steady_state_reached=False
                self.activated.add(individual)
                new_to_consider.update(to_consider)
        union_to_consider = new_to_consider

def analyze_network(self):
    """
    Analyze the network by identifying cascades and their properties.

    Returns:
        Tuple of the following: 
        - List of merged cascades (sets of nodes)
        - List of cascade sizes
        - List of polarized fractions in each cascade
    """
    self.alterations = 0
    sL, sR = self.generate_news_significance()

    all_samplers = self.pick_samplers()

    self.run_cascade(sL, sR, all_samplers, True)
    participating = [n.cascade_id for n in self.all_nodes if n.last_of_cascade]

    if len(participating) == 0: 
        # print("no cascades in this round")
        return [], [], []

    # merge sets of nodes that contain 1 or more of the same node -> cascade is overlapping and thus merged
    merged = []
    for current_set in participating:
        # check for all disjoint lists 
        overlapping_sets = [merged_set for merged_set in merged if not current_set.isdisjoint(merged_set)] 
        
        if overlapping_sets:
            # Merge all overlapping sets into one
            merged_set = set(current_set)  
            for overlap in overlapping_sets:
                merged_set.update(overlap) 
                merged.remove(overlap)     
            merged.append(merged_set)      
        else:
            # If no overlaps, add as a new set
            merged.append(current_set)
    
    number_nodes_within = sum(len(setje) for setje in merged)

    # overlapping cascades are merged, so no node can occur more than once in merged
    assert number_nodes_within == len(self.activated), f"All the nodes that are activated should be part of a cascade and vice versa"
            
    size_distiribution_cascades= [len(setje) for setje in merged]
    fractions_polarized = [
        sum(i for _, i in setje) / len(setje) if len(setje) > 0 else 0  
        for setje in merged
    ]

    for node in self.activated:
        node.reset_activation_state()
        node.reset_node()
        
    self.activated = set()
    
    return merged, size_distiribution_cascades, fractions_polarized

def run_cascade_for_visuals(self, sL, sR):
    """
    Continue responding to the news intensities until a steady state is reached (no changes in activation state).
    This is the cascade event.

    Args:
        sL: Normalized significance for the left media hub.
        sR: Normalized significance for the right media hub.

    Returns:
        List of activated agents in each round.
    """
    activated = []
    steady_state_reached = False
    while not steady_state_reached:
        round = []
        steady_state_reached = True  
        for agentL in self.agentsL:
            if agentL.respond_for_visuals(sL):
                round.append(agentL.ID)
                self.activated.add(agentL)
                steady_state_reached = False
        for agentR in self.agentsR:
            if agentR.respond_for_visuals(sR):
                round.append(agentR.ID)
                self.activated.add(agentR)
                steady_state_reached = False
        
        activated.append(round)