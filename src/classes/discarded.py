# # funcions from Networks, measuring cascades!
# def pick_samplers(self):
#         """
#         Pick samplers for the Depressed and Happy media hubs.

#         Returns:
#             Tuple of sets of samplers for the depressed and happy media hubs.
#         """
        
#         all_samplers_H, all_samplers_D = set(), set()
#         # for agent in self.rng.choice(list(self.all_agents), int(len(self.all_agents) * self.update_fraction), replace=False):
#         for agent in self.rng.choice(self.all_agents, int(len(self.all_agents) * self.update_fraction), replace=False):
#             if agent.identity == 'H':
#                 all_samplers_H.add(agent)
#             elif agent.identity == 'D':
#                 all_samplers_D.add(agent)
#             else:
#                 raise ValueError("agent identity should be assigned")
#             # assert agent.sampler_state == False, "at this point all samplers states should be false"
#             assert agent.activation_state == False, "at this point all agents should be inactive"
#             agent.sampler_state = True
#         return (all_samplers_H, all_samplers_D)
# def run_cascade(self, sL, sR, all_samplers, analyze=False):
#     """
#     Continue responding to the news intensities until a steady state is reached (no changes in activation state).
#     This is the cascade event.

#     Args:
#         sL: Normalized significance for the left media hub.
#         sR: Normalized significance for the right media hub.
#         all_samplers: Tuple of sets of samplers for the left and right media hubs.
#         analyze: Boolean flag to indicate whether to analyze the network.
    
#     Returns:
#         List of activated nodes in each round.
#     """
#     self.activated=set()
#     steady_state_reached = True
#     union_to_consider= set()
#     all_left, all_right = all_samplers

#     # inject news for left oriented nodes
#     for nodeL in all_left:
#         nodeL.reset_node()
#         active_state, to_consider_L = nodeL.respond(sL, analyze=analyze)
#         if active_state:
#             union_to_consider.update(to_consider_L)
#             steady_state_reached = False
#             self.activated.add(nodeL)

#     # inject news for right oriented nodes
#     for nodeR in all_right:
#         nodeR.reset_node()
#         active_state, to_consider_R = nodeR.respond(sR, analyze=analyze)
#         if active_state:
#             union_to_consider.update(to_consider_R)
#             steady_state_reached = False
#             self.activated.add(nodeR)

#     while not steady_state_reached:
#         steady_state_reached = True
#         new_to_consider = set()

#         for individual in union_to_consider:
#             # omit redundant checks by returning only the neighbors of newly activated nodes. 
#             active_state, to_consider = individual.respond(analyze=analyze)

#             if active_state:
#                 steady_state_reached=False
#                 self.activated.add(individual)
#                 new_to_consider.update(to_consider)
#         union_to_consider = new_to_consider

# def analyze_network(self):
#     """
#     Analyze the network by identifying cascades and their properties.

#     Returns:
#         Tuple of the following: 
#         - List of merged cascades (sets of nodes)
#         - List of cascade sizes
#         - List of polarized fractions in each cascade
#     """
#     self.alterations = 0
#     sL, sR = self.generate_news_significance()

#     all_samplers = self.pick_samplers()

#     self.run_cascade(sL, sR, all_samplers, True)
#     participating = [n.cascade_id for n in self.all_nodes if n.last_of_cascade]

#     if len(participating) == 0: 
#         # print("no cascades in this round")
#         return [], [], []

#     # merge sets of nodes that contain 1 or more of the same node -> cascade is overlapping and thus merged
#     merged = []
#     for current_set in participating:
#         # check for all disjoint lists 
#         overlapping_sets = [merged_set for merged_set in merged if not current_set.isdisjoint(merged_set)] 
        
#         if overlapping_sets:
#             # Merge all overlapping sets into one
#             merged_set = set(current_set)  
#             for overlap in overlapping_sets:
#                 merged_set.update(overlap) 
#                 merged.remove(overlap)     
#             merged.append(merged_set)      
#         else:
#             # If no overlaps, add as a new set
#             merged.append(current_set)
    
#     number_nodes_within = sum(len(setje) for setje in merged)

#     # overlapping cascades are merged, so no node can occur more than once in merged
#     assert number_nodes_within == len(self.activated), f"All the nodes that are activated should be part of a cascade and vice versa"
            
#     size_distiribution_cascades= [len(setje) for setje in merged]
#     fractions_polarized = [
#         sum(i for _, i in setje) / len(setje) if len(setje) > 0 else 0  
#         for setje in merged
#     ]

#     for node in self.activated:
#         node.reset_activation_state()
#         node.reset_node()
        
#     self.activated = set()
    
#     return merged, size_distiribution_cascades, fractions_polarized
# def update_round(self):
#         """
#         Update the network for one round by responding to news intensities and adjusting the network accordingly.
#         """
#         self.iterations +=1
#         # sL, sR = self.generate_news_significance()

#         allsamplers = self.pick_samplers()
#         # Respond to the news intensities, continue this untill steady state is reached
#         # self.run_cascade(sL, sR, allsamplers)

#         # Network adjustment
#         # self.network_adjustment(sL, sR)

#         # Reset states for next round
#         for agent in self.activated:
#             agent.reset_activation_state()

#         self.activated = set()

# def run_cascade_for_visuals(self, sL, sR):
#     """
#     Continue responding to the news intensities until a steady state is reached (no changes in activation state).
#     This is the cascade event.

#     Args:
#         sL: Normalized significance for the left media hub.
#         sR: Normalized significance for the right media hub.

#     Returns:
#         List of activated agents in each round.
#     """
#     activated = []
#     steady_state_reached = False
#     while not steady_state_reached:
#         round = []
#         steady_state_reached = True  
#         for agentL in self.agentsL:
#             if agentL.respond_for_visuals(sL):
#                 round.append(agentL.ID)
#                 self.activated.add(agentL)
#                 steady_state_reached = False
#         for agentR in self.agentsR:
#             if agentR.respond_for_visuals(sR):
#                 round.append(agentR.ID)
#                 self.activated.add(agentR)
#                 steady_state_reached = False
        
#         activated.append(round)

# # The scal-free Network generation functions 
#     def verify_scale_free_distribution(self, plot):
#         """
#         Check if the network exhibits scale-free characteristics
#         """
#         # Calculate node degrees
#         degrees = [len(node.node_connections) for node in self.all_nodes]
        
#         # Compute log-log plot for degree distribution
#         degree_counts = {}
#         for degree in degrees:
#             degree_counts[degree] = degree_counts.get(degree, 0) + 1
        
#         unique_degrees = list(degree_counts.keys())
#         frequencies = list(degree_counts.values())
        
#         if plot:
#             plt.figure(figsize=(10, 6))
#             plt.loglog(unique_degrees, frequencies, 'bo')
#             plt.title('Degree Distribution (Log-Log Scale)')
#             plt.xlabel('Degree')
#             plt.ylabel('Frequency')
#             plt.show()

#         assert all(degree >= self.m for degree in self.degree_distribution.values()), (
#         f"Some nodes have degree less than m={self.m}. Check initialization logic."
#         )
        
#         # Basic scale-free network indicators
#         assert max(degrees) > np.mean(degrees) * 2, "Network lacks high-degree nodes"
#         assert len([d for d in degrees if d > np.mean(degrees) * 2]) > 0, "No significant hub nodes"
#         print("Intializing a scale-free network with m:", self.m)
#         fit = Fit(degrees)
#         print(f"Power-law fit: alpha={fit.power_law.alpha}, KS={fit.power_law.KS()}")
#         assert fit.power_law.KS() < 0.5, f"Power-law fit is not significant; {fit.power_law.KS()}"
#         # assert fit.power_law.alpha < 7, f"Power-law exponent is too high; {fit.power_law.alpha}"



#########################################################
# In llama_activate.py main section
#########################################################
# def chat_turn(history, user_msg: str) -> (str, list):
#     # build messages with history + new user turn
#     msgs = history + [{"role": "user", "content": user_msg}]
#     prompt = tokenizer.apply_chat_template(
#         msgs, tokenize=False, add_generation_prompt=True
#     )
#     out = pipe(
#         prompt,
#         do_sample=False,          # deterministic; set True + temperature/top_p to sample
#         max_new_tokens=256,
#         return_full_text=False,   # only return the completion (safer than slicing)
#     )[0]["generated_text"].strip()
#     # append assistant reply to history
#     history.append({"role": "user", "content": user_msg})
#     history.append({"role": "assistant", "content": out})
#     return out, history

# def repl(history=None):
#     ''' 
#     Conversational function with this llama model, 
#     History of llama model is updated with each message.
#     when using /reset: history is forgotten
#     when using /exit: conversation is stopped. 
#     '''
#     print("Chat ready. Type your message. Commands: /reset, /exit")
#     while True:
#         try:
#             user = input("\nYou: ").strip()
#             if not user:
#                 continue
#             if user.lower() in {"/exit", "/quit"}:
#                 print("Bye!")
#                 break
#             if user.lower() == "/reset":
#                 history[:] = [{"role": "system", "content": SYSTEM_PROMPT}]
#                 print("(history reset)")
#                 continue
#             reply, history = chat_turn(history, user)
#             print(f"Assistant: {reply}")
#         except (KeyboardInterrupt, EOFError):
#             print("\nBye!")
#             break


# def act_llama2(text: str, prev=None) -> str:
#     """Raw continuation (no chat template)."""
#     out = pipe(text, do_sample=True)[0]["generated_text"]
#     return out[len(text):].strip()

# def chat_once(user_msg: str) -> str:
#     messages = [{"role": "user", "content": user_msg}]
#     prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     out = pipe(prompt, do_sample=False)[0]["generated_text"]
#     return out[len(prompt):].strip()