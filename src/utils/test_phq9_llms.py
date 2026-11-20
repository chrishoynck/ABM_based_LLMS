import numpy as np
import torch
from classes.agent import Agent


class TestLLMs:
    def __init__(self, seed: int = 42, num_agents: int = 5,
                 personas: list = None, well_being: list = None):
        self.rng = np.random.default_rng(seed)
        personas = self.rng.permutation(personas) if personas is not None else [None]*num_agents
        self.well_being = self.rng.permutation(well_being) if well_being is not None else [None]*num_agents
        self.all_agents = [Agent(i, rng=np.random.default_rng(seed + i), persona=personas[i], 
                              well_being=self.well_being[i]) for i in range(int(num_agents))]
        self.iterations = 0

    def _prepare_prompts(self, tokenizer) -> list:
        prompts = []
        for agent in self.all_agents:
            prompt = agent.step_llm_tweet(tokenizer, round_idx=self.iterations, force_active=True)
            prompts.append(prompt)
        return prompts, self.all_agents
    
    def generate_tweets(self, pipe, tokenizer, prompts: list) -> list:
        batch_size = 10
            
        out = self._generate_outputs(pipe, prompts, batch_size)
        for agent, answer in zip(self.all_agents, out):
            questionnaire_answers = answer[0]["generated_text"].strip()
            sum_score = agent.parse_phq9_answers(questionnaire_answers)
            agent.update_well_being(sum_score)

    
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
        )
        out = self.inference_w_batches(pipe, prompts, batch_size=batch_size, **gen_kwargs)
        return out
    
    def inference_w_batches(self,pipe, prompts, batch_size, **gen_kwargs):
        """
        Run LLM pipeline on a list of prompts in mini-batches.
        Returns a flat list of outputs, same order as prompts.
        """
        if len(prompts) == 0:
            return []
        
        all_outputs = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_outputs = pipe(batch_prompts, **gen_kwargs)
            all_outputs.extend(batch_outputs)
        

        return all_outputs
    

    def _phq9_questionnaire(self, tokenizer, pipe,mistakes, check_point):
        """
        Have all agents complete the PHQ-9 questionnaire via LLM and update their well-being scores.
        Args:
            tokenizer: The tokenizer for the LLM.
            pipe: The LLM pipeline for generating responses.
            check_point: The number of recent tweets to consider for the questionnaire.
        Returns:
            mistakes: A dictionary mapping old PHQ-9 scores to lists of changes in scores.
        """
        # prepare prompts for all agents
        prompts = []
        for agent in self.all_agents:
            prompt = agent.phq9_questionnaire_prompt(tokenizer, agent.tweethistory[-check_point:])
            prompts.append(prompt)
        
        # inference with LLM
        out = self._generate_outputs(pipe, prompts, batch_size=8)
        # update well-being scores based on responses
        for agent, answer in zip(self.all_agents, out):
            questionnaire_answers = answer[0]["generated_text"].strip()
            sum_score = agent.parse_phq9_answers(questionnaire_answers)

            old_sum_score = agent.all_phq9_sumscores[-1] if agent.all_phq9_sumscores else None
            if old_sum_score is not None:
                change = sum_score - old_sum_score
                mistakes[old_sum_score].append(change)

            sum_score = np.random.randint(0, 28)  # for testing, assign random score
            agent.update_well_being(sum_score)

        return mistakes

    def _apply_outputs_and_update_state(self, agents_w_prompt, out, n_grams, update_score):
        """
        Use LLM outputs to update agents' tweets and activation states,
        then compute distorted tweet statistics for this round.
        """
        # agents send out their tweets
        for agent, tweet in zip(agents_w_prompt, out):
            
            raw = tweet[0]["generated_text"].strip()
            agent.send_tweet(
                max_chars=240,
                raw_tweet=raw,
            )

        for agent in self.all_agents:
            _ = agent.commit(n_grams=n_grams, update_score=update_score)


    def update_round(self, mistake_dict, tokenizer, pipe, n_grams=[], check_point= 20):
        """
        Update the network for one round by responding to news intensities and adjusting the network accordingly.
        """
        self.iterations += 1
        batch_size = 8
        prompts, agents_w_prompt = self._prepare_prompts(tokenizer)
        update_score = False

        if self.iterations % check_point== 0:
            mistake_dict = self._phq9_questionnaire(tokenizer, pipe, mistake_dict, check_point)
            update_score = True
        
        # generate outputs in parallel
        out = self._generate_outputs(pipe, prompts, batch_size)

        # agents send out their tweets + state update + stats
        self._apply_outputs_and_update_state(
            agents_w_prompt, out, n_grams, update_score=update_score
        )

           
    def assess_performance(self, mistake_dict):
        """
        Assess the performance of the agents based on the mistake dictionary.
        """
        total_bias = []
        bias_per_phq9 = {}

        total_acc = []
        acc_per_phq9 = {}

        for old_score, changes in mistake_dict.items():
            bias_per_phq9[old_score] = np.mean(changes) if changes else 0
            total_bias.extend(changes)
            acc_per_phq9[old_score] = np.mean(np.abs(np.array(changes))) if changes else 0
            total_acc.extend(np.abs(np.array(changes)))
        
        if total_bias:
            avg_change = sum(total_bias) / len(total_bias)
        else:
            avg_change = 0
        
        if total_acc:
            total_acc = sum(total_acc) / len(total_acc)
        else:
            total_acc = 0
        
        return avg_change, bias_per_phq9, acc_per_phq9, total_acc
    
    def run_simulation(self, tokenizer, pipe, n_rounds=100, n_grams=[], check_point=20):
        """
        Run the full simulation for a specified number of rounds.
        """
        self.iterations = 0
        mistake_dict = {i: [] for i in range(0, 28)}  # PHQ-9 scores range from 0 to 27

        for round_idx in range(n_rounds):
            self.update_round(mistake_dict, tokenizer, pipe, n_grams=n_grams, check_point=check_point)
        
        avg_change, bias_per_phq9, acc_per_phq9, total_acc = self.assess_performance(mistake_dict)
        
        return avg_change, bias_per_phq9, acc_per_phq9, total_acc
        