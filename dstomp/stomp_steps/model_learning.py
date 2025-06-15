from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from dstomp.foundation import Foundation
from environment.gridworld import Actions


class ModelLearning:
    def __init__(
        self,
        foundation: Foundation,
        alpha_r: float = 0.1,
        alpha_p: float = 0.1,
        lambda_: float = 0,
        lambda_prime: float = 0,
    ):
        self.foundation = foundation
        self.alpha_r = alpha_r
        self.alpha_p = alpha_p
        self.lambda_ = lambda_
        self.lambda_prime = lambda_prime

    def learn_model(
        self, option_idx: int, off_policy_steps: int = 50_000
    ) -> Tuple[List[float], list[float]]:
        # We need to learn models for all primitive actions and for the full options.
        # In the primitive action case, stopping_function and the stopping value have a different treatment.
        is_primitive_action = option_idx < self.foundation.env.num_actions
        subgoal_idx = option_idx - self.foundation.env.num_actions

        # Initiating env
        state = self.foundation.env.reset()
        state_features = self.foundation.env.get_one_hot_state(state)

        # Lists to store the model errors
        reward_model_errors = []
        transition_model_errors = []

        for step in tqdm(range(off_policy_steps)):
            # Chose and execute an action from the equiprobable policy
            a = np.random.choice(
                self.foundation.env.num_actions, p=self.foundation.behavior_policy_probs
            )
            action = Actions(a)
            next_state, reward, done = self.foundation.env.step(action)

            # If we reach the goal, then we reset the state and the eligibility traces
            if done:
                state = self.foundation.env.reset()
                state_features = self.foundation.env.get_one_hot_state(state)
                self.foundation.e_options[subgoal_idx] = np.zeros(
                    self.foundation.env.num_states
                )
                self.foundation.e_policies[subgoal_idx] = np.zeros(
                    self.foundation.env.num_states * self.foundation.env.num_actions
                )
                continue

            next_state_features = self.foundation.env.get_one_hot_state(next_state)

            # Check the predicted values from the linear models
            predicted_reward = self.foundation.w_rewards[option_idx] @ state_features
            reward_model_error = predicted_reward - reward
            reward_model_errors.append(reward_model_error)

            predicted_transition = (
                self.foundation.W_transitions[option_idx] @ state_features
            )
            transition_model_error = np.linalg.norm(
                predicted_transition - next_state_features
            )
            transition_model_errors.append(transition_model_error)

            # Handling the stopping value and option probabilities
            if is_primitive_action:
                stopping_value = None
                option_probs = np.ones(self.foundation.env.num_actions)
            else:
                stopping_value = self.foundation.get_stopping_value(
                    next_state_features, subgoal_idx
                )
                option_probs = self.foundation.softmax_option_policy(
                    next_state, subgoal_idx
                )

            should_stop = self.foundation.should_stop(
                next_state_features, subgoal_idx, stopping_value
            )
            importance_sampling_ratio = (
                option_probs[action] / self.foundation.behavior_policy_probs[action]
            )

            # Learning the Reward model
            # TD Error for reward model
            delta_r = self.foundation.td_error(
                reward,
                0,
                float(self.foundation.w_rewards[option_idx] @ state_features),
                float(self.foundation.w_rewards[option_idx] @ next_state_features),
                should_stop,
            )

            # Learning Reward Model Weights
            (
                self.foundation.w_rewards[option_idx],
                self.foundation.e_rewards[option_idx],
            ) = self.foundation.UWT(
                self.foundation.w_rewards[option_idx],
                self.foundation.e_rewards[option_idx],
                state_features,
                self.alpha_r * delta_r,
                importance_sampling_ratio,
                self.foundation.gamma * self.lambda_ * (1 - should_stop),
            )

            # Learning the Transition model
            # For the transition model we update each row of the W_transitions matrix
            for j in range(self.foundation.env.num_states):
                # TD Error for transition model
                delta_n = self.foundation.td_error(
                    0,
                    next_state_features[j],
                    self.foundation.W_transitions[option_idx][j] @ state_features,
                    self.foundation.W_transitions[option_idx][j] @ next_state_features,
                    should_stop,
                )

                # Learning Transition Model Weights
                (
                    self.foundation.W_transitions[option_idx][j],
                    self.foundation.e_transitions[option_idx][j],
                ) = self.foundation.UWT(
                    self.foundation.W_transitions[option_idx][j],
                    self.foundation.e_transitions[option_idx][j],
                    state_features,
                    self.alpha_p * delta_n,
                    importance_sampling_ratio,
                    self.foundation.gamma * self.lambda_ * (1 - should_stop),
                )

            # Moving to the next state
            state = next_state
            state_features = next_state_features

        return reward_model_errors, transition_model_errors
