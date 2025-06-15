from typing import List

import numpy as np
from tqdm import tqdm

from dstomp.foundation import Foundation
from environment.gridworld import Actions


class OptionLearning:
    def __init__(
        self,
        foundation: Foundation,
        alpha: float = 0.1,
        alpha_prime: float = 0.1,
        lambda_: float = 0,
        lambda_prime: float = 0,
    ):
        self.foundation = foundation
        self.alpha = alpha
        self.alpha_prime = alpha_prime
        self.lambda_ = lambda_
        self.lambda_prime = lambda_prime

    def learn_options(
        self, subgoal_idx: int, off_policy_steps: int = 50_000
    ) -> List[float]:
        # Initiating env
        state = self.foundation.env.reset()
        state_features = self.foundation.env.get_one_hot_state(state)
        initial_state_features = state_features

        initial_state_estimative = []

        for step in tqdm(range(off_policy_steps)):
            initial_state_estimative.append(
                self.foundation.w_subgoal[subgoal_idx] @ initial_state_features
            )

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
            state_action_features = self.foundation.env.get_one_hot_state_action(
                state, action
            )

            # Calculating the stopping value and checking if the option needs to stop or not
            stopping_value = self.foundation.get_stopping_value(
                next_state_features, subgoal_idx
            )
            should_stop = self.foundation.should_stop(
                next_state_features, subgoal_idx, stopping_value
            )

            # Calculating the importance sampling ratio for off-policy learning
            option_probs = self.foundation.softmax_option_policy(state, subgoal_idx)
            importance_sampling_ratio = (
                option_probs[action] / self.foundation.behavior_policy_probs[action]
            )

            # Calculating TD Error
            delta = self.foundation.td_error(
                reward,
                stopping_value,
                float(self.foundation.w_subgoal[subgoal_idx] @ state_features),
                float(self.foundation.w_subgoal[subgoal_idx] @ next_state_features),
                should_stop,
            )

            # Learning Option Weights
            (
                self.foundation.w_subgoal[subgoal_idx],
                self.foundation.e_options[subgoal_idx],
            ) = self.foundation.UWT(
                self.foundation.w_subgoal[subgoal_idx],
                self.foundation.e_options[subgoal_idx],
                state_features,
                self.alpha * delta,
                importance_sampling_ratio,
                self.foundation.gamma * self.lambda_ * (1 - should_stop),
            )

            # Learning Option Policy
            (
                self.foundation.theta_subgoal[subgoal_idx],
                self.foundation.e_policies[subgoal_idx],
            ) = self.foundation.UWT(
                self.foundation.theta_subgoal[subgoal_idx],
                self.foundation.e_policies[subgoal_idx],
                state_action_features,
                self.alpha_prime * delta,
                importance_sampling_ratio,
                self.foundation.gamma * self.lambda_prime * (1 - should_stop),
            )

            # Moving to the next state
            state = next_state
            state_features = next_state_features

        return initial_state_estimative
