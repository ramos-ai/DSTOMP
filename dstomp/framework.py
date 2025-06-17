import os
import pickle
from os.path import join

from dstomp.foundation import Foundation
from dstomp.stomp_steps.model_learning import ModelLearning
from dstomp.stomp_steps.option_learning import OptionLearning
from dstomp.stomp_steps.planning import Planning
from dstomp.successor import Successor
from environment.gridworld import GridWorld


class STOMP:
    def __init__(
        self,
        foundation: Foundation,
        alpha: float = 0.1,
        alpha_prime: float = 0.1,
        alpha_r: float = 0.1,
        alpha_p: float = 0.1,
        alpha_step_size: float = 1.0,
        lambda_: float = 0,
        lambda_prime: float = 0,
    ):
        self.foundation = foundation
        self.option_learning = OptionLearning(
            foundation=foundation,
            alpha=alpha,
            alpha_prime=alpha_prime,
            lambda_=lambda_,
            lambda_prime=lambda_prime,
        )
        self.model_learning = ModelLearning(
            foundation=foundation,
            alpha_r=alpha_r,
            alpha_p=alpha_p,
            lambda_=lambda_,
            lambda_prime=lambda_prime,
        )
        self.planning = Planning(
            foundation=foundation,
            alpha_step_size=alpha_step_size,
        )


class DSTOMP:
    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.99,
        alpha: float = 0.1,
        alpha_prime: float = 0.1,
        alpha_r: float = 0.1,
        alpha_p: float = 0.1,
        alpha_step_size: float = 1.0,
        lambda_: float = 0,
        lambda_prime: float = 0,
        successor_alpha: float = 0.1,
        num_subgoals: int | None = None,
        seed: int | None = 42,
        off_policy_steps_for_successor_representation: int = 50_000,
        experiment_results_path: str | None = None,
    ):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_prime = alpha_prime
        self.alpha_r = alpha_r
        self.alpha_p = alpha_p
        self.alpha_step_size = alpha_step_size
        self.lambda_ = lambda_
        self.lambda_prime = lambda_prime
        self.num_subgoals = num_subgoals
        self.off_policy_steps_for_successor_representation = (
            off_policy_steps_for_successor_representation
        )
        self.seed = seed
        self.successor = Successor(env, successor_alpha, gamma)
        self.experiment_results_path = experiment_results_path

    def execute(
        self, num_lookahead_operations: int = 6_000, off_policy_steps: int = 50_000
    ):
        print("[INFO] Starting DSTOMP execution...\n")
        print("[INFO] Finding bottleneck states using Successor Representation")

        subagoals_state_idx, subgoals_state = self.successor.get_subgoals(
            self.num_subgoals,
            self.seed,
            self.off_policy_steps_for_successor_representation,
        )

        stomp_foundation = Foundation(
            env=self.env,
            subgoals_state=subgoals_state,
            subgoals_state_idx=subagoals_state_idx,
            behavior_policy_probs=self.successor.behavior_policy_probs,
            gamma=self.gamma,
            successor_alpha=self.successor.successor_alpha,
        )

        stomp_framework = STOMP(
            foundation=stomp_foundation,
            alpha=self.alpha,
            alpha_prime=self.alpha_prime,
            alpha_r=self.alpha_r,
            alpha_p=self.alpha_p,
            alpha_step_size=self.alpha_step_size,
            lambda_=self.lambda_,
            lambda_prime=self.lambda_prime,
        )

        option_learning_logs = []
        model_learning_logs = []

        for subgoal_idx in range(len(subagoals_state_idx)):
            print(
                f"\n[INFO] Learning options for subgoal {subgoal_idx + 1}/{self.num_subgoals}"
            )
            initial_state_estimative = stomp_framework.option_learning.learn_options(
                subgoal_idx, off_policy_steps
            )
            option_learning_logs.append(initial_state_estimative)

        for option_idx in range(stomp_foundation.num_options):
            print(
                f"\n[INFO] Learning model for option {option_idx + 1}/{stomp_foundation.num_options}, {'a Primitive Action' if option_idx < self.env.num_actions else 'a Full Option'}"
            )
            reward_model_errors, transition_model_errors = (
                stomp_framework.model_learning.learn_model(option_idx, off_policy_steps)
            )
            model_learning_logs.append((reward_model_errors, transition_model_errors))

        print("\n[INFO] Planning with learned options and models...")
        planning_logs = stomp_framework.planning.plan_with_options(
            num_lookahead_operations
        )

        if self.experiment_results_path is not None:
            os.makedirs(self.experiment_results_path, exist_ok=True)
            self.successor.save_successor(self.experiment_results_path)
            self.successor.env.save_room(self.experiment_results_path)
            stomp_foundation.save_vectors(self.experiment_results_path)
            with open(
                join(self.experiment_results_path, "option_learning_logs.pkl"),
                "wb",
            ) as f:
                pickle.dump(option_learning_logs, f)

            with open(
                join(self.experiment_results_path, "model_learning_logs.pkl"),
                "wb",
            ) as f:
                pickle.dump(model_learning_logs, f)

            with open(
                join(self.experiment_results_path, "planning_logs.pkl"), "wb"
            ) as f:
                pickle.dump(planning_logs, f)
            print(f"\n[INFO] Files saved on {self.experiment_results_path}")

        return option_learning_logs, model_learning_logs, planning_logs
