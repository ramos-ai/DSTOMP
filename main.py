import os
import pickle
from multiprocessing import Pool
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from dstomp.framework import DSTOMP, STOMP
from environment.gridworld import GridWorld
from experiments import ExperimentsAvailable, get_experiment

num_runs = 100
all_runs = []
experiment, experiment_results_path = get_experiment(
    ExperimentsAvailable.TWO_ROOM_WITH_SUCCESSOR
)


def run_experiment(run_idx: int):
    result_folder_path = join(experiment_results_path, f"run_{run_idx}")
    env = GridWorld(
        room_array=experiment.env_design, success_prob=experiment.env_success_prob
    )
    stomp = STOMP(
        env=env,
        subgoal_states_info=experiment.hallways_states_info,
        experiment_results_path=result_folder_path,
        alpha_step_size=experiment.alpha_step_size,
    )
    dstomp = DSTOMP(
        env=env,
        num_subgoals=experiment.num_subgoals_for_successor_representation,
        off_policy_steps_for_successor_representation=experiment.off_policy_steps_for_successor_representation,
        experiment_results_path=result_folder_path,
        alpha_step_size=experiment.alpha_step_size,
    )
    dstomp_reward_awareness = DSTOMP(
        env=env,
        num_subgoals=experiment.num_subgoals_for_successor_representation,
        off_policy_steps_for_successor_representation=experiment.off_policy_steps_for_successor_representation,
        experiment_results_path=result_folder_path,
        alpha_step_size=experiment.alpha_step_size,
        successor_reward_awareness=True,
    )

    env.reset()

    stomp.execute(
        off_policy_steps=experiment.off_policy_steps_for_stomp_progression,
        num_lookahead_operations=experiment.num_lookahead_operations,
    )

    env.reset()

    dstomp.execute(
        off_policy_steps=experiment.off_policy_steps_for_stomp_progression,
        num_lookahead_operations=experiment.num_lookahead_operations,
    )

    env.reset()

    dstomp_reward_awareness.execute(
        off_policy_steps=experiment.off_policy_steps_for_stomp_progression,
        num_lookahead_operations=experiment.num_lookahead_operations,
    )



cpus_available = os.cpu_count()
num_processes = 0 if cpus_available is None else cpus_available - 1
print(f"Running with {num_runs} runs, with {num_processes} processes\n")

with Pool(processes=num_processes) as pool:
    pool.map(run_experiment, range(num_runs))


def collect_logs(
    base_results_path: str, model_name: str, log_filename: str, num_runs: int = 30
):
    logs = []
    for idx in range(num_runs):
        log_path = os.path.join(
            base_results_path, f"run_{idx}", model_name, log_filename
        )
        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                logs.append(pickle.load(f))
        else:
            print(f"Warning: {log_path} does not exist.")
    return logs


stomp_planning_logs = collect_logs(
    experiment_results_path, "stomp", "planning_logs.pkl", num_runs
)
dstomp_planning_logs = collect_logs(
    experiment_results_path, "dstomp", "planning_logs.pkl", num_runs
)
dstomp_reward_awareness_planning_logs = collect_logs(
    experiment_results_path, "dstomp_reward_awareness", "planning_logs.pkl", num_runs
)


stomp_planning_mean, stomp_planning_std = (
    np.mean(stomp_planning_logs, axis=0),
    np.std(stomp_planning_logs, axis=0),
)
dstomp_planning_mean, dstomp_planning_std = (
    np.mean(dstomp_planning_logs, axis=0),
    np.std(dstomp_planning_logs, axis=0),
)
dstomp_reward_awareness_planning_mean, dstomp_reward_awareness_planning_std = (
    np.mean(dstomp_reward_awareness_planning_logs, axis=0),
    np.std(dstomp_reward_awareness_planning_logs, axis=0),
)


def plot_arrays(mean_arrays, std_arrays, colors, labels, plotting_info, plotting_name):
    # Create figure and axis
    plt.figure(figsize=(20, 6))

    for mean_array, std_array, color, label in zip(
        mean_arrays, std_arrays, colors, labels
    ):
        # Generate x-axis points (assuming these are sequential steps/episodes)
        x = np.arange(len(mean_array))

        # Plot mean line with shaded standard deviation
        plt.plot(x, mean_array, f"{color}-", label=f"{label}")
        plt.fill_between(
            x,
            mean_array - std_array,
            mean_array + std_array,
            color=f"{color}",
            alpha=0.2,
        )

    # Customize the plot
    plt.xlabel(plotting_info["xlabel"])
    plt.ylabel(f"{plotting_info['ylabel']}\n(Average Over 100 runs)")
    plt.title(plotting_info["title"])
    plt.legend()
    plt.grid(True)

    # Show the plot
    # plt.show()
    save_fig_path = join(experiment_results_path, f"{plotting_name}.png")
    plt.savefig(f"{save_fig_path}", bbox_inches="tight")


# for subgoal_idx in range(len(option_learning_logs_mean)):
#     plot_arrays(
#         option_learning_logs_mean[subgoal_idx],
#         option_learning_logs_std[subgoal_idx],
#         {
#             "xlabel": "Off-Policy Steps",
#             "ylabel": "Initial Estimate: v_hat(s0)",
#             "title": f"Subgoal {subgoal_idx + 1} Option Learning",
#         },
#         f"subgoal_{subgoal_idx + 1}_option_learning",
#     )

plot_arrays(
    [stomp_planning_mean, dstomp_planning_mean, dstomp_reward_awareness_planning_mean],
    [stomp_planning_std, dstomp_planning_std, dstomp_reward_awareness_planning_std],
    ["b", "g", "r"],
    ["STOMP", "Dynamic STOMP", "Dynamic STOMP - Reward Awareness"],
    {
        "xlabel": "Number of Planning Look-ahead Operations",
        "ylabel": "Initial State Estimative: v_hat(s0)",
        "title": "Planning with Options",
    },
    "planning_with_options",
)
