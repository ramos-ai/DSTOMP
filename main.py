import os
import pickle
from multiprocessing import Pool
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dstomp.framework import DSTOMP
from environment.gridworld import GridWorld
from experiments import ExperimentsAvailable, get_experiment

num_runs = 100
all_runs = []
experiment, experiment_results_path = get_experiment(
    ExperimentsAvailable.LARGER_HALLWAY_LARGER_ROOM_WITH_SUCCESSOR
)


def run_experiment(run_idx: int):
    result_folder_path = join(experiment_results_path, f"run_{run_idx}")
    env = GridWorld(experiment.env_design)
    dstomp = DSTOMP(
        env=env,
        num_subgoals=experiment.num_subgoals,
        off_policy_steps_for_successor_representation=experiment.off_policy_steps_for_successor_representation,
        experiment_results_path=result_folder_path,
    )

    env.reset()

    option_learning_logs, model_learning_logs, planning_logs = dstomp.execute(
        off_policy_steps=experiment.off_policy_steps_for_stomp_progression,
        num_lookahead_operations=experiment.num_lookahead_operations,
    )

    return option_learning_logs, model_learning_logs, planning_logs


cpus_available = os.cpu_count()
num_processes = 0 if cpus_available is None else cpus_available - 1
print(f"Running with {num_processes} processes\n")

with Pool(processes=num_processes) as pool:
    all_runs = list(tqdm(pool.map(run_experiment, range(num_runs)), total=num_runs))

with open(join(experiment_results_path, "all_runs.pkl"), "wb") as f:
    pickle.dump(all_runs, f)

option_learning_logs = [run[0] for run in all_runs]
option_learning_logs_mean = np.mean(option_learning_logs, axis=0)
option_learning_logs_std = np.std(option_learning_logs, axis=0)

planning = [run[2] for run in all_runs]
planning_mean = np.mean(planning, axis=0)
planning_std = np.std(planning, axis=0)


def plot_arrays(mean_array, std_array, plotting_info, plotting_name):
    # Create figure and axis
    plt.figure(figsize=(20, 6))

    # Generate x-axis points (assuming these are sequential steps/episodes)
    x = np.arange(len(mean_array))

    # Plot mean line with shaded standard deviation
    plt.plot(x, mean_array, "b-", label="Mean")
    plt.fill_between(
        x,
        mean_array - std_array,
        mean_array + std_array,
        color="b",
        alpha=0.2,
        label="Standard Deviation",
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

for subgoal_idx in range(len(option_learning_logs_mean)):
    plot_arrays(
        option_learning_logs_mean[subgoal_idx],
        option_learning_logs_std[subgoal_idx],
        {
            "xlabel": "Off-Policy Steps",
            "ylabel": "Initial Estimate: v_hat(s0)",
            "title": f"Subgoal {subgoal_idx + 1} Option Learning",
        },
        f"subgoal_{subgoal_idx + 1}_option_learning",
    )

plot_arrays(
    planning_mean,
    planning_std,
    {
        "xlabel": "Number of Planning Look-ahead Operations",
        "ylabel": "Initial State Estimative: v_hat(s0)",
        "title": "Planning with Options",
    },
    "planning_with_options",
)
