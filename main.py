import os
import pickle
from multiprocessing import Pool
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dstomp.framework import DSTOMP, STOMP
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
    stomp = STOMP(
        env=env,
        subgoal_states_info=experiment.hallways_states_info,
        experiment_results_path=result_folder_path,
    )
    dstomp = DSTOMP(
        env=env,
        num_subgoals=experiment.num_subgoals_for_successor_representation,
        off_policy_steps_for_successor_representation=experiment.off_policy_steps_for_successor_representation,
        experiment_results_path=result_folder_path,
    )

    env.reset()

    stomp_option_learning_logs, stomp_model_learning_logs, stomp_planning_logs = (
        stomp.execute(
            off_policy_steps=experiment.off_policy_steps_for_stomp_progression,
            num_lookahead_operations=experiment.num_lookahead_operations,
        )
    )

    dstomp_option_learning_logs, dstomp_model_learning_logs, dstomp_planning_logs = (
        dstomp.execute(
            off_policy_steps=experiment.off_policy_steps_for_stomp_progression,
            num_lookahead_operations=experiment.num_lookahead_operations,
        )
    )

    result = {
        "stomp": {
            "option_learning": stomp_option_learning_logs,
            "model_learning": stomp_model_learning_logs,
            "planning": stomp_planning_logs,
        },
        "dstomp": {
            "option_learning": dstomp_option_learning_logs,
            "model_learning": dstomp_model_learning_logs,
            "planning": dstomp_planning_logs,
        },
    }

    return result


cpus_available = os.cpu_count()
num_processes = 0 if cpus_available is None else cpus_available - 1
print(f"Running with {num_processes} processes\n")

with Pool(processes=num_processes) as pool:
    all_runs = list(tqdm(pool.map(run_experiment, range(num_runs)), total=num_runs))

with open(join(experiment_results_path, "all_runs.pkl"), "wb") as f:
    pickle.dump(all_runs, f)

stomp_option_learning_logs = []
stomp_planning_logs = []
dstomp_option_learning_logs = []
dstomp_planning_logs = []
for run in all_runs:
    stomp_option_learning_logs.append(run['stomp']['option_learning'])
    stomp_planning_logs.append(run['stomp']['planning'])
    dstomp_option_learning_logs.append(run['dstomp']['option_learning'])
    dstomp_planning_logs.append(run['dstomp']['planning'])

stomp_option_learning_logs_mean = np.mean(stomp_option_learning_logs, axis=0)
stomp_option_learning_logs_std = np.std(stomp_option_learning_logs, axis=0)
stomp_planning_mean = np.mean(stomp_planning_logs, axis=0)
stomp_planning_std = np.std(stomp_planning_logs, axis=0)

dstomp_option_learning_logs_mean = np.mean(dstomp_option_learning_logs, axis=0)
dstomp_option_learning_logs_std = np.std(dstomp_option_learning_logs, axis=0)
dstomp_planning_mean = np.mean(dstomp_planning_logs, axis=0)
dstomp_planning_std = np.std(dstomp_planning_logs, axis=0)


def plot_arrays(mean_arrays, std_arrays, colors, labels, plotting_info, plotting_name):
    # Create figure and axis
    plt.figure(figsize=(20, 6))

    for mean_array, std_array, color, label in zip(mean_arrays, std_arrays, colors, labels):
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
    [stomp_planning_mean, dstomp_planning_mean],
    [stomp_planning_std, dstomp_planning_std],
    ['b', 'g'],
    ['STOMP', 'Dynamic STOMP'],
    {
        "xlabel": "Number of Planning Look-ahead Operations",
        "ylabel": "Initial State Estimative: v_hat(s0)",
        "title": "Planning with Options",
    },
    "planning_with_options",
)

