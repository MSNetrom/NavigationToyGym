import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def read_and_plot_results(results_dir: Path):

    # Read the json file
    with open(results_dir / "results.json", "r") as f:
        results = json.load(f)

    # Plot the results
    results = np.array(results)

    # Create a 2D plot with alpha_1 and alpha_2 along the axes, and scatter crashes
    fig, ax = plt.subplots()
    #ax.scatter(results[:, 1], results[:, 2], c=results[:, 0], cmap='coolwarm')
    # Plot as red if crashed, green if not
    ax.scatter(results[results[:, 0] == 0, 1], results[results[:, 0] == 0, 2], c='green', label=f"No collision: {np.sum(results[:, 0] == 0)}", alpha=0.5)
    ax.scatter(results[results[:, 0] == 1, 1], results[results[:, 0] == 1, 2], c='red', label=f"Collision: {np.sum(results[:, 0] == 1)}", alpha=0.5)

    ax.legend(
        loc='upper left',
        bbox_to_anchor=(0, 1.11),
        borderaxespad=0.,  # No padding between the axes and the legend
        ncol=2,
        fontsize=15,
        #title='Crash Status'
    )

    ax.set_xlabel(r'$\alpha_1$', fontsize=15)
    ax.set_ylabel(r'$\alpha_2$', fontsize=15)

    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Save as pdf
    fig.savefig(results_dir / "scatter.pdf")

    # Make plot with histogram for lidar distance and crash
    fig, ax = plt.subplots()
    ax.hist(results[results[:, 0] == 1, 3], bins=20, color='red', alpha=0.5, label="Crashed")
    ax.set_xlabel("Lidar distance")
    ax.set_ylabel("Frequency")
    fig.savefig(results_dir / "hist_lidar.pdf")


    # Make scatter plot where we exclude crash distances smaller than 1
    fig, ax = plt.subplots()

    res_small_dist = results[(results[:, 4] > 0.5) | (results[:, 0] == 0)]

    ax.scatter(res_small_dist[res_small_dist[:, 0] == 0, 1], res_small_dist[res_small_dist[:, 0] == 0, 2], c='green', label=f"No collision: {np.sum(res_small_dist[:, 0] == 0)}", alpha=0.5)
    ax.scatter(res_small_dist[res_small_dist[:, 0] == 1, 1], res_small_dist[res_small_dist[:, 0] == 1, 2], c='red', label=f"Collision: {np.sum(res_small_dist[:, 0] == 1)}", alpha=0.5)

    #ax.legend()

    ax.legend(
        loc='upper left',
        bbox_to_anchor=(0, 1.11),
        borderaxespad=0.,  # No padding between the axes and the legend
        ncol=2,
        fontsize=15,
        #title='Crash Status'
    )

    ax.set_xlabel(r'$\alpha_1$', fontsize=15)
    ax.set_ylabel(r'$\alpha_2$', fontsize=15)

    ax.tick_params(axis='both', which='major', labelsize=14)

    fig.savefig(results_dir / "scatter_no_small_dist.pdf")

    # Plot distances of crashes
    fig, ax = plt.subplots()

    ax.hist(results[results[:, 0] == 1, 4], bins=20, color='red', alpha=0.5, label="Crashed")
    ax.set_xlabel("Crash distance")
    ax.set_ylabel("Frequency")
    fig.savefig(results_dir / "hist_crash_dist.pdf")

    print("Crash distances:", results[results[:, 0] == 1, 4])



if __name__ == "__main__":

    # Read the results
    results_dir = Path("ablation_results")
    read_and_plot_results(results_dir)
    
    #plt.show()