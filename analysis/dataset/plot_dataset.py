# %%
import json
import math
from collections import Counter, defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
    }
)

TRUNCATE = 50
CACHED = True
COLORS = ["green", "orange", "red"]
# %%

# %%


def entropy(dictionary, normalize=True):
    # sum of -p(x)log(p(x))
    # p(x) is the probability of a state
    total_size = sum(dictionary.values())
    entropy = 0
    for k in dictionary:
        p = dictionary[k] / total_size
        entropy += -p * math.log(p, 2)

    if normalize and len(dictionary.keys()) > 1:
        entropy /= math.log(len(dictionary.keys()), 2)
    return entropy


def savefig(name):
    plt.legend()
    plt.tight_layout()
    plt.savefig(name)
    plt.cla()
    plt.clf()


def generate_data(dataset_name):
    z = torch.load(dataset_name)
    z["state_visitation_distribution"] = json.loads(z["state_visitation_distribution"])
    z["state_action_distribution"] = json.loads(z["state_action_distribution"])

    # %%
    sd = z["state_visitation_distribution"]
    sa = z["state_action_distribution"]

    k = list(sd.keys())
    for i in k:
        if int(i) > TRUNCATE:
            del sd[i]
            del sa[i]

    # state action entropy calculation
    aggregated = Counter()
    aggregated_sa = defaultdict(Counter)
    for i in sd:
        if int(i) > 16:
            break
        print(f"aggregating {i}")
        aggregated += Counter(sd[i])
        for s in sa[i]:
            aggregated_sa[s] += Counter(sa[i][s])
    entropies = []

    print("calculating entropy...")
    for i in sorted(aggregated.items(), key=lambda x: x[1], reverse=True):
        if i[1] > 100:
            entropies.append(entropy(aggregated_sa[i[0]]))
    print("done")
    counts, bins = np.histogram(entropies, bins=100)

    return {
        "state_cardinality": [len(sd[i]) for i in sd],
        "max_freq_of_state": [max(sd[k].values()) for k in sd],
        "entropy_of_state_visitation": [entropy(sd[k]) for k in sd],
        "entropy_of_state_action_distribution": (
            counts,
            bins,
            np.mean(np.array(entropies)),
        ),
        "entropies": entropies,
    }


def generate_plots(data):
    plt.figure(figsize=(4, 2.8), dpi=300)

    # %%
    for i, k in enumerate(data):
        plt.plot(data[k]["state_cardinality"], label=k, c=COLORS[i])
    plt.title("State cardinality over moves")
    savefig("state_cardinality_over_moves.png")

    # %%
    # plt.plot(data['max_freq_of_state'], label=k)
    for i, k in enumerate(data):
        plt.plot(data[k]["max_freq_of_state"], label=k, c=COLORS[i])
    plt.yscale("log")
    plt.title("Max frequency of a given state over moves")
    savefig("max_frequency_of_state_over_moves.png")

    # %%
    # entropy calculatio

    # plt.plot(data['entropy_of_state_visitation'], label=k)
    for i, k in enumerate(data):
        plt.plot(data[k]["entropy_of_state_visitation"], label=k, c=COLORS[i])
    plt.title("Entropy of state visitation distribution over moves")
    savefig("entropy_of_state_visitation_distribution_over_moves.png")

    for i, k in enumerate(data):
        counts, bins, mean = data[k]["entropy_of_state_action_distribution"]
        counts, bins = np.histogram(data[k]["entropies"], bins=25)

        print(f"total points: {sum(counts)}")
        plt.axvline(x=mean, color=COLORS[i])

        plt.hist(
            bins[:-1],
            bins=bins,
            weights=counts / sum(counts),
            # density=True,
            histtype="bar",
            label=f"Max Rating: {k}",
            color=COLORS[i],
            alpha=0.2,
        )

    for i, k in enumerate(data):
        counts, bins, mean = data[k]["entropy_of_state_action_distribution"]
        plt.text(
            # mean - 0.4175,
            mean,
            plt.ylim()[1] * (0.75 - 0.2 * i),
            rf"$\mathbb{{E}}[\mathcal{{H}}]$ {k}: {mean:.2f}",
            color="black",
            verticalalignment="top",
            weight="bold",
        )
    # TODO plot expectation

    plt.xlabel(r"$\mathcal{H}(Y|X)$")
    plt.ylabel(r"$\mathbb{P}(\mathcal{H}(Y|X))$")
    plt.title(r"$\mathcal{H}$ of action distribution over common states")
    savefig("entropy_of_action_distribution_over_common_states.png")


# entropies = []

# for i in sd:
#     sorted_states = sorted(sd[i].items(), key=lambda x: x[1], reverse=True)
#     entropies.append(
#         entropy(z["state_action_distribution"][i][sorted_states[0][0]])
#     )
# plt.plot(entropies[:25])
# plt.title(
#     "Entropy of state action distribution over moves"
# )  # this one should be a bar chart, or avg'd across multiple moves. use more than just the first most common move as well. idea: just use several common states (with no meaning on x axis), and plot entropy across different datasets. then plot avg  as well. this can give us a table too!
# savefig("entropy_of_state_action_distribution_over_moves.png")

# %%

if __name__ == "__main__":
    if not CACHED:
        data_1000 = generate_data(
            "/path/to/state_action_dist_1000_1716063906.587322.pt"
        )
        data_1300 = generate_data(
            "/path/to/state_action_dist_1300_1716064171.990619.pt"
        )
        data_1500 = generate_data(
            "/path/to/state_action_dist_1500_1716064332.7873602.pt"
        )
        all_data = {"1000": data_1000, "1300": data_1300, "1500": data_1500}
        torch.save(all_data, "all_data.pt")
    else:
        all_data = torch.load("all_data.pt")

    generate_plots(all_data)
