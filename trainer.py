import numpy as np
import matplotlib.pyplot as plt

from policy import Policy
from policy_based_agents import hill_climbing, steepest_ascent_hill_climbing


class Trainer:
    def __init__(self, population_size=1, variant="hill_climbing"):
        self.population_size = population_size
        self.policies = [Policy() for _ in range(population_size)]
        self.variant = variant

    def train(self):
        if self.variant == "hill_climbing":
            assert (
                self.population_size == 1
            ), "Please make sure that we use one policy in hill climbing"
            self.score = hill_climbing(self.policies[0])
        elif self.variant == "steepest_ascent_hill_climbing":
            self.score = steepest_ascent_hill_climbing(
                policies=self.policies, population_size=self.population_size
            )


def main():
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)

    variants = ["hill_climbing", "steepest_ascent_hill_climbing"]
    for variant in variants:
        if variant == "steepest_ascent_hill_climbing":
            trainer = Trainer(population_size=8, variant=variant)
            trainer.train()
        else:
            trainer = Trainer()
            trainer.train()

        ax.plot(np.arange(len(trainer.score)), trainer.score, label=variant)

    plt.ylabel("Score")
    plt.xlabel("Episode #")
    ax.legend(loc="lower right", shadow=True, fontsize="small")
    plt.savefig("hill_climbing_variants")
    plt.show()


if __name__ == "__main__":
    main()
