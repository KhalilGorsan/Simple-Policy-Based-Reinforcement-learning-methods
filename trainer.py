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
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
