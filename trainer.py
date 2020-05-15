from policy import Policy
from policy_based_agents import hill_climbing


class Trainer:
    def __init__(self, policy):
        score = hill_climbing(policy=policy)


def main():
    policy = Policy()
    trainer = Trainer(policy)


if __name__ == "__main__":
    main()
