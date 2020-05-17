import numpy as np
import gym
from collections import deque
from policy import Policy


# CartPole-v0 Gym environment
env = gym.make("CartPole-v0")


def hill_climbing(
    policy, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-2
):
    """Implementation of hill climbing with adaptive noise scaling.
        
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        noise_scale (float): standard deviation of additive noise
    """
    scores_deque = deque(maxlen=100)
    scores = []
    best_R = -np.Inf
    best_w = policy.w
    for i_episode in range(1, n_episodes + 1):
        rewards = []
        state = env.reset()
        for _ in range(max_t):
            action = policy.act(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        if R >= best_R:  # found better weights
            best_R = R
            best_w = policy.w
            noise_scale = max(1e-3, noise_scale / 2)
            policy.w += noise_scale * np.random.rand(*policy.w.shape)
        else:  # did not find better weights
            noise_scale = min(2, noise_scale * 2)
            policy.w = best_w + noise_scale * np.random.rand(*policy.w.shape)

        if i_episode % print_every == 0:
            print(
                "Episode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )
        if np.mean(scores_deque) >= 195.0:
            print(
                "Environment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    i_episode - 100, np.mean(scores_deque)
                )
            )
            policy.w = best_w
            break

    return scores


def steepest_ascent_hill_climbing(
    policies,
    population_size=8,
    n_episodes=1000,
    max_t=1000,
    gamma=1.0,
    print_every=100,
    noise_scale=1e-2,
):
    """Implementation of steepest ascent hill climbing with adaptive noise scaling

    Params
    ======
        policies (list): a list of  initiliazed random policies
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        noise_scale (float): standard deviation of additive noise
    """
    scores_deque = deque(maxlen=100)
    scores = []
    best_R = -np.Inf
    for i_episode in range(1, n_episodes + 1):
        individuals_R = []
        R_sequence = []
        for policy in policies:
            state = env.reset()
            rewards = []
            for _ in range(max_t):
                action = policy.act(state)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break
            individuals_R.append(sum(rewards))
            R_sequence.append(rewards)

        # Select the best individual from the population
        best_individual_idx = np.argmax(individuals_R)
        b_policy = policies[best_individual_idx]
        # Keep track of rewards for the best individual
        scores_deque.append(individuals_R[best_individual_idx])
        scores.append(individuals_R[best_individual_idx])

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, R_sequence[best_individual_idx])])

        # create a new population
        policies = [Policy() for _ in range(population_size)]

        if R >= best_R:  # found better weights
            best_R = R
            best_w = b_policy.w
            noise_scale = max(1e-3, noise_scale / 2)
            for policy in policies:
                policy.w = b_policy.w + noise_scale * np.random.rand(*policy.w.shape)
        else:  # did not find better weights
            noise_scale = min(2, noise_scale * 2)
            for policy in policies:
                policy.w = best_w + noise_scale * np.random.rand(*policy.w.shape)

        if i_episode % print_every == 0:
            print(
                "Episode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )
        if np.mean(scores_deque) >= 195.0:
            print(
                "Environment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    i_episode - 100, np.mean(scores_deque)
                )
            )
            policy.w = best_w
            break

    return scores
