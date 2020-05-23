# Simple-Policy-Based-Reinforcement-learning-methods
Simple policy based RL methods to learn Cartpole

Install
--------------------------------------------------------------------------------
We use:
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
  to setup the environment,
- and python 3.7

Setup our environment:
```bash
conda --version

# Clone the repo
git clone https://github.com/KhalilGorsan/Simple-Policy-Based-Reinforcement-learning-methods.git
cd Simple-Policy-Based-Reinforcement-learning-methods

# Create a conda env
conda env create -f environment.yml

source activate deeprl_udacity
```

Environment
--------------------------------------------------------------------------------
**CartPole-v0**
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless
track.

The system is controlled by applying a force of +1 or -1 to the cart.

The pendulum starts upright, and the goal is to prevent it from falling over.

A reward of +1 is provided for every timestep that the pole remains upright.

The episode ends when the pole is more than 15 degrees from vertical, or the cart moves
more than 2.4 units from the center.

CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive
trials.

For more information, check [openAI official website](https://gym.openai.com/envs/CartPole-v0/)

Training
--------------------------------------------------------------------------------
We implemented three methods : **Vanilla Hill climbing, Hill climbing with steepest
ascent, and Cross entropy method.**

To run the training, use,
```bash
python trainer.py
```

Results
---------------------------------------------------------------------------------
![training results](https://github.com/KhalilGorsan/Simple-Policy-Based-Reinforcement-learning-methods/blob/master/hill_climbing_CEM.png)

- CEM converges faster compared to hill climbing with steepest ascent and vanilla
hill climbing.
- Using a population of candidate solution and average on a fraction of them to determine
the next one seems to stabilize the convergence.
