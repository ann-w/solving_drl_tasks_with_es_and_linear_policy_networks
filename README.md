# Solving Deep Reinforcement Learning Benchmarks with Linear Policy Networks

This is the repository for the paper Solving Deep Reinforcement Learning Tasks with
Evolution Strategies and Linear Policy Networks.

## Abstract

Although deep reinforcement learning methods can learn effective policies for challenging problems, the
underlying algorithms are complex, and training times are often long. This study investigates how several
state-of-the-art versions of Evolution Strategies perform compared to gradient-based deep reinforcement
learning methods. We use Evolution Strategies to optimize the weights of a neural network via neuroevolution,
performing direct policy search. We benchmark both deep policy networks and networks consisting of a
single linear layer from observations to actions for three gradient-based methods, such as Proximal Policy
Optimization, Soft Actor-Critic and Deep Q-Learning. These methods are evaluated against three classical
Evolution Strategies and Augmented Random Search, which all use linear policy networks. Our results reveal
that Evolution Strategies can find effective linear policies for many reinforcement learning benchmark tasks,
unlike deep reinforcement learning methods that can only find successful policies using much larger networks,
suggesting that current benchmarks are easier to solve than previously assumed. Interestingly, Evolution
Strategies, which does not use any gradient information, also achieve results comparable to gradient-based deep
reinforcement learning algorithms for higher-complexity tasks. Furthermore, we find that by directly accessing
the memory state of the game, Evolution Strategies can find successful policies in Atari that outperform the
policies found by Deep Q-Learning. Evolution Strategies also outperform Augmented Random Search in most
benchmarks, demonstrating superior sample efficiency and robustness in training linear policy networks.

##  Usage

The code in this repository can be used to run the Evolution Strategies (CSA-ES, CMA-ES, sep-CMA-ES) and Augmented Random Search.

1. In terminal go to the ES folder.
```
cd es
```

2. Install the code:
```
pip install .
```

3. All of the code can be run from `main.py`. For instance to run CSA on LanderLander:

  ```bash
  python main.py --env=LunarLander-v2 --strat=csa --normalized --sigma0=.5 --lamb=16 --seed=10 --test_every=25 
  ```

The parameters for the strategies are defined as follows:

- CSA-ES `--strat=csa`
- CMA-ES `--strat=cma-es`
- sep-CMA-ES `--strat=sep=cma-es`
- ARS-V1 `--strat=ars`
- ARS-V2 `--strat=ars-v2`


### Deep Reinforcement Learning (RL) and Linear RL Networks
We have used the [cleanrl repository](github.com/vwxyzjn/cleanrl) for the reinforcement learning algorithms. CleanRL is a Deep Reinforcement Learning library that provides high-quality single-file implementation with research-friendly features. To run the RL linear policies, remove the hidden layers from the network. All hyperparameters are reported in the paper.


<!-- ## Citation

```
@article{wong2024solving,
  title={Solving Deep Reinforcement Learning Benchmarks with Linear Policy Networks},
  author={Wong, Annie and de Nobel, Jacob and B{\"a}ck, Thomas and Plaat, Aske and Kononova, Anna V},
  journal={arXiv preprint arXiv:2402.06912},
  year={2024}
}
``` -->