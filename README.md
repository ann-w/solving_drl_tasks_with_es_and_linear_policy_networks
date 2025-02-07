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

Follow the instructions in the README in the ES folder for the installation and usage instructions. With this code, you can run the Evolution Strategies and Augmented Random Search. For the RL experiments, follow the installation instructions in the [CleanRL library](https://github.com/vwxyzjn/cleanrl). To run the RL linear policies, remove the hidden layers from the network.


<!-- ## Citation

```
@article{wong2024solving,
  title={Solving Deep Reinforcement Learning Benchmarks with Linear Policy Networks},
  author={Wong, Annie and de Nobel, Jacob and B{\"a}ck, Thomas and Plaat, Aske and Kononova, Anna V},
  journal={arXiv preprint arXiv:2402.06912},
  year={2024}
}
``` -->