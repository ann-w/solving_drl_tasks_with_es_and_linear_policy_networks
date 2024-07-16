# Solving Deep Reinforcement Learning Benchmarks with Linear Policy Networks

This is the repository for the paper Solving Deep Reinforcement Learning Tasks with
Evolution Strategies and Linear Policy Networks.

## Abstract
Although Deep Reinforcement Learning (DRL)
methods can learn effective policies for challenging problems
such as Atari games and robotics tasks, algorithms are complex
and training times are often long. This study investigates how
Evolution Strategies (ES) perform compared to gradient-based
deep reinforcement learning methods. We use ES to optimize
the weights of a neural network via neuroevolution, performing
direct policy search. We benchmark both regular (deep) policy
networks and networks consisting of a single linear layer from
observations to actions for three gradient-based methods, such
as PPO. These methods are evaluated against three classical
ES methods and Augmented Random Search (ARS), which all
use linear policy networks. Our results reveal that ES can find
effective linear policies for many RL benchmark tasks, unlike
DRL methods that can only find successful policies using much
larger networks, suggesting that current benchmarks are easier
to solve than previously assumed. Interestingly, ES also achieves
results comparable to gradient-based DRL algorithms for higher-
complexity tasks. Furthermore, we find that by directly accessing
the memory state of the game, ES can find successful policies
in Atari that outperform DQN. ES also outperforms ARS in
most benchmarks, demonstrating superior sample efficiency and
robustness in training linear policy networks. While gradient-
based methods have dominated the field in recent years, ES offers
an alternative that is easy to implement, parallelize, understand,
and tune.

##  Usage

Follow the instructions in the README in the ARS or ES folder for the installation and usage instructions.

## Citation

```
@article{wong2024solving,
  title={Solving Deep Reinforcement Learning Benchmarks with Linear Policy Networks},
  author={Wong, Annie and de Nobel, Jacob and B{\"a}ck, Thomas and Plaat, Aske and Kononova, Anna V},
  journal={arXiv preprint arXiv:2402.06912},
  year={2024}
}
```