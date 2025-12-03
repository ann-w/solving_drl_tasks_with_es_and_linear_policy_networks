# Installation

Please make sure the code in this folder is installed properly, using:

## 1. Create the conda environment (Python 3.11)

```bash
conda create -n solving-drl-with-es-py310 python=3.10 -y
conda activate solving-drl-with-es
```

## 2. Install the ES package (with all Gymnasium extras)

Run the following from the `es/` folder to pick up the MuJoCo/Atari/Box2d extras shipped with the project:

```bash
pip install -e .
```

## Usage

Then all of the code can be run via the ```main.py``` entrypoint, for example:

```bash
python main.py --env=LunarLander-v2 --strat=csa --normalized --sigma0=.5 --lamb=16 --seed=10 --test_every=25 
```

The following algorithms are included in this repostiory:

- CSA-ES (--strat=csa)
- CMA-ES (--strat=cma-es)
- sep-CMA-ES (--strat=sep=cma-es)
- ARS-V1 (--strat=ars)
- ARS-V2 (--strat=ars-v2)
- LM-MA-ES (--strat=lm-ma-es)
