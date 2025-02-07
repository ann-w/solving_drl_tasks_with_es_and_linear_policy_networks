# Installation
Please make sure the code in this folder is installed properly, using:
```bash
pip install .
```

# Usage
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


