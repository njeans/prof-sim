# Prof Simulations

### Edit Simulation in [prof_sim.py](prof_sim.py#L1556)

Example: 

`scenario_arbitrage(random_type='rndm', num_intervals=10, num_iters=100, debug= False, num_prof_users=3, mevshare_percentage=.9`

### Options for 'scenario_arbitrage'
* random_type ('same' or 'rndm', optional): whether each user has a different random ('rndm') preference or the 'same' random preference. Defaults to 'same'.
* num_intervals (int, optional): number of volatilities to try. Defaults to 10.
* num_iters (int, optional): number of iterations of each volatility. Defaults to 10.
debug (bool, optional): print extra debug information. Defaults to False.
* num_prof_users (int, optional): number of users or PROF/MEVShare. Defaults to 2.
* mevshare_percentage (float, optional): percentage of profit returned to MEVShare users. Defaults to .9.
* min_volatility (float, optional): minimum volatility. Defaults to 0.01.
* max_volatility (float, optional): maximum volatility. Defaults to 5.


Run with

```
python prof_sim.py
```