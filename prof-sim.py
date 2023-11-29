import math
import numpy as np
import copy
import matplotlib.pyplot as plt

"""
Toy model of an on-chain ecosystem with two tokens and one DEX.
Transactions on the DEX create an opportunity for sandwich attacks.
"""
class ExecutionException(Exception):
    pass
def require(cond, msg=None):
    if not cond: raise ExecutionException(msg)

class Chain():
    def __init__(self, poolA=1000., poolB=1000., accounts=None):
        self.poolA = poolA
        self.poolB = poolB
        if accounts is None:
            self.accounts = {"alice": [100.,100.],
                             "bob":   [100.,100.]}
        else:
            self.accounts = copy.deepcopy(accounts)

    def apply(self, tx, debug=True):
        # Apply the transaction, updating the pool price
        if (tx['type'] == 'swap'):
            if tx['qty'] >= 0:
                # Sell qty of tokenA, buy at least rsv of tokenB
                amtA = tx['qty']
                amtB = pool_swap(self.poolA, self.poolB, amtA)
                account = self.accounts[tx["sndr"]]
                require(self.accounts[tx['sndr']][0] >= amtA, f'not enough balance for trade {tx} {amtA} account {account}')
                require(amtB >= 0)
                require(amtB >= -tx['rsv'] or tx['rsv'] == 0, f"slippage exceeded amtB:{amtB}, -rsv:{-tx['rsv']}")
                require(self.poolB - amtB >= 0, 'exhausts pool')
                if debug:
                    print("\t",tx['sndr'], "sell", amtA, "of A and gets", amtB, "of B with slippage", amtB ,">=",  -tx['rsv'])
                amtB = -amtB
            else:
                # Sell qty of tokenB, buy at least rsv of tokenA
                amtB = -tx['qty']
                amtA = pool_swap(self.poolB, self.poolA, amtB)
                account = self.accounts[tx["sndr"]]
                require(self.accounts[tx['sndr']][1] >= amtB, f'not enough balance for trade {tx} {amtB} account {account}')
                require(amtA >= 0)
                require(amtA >= tx['rsv'] or tx['rsv'] == 0, f"slippage exceeded amtA:{amtA}, rsv:{tx['rsv']}")
                require(self.poolA + amtA >= 0, 'exhausts pool')
                if debug:
                    print("\t",tx['sndr'], "sell", amtB, "of B and gets", amtA, "of A with slippage", amtA ,">=",  tx['rsv'])
                amtA = -amtA

            self.poolA += amtA
            self.poolB += amtB
            self.accounts[tx['sndr']][0] -= amtA
            self.accounts[tx['sndr']][1] -= amtB

        else:
            raise ValueError("unknown tx type")


# This is the basic Uniswap v2 rule
def pool_swap(poolA, poolB, amtA):
    # Solve the constant product poolA*poolB == (poolA+amtA)*(poolB+amtB)
    # Convention: +amtA means amtA removed from the pool
    amtB = poolB - poolA*poolB / (poolA + amtA)
    assert np.isclose(poolA*poolB, (poolA + amtA)*(poolB - amtB))
    return amtB

def create_swap(sndr,qty,rsv):
    return dict(type="swap",sndr=sndr,qty=qty,rsv=rsv,auth="auth")

def utility(pref, portfolio):
    assert len(pref) == len(portfolio) == 2
    return pref[0] * portfolio[0] + pref[1] * portfolio[1]

# figure out the optimal frontrun transaction
def produce_sandwich(chain, tx_victim, attacker):
    """
    args: 
      chain: a copy of the current chainstate
      tx_victim: dict(type="swap",sender=_, qty=_, rsv=_}
      attacker: attacker name
    returns (tx_front, tx_back):
      tx_front: a front run swap
      tx_back: a backrun swap
    """
    # print("tx_victim", tx_victim)
    assert tx_victim["type"] == "swap"
    print("tx_victim", tx_victim)
    if tx_victim['qty'] == 0:
        return create_swap(attacker,0,0), create_swap(attacker,0,0)

    min_front = 1e-3
    if tx_victim["qty"] < 0: 
        max_front = abs(chain.accounts[attacker][1])
    else:
        max_front = chain.accounts[attacker][0]

    last_successful_front = 0
    chain = copy.deepcopy(chain)
    while True:
        chain_copy = copy.deepcopy(chain)
        frontrun_amt = (min_front + max_front) / 2. 
        if tx_victim["qty"] < 0:
            frontrun_amt = -frontrun_amt
        # print("try", frontrun_amt, "min", min_front , "max", max_front)
        tx = create_swap(attacker, frontrun_amt, 0)
        try:
            chain_copy.apply(tx, debug=False)
            chain_copy.apply(tx_victim, debug=False)
            min_front = abs(frontrun_amt)
            last_successful_front = frontrun_amt 
        except Exception as e:
            max_front = abs(frontrun_amt)
            # print("Exception", e)

        if abs(max_front - min_front) < 1e-3:
            # print("found", last_successful_front, max_front - min_front, 1e-3)
            break
        

    if tx_victim["qty"] < 0:
        backrun_amt = pool_swap(chain.poolB, chain.poolA, -last_successful_front)
    else:
        backrun_amt = -pool_swap(chain.poolA, chain.poolB, last_successful_front)

    return create_swap(attacker, last_successful_front, 0), create_swap(attacker, backrun_amt, 0)

def optimal_trade(poolA, poolB, prefA, prefB): #trade b for a
    assert prefA + prefB == 2
    print("optimal", poolA, poolB, prefA, prefB, (math.sqrt(poolB)*math.sqrt(2*prefA*poolA - (prefA**2)*poolA) - prefA * poolA)/prefA)
    return (math.sqrt(poolB)*math.sqrt(2*prefA*poolA - (prefA**2)*poolA) - prefA * poolA)/prefA

# The party tries to trade, in the direction of the top-of-block price
def make_trade(chain, sndr,prefs, portf):
    #token A: apples, token B: dollars
    chn = copy.deepcopy(chain)
    my_price =  prefs[0] / prefs[1] #utils/apple / utils/dollar -> dollars/apple
    pool_price = chain.poolB/chain.poolA #dollars/apple

    # slippage = .9
    print("pool_price", pool_price, "my_price", my_price)
    if pool_price > my_price: #trade A for B
        print("trade A for B")
        qty = min(1., abs(optimal_trade(chain.poolB, chain.poolA, prefs[1], prefs[0])))
        # slip =  -pool_swap(chain.poolA, chain.poolB, qty) * slippage
        slip = -prefs[0]* qty / prefs[1] # slippage s.t. new net utility will equal old net utility
        print("prefs", prefs[0], prefs[1], "pool",chain.poolA, chain.poolB, "optimal trade", optimal_trade(chain.poolB, chain.poolA, prefs[1], prefs[0]), "qty", qty, "slip", slip)
        assert abs(slip) <= abs(pool_swap(chain.poolA, chain.poolB, qty))
    elif my_price > pool_price: #trade B for A
        print("trade B for A")
        # slip = pool_swap(chain.poolB, chain.poolA, qty) * slippage
        qty = -min(1., abs(optimal_trade(chain.poolA, chain.poolB, prefs[0], prefs[1])))
        slip = -prefs[1]* qty / prefs[0] # slippage s.t. new net utility will equal old net utility
        print("prefs",prefs[0], prefs[1], "pool",chain.poolA, chain.poolB,"optimal trade", -optimal_trade(chain.poolA, chain.poolB, prefs[0], prefs[1]), "qty", qty, "slip", slip)
        assert abs(slip) <= abs(pool_swap(chain.poolB, chain.poolA, abs(qty)))
    else:
        qty = 0
        slip = 0
    return create_swap(sndr, qty, slip)

def scenario3():
    preferences = {"alice":[1.0130039880917086, 0.9869960119082914]}
    pool_A = 1000
    pool_B = 1000
    accounts = {"alice":[100,100],
                "bob":[1000000,1000000]}
    amtb = -optimal_trade(pool_A, pool_B, preferences['alice'][0], preferences['alice'][1])
    amta = optimal_trade(pool_B, pool_A, preferences['alice'][1], preferences['alice'][0])

    print("amtb", amtb)
    print("amta", amta)

    chainA = Chain(pool_A, pool_B, accounts)
    chainB = Chain(pool_A, pool_B, accounts)
    utility_old = utility(preferences['alice'], accounts['alice'])

    txA = create_swap('alice', amta, 0)
    txB = create_swap('alice', -amtb, 0)

    chainA.apply(txA, debug=True)
    chainB.apply(txB, debug=True)

    utility_newA = utility(preferences['alice'], chainA.accounts['alice'])
    utility_newB = utility(preferences['alice'], chainB.accounts['alice'])

    print("pool_price", pool_B/pool_A)
    print("my_price", preferences['alice'][0]/preferences['alice'][1])
    print("utility old", utility_old)
    print("utility A", utility_newA)
    print("utility B", utility_newB)



def scenario4A():
    # compare utility over time with/without sandwhich attacks
    preferences = {"alice":[1.01,0.99],  # Alice would prefer to buy tokenA
                    "bob":[1.0, 1.0]}      # Bob just wants to skim any and all tokens 
    pool_A = 1000
    pool_B = 1000
    accounts = {"alice":[100,100],
                "bob":[1000000,1000000]}
    chain = Chain(pool_A, pool_B, accounts)
    chain_sand = Chain(pool_A, pool_B, accounts)

    num_iters = 10
    driftAlice = np.exp(np.random.normal(loc=0., scale=.01, size=num_iters))

    utils_alice = []
    utils_alice_sand = []
    utils_bob_sand = []

    diffs_alice = []
    diffs_alice_sand = []
    diffs_bob_sand = []
    xs = []
    for i in range(num_iters):
        preferences['alice'][0] = preferences['alice'][0] * driftAlice[i]
        preferences['alice'][1] = 2.0 - preferences['alice'][0]

        txAlice = make_trade(chain, 'alice', preferences['alice'], chain.accounts['alice'])
        txAlice_sand = make_trade(chain_sand, 'alice', preferences['alice'], chain_sand.accounts['alice'])
        txFront, txBack = produce_sandwich(chain_sand, txAlice_sand, 'bob')

        util_a_old = utility(preferences['alice'], chain.accounts['alice'])
        util_a_old_sand = utility(preferences['alice'], chain_sand.accounts['alice'])
        util_b_old_sand = utility(preferences['bob'], chain_sand.accounts['bob'])
        try:
            print("Non-Sandwhich Chain")

            chain.apply(txAlice)

            print("Sandwhich Chain")
            print("chain_sand before apply", chain_sand.poolA, chain_sand.poolB, chain_sand.accounts)

            chain_sand.apply(txFront)
            chain_sand.apply(txAlice_sand)
            chain_sand.apply(txBack)
            print("chain_sand after apply", chain_sand.poolA, chain_sand.poolB, chain_sand.accounts)

        except ExecutionException as e: 
            print(e)
            raise e

        util_a_new = utility(preferences['alice'], chain.accounts['alice'])
        util_a_new_sand = utility(preferences['alice'], chain_sand.accounts['alice'])
        util_b_new_sand = utility(preferences['bob'], chain_sand.accounts['bob'])

        utils_alice.append(util_a_new)
        utils_alice_sand.append(util_a_new_sand)
        utils_bob_sand.append(util_b_new_sand)

        diffs_alice.append(util_a_new-util_a_old)
        diffs_alice_sand.append(util_a_new_sand-util_a_old_sand)
        diffs_bob_sand.append(util_b_new_sand-util_b_old_sand)
        xs.append(i)
        print(i, "util_alice_old", util_a_old, "util_alice_new", util_a_new, "diff_util_alice", util_a_new-util_a_old, "util_a_old_sand", util_a_old_sand, "util_a_new_sand", util_a_new_sand, "diff_util_alice_sand", util_a_new_sand-util_a_old_sand, "util_b_old_sand", util_b_old_sand, "util_b_new_sand", util_b_new_sand, "diff_util_bob", util_b_new_sand-util_b_old_sand)
        print(i,"driftAlice", driftAlice[i], "prefs_alice", preferences['alice'], "txAlice", txAlice_sand["qty"], "txFront", txFront['qty'], "txBack", txBack['qty'], "diff_util_alice", util_a_new-util_a_old, "diff_util_alice_sand", util_a_new_sand-util_a_old_sand, "diff_util_bob", util_b_new_sand-util_b_old_sand)

    print("utils_alice", utils_alice)
    print("utils_alice_sand", utils_alice_sand)
    print("utils_bob_sand", utils_bob_sand)

    print("diffs_alice", diffs_alice)
    print("diffs_alice_sand", diffs_alice_sand)
    print("diffs_bob_sand", diffs_bob_sand)

    plt.figure(0)
    plt.clf()
    plt.plot(xs,diffs_alice,xs,diffs_alice_sand, diffs_bob_sand)
    plt.xlabel('iter')
    plt.ylabel('difference in net utility')
    plt.title(f'difference in net utility after trading')
    plt.legend(['alice','alice_sandwhich', 'bob'])

    plt.figure(1)
    plt.clf()
    plt.plot(xs,utils_alice,xs,utils_alice_sand)
    plt.xlabel('iter')
    plt.ylabel('net utility')
    plt.title(f'net utility after trading')
    plt.legend(['alice','alice_sandwhich'])

    plt.show()


if __name__ == "__main__":
    # scenario3()
    scenario4A()
