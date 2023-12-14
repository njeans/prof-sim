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

    def __str__(self):
        return f"PoolA: {self.poolA} PoolB: {self.poolB} accounts: {self.accounts}"

    def price(self, token):
        if token == 'A' or token == 'a':
            return self.poolA/self.poolB
        else:
            return self.poolB/self.poolA
    
    def product(self):
        return self.poolA*self.poolB

    def transfer(self, sender, amount, chain, token):
        if token == 'A' or token == 'a':
            assert self.accounts[sender][0] >= amount
            if sender not in chain.accounts:
                chain.accounts[sender] = [0,0]
            chain.accounts[sender][0] += amount
            self.accounts[sender][0] -= amount
        else:
            assert self.accounts[sender][1] >= amount
            if sender not in chain.accounts:
                chain.accounts[sender] = [0,0]
            chain.accounts[sender][1] += amount
            self.accounts[sender][1] -= amount

# This is the basic Uniswap v2 rule
def pool_swap(poolA, poolB, amtA):
    # Solve the constant product poolA*poolB == (poolA+amtA)*(poolB+amtB)
    # Convention: +amtA means amtA removed from the pool
    amtB = poolB - poolA*poolB / (poolA + amtA)
    assert np.isclose(poolA*poolB, (poolA + amtA)*(poolB - amtB))
    return amtB

def create_swap(sndr,qty,rsv): 
    if qty < 0:
        assert rsv >= 0
    elif qty > 0:
        assert rsv <= 0
    return dict(type="swap",sndr=sndr,qty=qty,rsv=rsv,auth="auth")

def utility(pref, portfolio):
    assert len(pref) == len(portfolio) == 2
    return pref[0] * portfolio[0] + pref[1] * portfolio[1]

# figure out the optimal frontrun transaction
def produce_sandwich(chain, tx_victims, attacker, debug=False):
    """
    args: 
      chain: a copy of the current chainstate
      tx_victims: dict(type="swap",sender=_, qty=_, rsv=_}, or a list of dict
      attacker: attacker name
    returns (tx_front, tx_back):
      tx_front: a frontrun swap
      tx_back: a backrun swap
    """
    if not isinstance(tx_victims, list):
        tx_victims = [tx_victims]
    for tx_victim in tx_victims:
        assert tx_victim["type"] == "swap"
        # print("tx_victim", tx_victim)
    tx_sum = sum(list(map(lambda x: x["qty"], tx_victims)))
    if tx_sum == 0:
        return create_swap(attacker,0,0), create_swap(attacker,0,0)

    min_front = 1e-3
    if tx_sum < 0: 
        max_front = abs(chain.accounts[attacker][1])
    else:
        max_front = chain.accounts[attacker][0]

    last_successful_front = 0
    chain = copy.deepcopy(chain)
    while True:
        chain_copy = copy.deepcopy(chain)
        frontrun_amt = (min_front + max_front) / 2. 
        if tx_sum < 0:
            frontrun_amt = -frontrun_amt
        if debug:
            print("try", frontrun_amt, "min", min_front , "max", max_front)
        tx = create_swap(attacker, frontrun_amt, 0)
        try:
            chain_copy.apply(tx, debug=False)
            for tx_victim in tx_victims:
                chain_copy.apply(tx_victim, debug=False)
            min_front = abs(frontrun_amt)
            last_successful_front = frontrun_amt 
        except Exception as e:
            max_front = abs(frontrun_amt)
            if debug:
                print("Exception", e)

        if abs(max_front - min_front) < 1e-3:
            if debug:
                print("found", last_successful_front, max_front - min_front, 1e-3)
            break
        

    if tx_sum < 0:
        backrun_amt = pool_swap(chain.poolB, chain.poolA, -last_successful_front)
    else:
        backrun_amt = -pool_swap(chain.poolA, chain.poolB, last_successful_front)

    return create_swap(attacker, last_successful_front, 0), create_swap(attacker, backrun_amt, 0)

def optimal_trade(poolA, poolB, prefA, prefB): #trade b for a
    assert prefA + prefB == 2
    # print("optimal", poolA, poolB, prefA, prefB, (math.sqrt(poolB)*math.sqrt(2*prefA*poolA - (prefA**2)*poolA) - prefA * poolA)/prefA)
    return (math.sqrt(poolB)*math.sqrt(2*prefA*poolA - (prefA**2)*poolA) - prefA * poolA)/prefA

def optimal_arbitrage_algebra(chain1, chain2, prefs, attacker):

    if prefs[0] < prefs[1]: #todo fractional preferences?
        tokenTrade = 'A'
    else:
        tokenTrade = 'B'

    if chain1.price(tokenTrade) < chain2.price(tokenTrade):
        c1 = chain2 
        c2 = chain1
    else:
        c1 = chain1 
        c2 = chain2

    print("chain1", chain1.price(tokenTrade), "chain2", chain2.price(tokenTrade), "want" , tokenTrade)
    #how much of tokenTrade we can sell to chain1 before it has the same 
    #price as chain2
    a=c1.product()
    b=c2.product()
    if tokenTrade == 'B':
        c=c1.poolA
        d=c2.poolB 
        e=c1.poolB 
    else:
        c=c1.poolB
        d=c2.poolA 
        e=c1.poolA 
    dsq = d**2
    esq = e**2

    #https://www.wolframalpha.com/input?i=%28a%2Be-%28a%2F%28c%2Bx%29%29%29%5E2%2Fb%3Da%2F%28c%2Bx%29%5E2
    amtB1 = (math.sqrt(a*b*dsq + 2*a*b*d*e + a*b*esq) +a*d + a*e - c*dsq - 2*c*d*e - c*esq)/(dsq+ 2*d*e +esq)
    amtB2 = (-math.sqrt(a*b*dsq + 2*a*b*d*e + a*b*esq)+a*d + a*e - c*dsq - 2*c*d*e - c*esq)/(dsq+ 2*d*e +esq)
    print("amtB", amtB1, amtB2)
    amtB = max(amtB1, amtB2)
    assert amtB > 0

    amtA = pool_swap(c, e, amtB)
    print("amtA", amtA)
    if tokenTrade == 'A':
        tx_arb1 = create_swap(attacker, -amtB, 0)
        tx_arb2 = create_swap(attacker, amtA, 0)
    else:
        tx_arb1 = create_swap(attacker, amtB, 0)
        tx_arb2 = create_swap(attacker, -amtA, 0)

    if chain1.price(tokenTrade) > chain2.price(tokenTrade):
        return tx_arb1, tx_arb2
    else:
        return tx_arb2, tx_arb1

def optimal_arbitrage_search(chain1, chain2, prefs, attacker):
    min_b = .01
    max_b = chain1.accounts[attacker][1]+chain2.accounts[attacker][1]
    
    if prefs[0] < prefs[1]: #todo fractional preferences?
        tokenTrade = 'A'
    else:
        tokenTrade = 'B'

    switch_chains = chain1.price(tokenTrade) < chain2.price(tokenTrade)
    if switch_chains:
        chainA = chain2
        chainB = chain1
    else:
        chainA = chain1
        chainB = chain2

    if tokenTrade == 'A':
        chainB.transfer(attacker, chainA.accounts[attacker][0], chainA, 'B')
    else:
        chainB.transfer(attacker, chainA.accounts[attacker][1], chainA, 'A')

    c1 = copy.deepcopy(chainA) 
    c2 = copy.deepcopy(chainB)
        
    tx_arb1 = create_swap(attacker, 0,0)
    tx_arb2 = create_swap(attacker, 0,0)

    while not np.isclose(c1.price(tokenTrade), c2.price(tokenTrade)):
        c1 = copy.deepcopy(chainA) 
        c2 = copy.deepcopy(chainB)


        mid_b = (max_b+min_b)/2 
        if tokenTrade == 'A':
            tx_arb1 = create_swap(attacker, -mid_b, 0)
            amtA = pool_swap(c2.poolB, c2.poolA, mid_b)
            tx_arb2 = create_swap(attacker, amtA, 0)
        else:
            tx_arb1 = create_swap(attacker, mid_b, 0)
            amtA = pool_swap(c2.poolA, c2.poolB, mid_b)
            tx_arb2 = create_swap(attacker, -amtA, 0)           
        c1.apply(tx_arb1, debug=False)
        c2.apply(tx_arb2, debug=False)
        if c1.price(tokenTrade) > c2.price(tokenTrade):
            min_b = mid_b
        else:
            max_b = mid_b
        # print("amtB", mid_b, "c1", c1.price(tokenTrade), "c2", c2.price(tokenTrade), np.isclose(c1.price(tokenTrade), c2.price(tokenTrade)))
    if switch_chains:
        return tx_arb2, tx_arb1
    else:
        return tx_arb1, tx_arb2

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

def scenario_test():
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

def scenario_high_resource():
    # compare utility over time with/without sandwich attacks
    preferences = {"alice":[1.01,0.99],  # Alice would prefer to buy tokenA
                    "bob":[1.0, 1.0]}      # Bob just wants to skim any and all tokens 
    pool_A = 10000
    pool_B = 10000
    accounts = {"alice":[100,100],
                "bob":[1000000,1000000]}
    chain = Chain(pool_A, pool_B, accounts)
    chain_sand = Chain(pool_A, pool_B, accounts)

    num_iters = 10
    driftAlice = np.exp(np.random.normal(loc=0., scale=.01, size=num_iters))
    bob_norm = accounts['bob'][0]/accounts['alice'][0]

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
            print("Non-sandwich Chain")

            chain.apply(txAlice)

            print("sandwich Chain")
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
    normalized_utils_bob_sand = list(map(lambda x: x/bob_norm, utils_bob_sand))

    print("utils_alice", utils_alice)
    print("utils_alice_sand", utils_alice_sand)
    print("utils_bob_sand", normalized_utils_bob_sand)

    print("diffs_alice", diffs_alice)
    print("diffs_alice_sand", diffs_alice_sand)
    print("diffs_bob_sand", diffs_bob_sand)

    plt.figure(0)
    plt.clf()
    plt.plot(xs,diffs_alice,xs,diffs_alice_sand, diffs_bob_sand)
    plt.xlabel('iter')
    plt.ylabel('difference in net utility')
    plt.title(f'(high resource attacker) difference in net utility after trading')
    plt.legend(['alice','alice_sandwich', 'bob'])

    plt.figure(1)
    plt.clf()
    plt.plot(xs,utils_alice,xs,utils_alice_sand, xs, normalized_utils_bob_sand)
    plt.xlabel('iter')
    plt.ylabel('net utility')
    plt.title(f'(high resource attacker) net utility after trading')
    plt.legend(['alice','alice_sandwich', 'bob'])

    plt.show()

def scenario_low_resource():
    # compare utility over time with/without sandwich attacks
    preferences = {"alice":[1.01,0.99],  # Alice would prefer to buy tokenA
                    "bob":[1.0, 1.0]}      # Bob just wants to skim any and all tokens 
    pool_A = 10000
    pool_B = 10000
    accounts = {"alice":[100,100],
                "bob":[50,50]}
    chain = Chain(pool_A, pool_B, accounts)
    chain_sand = Chain(pool_A, pool_B, accounts)
    bob_norm = accounts['bob'][0]/accounts['alice'][0]

    num_iters = 100
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
            print("Non-sandwich Chain")

            chain.apply(txAlice)

            print("sandwich Chain")
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

    normalized_utils_bob_sand = list(map(lambda x: x/bob_norm, utils_bob_sand))
    print("utils_alice", utils_alice)
    print("utils_alice_sand", utils_alice_sand)
    print("utils_bob_sand", normalized_utils_bob_sand)

    print("diffs_alice", diffs_alice)
    print("diffs_alice_sand", diffs_alice_sand)
    print("diffs_bob_sand", diffs_bob_sand)

    plt.figure(0)
    plt.clf()
    plt.plot(xs,diffs_alice,xs,diffs_alice_sand, diffs_bob_sand)
    plt.xlabel('iter')
    plt.ylabel('difference in net utility')
    plt.title(f'(low resource attacker) difference in net utility after trading')
    plt.legend(['alice','alice_sandwich', 'bob'])

    plt.figure(1)
    plt.clf()
    plt.plot(xs,utils_alice,xs,utils_alice_sand, xs, normalized_utils_bob_sand)
    plt.xlabel('iter')
    plt.ylabel('net utility')
    plt.title(f'(low resource attacker) net utility after trading')
    plt.legend(['alice','alice_sandwich', 'bob'])

    plt.show()

def scenario_same_pref():
    # compare utility over time with/without sandwich attacks for 2 users
    preferences = {"alice":[1.1,0.9],  # Alice would prefer to buy tokenA
                    "carol":[1.1, 0.9],  # Carol also prefers to buy tokenA
                    "bob":[1.0, 1.0]}      # Bob just wants to skim any and all tokens 
    pool_A = 10000
    pool_B = 10000
    accounts = {"alice":[100,100],
                "bob":[10000,10000],
                "carol": [100,100]}
    chain = Chain(pool_A, pool_B, accounts)
    chain_sand = Chain(pool_A, pool_B, accounts)

    bob_norm = accounts['bob'][0]/accounts['alice'][0]
    num_iters = 100
    driftAlice = np.exp(np.random.normal(loc=0., scale=.01, size=num_iters))
    driftCarol = driftAlice#np.exp(np.random.normal(loc=0., scale=.01, size=num_iters))

    utils_alice = []
    utils_alice_sand = []
    utils_carol = []
    utils_carol_sand = []
    utils_bob_sand = []

    diffs_alice = []
    diffs_alice_sand = []
    diffs_carol = []
    diffs_carol_sand = []
    diffs_bob_sand = []
    xs = []
    for i in range(num_iters):
        preferences['alice'][0] = preferences['alice'][0] * driftAlice[i]
        preferences['alice'][1] = 2.0 - preferences['alice'][0]
        preferences['carol'][0] = preferences['carol'][0] * driftCarol[i]
        preferences['carol'][1] = 2.0 - preferences['carol'][0]

        txAlice = make_trade(chain, 'alice', preferences['alice'], chain.accounts['alice'])
        txAlice_sand = make_trade(chain_sand, 'alice', preferences['alice'], chain_sand.accounts['alice'])
        txCarol = make_trade(chain, 'carol', preferences['carol'], chain.accounts['carol'])
        txCarol_sand = make_trade(chain_sand, 'carol', preferences['carol'], chain_sand.accounts['carol'])
        txFront, txBack = produce_sandwich(chain_sand, [txAlice_sand, txCarol_sand], 'bob')

        util_a_old = utility(preferences['alice'], chain.accounts['alice'])
        util_a_old_sand = utility(preferences['alice'], chain_sand.accounts['alice'])
        util_b_old_sand = utility(preferences['bob'], chain_sand.accounts['bob'])
        util_c_old = utility(preferences['carol'], chain.accounts['carol'])
        util_c_old_sand = utility(preferences['carol'], chain_sand.accounts['carol'])
        try:
            print("Non-sandwich Chain")

            chain.apply(txAlice)
            chain.apply(txCarol)

            print("sandwich Chain")
            print("chain_sand before apply", chain_sand.poolA, chain_sand.poolB, chain_sand.accounts)

            chain_sand.apply(txFront)
            chain_sand.apply(txAlice_sand)
            chain_sand.apply(txCarol_sand)
            chain_sand.apply(txBack)
            print("chain_sand after apply", chain_sand.poolA, chain_sand.poolB, chain_sand.accounts)

        except ExecutionException as e: 
            print(e)
            raise e

        util_a_new = utility(preferences['alice'], chain.accounts['alice'])
        util_a_new_sand = utility(preferences['alice'], chain_sand.accounts['alice'])
        util_b_new_sand = utility(preferences['bob'], chain_sand.accounts['bob'])
        util_c_new = utility(preferences['carol'], chain.accounts['carol'])
        util_c_new_sand = utility(preferences['carol'], chain_sand.accounts['carol'])

        utils_alice.append(util_a_new)
        utils_alice_sand.append(util_a_new_sand)
        utils_bob_sand.append(util_b_new_sand)
        utils_carol.append(util_c_new)
        utils_carol_sand.append(util_c_new_sand)

        diffs_alice.append(util_a_new-util_a_old)
        diffs_alice_sand.append(util_a_new_sand-util_a_old_sand)
        diffs_bob_sand.append(util_b_new_sand-util_b_old_sand)
        diffs_carol.append(util_c_new-util_c_old)
        diffs_carol_sand.append(util_c_new_sand-util_c_old_sand)
        xs.append(i)
        print(i, "diff_util_alice", util_a_new-util_a_old, "diff_util_alice_sand", util_a_new_sand-util_a_old_sand, "diff_util_carol", util_c_new-util_c_old, "diff_util_carol_sand", util_c_new_sand-util_c_old_sand, "diff_util_bob", util_b_new_sand-util_b_old_sand)
        print(i,"driftAlice", driftAlice[i],"driftCarol", driftCarol[i], "prefs_alice", preferences['alice'], "txAlice", txAlice_sand["qty"], "txCarol", txCarol_sand["qty"], "txFront", txFront['qty'], "txBack", txBack['qty'])

    normalized_utils_bob_sand = list(map(lambda x: x/bob_norm, utils_bob_sand))
    print("utils_alice", utils_alice)
    print("utils_alice_sand", utils_alice_sand)
    print("utils_bob_sand", normalized_utils_bob_sand)
    print("utils_carol", utils_carol)
    print("utils_carol_sand", utils_carol_sand)

    print("diffs_alice", diffs_alice)
    print("diffs_alice_sand", diffs_alice_sand)
    print("diffs_bob_sand", diffs_bob_sand)
    print("diffs_carol", diffs_carol)
    print("diffs_carol_sand", diffs_carol_sand)


    plt.figure(0)
    plt.clf()
    plt.plot(xs,diffs_alice,label = 'alice')
    plt.plot(xs,diffs_alice_sand, label = 'alice_sandwich')
    plt.plot(xs, diffs_bob_sand, label = 'bob')
    plt.plot(xs, diffs_carol, label = 'carol')
    plt.plot(xs, diffs_carol_sand, label = 'carol_sandwich')
    plt.xlabel('iter')
    plt.ylabel('difference in net utility')
    plt.title(f'(same user preference) difference in net utility after trading')
    plt.legend(['alice','alice_sandwich', 'bob', 'carol', 'carol_sandwich'])

    plt.figure(1)
    plt.clf()
    plt.plot(xs,utils_alice,xs,utils_alice_sand, xs, normalized_utils_bob_sand, xs, utils_carol, xs, utils_carol_sand)
    plt.xlabel('iter')
    plt.ylabel('net utility')
    plt.title(f'(same user preference) net utility after trading')
    plt.legend(['alice','alice_sandwich', 'bob', 'carol', 'carol_sandwich'])

    plt.show()

def scenario_opposite_pref_sandwich_first():
    # compare utility over time with/without sandwich attacks for 2 users
    preferences = {"alice":[1.1,0.9],  # Alice would prefer to buy tokenA
                    "carol":[0.9, 1.1],  # Carol prefers to buy tokenB
                    "bob":[1.0, 1.0]}      # Bob just wants to skim any and all tokens 
    pool_A = 10000
    pool_B = 10000
    accounts = {"alice":[100,100],
                "bob":[10000,10000],
                "carol": [100,100]}
    chain = Chain(pool_A, pool_B, accounts)
    chain_sand = Chain(pool_A, pool_B, accounts)
    bob_norm = accounts['bob'][0]/accounts['alice'][0]

    num_iters = 10
    driftAlice = np.exp(np.random.normal(loc=0., scale=.01, size=num_iters))
    driftCarol = driftAlice#np.exp(np.random.normal(loc=0., scale=.01, size=num_iters))

    utils_alice = []
    utils_alice_sand = []
    utils_carol = []
    utils_carol_sand = []
    utils_bob_sand = []

    diffs_alice = []
    diffs_alice_sand = []
    diffs_carol = []
    diffs_carol_sand = []
    diffs_bob_sand = []
    xs = []
    for i in range(num_iters):
        preferences['alice'][0] = preferences['alice'][0] * driftAlice[i]
        preferences['alice'][1] = 2.0 - preferences['alice'][0]
        preferences['carol'][1] = preferences['carol'][1] * driftCarol[i]
        preferences['carol'][0] = 2.0 - preferences['carol'][1]

        txAlice = make_trade(chain, 'alice', preferences['alice'], chain.accounts['alice'])
        txAlice_sand = make_trade(chain_sand, 'alice', preferences['alice'], chain_sand.accounts['alice'])
        txCarol = make_trade(chain, 'carol', preferences['carol'], chain.accounts['carol'])
        txCarol_sand = make_trade(chain_sand, 'carol', preferences['carol'], chain_sand.accounts['carol'])
        txFront, txBack = produce_sandwich(chain_sand, [txAlice_sand], 'bob')

        util_a_old = utility(preferences['alice'], chain.accounts['alice'])
        util_a_old_sand = utility(preferences['alice'], chain_sand.accounts['alice'])
        util_b_old_sand = utility(preferences['bob'], chain_sand.accounts['bob'])
        util_c_old = utility(preferences['carol'], chain.accounts['carol'])
        util_c_old_sand = utility(preferences['carol'], chain_sand.accounts['carol'])
        try:
            print("Non-sandwich Chain")

            chain.apply(txAlice)
            chain.apply(txCarol)

            print("sandwich Chain")
            print("chain_sand before apply", chain_sand.poolA, chain_sand.poolB, chain_sand.accounts)

            chain_sand.apply(txFront)
            try:
                chain_sand.apply(txAlice_sand)
            except ExecutionException as e:
                print(e)
            try:
                chain_sand.apply(txCarol_sand)
            except ExecutionException as e:
                print(e)

            chain_sand.apply(txBack)
            print("chain_sand after apply", chain_sand.poolA, chain_sand.poolB, chain_sand.accounts)

        except ExecutionException as e: 
            print(e)
            raise e

        util_a_new = utility(preferences['alice'], chain.accounts['alice'])
        util_a_new_sand = utility(preferences['alice'], chain_sand.accounts['alice'])
        util_b_new_sand = utility(preferences['bob'], chain_sand.accounts['bob'])
        util_c_new = utility(preferences['carol'], chain.accounts['carol'])
        util_c_new_sand = utility(preferences['carol'], chain_sand.accounts['carol'])

        utils_alice.append(util_a_new)
        utils_alice_sand.append(util_a_new_sand)
        utils_bob_sand.append(util_b_new_sand/bob_norm)
        utils_carol.append(util_c_new)
        utils_carol_sand.append(util_c_new_sand)

        diffs_alice.append(util_a_new-util_a_old)
        diffs_alice_sand.append(util_a_new_sand-util_a_old_sand)
        diffs_bob_sand.append(util_b_new_sand-util_b_old_sand)
        diffs_carol.append(util_c_new-util_c_old)
        diffs_carol_sand.append(util_c_new_sand-util_c_old_sand)
        xs.append(i)
        print(i, "diff_util_alice", util_a_new-util_a_old, "diff_util_alice_sand", util_a_new_sand-util_a_old_sand, "diff_util_carol", util_c_new-util_c_old, "diff_util_carol_sand", util_c_new_sand-util_c_old_sand, "diff_util_bob", util_b_new_sand-util_b_old_sand)
        print(i,"driftAlice", driftAlice[i],"driftCarol", driftCarol[i], "prefs_alice", preferences['alice'], "txAlice", txAlice_sand["qty"], "txCarol", txCarol_sand["qty"], "txFront", txFront['qty'], "txBack", txBack['qty'])

    normalized_utils_bob_sand = list(map(lambda x: x/bob_norm, utils_bob_sand))
    print("utils_alice", max(utils_alice), min(utils_alice))#, utils_alice)
    print("utils_alice_sand", max(utils_alice_sand), min(utils_alice_sand))#, utils_alice_sand)
    print("utils_bob_sand", max(normalized_utils_bob_sand), min(normalized_utils_bob_sand))#, normalized_utils_bob_sand)
    print("utils_carol", max(utils_carol), min(utils_carol))#, utils_carol)
    print("utils_carol_sand", max(utils_carol_sand), min(utils_carol_sand))#, utils_carol_sand)

    print("diffs_alice", max(diffs_alice), min(diffs_alice), diffs_alice)
    print("diffs_alice_sand", max(diffs_alice_sand), min(diffs_alice_sand), diffs_alice_sand)
    print("diffs_bob_sand", max(diffs_bob_sand), min(diffs_bob_sand), diffs_bob_sand)
    print("diffs_carol", max(diffs_carol), min(diffs_carol), diffs_carol)
    print("diffs_carol_sand", max(diffs_carol_sand), min(diffs_carol_sand), diffs_carol_sand)


    plt.figure(0)
    plt.clf()
    plt.plot(xs,diffs_alice,label = 'alice')
    plt.plot(xs,diffs_alice_sand, label = 'alice_sandwich')
    plt.plot(xs, diffs_bob_sand, label = 'bob')
    plt.plot(xs, diffs_carol, label = 'carol')
    plt.plot(xs, diffs_carol_sand, label = 'carol_sandwich')
    plt.xlabel('iter')
    plt.ylabel('difference in net utility')
    plt.title(f'(opposite user preference, sandwich first)\ndifference in net utility after trading')
    plt.legend(['alice','alice_sandwich', 'bob', 'carol', 'carol_sandwich'])

    plt.figure(1)
    plt.clf()
    plt.plot(xs,utils_alice,xs,utils_alice_sand, xs, normalized_utils_bob_sand, xs, utils_carol, xs, utils_carol_sand)
    plt.xlabel('iter')
    plt.ylabel('net utility')
    plt.title(f'(opposite user preference, sandwich first)\nnet utility after trading')
    plt.legend(['alice','alice_sandwich', 'bob', 'carol', 'carol_sandwich'])

    plt.show()

def scenario_opposite_pref_sandwich_second():
    # compare utility over time with/without sandwich attacks for 2 users
    preferences = {"alice":[1.1,0.9],  # Alice would prefer to buy tokenA
                    "carol":[0.9, 1.1],  # Carol prefers to buy tokenB
                    "bob":[1.0, 1.0]}      # Bob just wants to skim any and all tokens 
    pool_A = 10000
    pool_B = 10000
    accounts = {"alice":[100,100],
                "bob":[10000,10000],
                "carol": [100,100]}
    chain = Chain(pool_A, pool_B, accounts)
    chain_sand = Chain(pool_A, pool_B, accounts)
    bob_norm = accounts['bob'][0]/accounts['alice'][0]

    num_iters = 10
    driftAlice = np.exp(np.random.normal(loc=0., scale=.01, size=num_iters))
    driftCarol = driftAlice#np.exp(np.random.normal(loc=0., scale=.01, size=num_iters))

    utils_alice = []
    utils_alice_sand = []
    utils_carol = []
    utils_carol_sand = []
    utils_bob_sand = []

    diffs_alice = []
    diffs_alice_sand = []
    diffs_carol = []
    diffs_carol_sand = []
    diffs_bob_sand = []
    xs = []
    for i in range(num_iters):
        preferences['alice'][0] = preferences['alice'][0] * driftAlice[i]
        preferences['alice'][1] = 2.0 - preferences['alice'][0]
        preferences['carol'][1] = preferences['carol'][1] * driftCarol[i]
        preferences['carol'][0] = 2.0 - preferences['carol'][1]

        txAlice = make_trade(chain, 'alice', preferences['alice'], chain.accounts['alice'])
        txAlice_sand = make_trade(chain_sand, 'alice', preferences['alice'], chain_sand.accounts['alice'])
        txCarol = make_trade(chain, 'carol', preferences['carol'], chain.accounts['carol'])
        txCarol_sand = make_trade(chain_sand, 'carol', preferences['carol'], chain_sand.accounts['carol'])
        txFront, txBack = produce_sandwich(chain_sand, [txCarol_sand], 'bob')

        util_a_old = utility(preferences['alice'], chain.accounts['alice'])
        util_a_old_sand = utility(preferences['alice'], chain_sand.accounts['alice'])
        util_b_old_sand = utility(preferences['bob'], chain_sand.accounts['bob'])
        util_c_old = utility(preferences['carol'], chain.accounts['carol'])
        util_c_old_sand = utility(preferences['carol'], chain_sand.accounts['carol'])
        try:
            print("Non-sandwich Chain")

            chain.apply(txAlice)
            chain.apply(txCarol)

            print("sandwich Chain")
            print("chain_sand before apply", chain_sand.poolA, chain_sand.poolB, chain_sand.accounts)

            chain_sand.apply(txFront)
            try:
                chain_sand.apply(txAlice_sand)
            except ExecutionException as e:
                print(e)
            try:
                chain_sand.apply(txCarol_sand)
            except ExecutionException as e:
                print(e)

            chain_sand.apply(txBack)
            print("chain_sand after apply", chain_sand.poolA, chain_sand.poolB, chain_sand.accounts)

        except ExecutionException as e: 
            print(e)
            raise e

        util_a_new = utility(preferences['alice'], chain.accounts['alice'])
        util_a_new_sand = utility(preferences['alice'], chain_sand.accounts['alice'])
        util_b_new_sand = utility(preferences['bob'], chain_sand.accounts['bob'])
        util_c_new = utility(preferences['carol'], chain.accounts['carol'])
        util_c_new_sand = utility(preferences['carol'], chain_sand.accounts['carol'])

        utils_alice.append(util_a_new)
        utils_alice_sand.append(util_a_new_sand)
        utils_bob_sand.append(util_b_new_sand/bob_norm)
        utils_carol.append(util_c_new)
        utils_carol_sand.append(util_c_new_sand)

        diffs_alice.append(util_a_new-util_a_old)
        diffs_alice_sand.append(util_a_new_sand-util_a_old_sand)
        diffs_bob_sand.append(util_b_new_sand-util_b_old_sand)
        diffs_carol.append(util_c_new-util_c_old)
        diffs_carol_sand.append(util_c_new_sand-util_c_old_sand)
        xs.append(i)
        print(i, "diff_util_alice", util_a_new-util_a_old, "diff_util_alice_sand", util_a_new_sand-util_a_old_sand, "diff_util_carol", util_c_new-util_c_old, "diff_util_carol_sand", util_c_new_sand-util_c_old_sand, "diff_util_bob", util_b_new_sand-util_b_old_sand)
        print(i,"driftAlice", driftAlice[i],"driftCarol", driftCarol[i], "prefs_alice", preferences['alice'], "txAlice", txAlice_sand["qty"], "txCarol", txCarol_sand["qty"], "txFront", txFront['qty'], "txBack", txBack['qty'])

    normalized_utils_bob_sand = list(map(lambda x: x/bob_norm, utils_bob_sand))
    print("utils_alice", max(utils_alice), min(utils_alice))#, utils_alice)
    print("utils_alice_sand", max(utils_alice_sand), min(utils_alice_sand))#, utils_alice_sand)
    print("utils_bob_sand", max(normalized_utils_bob_sand), min(normalized_utils_bob_sand))#, normalized_utils_bob_sand)
    print("utils_carol", max(utils_carol), min(utils_carol))#, utils_carol)
    print("utils_carol_sand", max(utils_carol_sand), min(utils_carol_sand))#, utils_carol_sand)

    print("diffs_alice", max(diffs_alice), min(diffs_alice), diffs_alice)
    print("diffs_alice_sand", max(diffs_alice_sand), min(diffs_alice_sand), diffs_alice_sand)
    print("diffs_bob_sand", max(diffs_bob_sand), min(diffs_bob_sand), diffs_bob_sand)
    print("diffs_carol", max(diffs_carol), min(diffs_carol), diffs_carol)
    print("diffs_carol_sand", max(diffs_carol_sand), min(diffs_carol_sand), diffs_carol_sand)


    plt.figure(0)
    plt.clf()
    plt.plot(xs,diffs_alice,label = 'alice')
    plt.plot(xs,diffs_alice_sand, label = 'alice_sandwich')
    plt.plot(xs, diffs_bob_sand, label = 'bob')
    plt.plot(xs, diffs_carol, label = 'carol')
    plt.plot(xs, diffs_carol_sand, label = 'carol_sandwich')
    plt.xlabel('iter')
    plt.ylabel('difference in net utility')
    plt.title(f'(opposite user preference, sandwich second)\ndifference in net utility after trading')
    plt.legend(['alice','alice_sandwich', 'bob', 'carol', 'carol_sandwich'])

    plt.figure(1)
    plt.clf()
    plt.plot(xs,utils_alice,xs,utils_alice_sand, xs, normalized_utils_bob_sand, xs, utils_carol, xs, utils_carol_sand)
    plt.xlabel('iter')
    plt.ylabel('net utility')
    plt.title(f'(opposite user preference, sandwich second)\nnet utility after trading')
    plt.legend(['alice','alice_sandwich', 'bob', 'carol', 'carol_sandwich'])

    plt.show()

def scenario_same_pref_diff_amounts():
    chain = Chain(10000,10000,{"alice":[100,100], "bob":[100,100], "carol":[10000,10000]})
    txAlice = create_swap('alice', 10, -9)
    txBob_same = create_swap('bob', 10, -9)
    txBob_close = create_swap('bob', 10, -8)
    txBob_far = create_swap('bob', 10, -1)
    sandwhich_txs = {}
    sandwhich_txs["close_both"] = produce_sandwich(chain, [txAlice, txBob_close], "carol")
    sandwhich_txs["close_both"]+= (txAlice, txBob_close)

    sandwhich_txs["far_both"] = produce_sandwich(chain, [txAlice, txBob_far], "carol")
    sandwhich_txs["far_both"]+=  (txAlice, txBob_far)

    sandwhich_txs["close_higher"] = produce_sandwich(chain, [txBob_close], "carol")
    sandwhich_txs["close_higher"]+= (txAlice, txBob_close)

    sandwhich_txs["far_higher"] = produce_sandwich(chain, [txBob_far], "carol")
    sandwhich_txs["far_higher"]+=  (txAlice, txBob_far)

    sandwhich_txs["same_both"] = produce_sandwich(chain, [txAlice, txBob_same], "carol")
    sandwhich_txs["same_both"]+= (txAlice, txBob_same)

    sandwhich_txs["same_higher"] = produce_sandwich(chain, [txBob_same], "carol")
    sandwhich_txs["same_higher"]+=  (txAlice, txBob_same)


    carol_utility = {}

    for k in sandwhich_txs.keys():
        chain = Chain(10000,10000,{"alice":[100,100], "bob":[100,100], "carol":[10000,10000]})
        carol_utility[k] = {}
        carol_utility[k]["util_c_old"] = utility([1.0,1.0], chain.accounts['carol'])
        chain.apply(sandwhich_txs[k][0])
        try:
            chain.apply(sandwhich_txs[k][2])
        except ExecutionException as e:
            print(e)
        try:
            chain.apply(sandwhich_txs[k][3])
        except ExecutionException as e:
            print(e)
        
        chain.apply(sandwhich_txs[k][1])

        carol_utility[k]["util_c_new"] = utility([1.0,1.0], chain.accounts['carol'])
        carol_utility[k]["util_c_diff"] = carol_utility[k]["util_c_new"] - carol_utility[k]["util_c_old"]

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(carol_utility)

def scenario_different_pref():
    chain = Chain(10000,10000,{"alice":[100,100], "bob":[100,100], "carol":[10000,10000]})
    txAlice = create_swap('alice', 10, -9)
    txBob_close = create_swap('bob', -10, 8)
    txBob_far = create_swap('bob', -10, 1)
    sandwhich_txs = {}
    sandwhich_txs["close_both"] = produce_sandwich(chain, [txAlice, txBob_close], "carol")
    sandwhich_txs["close_both"]+= (txAlice, txBob_close)

    sandwhich_txs["far_both"] = produce_sandwich(chain, [txAlice, txBob_far], "carol")
    sandwhich_txs["far_both"]+=  (txAlice, txBob_far)

    sandwhich_txs["close_higher"] = produce_sandwich(chain, [txBob_close], "carol")
    sandwhich_txs["close_higher"]+= (txAlice, txBob_close)

    sandwhich_txs["far_higher"] = produce_sandwich(chain, [txBob_far], "carol")
    sandwhich_txs["far_higher"]+=  (txAlice, txBob_far)

    carol_utility = {}

    for k in sandwhich_txs.keys():
        chain = Chain(10000,10000,{"alice":[100,100], "bob":[100,100], "carol":[10000,10000]})
        carol_utility[k] = {}
        carol_utility[k]["util_c_old"] = utility([1.0,1.0], chain.accounts['carol'])
        chain.apply(sandwhich_txs[k][0])
        try:
            chain.apply(sandwhich_txs[k][2])
        except ExecutionException as e:
            print(e)
        try:
            chain.apply(sandwhich_txs[k][3])
        except ExecutionException as e:
            print(e)
        
        chain.apply(sandwhich_txs[k][1])

        carol_utility[k]["util_c_new"] = utility([1.0,1.0], chain.accounts['carol'])
        carol_utility[k]["util_c_diff"] = carol_utility[k]["util_c_new"] - carol_utility[k]["util_c_old"]

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(carol_utility)

def scenario_arbitrage_test_alg():
    print("-----------------------------scenario_arbitrage_test_alg-----------------------------")
    c1 = Chain(100,100,{"a":[10000,10000], "b":[1000000,1000000], "c":[10000,10000]})
    c2 = Chain(200,200,{"a":[10000,10000], "b":[1000000,1000000], "c":[10000,10000]})
    tx = create_swap('c', 99, 0)
    c1.apply(tx, debug=False)

    print("---test 1---")
    prefB_1 = [0.0, 2.0]
    c1_1 = copy.deepcopy(c1)
    c2_1 = copy.deepcopy(c2)
    tx_arb1_1, tx_arb2_1 = optimal_arbitrage_algebra(c1_1, c2_1, prefB_1, 'b')
    c1_1.apply(tx_arb1_1)       
    c2_1.apply(tx_arb2_1)
    print("price c1", c1_1.price('A'), "c2", c2_1.price('A'))
    assert np.isclose(c1_1.price('A'), c2_1.price('A'))

    print("\n---test 2---")

    prefB_3 = [0.0, 2.0]
    c1_3 = copy.deepcopy(c2)
    c2_3 = copy.deepcopy(c1)
    tx_arb1_3, tx_arb2_3 = optimal_arbitrage_algebra(c1_3, c2_3, prefB_3, 'b')
    c1_3.apply(tx_arb1_3)       
    c2_3.apply(tx_arb2_3)
    print("price c1", c1_3.price('A'), "c2", c2_3.price('A'))
    assert np.isclose(c1_3.price('A'), c2_3.price('A'))


    print("\n---test 3---")
    prefB_2 = [2.0, 0.0]
    c1_2 = copy.deepcopy(c1)
    c2_2 = copy.deepcopy(c2)
    tx_arb1_2, tx_arb2_2 = optimal_arbitrage_algeb(c1_2, c2_2, prefB_2, 'b')
    c1_2.apply(tx_arb1_2)
    c2_2.apply(tx_arb2_2)       
    print("price c1", c1_2.price('A'), "c2", c2_2.price('A'))
    assert np.isclose(c1_2.price('A'), c2_2.price('A'))

    print("\n---test 4---")
    prefB_4 = [2.0, 0.0]
    c1_4 = copy.deepcopy(c2)
    c2_4 = copy.deepcopy(c1)
    tx_arb1_4, tx_arb2_4 = optimal_arbitrage_algeb(c1_4, c2_4, prefB_4, 'b')
    c1_4.apply(tx_arb1_4)       
    c2_4.apply(tx_arb2_4)
    print("price c1", c1_4.price('A'), "c2", c2_4.price('A'))
    assert np.isclose(c1_4.price('A'), c2_4.price('A'))

def scenario_arbitrage_test_search():
    print("-----------------------------scenario_arbitrage_test_search-----------------------------")
    c1 = Chain(100,100,{"a":[10000,10000], "b":[1000000,1000000]})
    c2 = Chain(200,200,{"a":[10000,10000], "b":[1000000,1000000]})
    tx = create_swap('a', 99, 0)
    c1.apply(tx, debug=False)

    print("---test 1---")
    prefB_1 = [0.0, 2.0]
    c1_1 = copy.deepcopy(c1)
    c2_1 = copy.deepcopy(c2)
    tx_arb1_1, tx_arb2_1 = optimal_arbitrage_search(c1_1, c2_1, prefB_1, 'b')
    c1_1.apply(tx_arb1_1)       
    c2_1.apply(tx_arb2_1)
    print("price c1", c1_1.price('A'), "c2", c2_1.price('A'))
    assert np.isclose(c1_1.price('A'), c2_1.price('A'))

    print("\n---test 2---")

    prefB_3 = [0.0, 2.0]
    c1_3 = copy.deepcopy(c2)
    c2_3 = copy.deepcopy(c1)
    tx_arb1_3, tx_arb2_3 = optimal_arbitrage_search(c1_3, c2_3, prefB_3, 'b')
    c1_3.apply(tx_arb1_3)       
    c2_3.apply(tx_arb2_3)
    print("price c1", c1_3.price('A'), "c2", c2_3.price('A'))
    assert np.isclose(c1_3.price('A'), c2_3.price('A'))


    print("\n---test 3---")
    prefB_2 = [2.0, 0.0]
    c1_2 = copy.deepcopy(c1)
    c2_2 = copy.deepcopy(c2)
    tx_arb1_2, tx_arb2_2 = optimal_arbitrage_search(c1_2, c2_2, prefB_2, 'b')
    c1_2.apply(tx_arb1_2)
    c2_2.apply(tx_arb2_2)       
    print("price c1", c1_2.price('A'), "c2", c2_2.price('A'))
    assert np.isclose(c1_2.price('A'), c2_2.price('A'))

    print("\n---test 4---")
    prefB_4 = [2.0, 0.0]
    c1_4 = copy.deepcopy(c2)
    c2_4 = copy.deepcopy(c1)
    tx_arb1_4, tx_arb2_4 = optimal_arbitrage_search(c1_4, c2_4, prefB_4, 'b')
    c1_4.apply(tx_arb1_4)       
    c2_4.apply(tx_arb2_4)
    print("price c1", c1_4.price('A'), "c2", c2_4.price('A'))
    assert np.isclose(c1_4.price('A'), c2_4.price('A'))

if __name__ == "__main__":
    # scenario_test()
    # scenario_high_resource()
    # scenario_low_resource()
    # scenario_same_pref()
    # scenario_opposite_pref_sandwich_first()
    # scenario_opposite_pref_sandwich_second()
    # scenario_same_pref_diff_amounts()
    # scenario_different_pref()
    scenario_arbitrage_test_alg()
    scenario_arbitrage_test_search()
