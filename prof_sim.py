import math
import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import defaultdict 
"""
Toy model of an on-chain ecosystem with two tokens and one DEX.
Transactions on the DEX create an opportunity for sandwich attacks.
"""
class ExecutionException(Exception):
    pass
def require(cond, msg=None):
    if not cond: raise ExecutionException(msg)

figure_num = 0

class Account():
    def __init__(self, username, tokens):
        self.username = username
        self.tokens = tokens
    def __str__(self):
        return f"{self.username}: {self.tokens}"
    def transfer(self, qty):
        if qty > 0:
            self.tokens[0] += qty 
        else:
            self.tokens[1] -= qty 

class Chain():
    def __init__(self, poolA=1000., poolB=1000., accounts=None, chainid=""):
        self.poolA = poolA
        self.poolB = poolB
        self.chainid = chainid
        if accounts is None:
            self.accounts = {"alice": Account("alice", [100.,100.]),
                             "bob":   Account("bob", [100.,100.])}
        else:
            self.accounts = accounts

    def apply(self, tx, debug=True, accounts=None):
        # Apply the transaction, updating the pool price
        if (tx['type'] == 'swap'):
            if tx['qty'] >= 0:
                # Sell qty of tokenA, buy at least rsv of tokenB
                amtA = tx['qty']
                amtB = pool_swap(self.poolA, self.poolB, amtA)
                if accounts is None:
                    account = self.accounts[tx["sndr"]].tokens
                else:
                    account = accounts[tx["sndr"]].tokens
                require(account[0] >= amtA, f'not enough balance for trade {tx} {amtA} account {account}')
                require(amtB >= 0)
                require(amtB >= -tx['rsv'] or tx['rsv'] == 0, f"slippage exceeded amtB:{amtB}, -rsv:{-tx['rsv']}")
                require(self.poolB - amtB >= 0, 'exhausts pool')
                if debug:
                    print(self.chainid,"\t",tx['sndr'], "sell", amtA, "of A and gets", amtB, "of B with slippage", amtB ,">=",  -tx['rsv'])
                amtB = -amtB
            else:
                # Sell qty of tokenB, buy at least rsv of tokenA
                amtB = -tx['qty']
                amtA = pool_swap(self.poolB, self.poolA, amtB)
                if accounts is None:
                    account = self.accounts[tx["sndr"]].tokens
                else:
                    account = accounts[tx["sndr"]].tokens
                require(account[1] >= amtB, f'not enough balance for trade {tx} {amtB} account {account}')
                require(amtA >= 0)
                require(amtA >= tx['rsv'] or tx['rsv'] == 0, f"slippage exceeded amtA:{amtA}, rsv:{tx['rsv']}")
                require(self.poolA + amtA >= 0, 'exhausts pool')
                if debug:
                    print(self.chainid,"\t",tx['sndr'], "sell", amtB, "of B and gets", amtA, "of A with slippage", amtA ,">=",  tx['rsv'])
                amtA = -amtA

            self.poolA += amtA
            self.poolB += amtB
            account[0] -= amtA
            account[1] -= amtB
        else:
            raise ValueError("unknown tx type")

    def __str__(self):
        return f"PoolA: {self.poolA} PoolB: {self.poolB} accounts: {[str(self.accounts[acc]) for acc in self.accounts]}"

    def price(self, token):
        if token == 'A' or token == 'a':
            return self.poolA/self.poolB
        else:
            return self.poolB/self.poolA
    
    def product(self):
        return self.poolA*self.poolB

def pool_swap(poolA, poolB, amtA):
    """Uniswap v2 rule trade A for B

    Args:
        poolA (number): amount of token A
        poolB (number): amount of token B
        amtA (number): amount of A to trade

    Returns:
        number: amount of B returned
    """
    # trade A for B
    amtB = poolB - poolA*poolB / (poolA + amtA)
    assert np.isclose(poolA*poolB, (poolA + amtA)*(poolB - amtB))
    return amtB

def create_swap(sndr,qty,rsv):
    """create a swap transaction

    Args:
        sndr (string): username of sender
        qty (number): amount to trade (Positive to trade token A, Negative to trade token B)
        rsv (number): slippage limit (minimum of other token to get) (Positive to get token A, Negative to get token B)

    Returns:
        dict: transaction
    """
    if qty < 0:
        assert rsv >= 0
    elif qty > 0:
        assert rsv <= 0
    return dict(type="swap",sndr=sndr,qty=qty,rsv=rsv,auth="auth")

def utility(pref, portfolio):
    """calculate net utility

    Args:
        pref (list of 2 numbers): preference for each token
        portfolio (list of two numbers): amount owned of each token

    Returns:
        number: net utility
    """
    assert len(pref) == len(portfolio) == 2
    return pref[0] * portfolio[0] + pref[1] * portfolio[1]

def logistic_function(x, k, max_y, mid_y):
    """logistic curve function

    Args:
        x (number): x value
        k (number): growth rate of function
        max_y (_type_): max y value for x>=0
        mid_y (_type_): mid point of function for x==0

    Returns:
        number: evaluation of function at x
    """
    
    y = max_y*(1/(1+np.exp(-x*k))-.5+(mid_y/max_y))
    return y

def produce_sandwich(chain, tx_victims, attacker, debug=False):
    """figure out the optimal frontrun/backrun transactions for
    a sandwhich attack

    Args:
        chain (Chain): chain the sandwhich is on
        tx_victims (list of dict or dict): victim transaction(s) to sandwhich
        attacker (string): attacker name
        debug (bool, optional): print debug information. Defaults to False.

    Returns:
        tuple of dict swap transactions: frontrun swap transaction, backrun swap transaction
    """


    if not isinstance(tx_victims, list):
        tx_victims = [tx_victims]
    for tx_victim in tx_victims:
        assert tx_victim["type"] == "swap"

    #if the transaction amounts sum up to 0 there is no sandwhich attack
    tx_sum = sum(list(map(lambda x: x["qty"], tx_victims)))
    if tx_sum == 0:
        return create_swap(attacker,0,0), create_swap(attacker,0,0)

    min_front = 0

    if tx_sum < 0:
        #victim transactions are net trading token B for token A so frontrun
        #transaction will trade token B
        max_front = chain.accounts[attacker].tokens[1]
    else:
        max_front = chain.accounts[attacker].tokens[0]

    last_successful_front = 0
    # chain = copy.deepcopy(chain)
    while True:
        chain_copy = copy.deepcopy(chain)
        frontrun_amt = (min_front + max_front) / 2. 
        if tx_sum < 0:
            frontrun_amt = -frontrun_amt #trading token B for A so qty parameter is negative
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

        if np.isclose(max_front, min_front):
            if debug:
                print("found", last_successful_front)
            break
        

    if tx_sum < 0:
        backrun_amt = pool_swap(chain.poolB, chain.poolA, -last_successful_front)
    else:
        #backrun is trading token B for A so qty parameter is negative
        backrun_amt = -pool_swap(chain.poolA, chain.poolB, last_successful_front)

    return create_swap(attacker, last_successful_front, 0), create_swap(attacker, backrun_amt, 0)

 # optimal amount of b to trade for a to get my preference

def optimal_trade_amt(poolA, poolB, prefA, prefB, portfA, portfB):
    """figure out the optimal amount of token B to trade to maximize increase in net utility 
    using algebra.

    Args:
        poolA (number): amount of token A in pool
        poolB (number): amount of token B in pool
        prefA (number): preference for token A
        prefB (number): preference for token B
        portfA (number): amount of token A in owned
        portfB (number): amount of token B in owned

    Returns:
        number: amount of token B to trade
    """
    assert prefA + prefB == 2
    a=prefA 
    asq = prefA**2
    c=poolA 
    d=poolB
    e=portfA
    f=portfB
    amtB1 = (math.sqrt(2*a*c*d - asq*c*d)-a*d+2*d)/(a-2)
    amtB2 = (-math.sqrt(2*a*c*d - asq*c*d)-a*d+2*d)/(a-2)
    res = max(amtB1, amtB2)
    # print("optimal", amtB1, amtB2)
    assert res >= 0
    return res

def optimal_trade_amt_search(poolA, poolB, prefA, prefB, portfA, portfB):
    """figure out the optimal amount of token B to trade to maximize increase in net utility 
    using bisection search.

    Args:
        poolA (number): amount of token A in pool
        poolB (number): amount of token B in pool
        prefA (number): preference for token A
        prefB (number): preference for token B
        portfA (number): amount of token A in owned
        portfB (number): amount of token B in owned

    Returns:
        number: amount of token B to trade
    """
    min_b=0.0
    max_b=portfB
    util_init = utility([prefA, prefB],[portfA, portfB])
    util_old = util_init
    util_max = util_init
    while True:
        mid_b = (max_b+min_b)/2
        amtA = pool_swap(poolB, poolA, mid_b)
        util = utility([prefA, prefB],[portfA+amtA, portfB-mid_b])
        # print("amtB", mid_b, "util", util)
        if util > util_max:
            min_b = mid_b
            util_max = util
        else:
            max_b = mid_b
        if np.isclose(util, util_old):
            # print("best is", mid_b, "resulting in util", util, "util_old", util_old, "util_max", util_max, "util_init", util_init)
            if util >= util_init:
                return mid_b
            else:
                return 0
        util_old = util

def optimal_arbitrage_algebra(chain1, chain2, prefs, attacker):
    """figure out the optimal arbitrage transaction between two token pools
    using algebra.

    Args:
        chain1 (Chain): 1st chain with token pool
        chain2 (Chain): 2nd chain with other token pool
        prefs (list of numbers): [preference for token A, preference for token B]
        attacker (string): arbitrager name

    Returns:
        tuple of dict (swap txs): (transaction for chain1 and transaction for chain2)
    """
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

    # print("chain1", chain1.price(tokenTrade), "chain2", chain2.price(tokenTrade), "trading token" , tokenTrade)
    #how much of tokenTrade we can sell to c1 and buy from c2 until they have the same price
    a=c1.product()
    b=c2.product()
    if tokenTrade == 'B':
        c=c1.poolB
        d=c1.poolA
        g=c2.poolB 
    else:
        c=c1.poolA
        d=c1.poolB 
        g=c2.poolA 
    # print("a", a ,'b', b,'c', c,'d', d, 'g', g)
    csq = c**2
    gsq = g**2

    #https://www.wolframalpha.com/input?i=%28c-x%29%2F%28a%2F%28c-x%29%29+%3D+%28g%2Bx%29%2F%28b%2F%28g%2Bx%29%29
    amtB1 = (-math.sqrt(a*b*csq + 2*a*b*c*g + a*b*gsq)+a*c+a*g+csq*(-d)-2*c*d*g-d*gsq)/(csq+2*c*g+gsq)
    amtB2 = ( math.sqrt(a*b*csq + 2*a*b*c*g + a*b*gsq)+a*c+a*g+csq*(-d)-2*c*d*g-d*gsq)/(csq+2*c*g+gsq)

    amtB = max(amtB1, amtB2)
    amtA = pool_swap(d,c,amtB)

    assert amtA > 0

    assert amtB > 0

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
    """figure out the optimal arbitrage transaction between two token pools
    using bisection search. 
    TODO remove not possible to calcu;ate with search

    Args:
        chain1 (Chain): 1st chain with token pool
        chain2 (Chain): 2nd chain with other token pool
        prefs (list of numbers): [preference for token A, preference for token B]
        attacker (string): arbitrager name

    Returns:
        tuple of dict (swap txs): (transaction for chain1 and transaction for chain2)
    """
    min_b = 0
    
    if prefs[0] < prefs[1]: #todo fractional preferences?
        tokenTrade = 'A'
        max_b = chain1.accounts[attacker].tokens[1]
    else:
        tokenTrade = 'B'
        max_b = chain1.accounts[attacker].tokens[0]

    switch_chains = chain1.price(tokenTrade) < chain2.price(tokenTrade)
    if switch_chains:
        chainA = chain2
        chainB = chain1
    else:
        chainA = chain1
        chainB = chain2

    

    c1 = copy.deepcopy(chainA) 
    c2 = copy.deepcopy(chainB)

    tx_arb1 = create_swap(attacker, 0,0)
    tx_arb2 = create_swap(attacker, 0,0)
    last_successful_tx1 = tx_arb1
    last_successful_tx2 = tx_arb2


    while not np.isclose(c1.price(tokenTrade), c2.price(tokenTrade)):
        c1 = copy.deepcopy(chainA) 
        c2 = copy.deepcopy(chainB)

        mid_b = (max_b+min_b)/2 
        if tokenTrade == 'A':
            tx_arb1 = create_swap(attacker, -mid_b, 0)
            amtA = pool_swap(c1.poolB, c1.poolA, mid_b)
            tx_arb2 = create_swap(attacker, amtA, 0)
        else:
            tx_arb1 = create_swap(attacker, mid_b, 0)
            amtA = pool_swap(c1.poolA, c1.poolB, mid_b)
            tx_arb2 = create_swap(attacker, -amtA, 0)
        try:    
            c1.apply(tx_arb1, debug=False)
            c2.apply(tx_arb2, debug=False)
            last_successful_tx1 = tx_arb1
            last_successful_tx2 = tx_arb2
        except ExecutionException as e:
            print(e)
            pass
        if c1.price(tokenTrade) > c2.price(tokenTrade):
            if min_b == mid_b: #nothing better
                break
            min_b = mid_b
        else:
            if max_b == mid_b: #nothing better
                break
            max_b = mid_b
        # print("amtB", mid_b, "c1", c1.price(tokenTrade), "c2", c2.price(tokenTrade), tx_arb1['qty'], tx_arb2['qty'])

    if switch_chains:
        return last_successful_tx2, last_successful_tx1
    else:
        return last_successful_tx1, last_successful_tx2

def make_trade(chain, sndr, prefs, optimal=False, static_value=1., percent_optimal=1., accounts=None):
    """generate a swap transaction based on the user's preference
    and the top of block pool price on the chain

    Args:
        chain (Chain): chain the swap will occur on
        sndr (string): sender name
        prefs (list of numbers): preferences of sndr for the token. (must add up to 2.0)
        optimal (bool, optional): Use the optimal trade to increase net utility. Defaults to False.
        static_value (number, optional): Max amount to trade in the optimal direction. Defaults to 1..
        percent_optimal (number, optional): Percentage of the optimal trade. Defaults to 1..
        accounts (dict of str -> Account, optional): accounts associated with users

    Returns:
        dict: swap transaction
    """
    assert percent_optimal <= 1.0 and percent_optimal >= 0

    #token A: apples, token B: dollars
    if prefs[1] == 0:
        my_price = np.inf
    else:
        my_price =  prefs[0] / prefs[1] #utils/apple / utils/dollar -> dollars/apple
    pool_price = chain.poolB/chain.poolA #dollars/apple 
    if accounts is None:
        accounts = chain.accounts

    # print("pool_price", pool_price, "my_price", my_price)
    if pool_price > my_price: #trade A for B
        # print("trade A for B")
        if optimal == False:
            qty = min(static_value, optimal_trade_amt(chain.poolB, chain.poolA, prefs[1], prefs[0], accounts[sndr].tokens[1], accounts[sndr].tokens[0]))
        else:
            optimal_val = optimal_trade_amt(chain.poolB, chain.poolA, prefs[1], prefs[0], accounts[sndr].tokens[1], accounts[sndr].tokens[0])
            # print("optimal_val", optimal_val, "*", percent_optimal, "=", optimal_val*percent_optimal )
            optimal_val *= percent_optimal
            qty = optimal_val
        qty = min(qty, accounts[sndr].tokens[0])
        slip = -prefs[0]* qty / prefs[1] # slippage s.t. new net utility will equal old net utility
        if optimal:
            print(sndr, f"\t{percent_optimal*100}% optimal tx:", "qty", qty, "slip", slip, "for prefs", prefs, "tokens", accounts[sndr].tokens, "pool", [chain.poolA, chain.poolB])
        else:
            print(sndr, f"\ttx:", "qty", qty, "slip", slip, "for prefs", prefs, "tokens", accounts[sndr].tokens, "pool", [chain.poolA, chain.poolB])

        assert abs(slip) <= abs(pool_swap(chain.poolA, chain.poolB, qty))
    elif my_price > pool_price: #trade B for A
        # print("trade B for A")
        if optimal == False:
            qty = min(static_value, optimal_trade_amt(chain.poolA, chain.poolB, prefs[0], prefs[1], accounts[sndr].tokens[0], accounts[sndr].tokens[1]))
        else:
            optimal_val = optimal_trade_amt(chain.poolA, chain.poolB, prefs[0], prefs[1], accounts[sndr].tokens[0], accounts[sndr].tokens[1])
            # print("optimal_val", optimal_val, "*", percent_optimal, "=", optimal_val*percent_optimal )
            optimal_val *= percent_optimal
            qty = optimal_val
        qty = -min(qty, accounts[sndr].tokens[1])
        slip = -prefs[1]* qty / prefs[0] # slippage s.t. new net utility will equal old net utility

        if optimal:
            print(sndr, f"\t{percent_optimal*100}% optimal tx:", "qty", qty, "slip", slip, "for prefs", prefs, "tokens", accounts[sndr].tokens, "pool", [chain.poolA, chain.poolB])
        else:
            print(sndr, f"\ttx:", "qty", qty, "slip", slip, "for prefs", prefs, "tokens", accounts[sndr].tokens, "pool", [chain.poolA, chain.poolB])

        assert abs(slip) <= abs(pool_swap(chain.poolB, chain.poolA, abs(qty)))
    else:
        qty = 0
        slip = 0
    return create_swap(sndr, qty, slip)

def scenario_test_optimal_trade():
    preferences = {"alice":[1.1, 0.9]}
    pool_A = 1001
    pool_B = 1000
    accounts = {"alice":Account('alice',[201,200])}
    chain1 = Chain(pool_A, pool_B, accounts)
    chain2 = copy.deepcopy(chain1)
    my_price =  preferences['alice'][0] / preferences['alice'][1] #utils/apple / utils/dollar -> dollars/apple
    pool_price = chain1.poolB/chain1.poolA #dollars/apple

    if pool_price > my_price: #trade A for B
        print('trade A for B')
        amtb1 = optimal_trade_amt_search(pool_B, pool_A, preferences['alice'][1], preferences['alice'][0], accounts['alice'].tokens[1], accounts['alice'].tokens[0])
        amtb2 = optimal_trade_amt(pool_B, pool_A, preferences['alice'][1], preferences['alice'][0], accounts['alice'].tokens[1], accounts['alice'].tokens[0])
    else:
        print('trade B for A')
        amtb1 = -optimal_trade_amt_search(pool_A, pool_B, preferences['alice'][0], preferences['alice'][1], accounts['alice'].tokens[0], accounts['alice'].tokens[1])
        amtb2 = -optimal_trade_amt(pool_A, pool_B, preferences['alice'][0], preferences['alice'][1], accounts['alice'].tokens[0], accounts['alice'].tokens[1])

    print("amtb", amtb1, amtb2)
    # assert amtb1 == amtb2
    utility_old = utility(preferences['alice'], accounts['alice'].tokens)

    tx1 = create_swap('alice', amtb1, 0)
    tx2= create_swap('alice', amtb2, 0)

    chain1.apply(tx1, debug=True)
    chain2.apply(tx2, debug=True)

    utility_new1 = utility(preferences['alice'], chain1.accounts['alice'].tokens)
    utility_new2 = utility(preferences['alice'], chain2.accounts['alice'].tokens)

    print("pool_price", pool_B/pool_A)
    print("my_price", preferences['alice'][0]/preferences['alice'][1])
    print("utility old", utility_old)
    # print("utility A", utility_newA)
    print("utility new", utility_new1, utility_new2)
    assert utility_new1 > utility_old
    assert utility_new2 > utility_old
    assert np.isclose(utility_new1, utility_new2)

def scenario_high_resource():
    # compare utility over time with/without sandwich attacks
    preferences = {"alice":[1.01,0.99],  # Alice would prefer to buy tokenA
                    "bob":[1.0, 1.0]}      # Bob just wants to skim any and all tokens 
    pool_A = 10000
    pool_B = 10000
    accounts = {"alice":Account('alice',[100,100]),
                "bob":Account('bob',[1000000,1000000])}
    chain = Chain(pool_A, pool_B, accounts)
    chain_sand = copy.deepcopy(chain)

    num_iters = 10
    driftAlice = np.exp(np.random.normal(loc=0., scale=.01, size=num_iters))
    bob_norm = accounts['bob'].tokens[0]/accounts['alice'].tokens[0]

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

        txAlice = make_trade(chain, 'alice', preferences['alice'])
        txAlice_sand = make_trade(chain_sand, 'alice', preferences['alice'])
        txFront, txBack = produce_sandwich(chain_sand, txAlice_sand, 'bob')

        util_a_old = utility(preferences['alice'], chain.accounts['alice'].tokens)
        util_a_old_sand = utility(preferences['alice'], chain.accounts['alice'].tokens)
        util_b_old_sand = utility(preferences['bob'], chain_sand.accounts['bob'].tokens)
        try:
            print("Non-sandwich Chain")

            chain.apply(txAlice)

            print("sandwich Chain")
            print("chain_sand before apply", chain_sand)

            chain_sand.apply(txFront)
            chain_sand.apply(txAlice_sand)
            chain_sand.apply(txBack)
            print("chain_sand after apply", chain_sand)

        except ExecutionException as e: 
            print(e)
            raise e

        util_a_new = utility(preferences['alice'], chain.accounts['alice'].tokens)
        util_a_new_sand = utility(preferences['alice'], chain_sand.accounts['alice'].tokens)
        util_b_new_sand = utility(preferences['bob'], chain_sand.accounts['bob'].tokens)

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
    accounts = {"alice":Account('alice',[100,100]),
                "bob":Account('bob',[50,50])}
    chain = Chain(pool_A, pool_B, accounts)
    chain_sand = copy.deepcopy(chain)
    bob_norm = accounts['bob'].tokens[0]/accounts['alice'].tokens[0]

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

        txAlice = make_trade(chain, 'alice', preferences['alice'])
        txAlice_sand = make_trade(chain_sand, 'alice', preferences['alice'])
        txFront, txBack = produce_sandwich(chain_sand, txAlice_sand, 'bob')

        util_a_old = utility(preferences['alice'], chain.accounts['alice'].tokens)
        util_a_old_sand = utility(preferences['alice'], chain_sand.accounts['alice'].tokens)
        util_b_old_sand = utility(preferences['bob'], chain_sand.accounts['bob'].tokens)
        try:
            print("Non-sandwich Chain")

            chain.apply(txAlice)

            print("sandwich Chain")
            print("chain_sand before apply", chain_sand)

            chain_sand.apply(txFront)
            chain_sand.apply(txAlice_sand)
            chain_sand.apply(txBack)
            print("chain_sand after apply", chain_sand)

        except ExecutionException as e: 
            print(e)
            raise e

        util_a_new = utility(preferences['alice'], chain.accounts['alice'].tokens)
        util_a_new_sand = utility(preferences['alice'], chain_sand.accounts['alice'].tokens)
        util_b_new_sand = utility(preferences['bob'], chain_sand.accounts['bob'].tokens)

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
    accounts = {"alice":Account('alice',[100,100]),
                "bob":Account('bob',[10000,10000]),
                "carol":Account('carol', [100,100])}
    chain = Chain(pool_A, pool_B, accounts)
    chain_sand = copy.deepcopy(chain)

    bob_norm = accounts['bob'].tokens[0]/accounts['alice'].tokens[0]
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

        txAlice = make_trade(chain, 'alice', preferences['alice'])
        txAlice_sand = make_trade(chain_sand, 'alice', preferences['alice'])
        txCarol = make_trade(chain, 'carol', preferences['carol'])
        txCarol_sand = make_trade(chain_sand, 'carol', preferences['carol'])
        txFront, txBack = produce_sandwich(chain_sand, [txAlice_sand, txCarol_sand], 'bob')

        util_a_old = utility(preferences['alice'], chain.accounts['alice'].tokens)
        util_a_old_sand = utility(preferences['alice'], chain_sand.accounts['alice'].tokens)
        util_b_old_sand = utility(preferences['bob'], chain_sand.accounts['bob'].tokens)
        util_c_old = utility(preferences['carol'], chain.accounts['carol'].tokens)
        util_c_old_sand = utility(preferences['carol'], chain_sand.accounts['carol'].tokens)
        try:
            print("Non-sandwich Chain")

            chain.apply(txAlice)
            chain.apply(txCarol)

            print("sandwich Chain")
            print("chain_sand before apply", chain_sand)

            chain_sand.apply(txFront)
            chain_sand.apply(txAlice_sand)
            chain_sand.apply(txCarol_sand)
            chain_sand.apply(txBack)
            print("chain_sand after apply", chain_sand)

        except ExecutionException as e: 
            print(e)
            raise e

        util_a_new = utility(preferences['alice'], chain.accounts['alice'].tokens)
        util_a_new_sand = utility(preferences['alice'], chain_sand.accounts['alice'].tokens)
        util_b_new_sand = utility(preferences['bob'], chain_sand.accounts['bob'].tokens)
        util_c_new = utility(preferences['carol'], chain.accounts['carol'].tokens)
        util_c_new_sand = utility(preferences['carol'], chain_sand.accounts['carol'].tokens)

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
    plt.plot(xs, diffs_alice,label = 'alice')
    plt.plot(xs, diffs_alice_sand, label = 'alice_sandwich')
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
    accounts = {"alice":Account('alice',[100,100]),
                "bob":Account('bob',[10000,10000]),
                "carol":Account('carol', [100,100])}
    chain = Chain(pool_A, pool_B, accounts)
    chain_sand = copy.deepcopy(chain)

    bob_norm = accounts['bob'].tokens[0]/accounts['alice'].tokens[0]

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

        util_a_old = utility(preferences['alice'], chain.accounts['alice'].tokens)
        util_a_old_sand = utility(preferences['alice'], chain_sand.accounts['alice'].tokens)
        util_b_old_sand = utility(preferences['bob'], chain_sand.accounts['bob'].tokens)
        util_c_old = utility(preferences['carol'], chain.accounts['carol'].tokens)
        util_c_old_sand = utility(preferences['carol'], chain_sand.accounts['carol'].tokens)
        try:
            print("Non-sandwich Chain")

            chain.apply(txAlice)
            chain.apply(txCarol)

            print("sandwich Chain")
            print("chain_sand before apply", chain_sand)

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
            print("chain_sand after apply", chain_sand)

        except ExecutionException as e: 
            print(e)
            raise e

        util_a_new = utility(preferences['alice'], chain.accounts['alice'].tokens)
        util_a_new_sand = utility(preferences['alice'], chain_sand.accounts['alice'].tokens)
        util_b_new_sand = utility(preferences['bob'], chain_sand.accounts['bob'].tokens)
        util_c_new = utility(preferences['carol'], chain.accounts['carol'].tokens)
        util_c_new_sand = utility(preferences['carol'], chain_sand.accounts['carol'].tokens)

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
    accounts = {"alice":Account('alice',[100,100]),
                "bob":Account('bob',[10000,10000]),
                "carol":Account('carol', [100,100])}
    chain = Chain(pool_A, pool_B, accounts)
    chain_sand = copy.deepcopy(chain)

    bob_norm = accounts['bob'].tokens[0]/accounts['alice'].tokens[0]

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

        util_a_old = utility(preferences['alice'], chain.accounts['alice'].tokens)
        util_a_old_sand = utility(preferences['alice'], chain_sand.accounts['alice'].tokens)
        util_b_old_sand = utility(preferences['bob'], chain_sand.accounts['bob'].tokens)
        util_c_old = utility(preferences['carol'], chain.accounts['carol'].tokens)
        util_c_old_sand = utility(preferences['carol'], chain_sand.accounts['carol'].tokens)
        try:
            print("Non-sandwich Chain")

            chain.apply(txAlice)
            chain.apply(txCarol)

            print("sandwich Chain")
            print("chain_sand before apply", chain_sand)

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
            print("chain_sand after apply", chain_sand)

        except ExecutionException as e: 
            print(e)
            raise e

        util_a_new = utility(preferences['alice'], chain.accounts['alice'].tokens)
        util_a_new_sand = utility(preferences['alice'], chain_sand.accounts['alice'].tokens)
        util_b_new_sand = utility(preferences['bob'], chain_sand.accounts['bob'].tokens)
        util_c_new = utility(preferences['carol'], chain.accounts['carol'].tokens)
        util_c_new_sand = utility(preferences['carol'], chain_sand.accounts['carol'].tokens)

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
    chain = Chain(10000,10000, {"alice":Account('alice',[100,100]), 
        "bob":Account('bob',[100,100]), 
        "carol":Account('carol',[10000,10000])})
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
        chain = Chain(10000,10000, {"alice":Account('alice',[100,100]), 
        "bob":Account('bob',[100,100]), 
        "carol":Account('carol',[10000,10000])})
        carol_utility[k] = {}
        carol_utility[k]["util_c_old"] = utility([1.0,1.0], chain.accounts['carol'].tokens)
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

        carol_utility[k]["util_c_new"] = utility([1.0,1.0], chain.accounts['carol'].tokens)
        carol_utility[k]["util_c_diff"] = carol_utility[k]["util_c_new"] - carol_utility[k]["util_c_old"]

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(carol_utility)

def scenario_different_pref():
    chain = Chain(10000,10000,
        {"alice":Account('alice',[100,100]), 
        "bob":Account('bob',[100,100]), 
        "carol":Account('carol',[10000,10000])})
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
        chain =  Chain(10000,10000,
        {"alice":Account('alice',[100,100]), 
        "bob":Account('bob',[100,100]), 
        "carol":Account('carol',[10000,10000])})
        carol_utility[k] = {}
        carol_utility[k]["util_c_old"] = utility([1.0,1.0], chain.accounts['carol'].tokens)
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

        carol_utility[k]["util_c_new"] = utility([1.0,1.0], chain.accounts['carol'].tokens)
        carol_utility[k]["util_c_diff"] = carol_utility[k]["util_c_new"] - carol_utility[k]["util_c_old"]

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(carol_utility)

def scenario_arbitrage_test_alg():
    print("-----------------------------scenario_arbitrage_test_alg-----------------------------")
    accounts = {"a":Account('a', [10000,10000]), "b":Account('b',[1000000,1000000]), "c":Account('c', [10000,10000])}
    c1 = Chain(100,100,accounts,"c1")
    c2 = Chain(200,200,accounts,"c2")
    tx = create_swap('c', 99, 0)
    c1.apply(tx, debug=False)

    print("---test 1---")
    prefB_1 = [0.0, 2.0]
    c1_1 = copy.copy(c1)
    c2_1 = copy.copy(c2)
    util_old_1 = utility(prefB_1, c1_1.accounts['b'].tokens)
    tx_arb1_1, tx_arb2_1 = optimal_arbitrage_algebra(c1_1, c2_1, prefB_1, 'b')
    c1_1.apply(tx_arb1_1)       
    c2_1.apply(tx_arb2_1)
    util_new_1 = utility(prefB_1, c1_1.accounts['b'].tokens)
    print("price c1", c1_1.price('A'), "c2", c2_1.price('A'), "util old", util_old_1, "new", util_new_1)
    assert np.isclose(c1_1.price('A'), c2_1.price('A'))


    print("\n---test 2---")

    prefB_3 = [0.0, 2.0]
    c1_3 = copy.copy(c2)
    c2_3 = copy.copy(c1)
    tx_arb1_3, tx_arb2_3 = optimal_arbitrage_algebra(c1_3, c2_3, prefB_3, 'b')
    c1_3.apply(tx_arb1_3)       
    c2_3.apply(tx_arb2_3)
    print("price c1", c1_3.price('A'), "c2", c2_3.price('A'))
    assert np.isclose(c1_3.price('A'), c2_3.price('A'))


    print("\n---test 3---")
    prefB_2 = [2.0, 0.0]
    c1_2 = copy.copy(c1)
    c2_2 = copy.copy(c2)
    tx_arb1_2, tx_arb2_2 = optimal_arbitrage_algebra(c1_2, c2_2, prefB_2, 'b')
    c1_2.apply(tx_arb1_2)
    c2_2.apply(tx_arb2_2)       
    print("price c1", c1_2.price('A'), "c2", c2_2.price('A'))
    assert np.isclose(c1_2.price('A'), c2_2.price('A'))

    print("\n---test 4---")
    prefB_4 = [2.0, 0.0]
    c1_4 = copy.copy(c2)
    c2_4 = copy.copy(c1)
    tx_arb1_4, tx_arb2_4 = optimal_arbitrage_algebra(c1_4, c2_4, prefB_4, 'b')
    c1_4.apply(tx_arb1_4)       
    c2_4.apply(tx_arb2_4)
    print("price c1", c1_4.price('A'), "c2", c2_4.price('A'))
    assert np.isclose(c1_4.price('A'), c2_4.price('A'))

def scenario_arbitrage_test_search():
    print("-----------------------------scenario_arbitrage_test_search-----------------------------")

    accounts = {"a":Account('a', [10000,10000]), "b":Account('b',[1000000,1000000]), "c":Account('c', [10000,10000])}
    c1 = Chain(100,100,accounts,"c1")
    c2 = Chain(200,200,accounts,"c2")
    tx = create_swap('a', 99, 0)
    c1.apply(tx, debug=False)

    print("---test 1---")
    prefB_1 = [0.0, 2.0]
    c1_1 = copy.deepcopy(c1)
    c2_1 = copy.deepcopy(c2)
    util_old_1 = utility(prefB_1, c2_1.accounts['b'].tokens)
    tx_arb1_1, tx_arb2_1  = optimal_arbitrage_search(c1_1, c2_1, prefB_1, 'b')
    c1_1.apply(tx_arb1_1)       
    c2_1.apply(tx_arb2_1)
    util_new_1 = utility(prefB_1, c2_1.accounts['b'].tokens)
    print("price c1", c1_1.price('A'), "c2", c2_1.price('A'), "util old", util_old_1, "new", util_new_1)
    assert np.isclose(c1_1.price('A'), c2_1.price('A'))
    assert util_new_1 >= util_old_1


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

def scenario_arbitrage(random_type='same', num_intervals=10, num_iters=10, debug=False, num_prof_users=2, mevshare_percentage=.9, min_volatility=0.01, max_volatility=5):
    np.random.seed(0)
    num_block_users = 0

    volatilities = [np.exp(np.log(min_volatility + k*max_volatility/num_intervals)) for k in range(0, num_intervals)]
    print("volatilities", volatilities)

    util_old = {}
    util_new = {}
    util_new_kickback = {}
    
    def setup_users(scenario, coorelation_type='same',random_type='rndm',debug=False):
        """setup users accounts and preferences

        Args:
            scenario (string): scenario name
            coorelation_type ('same' or 'diff'): whether all users have the same or half have 'diff'erent preferences
            random_type ('rndm' or 'same'): whether all users have different random preferences (rndm) or the 'same' random preference 

        Returns:
            None
        """
        accounts = {}
        preferences = {}
        if random_type == 'same':
            pref = [[logistic_function(abs(np.random.normal(loc=0., scale=scale)), .5, 2., 0.) for _ in range(num_iters)] for scale in volatilities]
        for i in range(num_prof_users):
            username=f"user_{scenario}_{coorelation_type}_{i}"
            accounts[username] = Account(username, [10000, 10000])
            if random_type == 'rndm': #every user gets a different random preference
                preferences[username] = [[logistic_function(abs(np.random.normal(loc=0., scale=scale)), .5, 2., 0.) for _ in range(num_iters)] for scale in volatilities]
            else: #users have the same random preference
                preferences[username] = [[x for x in l] for l in pref]
            
            if coorelation_type == "diff" and i < num_prof_users/2: #half of users have different/opposite preferences
                preferences[username] = [list(map(lambda x: [x, 2.0-x], preferences[username][j])) for j in range(len(preferences[username]))]
            else: #all users have the same preference direction
                preferences[username] = [list(map(lambda x: [2.0-x, x], preferences[username][j])) for j in range(len(preferences[username]))]

            util_old[username] = [[None for _ in range(num_iters)]for _ in range(num_intervals)]
            util_new[username] = [[None for _ in range(num_iters)]for _ in range(num_intervals)]
            util_new_kickback[username]  = [[None for _ in range(num_iters)]for _ in range(num_intervals)]
        if scenario == "mevs":
            for i in range(num_prof_users): #each user has their own arbitrager
                username=f"arbi_{scenario}_{coorelation_type}_{i}"
                accounts[username] =Account(username, [10000, 10000])
                util_old[username] = [[None for _ in range(num_iters)]for _ in range(num_intervals)]
                util_new[username] = [[None for _ in range(num_iters)]for _ in range(num_intervals)]
        else:
            username=f"arbi_{scenario}_{coorelation_type}" #one arbitrager for all
            accounts[username] = Account(username, [10000, 10000])
            util_old[username] = [[None for _ in range(num_iters)]for _ in range(num_intervals)]
            util_new[username] = [[None for _ in range(num_iters)]for _ in range(num_intervals)]            
        return accounts, preferences

    util_diff_avg_users = defaultdict(lambda : [0 for _ in range(num_intervals)])
    util_diff_avg_users_kickback = defaultdict(lambda : [0 for _ in range(num_intervals)])
    util_diff_avg_users_kickback_diff = defaultdict(lambda : [0 for _ in range(num_intervals)])
    util_diff_individual_users_kickback = defaultdict(lambda : [[0 for _ in range(num_intervals)] for _ in range(num_prof_users)])
    util_diff_avg_attackers = defaultdict(lambda : [0 for _ in range(num_intervals)])

    for interval_num in range(0, num_intervals):
        for scenario in ['prof', 'mevs', 'prof_nokickback']:
            for coorelation_type in ['same', 'diff']:
                print(f"-----------scenario: {scenario} | user preferences: {coorelation_type} | volatitiy: {volatilities[interval_num]}-----------")
                for iter_num in range(num_iters):
                    txs = {}
                    kickbacks = {}
                    accounts, preferences = setup_users(scenario, coorelation_type, random_type)
                    chain1 = Chain(poolA=10000., poolB=10000., accounts=accounts, chainid=f"chain1")
                    chain2 = Chain(poolA=10000., poolB=10000., accounts=accounts, chainid=f"chain2")
                    tokens = {}

                    #figure out what transactions each user will make
                    for i in range(num_prof_users): 
                        username = f"user_{scenario}_{coorelation_type}_{i}"
                        pref = preferences[username][interval_num][iter_num]
                        util_old[username][interval_num][iter_num] = utility(pref, accounts[username].tokens)
                        #users only use chain1 
                        txs[username] = make_trade(chain1, username, pref, optimal=True, percent_optimal=.2, accounts=accounts)


                    #execute transactions
                    for i in range(num_prof_users): 
                        username = f"user_{scenario}_{coorelation_type}_{i}"
                        pref = preferences[username][interval_num][iter_num]
                        try:
                            chain1.apply(txs[username])
                        except ExecutionException as e:
                            print(username, e)
                            pass
                        if scenario == 'mevs':
                            attacker_username=f"arbi_{scenario}_{coorelation_type}_{i}"
                            tokens[attacker_username] = copy.copy(accounts[attacker_username].tokens)
                            if abs(chain1.price('A')- chain2.price('A')) < abs(chain1.price('B')- chain2.price('B')):
                                tx1, tx2 = optimal_arbitrage_algebra(chain1, chain2, [0.0, 2.0], attacker_username)
                            else:
                                tx1, tx2 = optimal_arbitrage_algebra(chain1, chain2, [2.0, 0.0], attacker_username)
                            chain1.apply(tx1)
                            chain2.apply(tx2)
                    if 'prof' in scenario:
                        attacker_username=f"arbi_{scenario}_{coorelation_type}"
                        tokens[attacker_username] = copy.copy(accounts[attacker_username].tokens)
                        if abs(chain1.price('A')- chain2.price('A')) < abs(chain1.price('B')- chain2.price('B')):
                            tx1, tx2 = optimal_arbitrage_algebra(chain1, chain2, [0.0, 2.0], attacker_username)
                        else:
                            tx1, tx2 = optimal_arbitrage_algebra(chain1, chain2, [2.0, 0.0], attacker_username)
                        chain1.apply(tx1)
                        chain2.apply(tx2)


                    #calculate utility for arbitrager


                    if scenario == 'mevs':
                        for i in range(num_prof_users):
                            username=f"arbi_{scenario}_{coorelation_type}_{i}"
                            util_new[username][interval_num][iter_num] = utility([1.0,1.0], accounts[username].tokens)
                            util_diff_avg_attackers[f'{scenario}_{coorelation_type}'][interval_num] += (util_new[username][interval_num][iter_num] - 20000.)
                            print("utility", username, 20000., "->", util_new[username][interval_num][iter_num])
                    else:
                        username=f"arbi_{scenario}_{coorelation_type}"
                        util_new[username][interval_num][iter_num] = utility([1.0,1.0], accounts[username].tokens)
                        util_diff_avg_attackers[f'{scenario}_{coorelation_type}'][interval_num] += (util_new[username][interval_num][iter_num] - 20000.)
                        print("utility", username, 20000., "->", util_new[username][interval_num][iter_num])
                    #calculate utility for user
                    for i in range(num_prof_users): 
                        username = f"user_{scenario}_{coorelation_type}_{i}"
                        util_new[username][interval_num][iter_num] = utility(preferences[username][interval_num][iter_num], accounts[username].tokens)
                        print("utility b4 kickback", username, util_old[username][interval_num][iter_num], "->", util_new[username][interval_num][iter_num])
                        util_diff_avg_users[f'{scenario}_{coorelation_type}'][interval_num] += (util_new[username][interval_num][iter_num] - util_old[username][interval_num][iter_num])


                    #execute kickbacks
                    for i in range(num_prof_users):
                        username = f"user_{scenario}_{coorelation_type}_{i}"
                        if 'prof' == scenario: #not no kickback prof
                            attacker_username=f"arbi_{scenario}_{coorelation_type}"
                            profit = [accounts[attacker_username].tokens[0] - tokens[attacker_username][0], accounts[attacker_username].tokens[1] - tokens[attacker_username][1]]
                            print(f"arbitrage profit {attacker_username} -> {username}: {[profit[0]/num_prof_users, profit[1]/num_prof_users]}")
                            accounts[username].tokens[0] += profit[0]/num_prof_users 
                            accounts[username].tokens[1] += profit[1]/num_prof_users
                        elif scenario == 'mevs':
                            attacker_username=f"arbi_{scenario}_{coorelation_type}_{i}"
                            profit = [accounts[attacker_username].tokens[0] - tokens[attacker_username][0], accounts[attacker_username].tokens[1] - tokens[attacker_username][1]]
                            print(f"arbitrage profit {attacker_username} -> {username}: {profit} * {mevshare_percentage} = {[profit[0]*mevshare_percentage, profit[1]*mevshare_percentage]}")
                            accounts[username].tokens[0] += profit[0]*mevshare_percentage
                            accounts[username].tokens[1] += profit[1]*mevshare_percentage

                    #calculate utility with kickbacks
                    for i in range(num_prof_users): 
                        username = f"user_{scenario}_{coorelation_type}_{i}"
                        util_new_kickback[username][interval_num][iter_num] = utility(preferences[username][interval_num][iter_num], accounts[username].tokens)
                        print("utility w/ kickback", username, util_old[username][interval_num][iter_num], "->", util_new_kickback[username][interval_num][iter_num])
                        util_diff_avg_users_kickback[f'{scenario}_{coorelation_type}'][interval_num] += (util_new_kickback[username][interval_num][iter_num] - util_old[username][interval_num][iter_num])
                        util_diff_avg_users_kickback_diff[f'{scenario}_{coorelation_type}'][interval_num] += (util_new_kickback[username][interval_num][iter_num] - util_new[username][interval_num][iter_num])
                        util_diff_individual_users_kickback[f'{scenario}_{coorelation_type}'][i][interval_num] += (util_new_kickback[username][interval_num][iter_num] - util_old[username][interval_num][iter_num])

                util_diff_avg_attackers[f'{scenario}_{coorelation_type}'][interval_num] /= num_intervals
                util_diff_avg_users[f'{scenario}_{coorelation_type}'][interval_num] /= (num_prof_users*num_intervals)
                util_diff_avg_users_kickback[f'{scenario}_{coorelation_type}'][interval_num] /= (num_prof_users*num_intervals)
                util_diff_avg_users_kickback_diff[f'{scenario}_{coorelation_type}'][interval_num] /= (num_prof_users*num_intervals)
 
    print("\n\n\n----------------------------------------results----------------------------------------")
    print("volatilities", volatilities)
    for coorelation_type in ['same', 'diff']:
        for scenario in ['prof', 'mevs', 'prof_nokickback']:
            # print("util_diff_avg_users w/o kickback", f'{scenario}_{coorelation_type}', util_diff_avg_users[f'{scenario}_{coorelation_type}'])
            print("util_diff_avg_users", scenario, coorelation_type, util_diff_avg_users_kickback[f'{scenario}_{coorelation_type}'])
            # print("util_diff_avg_users diff w/ & w/o kickback", f'{scenario}_{coorelation_type}', util_diff_avg_users_kickback_diff[f'{scenario}_{coorelation_type}'])
            # print("util_diff_avg_arbitrager", f'{scenario}_{coorelation_type}', util_diff_avg_attackers[f'{scenario}_{coorelation_type}'])

    global figure_num    
    for coorelation_type in ['same', 'diff']:
        plt.figure(figure_num)
        for scenario in ['prof', 'mevs', 'prof_nokickback']:
            plt.plot(volatilities, util_diff_avg_users_kickback[f'{scenario}_{coorelation_type}'], label = f'{scenario}')
        plt.xlabel('volatility (std of preferences)')
        plt.ylabel('difference in net utility')
        plt.title(f'average difference in net utility \nuser preferences \'{coorelation_type}\'\nmevshare percent {mevshare_percentage}')
        # plt.xscale("log")
        plt.legend([f'{s}' for s in ['prof', 'mevshare', 'prof_nokickback']])
        figure_num+=1

    print("\n")
    for coorelation_type in ['same', 'diff']:
            for i in range(num_prof_users):
                for scenario in ['prof', 'mevs', 'prof_nokickback']:
                    print('user', i , scenario, coorelation_type, util_diff_individual_users_kickback[f'{scenario}_{coorelation_type}'][i])

    for coorelation_type in ['same', 'diff']:
            for i in range(num_prof_users):
                plt.figure(figure_num)
                for scenario in ['prof', 'mevs', 'prof_nokickback']:
                    plt.plot(volatilities, util_diff_individual_users_kickback[f'{scenario}_{coorelation_type}'][i], label = f'{scenario}')
                plt.xlabel('volatility (std of preferences)')
                plt.ylabel(f'difference in net utility')
                plt.title(f'difference in net utility, user preferences \'{coorelation_type}\'\nuser {i}, mevshare percent {mevshare_percentage}')
                plt.legend([f'{s}' for s in ['prof', 'mevshare', 'prof_nokickback']])
                figure_num+=1



if __name__ == "__main__":
    # scenario_test_optimal_trade()
    # scenario_high_resource()
    # scenario_low_resource()
    # scenario_same_pref()
    # scenario_opposite_pref_sandwich_first()
    # scenario_opposite_pref_sandwich_second()
    # scenario_same_pref_diff_amounts()
    # scenario_different_pref()
    # scenario_arbitrage_test_search()
    # scenario_arbitrage_test_alg()

    # scenario_arbitrage()
    # scenario_arbitrage(random_type='same', num_intervals=1, num_iters=1, debug= True)
    # scenario_arbitrage(random_type='same', num_intervals=10, num_iters=100, debug= False, num_prof_users=4, mevshare_percentage=.7)
    # scenario_arbitrage(random_type='same', num_intervals=10, num_iters=100, debug= False, num_prof_users=3, mevshare_percentage=.95)
    # scenario_arbitrage(random_type='same', num_intervals=10, num_iters=100, debug= False, num_prof_users=4, mevshare_percentage=.9)
    scenario_arbitrage(random_type='same', num_intervals=3, num_iters=1, debug= False, num_prof_users=2, mevshare_percentage=.9)
    plt.show()
