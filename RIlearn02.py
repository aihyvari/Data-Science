# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 08:54:10 2018

@author: hyvarant
"""

from yahoo_finance import Share 
#YAHOO FINANCE DISCONTINUED TO WORK AS PREVIOUSLY
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override() # <== that's all it takes :-)

####
from matplotlib import pyplot as plt 
import numpy as np 
import random 
import tensorflow as tf 
import random 

class DecisionPolicy: 
    def select_action(self, current_state, step): 
        pass 

    def update_q(self, state, action, reward, next_state): 
        pass 

class RandomDecisionPolicy(DecisionPolicy): 
    def __init__(self, actions): 
        self.actions = actions 

    def select_action(self, current_state, step): 
        action = self.actions[random.randint(0, len(self.actions) - 1)] 
        return action 

class QLearningDecisionPolicy(DecisionPolicy): 
    def __init__(self, actions, input_dim): 
        self.epsilon = 0.9 
        self.gamma = 0.001 
        self.actions = actions 
        output_dim = len(actions) 
        h1_dim = 200 
 
        self.x = tf.placeholder(tf.float32, [None, input_dim]) 
        self.y = tf.placeholder(tf.float32, [output_dim]) 
        W1 = tf.Variable(tf.random_normal([input_dim, h1_dim])) 
        b1 = tf.Variable(tf.constant(0.1, shape=[h1_dim])) 
        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1) 
        W2 = tf.Variable(tf.random_normal([h1_dim, output_dim])) 
        b2 = tf.Variable(tf.constant(0.1, shape=[output_dim])) 
        self.q = tf.nn.relu(tf.matmul(h1, W2) + b2) 

        loss = tf.square(self.y - self.q) 
        self.train_op = tf.train.AdagradOptimizer(0.01).minimize(loss) 
        self.sess = tf.Session() 
        self.sess.run(tf.global_variables_initializer()) 

    def select_action(self, current_state, step): 
        threshold = min(self.epsilon, step / 1000.) 
        if random.random() < threshold: 
          # Exploit best option with probability epsilon 
          action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state}) 
          action_idx = np.argmax(action_q_vals)  # TODO: replace w/ tensorflow's argmax 
          action = self.actions[action_idx] 
        else: 
          # Explore random option with probability 1 - epsilon 
              action = self.actions[random.randint(0, len(self.actions) - 1)] 
        return action 


    def update_q(self, state, action, reward, next_state): 
        action_q_vals = self.sess.run(self.q, feed_dict={self.x: state}) 
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state}) 
        next_action_idx = np.argmax(next_action_q_vals) 
        action_q_vals[0, next_action_idx] = reward + self.gamma * next_action_q_vals[0, next_action_idx] 
        action_q_vals = np.squeeze(np.asarray(action_q_vals)) 
        self.sess.run(self.train_op, feed_dict={self.x: state, self.y: action_q_vals}) 

#################

def run_simulation(policy, initial_budget, initial_num_stocks, prices, hist, debug=False): 
    budget = initial_budget 
    num_stocks = initial_num_stocks 
    share_value = 0 
    transitions = list() 
    for i in range(len(prices) - hist - 1): 
        if i % 100 == 0: 
             print('progress {:.2f}%'.format(float(100*i) / (len(prices) - hist - 1))) 
        current_state = np.asmatrix(np.hstack((prices[i:i+hist,0], budget, num_stocks))) #VITTU KUN KESTI: LAITA NOLLA
        current_portfolio = budget + num_stocks * share_value 
        action = policy.select_action(current_state, i) 
        share_value = float(prices[i + hist + 1]) 
        if action == 'Buy' and budget >= share_value: 
            budget -= share_value + 2  #ADD TRANSACTION
            num_stocks += 1 
        elif action == 'Sell' and num_stocks > 0: 
            budget += share_value      #ADD TRANSACTION
            num_stocks -= 1 
        else: 
             action = 'Hold' 
        new_portfolio = budget + num_stocks * share_value 
        reward = new_portfolio - current_portfolio                                  #REWARD
        next_state = np.asmatrix(np.hstack((prices[i+1:i+hist+1,0], budget, num_stocks))) #PRKLE TOINEN
        transitions.append((current_state, action, reward, next_state)) 
        policy.update_q(current_state, action, reward, next_state) 

    portfolio = budget + num_stocks * share_value 
    if debug: 
        print('${}\t{} shares'.format(budget, num_stocks)) 
    return portfolio 
 
#####################################
 
def run_simulations(policy, budget, num_stocks, prices, hist): 
     num_tries = 4   #kuinka monta simulaatiota, joista ka:t ja sd:t
     final_portfolios = list() 
     for i in range(num_tries): 
         final_portfolio = run_simulation(policy, budget, num_stocks, prices, hist) 
         final_portfolios.append(final_portfolio) 
     avg, std = np.mean(final_portfolios), np.std(final_portfolios) 
     return avg, std 

##########################
#import struct
def get_prices(osake, start_date, end_date, cache_filename='osake_hinta.npy'): 
     try: 
         avaushinnat = np.load('xxx.npy', encoding='bytes') 
     #    avaushinnat.to_records(index=False)
     except IOError: 
         #share = Share(share_symbol) ##VANHA
         # download dataframe
         histData = pdr.get_data_yahoo(osake, start_date, end_date)
         #stock_hist = share.get_historical(start_date, end_date) 
         avaushinnat = histData[['Open']]
         #avaushinnat.to_records(index=False)
         #avaushinnat= avaushinnat.tostring()
         #avaushinnat = [stock_price['Open'] for stock_price in histData] 
         np.save(cache_filename, avaushinnat) 
         avaushinnat = np.load(cache_filename, encoding='bytes')
     return avaushinnat 
#######################

def plot_prices(prices): 
     plt.title('Opening stock prices') 
     plt.xlabel('day') 
     plt.ylabel('price ($)') 
     plt.plot(prices) 
#     plt.savefig('prices.png') 

if __name__ == '__main__': 
     prices = get_prices('BMY', '2010-07-22', '2018-07-12') 
     plot_prices(prices) 
#     
     actions = ['Buy', 'Sell', 'Hold'] 
     hist = 100 
     # policy = RandomDecisionPolicy(actions) 
     policy = QLearningDecisionPolicy(actions, hist + 2) 
     budget = 1000.0 
     num_stocks = 0 
     
     
     avg, std = run_simulations(policy, budget, num_stocks, prices, hist) 
     print(avg, std) 



