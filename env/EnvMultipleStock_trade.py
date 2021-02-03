import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE=1000000
# total number of stocks in our portfolio

# 37 for CAC 40
# STOCK_DIM = 37
# 30 for DJI
STOCK_DIM = 30

# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001

# turbulence index: 90-150 reasonable threshold
#TURBULENCE_THRESHOLD = 140
REWARD_SCALING = 1e-4

class StockEnvTrade(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df,day = 0, turbulence_threshold=140 ,initial=True, previous_state=[], model_name='', iteration='', df_trace=''):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.df = df
        self.initial = initial
        self.previous_state = previous_state
        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,)) 
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30] 
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (181,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold
        # initalize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.adjcp.values.tolist() + \
                      [0]*STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist()
        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        #self.reset()
        self._seed()
        self.model_name=model_name        
        self.iteration=iteration
        self.df_trace = df_trace


    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.turbulence<self.turbulence_threshold:
            if self.state[index+STOCK_DIM+1] > 0:
                #update balance
                self.state[0] += \
                self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
                 (1- TRANSACTION_FEE_PERCENT)
                
                self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])
                self.cost +=self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
                 TRANSACTION_FEE_PERCENT
                self.trades+=1
                # CD Tracing with dataframe
                df_sell_row = pd.DataFrame(data = np.array([[self.data.datadate.values[0], self.data.tic.values[index] , self.iteration , "sell"      ,  "0"  , self.state[index+1] ,  min(abs(action), self.state[index+STOCK_DIM+1]) , self.cost    ]]), columns= ['datadate'                  , 'tic'                , 'iter_num'     , 'action_performed' , 'available_amount' , 'stock_value'       ,  'nb_stock_traded'             , 'trade_cost'  ]) 
                self.df_trace.insert_row_to_stock_daily_trading_step(df_sell_row) 
            else:
                # if turbulence goes over threshold, just stop buying
                # CD Tracing with dataframe         

                df_step_row = pd.DataFrame(data = np.array([[self.data.datadate.values[0], self.data.tic.values[index] , self.iteration , "step_sell" ,  "0"  , self.state[index+1] ,  "0"                                              , self.cost    ]]), columns= ['datadate'                  , 'tic'                , 'iter_num'     , 'action_performed' , 'available_amount' , 'stock_value'       ,  'nb_stock_traded'             , 'trade_cost'  ]) 
                self.df_trace.insert_row_to_stock_daily_trading_step(df_step_row) 
                pass
        else:
            # if turbulence goes over threshold, just clear out all positions 
            if self.state[index+STOCK_DIM+1] > 0:
                #update balance
                self.state[0] += self.state[index+1]*self.state[index+STOCK_DIM+1]* \
                              (1- TRANSACTION_FEE_PERCENT)
                self.state[index+STOCK_DIM+1] =0
                self.cost += self.state[index+1]*self.state[index+STOCK_DIM+1]* \
                              TRANSACTION_FEE_PERCENT
                self.trades+=1

                df_sell_row = pd.DataFrame(data = np.array([[self.data.datadate.values[0], self.data.tic.values[index] , self.iteration , "sell_all"       ,  "0"  , self.state[index+1] ,  min(abs(action), self.state[index+STOCK_DIM+1]) , self.cost    ]]), columns= ['datadate'                  , 'tic'                , 'iter_num'     , 'action_performed' , 'available_amount' , 'stock_value'       ,  'nb_stock_traded'             , 'trade_cost'  ]) 
                self.df_trace.insert_row_to_stock_daily_trading_step(df_sell_row) 
            else:
                # if turbulence goes over threshold, just stop buying
                # CD Tracing with dataframe

                df_step_row = pd.DataFrame(data = np.array([[self.data.datadate.values[0], self.data.tic.values[index] , self.iteration , "stop_all_turb" ,  "0"  , self.state[index+1] ,  "0"                                              , self.cost    ]]), columns= ['datadate'                  , 'tic'                , 'iter_num'     , 'action_performed' , 'available_amount' , 'stock_value'       ,  'nb_stock_traded'             , 'trade_cost'  ]) 
                self.df_trace.insert_row_to_stock_daily_trading_step(df_step_row)                 
                pass
    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        if self.turbulence< self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index+1]
            # print('available_amount:{}'.format(available_amount))
            
            #update balance
            self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                              (1+ TRANSACTION_FEE_PERCENT)

            self.state[index+STOCK_DIM+1] += min(available_amount, action)
            
            self.cost+=self.state[index+1]*min(available_amount, action)* \
                              TRANSACTION_FEE_PERCENT
            self.trades+=1
            
            # CD Tracing with dataframe
            df_buy_row = pd.DataFrame(data = np.array([[self.data.datadate.values[0], self.data.tic.values[index] , self.iteration , "buy"              ,  available_amount  , self.state[index+1] ,  min(available_amount, action) , self.cost    ]]), columns= ['datadate'                  , 'tic'                , 'iter_num'     , 'action_performed' , 'available_amount' , 'stock_value'       ,  'nb_stock_traded'             , 'trade_cost'  ]) 
            self.df_trace.insert_row_to_stock_daily_trading_step(df_buy_row)            
           
        else:
            # if turbulence goes over threshold, just stop buying
            # CD Tracing with dataframe
            df_step_row = pd.DataFrame(data = np.array([[self.data.datadate.values[0], self.data.tic.values[index] , self.iteration , "step_buy"        ,  "0"                  , self.state[index+1] ,  "0"                            , self.cost    ]]), columns= ['datadate'                  , 'tic'                , 'iter_num'     , 'action_performed' , 'available_amount' , 'stock_value'       ,  'nb_stock_traded'             , 'trade_cost'  ]) 
            self.df_trace.insert_row_to_stock_daily_trading_step(df_step_row) 
            
            pass
        
    def step(self, actions):
        if(self.df_trace.trace_mode == "LOG"):
            print("***************  Step:         ",self.day)
            print("Action:       ",actions)
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            if(self.df_trace.trace_mode == "LOG"):
                plt.plot(self.asset_memory,'r')
                plt.savefig('results/account_value_trade_{}_{}.png'.format(self.model_name, self.iteration))
                plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            
            if(self.df_trace.trace_mode == "LOG"):
                df_total_value.to_csv('results/account_value_trade_{}_{}.csv'.format(self.model_name, self.iteration))
            
            end_total_asset = self.state[0]+ sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            total_reward = self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))- self.asset_memory[0]  

            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            sharpe = (4**0.5)*df_total_value['daily_return'].mean()/ \
                  df_total_value['daily_return'].std()

            if(self.df_trace.trace_mode == "LOG"):
                print("~~~~~~~~~~~~~~ Terminal Step ~~~~~~~~~~~~~~")
                print("previous_total_asset: {}".format(self.asset_memory[0]))           
                print("end_total_asset:      {}".format(end_total_asset))
                print("total_reward:         {}".format(total_reward))
                print("total_cost:           ", self.cost)
                print("total trades:         ", self.trades)
                print("Sharpe: ",sharpe) 
                
            df_rewards = pd.DataFrame(self.rewards_memory)
            
            if(self.df_trace.trace_mode == "LOG"):           
                df_rewards.to_csv('results/account_rewards_trade_{}_{}.csv'.format(self.model_name, self.iteration))
            
            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            # with open('obs.pkl', 'wb') as f:  
            #    pickle.dump(self.state, f)
            
            # Trace by CD --------------------------
            # Add new entries in df_trace struscture
            df_new_row = pd.DataFrame(data = np.array([[self.data.datadate.values[0], self.iteration, self.asset_memory[0], end_total_asset, total_reward, self.cost, self.trades, sharpe ]]), columns=['datadate','iter_num','previous_total_asset','end_total_asset','total_reward','total_cost','total_trade','sharpe'])           
            self.df_trace.insert_row_to_terminal_state(df_new_row)
            
            return self.state, self.reward, self.terminal,{}

        else:
            # print("np.array(self.state[1:29]) -> ",np.array(self.state[1:29]))

            actions = actions * HMAX_NORMALIZE
            #actions = (actions.astype(int))
            if self.turbulence>=self.turbulence_threshold:
                actions=np.array([-HMAX_NORMALIZE]*STOCK_DIM)
                
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            
            if(self.df_trace.trace_mode == "LOG"):            
                print("begin_total_asset:  {}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action: {}'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day,:]         
            self.turbulence = self.data['turbulence'].values[0]
            
            if(self.df_trace.trace_mode == "LOG"):
                print("turbulence:      ", self.turbulence)
            
            #load next state
            # print("stock_shares:    {}".format(self.state[29:]))
            self.state =  [self.state[0]] + \
                    self.data.adjcp.values.tolist() + \
                    list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                    self.data.macd.values.tolist() + \
                    self.data.rsi.values.tolist() + \
                    self.data.cci.values.tolist() + \
                    self.data.adx.values.tolist()
            
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            self.asset_memory.append(end_total_asset)
            
            if(self.df_trace.trace_mode == "LOG"):
                print("end_total_asset: {}".format(end_total_asset))
            
            self.reward = end_total_asset - begin_total_asset            
            
            if(self.df_trace.trace_mode == "LOG"):
                print("step_reward:     {}".format(self.reward))
            
            self.rewards_memory.append(self.reward)

            # Trace by CD --------------------------
            # Add new entries in df_trace struscture
            
            df_row = pd.DataFrame(data = np.array([[self.data.datadate.values[0], self.iteration, begin_total_asset, end_total_asset, self.reward, self.turbulence, self.turbulence_threshold]]), columns=['datadate', 'iter_num', 'begin_total_asset', 'end_total_asset', 'step_reward', 'turbulence', 'turbulence_threshold'])           
            self.df_trace.insert_row_to_portfolio_daily_step(df_row)

            self.reward = self.reward*REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def reset(self):  
        if self.initial:
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False 
            #self.iteration=self.iteration
            self.rewards_memory = []
            #initiate state
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                          self.data.adjcp.values.tolist() + \
                          [0]*STOCK_DIM + \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist()  + \
                          self.data.cci.values.tolist()  + \
                          self.data.adx.values.tolist() 
        else:
            previous_total_asset = self.previous_state[0]+ \
            sum(np.array(self.previous_state[1:(STOCK_DIM+1)])*np.array(self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            self.asset_memory = [previous_total_asset]
            #self.asset_memory = [self.previous_state[0]]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False 
            #self.iteration=iteration
            self.rewards_memory = []
            #initiate state
            #self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]
            #[0]*STOCK_DIM + \

            self.state = [ self.previous_state[0]] + \
                          self.data.adjcp.values.tolist() + \
                          self.previous_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]+ \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist()  + \
                          self.data.cci.values.tolist()  + \
                          self.data.adx.values.tolist() 
            
        return self.state
    
    def render(self, mode='human',close=False):
        return self.state
    

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]