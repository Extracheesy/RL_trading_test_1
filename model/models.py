# common library
import pandas as pd
import numpy as np
import time
import gym

from stable_baselines import PPO2
from stable_baselines import A2C



# from stable_baselines import GAIL, SAC
from stable_baselines import ACER
from stable_baselines import PPO2
from stable_baselines import A2C
# from stable_baselines import DDPG
# from stable_baselines import TD3
  
#from stable_baselines import DDPG
#from stable_baselines import TD3

# from stable_baselines.ddpg.policies import DDPGPolicy
# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
# from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
# from stable_baselines.common.vec_env import DummyVecEnv

from stable_baselines.common.vec_env import DummyVecEnv

from preprocessing.preprocessors import *

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade

import winsound  


""" CD Comment tmp
# RL models from stable-baselines
from stable_baselines import GAIL, SAC
from stable_baselines import ACER
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import TD3

from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv

    
from preprocessing.preprocessors import *

from config import config

# customized env
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade

"""
def train_A2C(env_train, model_name, timesteps=50000):
#    A2C model

    start = time.time()
    #print("A2C Model Start")
    #print("start:                  ",start)
    print("Train A2C Model with MlpPolicy")
        
    model = A2C('MlpPolicy', env_train, verbose=0)

    print("A2C Model finish           :")
    print("A2C Learn Start            :")

    model.learn(total_timesteps=timesteps)

    print("A2C Model learn finish     :")

    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")

    print("A2C Model save finish     :")


    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model


def train_A2C_MlpLstmPolicy(env_train, model_name, timesteps=50000):
#    A2C model with MlpLstmPolicy

    start = time.time()
    # print("A2C Model")
    # print("start:                  ",start)
    print("Train A2C Model with MlpLstmPolicy")
    
    model = A2C('MlpLstmPolicy', env_train, verbose=0)

    print("A2C Model finish           :")
    print("A2C Learn Start            :")

    model.learn(total_timesteps=timesteps)

    print("A2C Model learn finish     :")

    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")

    print("A2C Model save finish     :")

    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    print("End Training A2C Model with MlpLstmPolicy")
    
    return model


def train_A2C_MlpLnLstmPolicy(env_train, model_name, timesteps=50000):
#    A2C model with MlpLnLstmPolicy

    start = time.time()
    # print("A2C Model")
    # print("start:                  ",start)
  
    print("Train A2C Model with MlpLnLstmPolicy")

    model = A2C('MlpLnLstmPolicy', env_train, verbose=0)

    print("A2C Model finish           :")
    print("A2C Learn Start            :")

    model.learn(total_timesteps=timesteps)

    print("A2C Model learn finish     :")

    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")

    print("A2C Model save finish     :")

    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    print("End Training A2C Model with MlpLnLstmPolicy")
    
    return model








"""
def train_ACER(env_train, model_name, timesteps=25000):
    start = time.time()
    model = ACER('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model


def train_DDPG(env_train, model_name, timesteps=10000):
#    DDPG model

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', (end-start)/60,' minutes')
    return model
"""

def train_PPO(env_train, model_name, policy, timesteps=50000):
#    PPO model

    start = time.time()
    
    print("debug CD - PPO model creation with policy: ",policy)
    
    #model = PPO2(policy, env_train, ent_coef = 0.005, nminibatches = 8)
    model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

    print("debug CD - PPO model creation completed with policy: ",policy)

    print("debug CD start PPO train start with policy: ",policy)

    model.learn(total_timesteps=timesteps)
    
    print("debug CD start PPO train completed start with policy: ",policy)
     
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

"""
def train_GAIL(env_train, model_name, timesteps=1000):
#    GAIL Model
    #from stable_baselines.gail import ExportDataset, generate_expert_traj
    start = time.time()
    # generate expert trajectories
    model = SAC('MLpPolicy', env_train, verbose=1)
    generate_expert_traj(model, 'expert_model_gail', n_timesteps=100, n_episodes=10)

    # Load dataset
    dataset = ExpertDataset(expert_path='expert_model_gail.npz', traj_limitation=10, verbose=1)
    model = GAIL('MLpPolicy', env_train, dataset, verbose=1)

    model.learn(total_timesteps=1000)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

"""

def DRL_prediction(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   turbulence_threshold,
                   initial,
                   df_trace):
    ### make a prediction based on trained model###
    
    if df_trace.trace_mode == "LOG" :
        print("+++++++++++++++++++++++++++++++++++++++++")
        print("DRL_prediction")
        print("iter_num             ",iter_num)
    ## trading env
    trade_data = data_split(df, start=unique_trade_date[iter_num - rebalance_window], end=unique_trade_date[iter_num])
    
    if df_trace.trace_mode == "LOG" :
        print("split start          ",unique_trade_date[iter_num - rebalance_window])
        print("split end            ",unique_trade_date[iter_num])
        print("trade_data           ", trade_data)
 
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num,
                                                   df_trace=df_trace)])
    if df_trace.trace_mode == "LOG" :
        print("Stock env trade           ")
        print("env_trade                 ", env_trade)
    obs_trade = env_trade.reset()
    # print("obs_trade           ", obs_trade)  

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            # print(env_test.render())
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    
    if df_trace.trace_mode == "LOG" :
        print("df_last_state           ", df_last_state)     
        df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
    
    return last_state



def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    
    # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    # print("DRL_validation")
    # print("model                            ", model)
    # print("test_data                        ", test_data)
    # print("test_env                         ", test_env)
    # print("test_obs                         ", test_obs)
    # print("test_data.index.unique           ", test_data.index.unique())
    # print("range len test_data.index.unique ", range(len(test_data.index.unique())))
    
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)




def get_validation_sharpe(iteration):
    ###Calculate Sharpe ratio based on validation results###
 
    # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    # print("get_validation_sharpe")
    
    # print("iteration             ", iteration)
 
    df_total_value = pd.read_csv('results/account_value_validation_{}.csv'.format(iteration), index_col=0)
    
    # print("read cvs total value  ",df_total_value)
    
    df_total_value.columns = ['account_value_train']

    #print("df_total_value.columns        ",df_total_value.columns)
    #print("df_total_value daily_return   ",df_total_value['daily_return'])
    #print("df_total_value.pct_change 1   ",df_total_value.pct_change(1))
   
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
             
    # print("sharpe", sharpe)
    return sharpe

 

def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window, trace_df) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""


    tace_mode = trace_df.trace_mode
    if tace_mode == "LOG" :
        print("============Start Ensemble Strategy============")
        print("run_ensemble_strateg:")
        print("  -- df                          Data from csv file")
        # print("  -- df datadate:               ",df.datadate)
        print("  -- df datadate size:          ",df.datadate.size)
        print("  -- unique_trade_date:         ",unique_trade_date)
        # print("  -- unique_trade_date size:    ",unique_trade_date.size)
        # print("  -- rebalance_window:          ",rebalance_window)
        # print("  -- validation_window:         ", validation_window)


    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []

    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    a2c_sharpe_list = []

    model_use = []
    
    # print("*********************************************")
    # print(" turbulence")
    # print("  -- df datadate", df.datadate)   
 
    # based on the analysis of the in-sample data
    turbulence_threshold = 140
    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)
   
    start = time.time()    
         
         
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        ## initial state is empty
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # print(" --> i = ", i, " in range of ", len(unique_trade_date), " with step of ", rebalance_window)
        if i - rebalance_window - validation_window == 0:
            # inital state
            # print(" -->  initial state")
            initial = True
        else:
            # previous state
            # print(" -->  previous state")
            initial = False
 
 
        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        # Comment CD:
        # Calcul le i - rebalance_window - validation_window eme jour dans la table de unique date
        # Cree une table de booleen que met a true cette date dans df datadate 
        # -> retourne la lite des indexes df de cette date
        # le dernier index est pris en tant que end_date
        # la start date est prise en comptant a rebour 63 days

        end_date_index = df.index[   df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]   ].to_list()[-1]
        start_date_index = end_date_index - validation_window*30 + 1
        
        # Comment CD:
        # Remplir historical_turbulence avec les data 
        # du dataframe df comprises entre start_date_index et end_date_index + 1
        # sortir les duplicate en ne gardant que les valeurs de la premier Stock ici AAPL
        # calculer la moyenne mean sur la plage de valeur des turbulences filtrée
        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
        #print(" --> historical_turbulence                      ",historical_turbulence)

        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
        #print(" --> historical_turbulence drop_dup             ",historical_turbulence)

        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)
        # print(" --> historical_turbulence np.mean ",historical_turbulence_mean)
        # print(" --> turbulence_threshold:         ", turbulence_threshold)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            print(" --> historical_turbulence_mean > insample_turbulence_threshold")
            print("     ",historical_turbulence_mean, ">", insample_turbulence_threshold)
            turbulence_threshold = insample_turbulence_threshold
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            print(" --> historical_turbulence_mean < insample_turbulence_threshold")
            print(historical_turbulence_mean, "<", insample_turbulence_threshold)
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
            print(" --> turbulence_threshold: ", turbulence_threshold)

        # Modif CD test turbulence threshold
        # turbulence_threshold = 180

        ############## Environment Setup starts ##############
        ## training env
        if tace_mode == "LOG" :
            print(" --> Split train      dataset from 20090000 to ", unique_trade_date[i - rebalance_window - validation_window])
            print(" --> Split validation dataset from ", unique_trade_date[i - rebalance_window - validation_window] ," to ", unique_trade_date[i - rebalance_window])

        train = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])

        if tace_mode == "LOG" :
            print(" ==> StockEnvTrain")
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        ## validation env
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        if tace_mode == "LOG" :
            print(" ==> StockEnvValidation")
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()
       
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        if tace_mode == "LOG" :
            print("============== Model Training===========")
            print("======  Model training from: ", 20090000, " to ",
                  unique_trade_date[i - rebalance_window - validation_window])
        
        model_to_test  = trace_df.model_to_test
        policy_to_test = trace_df.policy
        
        if model_to_test == "A2C":
            ############## A2C Model Start ##############
            A2C_File_Name = "A2C_30k_dow_" + str(i)
            #print("A2C_30k_dow_{}", i)
            print("======  A2C file: ", A2C_File_Name)
            print("======  A2C Training Start  ========")        
            # model_a2c = train_A2C(env_train, model_name="A2C_30k_dow_{}".format(i), timesteps=30000)
            # model_a2c = train_A2C(env_train, model_name=A2C_File_Name, timesteps=30000)
  
            if policy_to_test == "MlpPolicy":
                model_a2c = train_A2C(env_train, model_name=A2C_File_Name, timesteps=50000)
            elif policy_to_test == "MlpLstmPolicy":
                model_a2c = train_A2C_MlpLstmPolicy(env_train, model_name=A2C_File_Name, timesteps=50000)
            else :
                    model_a2c = train_A2C_MlpLnLstmPolicy(env_train, model_name=A2C_File_Name, timesteps=50000)
 
            winsound.Beep(2000, 1000)
            
            print("======  Model Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
                  unique_trade_date[i - rebalance_window])
            print("======  A2C DRL_Validation Start ========")
            DRL_validation(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
            sharpe_a2c = get_validation_sharpe(i)
            print("======  A2C Sharpe Ratio: ", sharpe_a2c)
            
            a2c_sharpe_list.append(sharpe_a2c)
            #print("======  a2c_sharpe_list:       ", a2c_sharpe_list)   
     
            model_ensemble = model_a2c
            model_use.append('A2C')
            print("======  model_use:        ", model_use)
            ############## A2C Model End ##############
        elif  model_to_test == "PPO":
            ############## PPO Model Start ############## 
            PPO_File_Name = "PPO_30k_dow_" + str(i)
            print("======  PPO Training========")
            print("======  PPO file: ", PPO_File_Name)
            print("======  PPO Training Start ========")
            # model_ppo = train_PPO(env_train, model_name="PPO_100k_dow_{}".format(i), timesteps=100000)
            # model_ppo = train_PPO(env_train, model_name=PPO_File_Name, timesteps=100000)
            model_ppo = train_PPO(env_train, model_name=PPO_File_Name, policy=policy_to_test, timesteps=100000)
            winsound.Beep(2000, 1000)
            print("======  PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
            unique_trade_date[i - rebalance_window])
            print("======  PPO DRL_Validation Start  ========")
            DRL_validation(model=model_ppo, test_data=validation, test_env=env_val, test_obs=obs_val)
            sharpe_ppo = get_validation_sharpe(i)
            print("PPO Sharpe Ratio: ", sharpe_ppo)    
            ppo_sharpe_list.append(sharpe_ppo)           
            #print("======  a2c_sharpe_list:       ", a2c_sharpe_list) 
            model_ensemble = model_ppo
            model_use.append('PPO')
            print("======  model_use:        ", model_use)
            ############## PPO Model End ##############
        else :
                ############## DDPG Model Start ############## 
                DDPG_File_Name = "DDPG_30k_dow_" + str(i)
                print("======  DDPG Training  ========")
                print("======  DDPG file: ", DDPG_File_Name)
                print("======  DDPG Training Start ========")       
                #model_ddpg = train_DDPG(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=10000)
                #model_ddpg = train_DDPG(env_train,  DDPG_File_Name, timesteps=10000)
                #model_ddpg = train_TD3(env_train, model_name="DDPG_10k_dow_{}".format(i), timesteps=20000)
                #model_ddpg = train_TD3(env_train, DDPG_File_Name, timesteps=30000)
                winsound.Beep(2000, 1000)
                print("======  DDPG Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
                unique_trade_date[i - rebalance_window])
                print("======  DDPG DRL_Validation Start  ========")
                # DRL_validation(model=model_ddpg, test_data=validation, test_env=env_val, test_obs=obs_val)
                # sharpe_ddpg = get_validation_sharpe(i)
                print("DDPG Sharpe Ratio: ", sharpe_ddpg)          
                ddpg_sharpe_list.append(sharpe_ddpg)
                model_ensemble = model_ddpg
                model_use.append('DDPG')           
                print("======  model_use:        ", model_use)     
        
        ############## Training and Validation ends ##############
        winsound.Beep(2000, 1000)
 
        if tace_mode == "LOG" :
            ############## Trading starts ##############
            print("======  Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
            # print("======  Model Used Model: ", model_ensemble)   
            print("======  DRL_Prediction Trading Start ========")
        
        last_state_ensemble = DRL_prediction(df=df, model=model_ensemble, name="ensemble",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial,
                                             df_trace = trace_df)  
        # print("last_state_ensemble        ", last_state_ensemble)

        print("======  DRL_Prediction Trading Completed ========")
        
        winsound.Beep(2000, 1000)
        ############## Trading end ##############
 
    end = time.time()

    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")
    print("============End Ensemble Strategy============")
    
    
    












