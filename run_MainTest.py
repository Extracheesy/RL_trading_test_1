import numpy as np
import time
import pandas as pd

# model
from model.models import *

import os
import sys
import datetime
import pathlib
  
from traces.DfTracesClass import Df_Traces


def run_model() -> None:
    
    #preprocessed_path = "done_data_DJI.csv"
    #preprocessed_path = "done_data_CAC40.csv"
    #preprocessed_path = "done_data_DAX.csv"
  
    # Creat the trace instance from Df_Traces
    df_trace = Df_Traces()
    df_trace = Df_Traces.new(df_trace)
    tace_mode = df_trace.trace_mode
    
   
    #MARKET = "DJI"
    #MARKET = "CAC_40"
    #MARKET = "DAX"
    
    #MODEL_TO_TEST = "A2C"
    # MODEL_TO_TEST = "PPO"
    
    #POLICY = "MlpPolicy"
    # POLICY = "MlpLstmPolicy"
    # POLICY = "MlpLnLstmPolicy"
    
    #START_RUN_TIME = time.strftime("%Y%m%d-%H%M%S")
    
    run_id = 0
    
    if tace_mode == "LOG" :
        LOG_FILE = "log_" + MODEL_TO_TEST + "_" + POLICY + "_" + START_RUN_TIME + ".txt"
        sys.stdout = open(LOG_FILE, 'w') 
        print("########################################")    
        print("Model to Test:       ",MODEL_TO_TEST)    
        print("Market:              ",MARKET)
        print("Policy:              ",POLICY)
        print("log file:            ",LOG_FILE) 
        print("########################################")
        print("Creat log file ",LOG_FILE)

    #df_trace.start_time = START_RUN_TIME
    #df_trace.model_to_test = MODEL_TO_TEST
    #df_trace.policy = POLICY
    #df_trace.market = MARKET
    toto = "true"

    #for market in ["DJI", "CAC", "DAX"] :
    #        for policy in ["MlpPolicy", "MlpLstmPolicy", "MlpLnLstmPolicy"]:
    #            for model_to_test in ["PPO" , "A2C"] :
    
    #while toto == "true" :
    #    for market in ["CAC", "DAX", "DJI"] :
    #        for policy in ["MlpPolicy"]:
    #            for model_to_test in ["A2C", "PPO"] :

                    
    while toto == "true" :
        for market in ["DAX", "DJI"] :
            for policy in [ "MlpLstmPolicy", "MlpLnLstmPolicy"]:
                for model_to_test in ["PPO"] :
                    
                    window_size = 63

                    #if model_to_test == "PPO" :
                    #    policy = "MlpPolicy"

                    print("New run - Policy:    ", policy ," model:   ", model_to_test, "window size:  ",window_size)               
                
                    START_RUN_TIME = time.strftime("%Y%m%d-%H%M%S")
                    
                    df_trace.start_time = START_RUN_TIME
                    df_trace.model_to_test = model_to_test
                    df_trace.policy = policy
                    df_trace.market = market
    
                    preprocessed_path = "done_data_" + df_trace.market + ".csv"
                  
                    if os.path.exists(preprocessed_path):
                        data = pd.read_csv(preprocessed_path, index_col=0)
                        print("read csv file:   ", preprocessed_path)
                    else:
                        print("NOK No csv sata file available")
                
                    data['datadate'] = data['datadate'].astype(np.int64)
                    data['adjcp']    = data['adjcp'].astype(np.float64)    
                    data['open']     = data['open'].astype(np.float64)   
                    data['high']     = data['high'].astype(np.float64)   
                    data['low']      = data['low'].astype(np.float64)   
                    data['volume']   = data['volume'].astype(np.float64) 
                  
                    # 2015/10/01 is the date that validation starts
                    # 2016/01/01 is the date that real trading starts
                    # unique_trade_date needs to start from 2015/10/01 for validation purpose
                    unique_trade_date = data[(data.datadate > 20151001)&(data.datadate <= 20200707)].datadate.unique()
            
                    if tace_mode == "LOG" :
                        print( "datadate", data['datadate'].dtypes )   
                        print( "adjcp",    data['adjcp'].dtypes )
                        print( "open",     data['open'].dtypes )   
                        print( "high",     data['high'].dtypes )
                        print( "low",      data['low'].dtypes )   
                        print( "volume",   data['volume'].dtypes )
                        print(" -- unique_trade_date       ",unique_trade_date)
                        print(" -- unique_trade_date size  ",unique_trade_date.size)
                    
                    
                    # rebalance_window is the number of months to retrain the model
                    # validation_window is the number of months to validation the model and select for trading
                   
                    # rebalance_window = 63
                    # validation_window = 63
            
                    rebalance_window = window_size
                    validation_window = window_size
            
                    # MODEL_TO_TEST = "A2C"  
                    # MODEL_TO_TEST = "PPO"
                    # POLICY = "MlpPolicy"
                    # POLICY = "MlpLstmPolicy"
                    # POLICY = "MlpLnLstmPolicy"
            
                    ## Ensemble Strategy
                    if tace_mode == "LOG" :
                        print("------------------------------------------------------")
                        print("run_ensemble_strategy ")
                        print("Market         ",MARKET)
                        print("Model set to   ",MODEL_TO_TEST)
                        print("Policy set to  ",POLICY) 
                        print("------------------------------------------------------")        
             
                    
                    ## Ensemble Strategy
                    run_ensemble_strategy(df=data, 
                                          unique_trade_date = unique_trade_date,
                                          rebalance_window  = rebalance_window,
                                          validation_window = validation_window,
                                          trace_df          = df_trace)
                
                    if tace_mode == "LOG" :    
                        print("------------------------------------------------------")
                        print("Market         ",MARKET)
                        print("Model set to   ",MODEL_TO_TEST," completed")
                        print("Policy set to  ",POLICY," completed") 
                        print("run_ensemble_strategy completed")
                        print("------------------------------------------------------")
                     
                    
                    df_trace.portfolio_terminal_step_state.to_csv('trace_csv_results/portfolio_terminal_step_state_{}_{}_{}_{}.csv'.format(df_trace.start_time, df_trace.market , df_trace.model_to_test , df_trace.policy))
                    
                    df_trace.portfolio_daily_status.to_csv('trace_csv_results/portfolio_daily_status_{}_{}_{}_{}.csv'.format(df_trace.start_time, df_trace.market , df_trace.model_to_test , df_trace.policy))
                    
                    df_trace.stock_daily_trading_status.to_csv('trace_csv_results/stock_daily_trading_status_{}_{}_{}_{}.csv'.format(df_trace.start_time, df_trace.market , df_trace.model_to_test , df_trace.policy))
                   
                    """
                    df_trace.portfolio_terminal_step_state.to_csv('trace_csv_results/no_turbulence_test_portfolio_terminal_step_state_{}_{}_{}_{}_{}.csv'.format(df_trace.start_time, df_trace.market , df_trace.model_to_test , df_trace.policy   , run_id))
                    
                    df_trace.portfolio_daily_status.to_csv('trace_csv_results/no_turbulence_test_portfolio_daily_status_{}_{}_{}_{}_{}.csv'.format(df_trace.start_time, df_trace.market , df_trace.model_to_test , df_trace.policy   , run_id))
                    
                    df_trace.stock_daily_trading_status.to_csv('trace_csv_results/no_turbulence_test_stock_daily_trading_status_{}_{}_{}_{}_{}.csv'.format(df_trace.start_time, df_trace.market , df_trace.model_to_test , df_trace.policy   , run_id))
                     """
                    
                    print("csv ready done")
                  
                    START_RUN_TIME = time.strftime("%Y%m%d-%H%M%S")
    
                    run_id = run_id + 1
                    
                    # re-init
                    df_trace = Df_Traces.new(df_trace)

    if tace_mode == "LOG" :
        sys.stdout.close()





    #_logger.info(f"saving model version: {_version}")
   
    
if __name__ == "__main__":
    
    
    run_model()
























