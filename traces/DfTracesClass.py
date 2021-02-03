import pandas as pd



class Df_Traces :
# stock_data_structure: datadate - tic - Price - Action - nb_Stock - profit_loss
# portfolio_data_structure: datadate - begin_total_asset - end_total_asset - step_reward - turbulence
# terminal_step_state: datadate - end_total_asset - total_reward - total_cost - total_trade


    def new(self):

        # Global info at the end of each trade widows
        self.portfolio_terminal_step_state = pd.DataFrame(columns = ['datadate','iter_num','previous_total_asset','end_total_asset','total_reward','total_cost','total_trade','sharpe'])
                                                    
        # Portfolio variance per trade day
        self.portfolio_daily_status = pd.DataFrame(columns = ['datadate', 'iter_num', 'begin_total_asset', 'end_total_asset', 'step_reward', 'turbulence', 'turbulence_threshold'])

        # Stock detailled trade infos
        self.stock_daily_trading_status = pd.DataFrame(columns = ['datadate', 'tic', 'iter_num', 'action_performed', 'available_amount', 'stock_value', 'nb_stock_traded', 'trade_cost'])
            
        
        # race_mode: LOG or DATAFRAME    
        # trace_mode = "LOG"
        self.trace_mode = "DATAFRAME"
    
        self.start_time = ""
        self.run_id = 0
        self.model_to_test = ""
        self.market = ""
        self.policy = ""
        
        return (self)
    

    def insert_row_to_terminal_state(self, df_new_row) :
        
        self.portfolio_terminal_step_state = self.portfolio_terminal_step_state.append(df_new_row, ignore_index=True)

            
    def insert_row_to_portfolio_daily_step(self, df_new_row) :
        
        self.portfolio_daily_status = self.portfolio_daily_status.append(df_new_row, ignore_index=True)


    def insert_row_to_stock_daily_trading_step(self, df_new_row) :
        
        self.stock_daily_trading_status = self.stock_daily_trading_status.append(df_new_row, ignore_index=True)