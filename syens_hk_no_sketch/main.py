import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime
#predict heng seng index
#predict the best stock
#for hongkong stock

from finrl.config_tickers import DOW_30_TICKER,HSI_50_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
#from finrl.agents.stablebaselines3.models import DRLAgent,DRLEnsembleAgent
from finrl.agents.stablebaselines3.models_syn import DRLAgent,DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

from pprint import pprint

import sys
sys.path.append("../FinRL-Library")

import itertools
import argparse
import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)
import random
import torch
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--summary-path", 
                    help="training summary df path",
                    type=str,
                    default='./summary/sum.csv')
parser.add_argument("--seed", 
                    help="training seed",
                    type=int,
                    default=0)

args = parser.parse_args()

##One call at beginning is enough
seed_everything(args.seed)

def data_download():
    # Download data
    TRAIN_START_DATE = '2009-01-01'
    TRAIN_END_DATE = '2015-09-26'
    TEST_START_DATE = '2015-09-26'
    TEST_END_DATE = '2023-06-01'

    
    df = YahooDownloader(start_date = TRAIN_START_DATE,
                        end_date = TEST_END_DATE,
                        ticker_list = HSI_50_TICKER).fetch_data()

    return df

def data_preprocess(df):
    # Preprocess data
    fe = FeatureEngineer(use_technical_indicator=True,
                        tech_indicator_list = INDICATORS,
                        use_turbulence=True,
                        user_defined_feature = True)

    processed = fe.preprocess_data(df)
    processed = processed.copy()
    processed = processed.fillna(0)
    processed = processed.replace(np.inf,0)

    return processed

def set_env(processed, INDICATORS):
    # Design env
    stock_dimension = len(processed.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    # print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "buy_cost_pct": 0.001, 
        "sell_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4,
        "print_verbosity":5
    }

    return env_kwargs
    
def set_drl_agent(processed, env_kwargs):
    # Implement DRL
    rebalance_window = 66 # rebalance_window is the number of days to retrain the model
    validation_window = 66 # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)
    TRAIN_START_DATE = '2009-01-01'
    TRAIN_END_DATE = '2015-09-26'
    TEST_START_DATE = '2015-09-26'
    TEST_END_DATE = '2023-06-01'



    ensemble_agent = DRLEnsembleAgent(df=processed,
                    train_period=(TRAIN_START_DATE,TRAIN_END_DATE),
                    val_test_period=(TEST_START_DATE,TEST_END_DATE),
                    rebalance_window=rebalance_window, 
                    validation_window=validation_window, 
                    **env_kwargs)

    A2C_model_kwargs = {
                        'n_steps': 5,
                        'ent_coef': 0.005,
                        'learning_rate': 0.0007
                        }

    PPO_model_kwargs = {
                        "ent_coef":0.01,
                        "n_steps": 2048,
                        "learning_rate": 0.00025,
                        "batch_size": 128
                        }

    DDPG_model_kwargs = {
                        #"action_noise":"ornstein_uhlenbeck",
                        "buffer_size": 10_000,
                        "learning_rate": 0.0005,
                        "batch_size": 64
                        }

    timesteps_dict = {'a2c' : 10_000, 
                    'ppo' : 10_000, 
                    'ddpg' : 10_000
                    }

    return ensemble_agent, A2C_model_kwargs, PPO_model_kwargs,\
           DDPG_model_kwargs, timesteps_dict

def train(ensemble_agent, 
          A2C_model_kwargs, 
          PPO_model_kwargs,
          DDPG_model_kwargs, 
          timesteps_dict):
    df_summary = ensemble_agent.run_ensemble_strategy(A2C_model_kwargs,
                                                    PPO_model_kwargs,
                                                    DDPG_model_kwargs,
                                                    timesteps_dict)
    print(df_summary)
    df_summary.to_csv(args.summary_path)

if __name__ == '__main__':

    print('*'*20+'data downloading'+'*'*20)
    df = data_download()

    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

    processed = data_preprocess(df)

    env_kwargs = set_env(processed, INDICATORS)

    ensemble_agent, A2C_model_kwargs, PPO_model_kwargs,\
    DDPG_model_kwargs, timesteps_dict = set_drl_agent(processed, env_kwargs)

    print('*'*20+'start training'+'*'*20)
    train(ensemble_agent, 
          A2C_model_kwargs, 
          PPO_model_kwargs,
          DDPG_model_kwargs, 
          timesteps_dict)
   