import pandas as pd
from glob import glob
from types import SimpleNamespace

from qps.models.summary import Summary, SummaryDB
from modelling_config import model_config


#%% Setting ====================================================================

DATA = SimpleNamespace(**model_config['DATA'])
PREP = SimpleNamespace(**model_config['PREPROCESSING'])


#%% Extract Y ==================================================================
# lot id를 읽어 시간순으로 train / test split
y = pd.read_csv('./data/CMP115_POST.csv')
y_df = y.sort_values(['TRACK_IN_DTTS']).reset_index(drop=True)

train_idx = int(len(y_df)*0.7)
train_y = y_df.loc[:873, :]
test_y = y_df.loc[874:, :]


train_y.to_csv('./data/train_y.csv', index=False)
test_y.to_csv('./data/test_y.csv', index=False)


#%% Create Summary Data ========================================================
train_lots = pd.read_csv('./data/train_y.csv', usecols=['LOT_ID'])['LOT_ID'].unique()

total_summ = pd.DataFrame()
x_dirs = glob(f'./data/X/*.csv')
for lot in train_lots:
    print(lot)
    train_x = pd.read_csv(glob(f'./data/X/{lot}*')[0])
    summ = Summary(DATA.raw_id, DATA.process_order, DATA.raw_time)
    summ_data = summ.create_summary_data(train_x)
    total_summ = pd.concat([total_summ, summ_data], ignore_index=True)







#%% Merge y ====================================================================
y = pd .read_csv('./data/train_y.csv')
y[DATA.raw_id] = y['LOT_ID'] + '_' + y['WAFER_ID'].apply(lambda x: x.split('.')[1])
train_summary = pd.merge(y.loc[:, [DATA.raw_id, DATA.raw_y]], total_summ,
                 on=DATA.raw_id, how='inner')

train_summary.rename(columns={DATA.raw_id: DATA.id, DATA.raw_time: DATA.time,
                              DATA.raw_y: DATA.y},
                  inplace=True)


#%% Save Summary Data ==========================================================

train_summary.to_csv('./data/train_summary.csv', index=False)

SummaryDB().save_data(train_summary)