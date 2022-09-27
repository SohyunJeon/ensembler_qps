from glob import glob
import pandas as pd
import json

#%% load sample data

test_lots = pd.read_csv('./data/test_y.csv', usecols=['LOT_ID'])['LOT_ID'].unique()

x_dirs = glob(f'./data/X/{test_lots[0]}*.csv')

test_df = pd.read_csv(x_dirs[0])
test_df = test_df.loc[test_df['SLOT']==3, :].reset_index(drop=True)


#%% Create inference input

input = {"company": "BRIUQE", "target": "CMP", "service_type": "QPS",
 "result_type": "REGRESSION", "master_id": test_df['SUBSTRATE_ID'][0],
 "time": "2010-01-01T00:00:00Z"}

x = []
for idx, row in test_df.iterrows():
    x.append(dict(row))

input['x'] = x

esbr_input = json.dumps(input)