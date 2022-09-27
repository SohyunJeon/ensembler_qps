import pandas as pd
import numpy as np
from types import SimpleNamespace

from qps.models.connection_obj import DBConn
from modelling_config import model_config




class Summary:
    def __init__(self, id_name, step_name, time_name):
        self.id_name = id_name
        self.step_name = step_name
        self.time_name = time_name
        _rem_cols = ['TIME', 'PRODUCT', 'PPID', 'STEP_NAME', 'port4_lot',
                    'TIME_EPD_SUBTRACT_VIR',
                    'TIME_EPD_MIN_VIR', 'Step', 'SLOT', 'MES_RECIPE_LIST',
                    'MirraP1WID',
                    'Step Duration', 'port3_lot', 'SVPlaten1EPData2',
                    'EOR1PolishTitle',
                    'port2_lot', 'Platen1StepNum', 'MES_LOT_LIST',
                    'SVPlaten1EPData',
                    'port1_lot']
        _timestamp_cols = ['TIME_1STEP_BEFORE', 'TIME_1STEP_BEFORE2',
                          'WAFER_START_TIME_VIR',
                          'PRESSURE_PAD_COND_LBF_X_VIR', 'TIME_1STEP_VIR',
                          'StepStartTime',
                          'PRESSURE_PAD_COND_LBF_STEP99_VIR', 'CurrentTime']
        self._useless_cols = _rem_cols + _timestamp_cols
        self._useless_steps = [99, 6]
        self._cat_cols = ['EQP', 'MODULE', 'LOT_ID', 'RECIPE_ID', 'OPERATION']

    def create_summary_data(self, data: pd.DataFrame) -> pd.DataFrame:
        time_summ = self.create_time_summ(data)
        df = self.preprocess(data)
        cat_summ = self.create_category_summ(df)
        num_summ = self.create_numeric_summ(df)
        summ_df = pd.merge(time_summ, num_summ, on=self.id_name)

        print(f'Summary data shape: {summ_df.shape}')
        return summ_df


    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.loc[data[self.id_name].dropna().index, :]
        df = df.drop(self._useless_cols, axis=1, errors='ignore')
        df = df.sort_values([self.id_name, self.step_name]).reset_index(drop=True)
        df = df.loc[~df[self.step_name].isin(self._useless_steps), :]
        return df

    def create_time_summ(self, data:pd.DataFrame):
        df = data.loc[:, [self.id_name, self.time_name]]
        df = df.groupby(self.id_name).last()
        df = df.reset_index()
        return df


    def create_category_summ(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.concat([data.loc[:, [self.id_name, self.step_name]],
                        data.loc[:, self.cat_cols]], axis=1)
        df = df.groupby(self.id_name).first()
        df = df.drop([self.step_name], axis=1, errors='ignore')
        return df

    def create_numeric_summ(self, data:pd.DataFrame) -> pd.DataFrame:
        def range_func(x):
            return abs(np.max(x) - np.min(x))

        def iqr_func(x):
            q3, q1 = np.percentile(x, [75, 25])
            return abs(q3 - q1)

        num_cols = [col for col in data.columns if col not in self._cat_cols]
        num_df = data.loc[:, num_cols]

        num_summ1 = num_df.groupby([self.id_name, self.step_name]).agg(
            ['mean', 'min', 'max', 'std', 'median', 'skew'])

        num_summ2 = num_df.groupby([self.id_name, self.step_name]).\
            agg([lambda x: range_func(x)])
        num_summ2.columns = [(x[0], 'range') for x in num_summ2.columns]

        num_summ3 = num_df.groupby([self.id_name, self.step_name]).\
            agg([lambda x: iqr_func(x)])
        num_summ3.columns = [(x[0], 'iqr') for x in num_summ3.columns]

        num_summ = pd.concat([num_summ1, num_summ2, num_summ3], axis=1)
        num_summ = num_summ.reset_index()

        num_summ_pv = pd.pivot(num_summ, index=self.id_name,
                               columns=self.step_name)
        new_cols = [x + '-' + str(int(z)) + '-' + y for x, y, z in
                    num_summ_pv.columns]
        num_summ_pv.columns = new_cols
        num_summ_pv = num_summ_pv.reset_index()
        return num_summ_pv


    def pop_useless_column(self, name: str):
        self._useless_cols.remove(name)

    def add_useless_column(self, name: str):
        self._useless_cols.append(name)

    def pop_useless_step(self, number: int):
        self._useless_steps.remove(number)

    def add_useless_step(self, number: int):
        self._useless_steps.append(number)

    def pop_cat_col(self, name: str):
        self._cat_cols.remove(name)

    def add_cat_col(self, name: str):
        self._cat_cols.append(name)

    @property
    def useless_cols(self):
        return self._useless_cols

    @property
    def useless_steps(self):
        return self._useless_cols

    @property
    def cat_cols(self):
        return self._cat_cols



class SummaryDB:
    def __init__(self):
        self.DATA = SimpleNamespace(**model_config['DATA'])
        self.SERVICE = SimpleNamespace(**model_config['SERVICE'])
        self.collection = DBConn().esbr_summ_db[self.SERVICE.company]



    def save_data(self, data:pd.DataFrame):
        for label, content in data.iterrows():
            resp = self.collection.update_one({self.DATA.id: content[self.DATA.id]},
                                                 {'$set': dict(content)},
                                              upsert=True)
            print(f'{label}: {resp.raw_result}')


    def load_data(self, count: int)-> pd.DataFrame:
        docs = self.collection.find({}, {'_id': 0}).sort(self.DATA.time, -1).\
            limit(count)
        result = pd.DataFrame(docs)
        result = result.sort_values(self.DATA.time).reset_index(drop=True)
        return result



if __name__=='__main__':
    # train_summary = pd.read_csv('./data/train_summary.csv')
    # SummaryDB().save_data(train_summary)

    data = SummaryDB().load_data(100)

