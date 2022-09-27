from types import SimpleNamespace
import pandas as pd
import timeit
from time import sleep
import json
import traceback

from qpslib_history_manager import history_client
from common.error import make_error_msg
from qps.models import evaluation
from common.handler import Handler
from modelling_config import model_config
from qps.models.summary import SummaryDB

from config import Config
import config



class QPSFeedback(Handler):
    def __init__(self):
        cfg = Config()
        self.SERVICE = SimpleNamespace(**model_config['SERVICE'])
        self.DATA = SimpleNamespace(**model_config['DATA'])
        self.PREP = SimpleNamespace(**model_config['PREPROCESSING'])
        self.SETTING = SimpleNamespace(**model_config['FEEDBACK'])

    # @concurrent.process(timeout=30) # not work with
    def run(self, data: SimpleNamespace):
        y = data.y.value
        self.comments = []
        output = {}
        output['y'] = y

        ## Save y to SummaryDB
        try:
            y_df = pd.DataFrame({self.DATA.id: data.master_id,
                                 self.DATA.y: y}, index=[0])
            SummaryDB().save_data(y_df)
        except Exception as e:
            output['error'] = {'message': e.details()}
            print(f'output: {output}')
            return output

        ## Load yhat
        wait = True
        check_iter = 0
        start = timeit.default_timer()
        while(wait):
            try:
                client = history_client.QPSHistoryClient(self.SERVICE.host,
                                                         data.company,
                                                         data.target,
                                                         self.SERVICE.result_type)
                yhat = client.get_inference_result(data.master_id).y_hat
                comment = f'Waiting time to get yhat : ' \
                          f'{check_iter * self.SETTING.sleep_timer}'
                self.comment.append(comment)
                wait = False
            except Exception as e:
                end = timeit.default_timer()
                if end - start > self.SETTING.time_limit:
                    wait = False
                    output['error'] = make_error_msg(str(e),
                                                     f'Load yhat: {traceback.format_exc()}')
                    return output
                else:
                    check_iter += 1
                    sleep(self.SETTING.sleep_timer)



        ## Evaluation
        residual = evaluation.cal_residual(y, yhat)

        ## Output
        output['residual'] = residual
        output['comment'] = '||'.join(self.comments)


        return output








if __name__ == '__main__':
    esbr_input = '''{
  "company": "BRIUQE",
  "target": "TTA",
  "master_id": "12JB439A_03",
  "y": {
    "time": "2021-07-28T05:53:30.284331Z",
    "value": 2186.5462
  }
}'''


    config.load_config('../../config.yml')

    data = SimpleNamespace(**json.loads(esbr_input))
    data.y = SimpleNamespace(**data.y)

    runner = QPSFeedback()
    output = runner.run(data)
    print(output)

