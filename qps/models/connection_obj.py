from types import SimpleNamespace
from common.connect_db import MongoDBConnection
from modelling_config import model_config


class DBConn():
    def __init__(self):
        DB = SimpleNamespace(**model_config['DB'])

        self.esbr_raw_conn, self.esbr_raw_db = MongoDBConnection(host=DB.host,
                                     port=DB.port,
                                     username=DB.username,
                                     password=DB.password,
                                     database='pjems').get_db_conn()

        self.esbr_summ_conn, self.esbr_summ_db = MongoDBConnection(host=DB.host,
                                     port=DB.port,
                                     username=DB.username,
                                     password=DB.password,
                                     database='summary').get_db_conn()

        self.esbr_history_conn, self.esbr_history_db = MongoDBConnection(host=DB.host,
                                     port=DB.port,
                                     username=DB.username,
                                     password=DB.password,
                                     database='history').get_db_conn()

        self.esbr_model_conn, self.esbr_model_db = MongoDBConnection(host=DB.host,
                                                                         port=DB.port,
                                                                         username=DB.username,
                                                                         password=DB.password,
                                                                         database='model').get_db_conn()



if __name__=='__main__':
    result = DBConn().esbr_summ_db