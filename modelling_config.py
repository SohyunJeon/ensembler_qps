
model_config = \
{
    'DATA': {
        'raw_id': 'SUBSTRATE_ID',
        'raw_time': 'TIME',
        'raw_y': 'OUTPUT_VALUE',
        'process_order': 'STEP_ID',
        'id': 'master_id',
        'y': 'y',
        'time': 'time'
    },
    'PREPROCESSING': {
        'same_value_ratio': 0.9,
        'high_corr': 0.7
    },
    'SERVICE': {
        'host': 'esbr-dev.brique.kr:8000',
        'company': 'BRIQUE',
        'target': 'TTA',
        'service_type': 'QPS',
        'result_type': 'REGRESSION'
    },
    'DB': {
        'host': 'esbr-dev.brique.kr',
        'port': 27017,
        'username': '',
        'password': '',
        'summ_database': 'summary',
        'history_database': 'history'
    },
    'FEEDBACK': {
        'time_limit': 600,
        'sleep_timer': 30
    },
    'UPDATE': {
        'eval_data_count': 100,
        'retain_data_count': 5000,
        'r2_limit': 0.5,
        'stacked_model_count': 3,
        'sub_model_count': 3

    }
}


