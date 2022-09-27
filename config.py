import logging

import yaml

from common.singleton import SingletonMeta


class Config(metaclass=SingletonMeta):
    _port: int
    _max_workers: int
    _service_host: str

    def parse_config(self, config_file: str):
        with open(config_file, 'r') as f:
            loaded_cfg = yaml.safe_load(f)
            self._port = loaded_cfg['server']['port']
            self._max_workers = loaded_cfg['server']['max_workers']
            self._service_host = loaded_cfg['service']['host']


    def get_port(self) -> int:
        return self._port

    def get_max_workers(self) -> int:
        return self._max_workers

    def get_service_host(self) -> str:
        return self._service_host


def load_config(config_file: str):
    Config().parse_config(config_file)


if __name__ == '__main__':
    load_config('config.yml')
    cfg = Config()
    logging.info(cfg.get_port(), cfg.get_service_host())
