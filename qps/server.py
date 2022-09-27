import logging
import json
from types import SimpleNamespace
import pandas as pd
from plus_ba import plus_ba_pb2_grpc as pb2_grpc
from plus_ba import plus_ba_pb2 as pb2
import timeit

from qps.online.feedback import QPSFeedback
from qps.online.inference import QPSInference
from qps.online.model_update import QPSModelUpdate


class PlusBAServer(pb2_grpc.PlusBAServicer):
    def __init__(self):
        super()
        logging.info("qps_cls service is ready")

    def Inference(self, request, context):
        start = timeit.default_timer()
        data = SimpleNamespace(**json.loads(request.input_json))

        output = QPSInference().run(data)
        print('Elapse : ', timeit.default_timer() - start)
        print(output)
        return pb2.Output(output_json=json.dumps(output))

    def Feedback(self, request, context):
        data = SimpleNamespace(**json.loads(request.input_json))
        data.y = SimpleNamespace(**data.y)
        output = QPSFeedback().run(data)
        print(output)
        return pb2.Output(output_json=json.dumps(output))
        # return common.no_future.make_result(QPSFeedback().run, 'QPS Feedback', request, context)

    def ModelUpdate(self, request, context):
        start = timeit.default_timer()
        data = SimpleNamespace(**json.loads(request.input_json))
        output = QPSModelUpdate().run(data)
        print('Elapse : ', timeit.default_timer() - start)
        print(output)

        return pb2.Output(output_json=json.dumps(output))
        # return common.no_future.make_result(QPSModelUpdate().run, 'QPS ModelUpdate', request, context)
             