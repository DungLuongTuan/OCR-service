import grpc
from concurrent import futures
import time

# import the generated classes
import predict_pb2
import predict_pb2_grpc

# import the original calculator.py
import test_eval

# create a class to define the server functions, derived from
# predict_pb2_grpc.CalculatorServicer
class OCRServicer(predict_pb2_grpc.OCRServicer):
    def Predict(self, request, context):
        response = predict_pb2.response()
        response = test_eval.predict(request)
        return response


# create a gRPC server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

# use the generated function `add_CalculatorServicer_to_server`
# to add the defined class to the server
predict_pb2_grpc.add_OCRServicer_to_server(
        OCRServicer(), server)

# listen on port 50051
print('Starting server. Listening on port 50051.')
server.add_insecure_port('[::]:50051')
server.start()

# since server.start() will not block,
# a sleep-loop is added to keep alive
try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)