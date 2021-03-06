# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import predict_pb2 as predict__pb2


class OCRStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Predict = channel.unary_unary(
        '/OCR/Predict',
        request_serializer=predict__pb2.request.SerializeToString,
        response_deserializer=predict__pb2.response.FromString,
        )


class OCRServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Predict(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_OCRServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Predict': grpc.unary_unary_rpc_method_handler(
          servicer.Predict,
          request_deserializer=predict__pb2.request.FromString,
          response_serializer=predict__pb2.response.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'OCR', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
