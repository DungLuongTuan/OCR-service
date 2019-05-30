import grpc

# import the generated classes
import predict_pb2
import predict_pb2_grpc

# open a gRPC channel
channel = grpc.insecure_channel('localhost:50051')

# create a stub (client)
stub = predict_pb2_grpc.OCRStub(channel)

# create a valid request message
test_request = {
    "info": {
        "file": "images/image_actual_data.png"
    },
    "data": {
        "fields": {
            "name": {
                "cuttype": "word",
                "images": [
                    "images/image_actual_data.png"
                ]
            },
            "idnumber": {
                "cuttype": "word",
                "images": [
                    "images/image_actual_data.png"
                ]
            },
            "birthday": {
                "cuttype": "word",
                "images": [
                    "images/image_actual_data.png"
                ]
            },
            "residence": {
                "cuttype": "word",
                "images": [
                    "images/image_actual_data.png"
                ]
            }
        }
    }
}
request = predict_pb2.request(info=test_request["info"], data=test_request["data"])

# make the call
response = stub.Predict(request)

# response
print(response)