# Model Serving

Serve your Deep Learning models using Triton Inference Server, introduced by NVIDA. 

This project aims to offer a High-Level tool to the user to serve DL models in the one hand and perform inferences through a Python API in the other hand. 

## Start Triton Inferencer Server


$ triton start+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| <model_name>         | <v>     | READY  |
| ..                   | .       | ..     |
| ..                   | .       | ..     |
+----------------------+---------+--------+
...
...
...
I1002 21:58:57.891440 62 grpc_server.cc:3914] Started GRPCInferenceService at 0.0.0.0:8001
I1002 21:58:57.893177 62 http_server.cc:2717] Started HTTPService at 0.0.0.0:8000
I1002 21:58:57.935518 62 http_server.cc:2736] Started Metrics Service at 0.0.0.0:8002

## Stop Trtion Inference Server
$ triton stop
## Check if the server is running 

$ triton is_aliveTHE SERVER IS NOT RUNNING

```python

```
