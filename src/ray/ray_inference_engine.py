import sys
import os 

MODEL_RESPOSITORY = os.getenv('MODEL_REPOSITORY')
print(MODEL_RESPOSITORY)

import sys
sys.path.insert(0, '/home/amine/model_serving')

import numpy as np 
import time
import importlib

from src.ModelServing import ModelServing

import argparse
import asyncio
import time 
import multiprocessing 
import pickle
from functools import partial

import ray
from ray import serve 

import requests  

class Inference_engine(ModelServing):
	def __init__(self, deployed_class, backend_name,  *args):
		'''
			FLAGS:
				requiered:
					model_name: the name of the model to use for inference
					models_utils:
					input_tensor_name: name of the model input tensor
					out_tensor_name: name of the model output tensor
					output_shape : shape of the model output tensor
					url: the url where requests will be sent 
				optional:
					please refer to the doc for the hole FLAGS options	
		'''
		super()
		self.get_flags()
		self.backend_name = backend_name
		self.client = None
		try:
			if self.FLAGS.long_live:
				ray.init(address='auto')
				self.client = serve.connect()
			else:
				ray.init()
				self.client = serve.start()
			self.client.create_backend(backend_name, deployed_class, args)
			self.client.create_endpoint(backend_name, backend=backend_name, route="/"+backend_name, methods=["POST", "GET"])

		except Exception as e:
			print("Error: " + str(e))
			sys.exit(1)

	def run_inference(self, data):
		@ray.remote
		def send_request(d):
			return requests.post("http://127.0.0.1:8000/"+backend_name, params=d)

		predictions = ray.get([send_request(d) for d in data])
		return predictions

	def get_flags(self):
	    # parse the arguments and return them as FLAGS
	    parser = argparse.ArgumentParser()	        
	    parser.add_argument(
	        '--long_live',
	        type=bool,
	        required=False,
	        default=False,
	        help='Launch Ray in long live version')
	    self.FLAGS = parser.parse_args()