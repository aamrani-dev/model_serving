import sys
import os 

MODEL_RESPOSITORY = os.getenv('MODEL_REPOSITORY')
print(MODEL_RESPOSITORY)

import sys
sys.path.insert(0, os.getenv("MODEL_SERVING"))

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
from concurrent.futures import ThreadPoolExecutor

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
				ray.init(address="auto", namespace="serve")
				self.client = serve.connect()
			else:
				ray.init()
				self.client = serve.start()
			try:
				backend_config = serve.BackendConfig(num_replicas=self.FLAGS.num_replicas)
				self.client.create_backend(backend_name, deployed_class, args, config=backend_config)
				self.client.create_endpoint(backend_name, backend=backend_name, route="/"+backend_name, methods=["POST"])
			except Exception as e:
				print(e) 
				print("A model with a same name is already registered. Be sure that there is no conflict")

			# self.handler = 
			
		except Exception as e:
			print("Error: " + str(e))
			sys.exit(1)

	def __send_request(self,inputs):
			r  = requests.post("http://127.0.0.1:8000/"+self.backend_name, json=inputs)
			return r.text

	def run_inference(self, data):
		data = data[:int(3000/self.FLAGS.b)]
		with ThreadPoolExecutor(max_workers = 3) as executor:
			results = executor.map(self.__send_request, data)
		return [r for r in results]

	def get_flags(self):
	    # parse the arguments and return them as FLAGS
	    parser = argparse.ArgumentParser()	        
	    parser.add_argument(
	        '--long_live',
	        type=bool,
	        required=False,
	        default=False,
	        help='Launch Ray in long live version')
	    parser.add_argument(
	        '--b',
	        type=int,
	        required=False,
	        default=4,
	        help=
	        'batch size to use. Default: 4'
	    )	
	    parser.add_argument(
	        '--num_replicas',
	        type=int,
	        required=False,
	        default=4,
	        help=
	        'number of replicas (workers)'
	    )		        
	    self.FLAGS = parser.parse_args()