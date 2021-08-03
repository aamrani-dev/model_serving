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
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

import argparse
import asyncio
import time 
import multiprocessing 
import pickle
from functools import partial


class Inference_engine(ModelServing):
	def __init__(self):
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

		self.model_name = self.FLAGS.model_name
		sys.path.insert(0, MODEL_RESPOSITORY+'/'+self.model_name)

		self.models_utils = importlib.import_module(self.model_name)
		self.input_tensor_name = self.FLAGS.input_tensor_name
		self.output_tensor_name = self.FLAGS.output_tensor_name
		self.output_shape = int(self.FLAGS.output_shape)
		try:
			# We take into account the usage of an SSL communication (for security reasons)
			if self.FLAGS.ssl:
				self.triton_client = httpclient.InferenceServerClient(
					url=self.FLAGS.url,
					verbose=self.FLAGS.verbose,
					ssl=True,
					ssl_context_factory=gevent.ssl._create_unverified_context,
					insecure=True)
			else:
				self.triton_client = httpclient.InferenceServerClient(
					url=self.FLAGS.url, verbose=self.FLAGS.verbose)
			# self.setup()

		except Exception as e:
			print("channel creation failed: " + str(e))
			sys.exit(1)

	def run_inference(self, data):
		try:
			inference_input =  self.preprocessing(data)
			postprocessing_input = self.inference(inference_input)
			return self.postprocessing(postprocessing_input)

		except Exception as e:
			return [{"error": str(e)} for _ in data]

	def preprocessing(self, data): 
		try:
			return self.models_utils.preprocessing(data)
		except Exception as e: 
			print(e)

	def inference(self, data): 
		'''
			data: list of model inputs 
		'''
		data = data.astype(np.float32)
		inputs = []
		try:
			inputs.append(httpclient.InferInput(self.input_tensor_name, list(data.shape), "FP32"))
			inputs[0].set_data_from_numpy(data, binary_data=False)
		except Exception as e: 
			print(e)

		try:
			outputs = []
			outputs.append(httpclient.InferRequestedOutput(self.output_tensor_name, binary_data=False))
			results = self.triton_client.infer(self.model_name,
		        inputs = inputs,
		        outputs=outputs,
		        query_params=None,
		        headers=None
		    )
			results = results.get_response()
			outputs = results["outputs"]
			predictions = outputs[0]["data"]
			predictions  = [predictions[i:i+self.output_shape] for i in range(0, len(predictions), self.output_shape)]  
			return predictions

		except Exception as e: 
			print(e)

	def postprocessing(self, predictions): 
		return self.models_utils.postprocessing(predictions)


	def get_flags(self):
	    # parse the arguments and return them as FLAGS
	    parser = argparse.ArgumentParser()
	    parser.add_argument('-v',
	        '--verbose',
	        action="store_true",
	        required=False,
	        default=False,
	        help='Enable verbose output')
	    parser.add_argument('-u',
	        '--url',
	        type=str,
	        required=False,
	        default='localhost:8000',
	        help='Inference server URL. Default is localhost:8000.')
	    parser.add_argument('-s',
	        '--ssl',
	        action="store_true",
	        required=False,
	        default=False,
	        help='Enable encrypted link to the server using HTTPS')
	    parser.add_argument(
	        '-H',
	        dest='http_headers',
	        metavar="HTTP_HEADER",
	        required=False,
	        action='append',
	        help='HTTP headers to add to inference server requests. ' +
	        'Format is -H"Header:Value".')
	    parser.add_argument(
	        '--request-compression-algorithm',
	        type=str,
	        required=False,
	        default=None,
	        help=
	        'The compression algorithm to be used when sending request body to server. Default is None.'
	    )
	    parser.add_argument(
	        '--response-compression-algorithm',
	        type=str,
	        required=False,
	        default=None,
	        help=
	        'The compression algorithm to be used when receiving response body from server. Default is None.'
	    )
	    parser.add_argument(
	        '--model_name',
	        type=str,
	        required=True,
	        default=None,
	        help=
	        'model name'
	    )
	    parser.add_argument(
	        '--data',
	        type=str,
	        required=False,
	        default=None,
	        help=
	        'Pickle file containing a list of inputs'
	    )
	    parser.add_argument(
	        '--b',
	        type=str,
	        required=False,
	        default=4,
	        help=
	        'batch size to use. Default: 4'
	    )
	    parser.add_argument(
	        '--input_tensor_name',
	        type=str,
	        required=True,
	        default=None,
	        help=
	        'input tensor name'
	    )
	    parser.add_argument(
	        '--output_tensor_name',
	        type=str,
	        required=True,
	        default=None,
	        help=
	        'output tensor name'
	    )
	    parser.add_argument(
	        '--save',
	        type=str,
	        required=False,
	        default=None,
	        help=
	        'Path where predictions will be saved'
	    )
	    parser.add_argument(
	        '--output_shape',
	        type=str,
	        required=True,
	        default=None,
	        help=
	        'Output tensor shape'
	    )    
	    self.FLAGS = parser.parse_args()

	    
	