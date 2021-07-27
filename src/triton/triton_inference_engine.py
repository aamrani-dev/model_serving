import sys
import os 

MODEL_RESPOSITORY = os.getenv('MODEL_REPOSITORY')
print(MODEL_RESPOSITORY)
import numpy as np 
import time
import importlib

from src.triton.ModelServing import ModelServing
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

class Inference_engine(ModelServing):

	def __init__(self, FLAGS):
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

		self.model_name = FLAGS.model_name
		sys.path.insert(0, MODEL_RESPOSITORY+'/'+self.model_name)

		self.models_utils = importlib.import_module(self.model_name)
		self.input_tensor_name = FLAGS.input_tensor_name
		self.output_tensor_name = FLAGS.output_tensor_name
		self.output_shape = int(FLAGS.output_shape)
		try:
			# We take into account the usage of an SSL communication (for security reasons)
			if FLAGS.ssl:
				self.triton_client = httpclient.InferenceServerClient(
					url=FLAGS.url,
					verbose=FLAGS.verbose,
					ssl=True,
					ssl_context_factory=gevent.ssl._create_unverified_context,
					insecure=True)
			else:
				self.triton_client = httpclient.InferenceServerClient(
					url=FLAGS.url, verbose=FLAGS.verbose)

		except Exception as e:
			print("channel creation failed: " + str(e))
			sys.exit(1)

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