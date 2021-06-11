import numpy 
from abc import ABC, abstractmethod

class ModelServing(ABC):
	def __init__(self):
		pass 
	@abstractmethod
	def preprocessing(self, data): 
		pass
	@abstractmethod
	def inference(self, data): 
		pass
	@abstractmethod	
	def postprocessing(self, data): 
		pass 

	def run_inference(self, data):
		try:
			inference_input =  self.preprocessing(data)
			postprocessing_input = self.inference(inference_input)
			return self.postprocessing(postprocessing_input)

		except Exception as e:
			return [{"error": str(e)} for _ in data]
