import numpy 
from ray import serve
from abc import ABC, abstractmethod

class ModelServing(ABC):
	@abstractmethod
	def __init__(self):
		pass 

	@abstractmethod
	async def preprocessing(self, data): 
		pass
	@abstractmethod
	async def inference(self, data): 
		pass
	@abstractmethod
	async def postprocessing(self, data): 
		pass 

	@abstractmethod
	async def extract_data(self, requests):
		pass

	@serve.batch(max_batch_size=8, batch_wait_timeout_s=1)
	async def handler(self, requests):
		try:
			print(len(requests))
			data = await self.extract_data(requests)
			print("========================= extract_data done =====================")
			inference_input = await self.preprocessing(data)
			print("========================= preprocessing done =====================")		
			postprocessing_input = await self.inference(inference_input)
			print("========================= inference done =====================")		
			return await self.postprocessing(postprocessing_input)
		except Exception as e:
			return [{"error": str(e)} for _ in requests]		

	async def __call__(self, request): 
		return await self.handler(request)

