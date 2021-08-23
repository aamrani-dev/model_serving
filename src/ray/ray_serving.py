from abc import ABC, abstractmethod
import sys
sys.path.insert(0, '/home/amine/model_serving')
from src.ModelServing import ModelServing
import ray 
from ray import serve 


class Ray_serving(ABC): 

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
			data = await self.extract_data(requests)
			predictions = await self.inference(data)

			predictions = await self.postprocessing(predictions)
			return predictions
		except Exception as e:
			return [{"error": str(e)} for _ in requests]		

	async def __call__(self, request): 
		return await self.handler(request)
