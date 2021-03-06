import os 
import sys
sys.path.insert(0, os.getenv("MODEL_SERVING"))
from src.ray.ray_serving import Ray_serving

import ray
from ray import serve 
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
from cv2 import imread,imwrite
from cv2 import cvtColor
from cv2 import COLOR_BGR2RGB
import cv2
import numpy as np 
import operator
import time
from starlette.responses import JSONResponse

# import requests
from scipy.ndimage import zoom


class MNIST(Ray_serving):
	def __init__(self, args):
		self.model = tf.keras.models.load_model(args[0])


	async def preprocessing(self, data): 
		return data

	async def inference(self, data):
		return [self.model.predict(d) for d in data]

	async def postprocessing(self, predictions, requests=None): 
	    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
	               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
	    
	    res = []
	    for prediction in predictions:
		    res.append(JSONResponse({"predictions": [class_names[np.argmax(pred)] for pred in prediction]}))
	    
	    return res 

	async def extract_data(self, requests):
		inputs = []
		try:
			for request in requests:
				data  = await request.json()
				inputs.append(np.array(data["inputs"]))
			return inputs

		except Exception as e: 
			return [{"error": str(e)}]

