import os 
import sys
sys.path.insert(0, '/home/amine/model_serving')
import requests

from tests.MNIST import MNIST
from src.ray.ray_inference_engine import Inference_engine
import ray 
from ray import serve
class Toto():
	def __init__(self):
		self.toz = None
	def __call__(self):
		print("hello")

if __name__ == '__main__':
	ray.init()
	client = serve.start()
	client.create_backend("to", Toto)
	client.create_endpoint("to", "to", "/to")

	# engine = Inference_engine(MNIST, "MNIST","/home/amine/Desktop/examples/model_repository/mnist/1/model.savedmodel")
	# print("heeeeeeeeeeeeeeeeeeeeeeeeeeeellllllllllllloooooooooooooooooooo")

	# # fashion_mnist = keras.datasets.fashion_mnist
	# # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	# # test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

	# # requests = engine.prepare_requests(test_images)

	# # requests = [{"data": request} for request in requests]

	# # predictions = engine.run_inference(requests)

	# # print(predictions)