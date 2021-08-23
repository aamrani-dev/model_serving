import sys
import os
sys.path.insert(0, os.getenv("MODEL_SERVING"))
import requests
from tensorflow import keras
from tests.MNIST import MNIST
from src.ray.ray_inference_engine import Inference_engine
import ray 
from ray import serve

import requests

PATH_TO_MODEL = "/data/appli_PITSI/users/amrani/model_serving/tests/model_repository/mnist/1/model.savedmodel"
MODEL_NAME = "MNIST"

if __name__ == '__main__':

	engine = Inference_engine(MNIST, MODEL_NAME, PATH_TO_MODEL)

	fashion_mnist = keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

	req = engine.prepare_requests(test_images)

	req = [{"inputs": r.tolist()} for r in req]
	predictions = engine.run_inference(req[:2])

	print(predictions)