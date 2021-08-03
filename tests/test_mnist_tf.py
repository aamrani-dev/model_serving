# python3 test_triton.py --model_name POL_FoN --data syraco_data.pickle --input_tensor_name input_1 --output_tensor_name image_predictions/Softmax --save toto.pckl --output_shape 2
import sys
import os
sys.path.insert(0, os.getenv("TRITON"))
from src.triton.triton_inference_engine import Inference_engine
from tensorflow import keras
import time

if __name__ == '__main__':

	engine = Inference_engine()

	# engine.setup(FLAGS)

	fashion_mnist = keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

	requests = engine.prepare_requests(test_images)

	start = time.time()
	
	predictions = [engine.run_inference(req) for req in requests]


	predictions = [item for batch in predictions for item in batch]
	end = time.time()

	print(end-start)

	for i in range(len(predictions)):
		print( "predicted: ",  predictions[i] ,". Actual : " , test_labels[i])