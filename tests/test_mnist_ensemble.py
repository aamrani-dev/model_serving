# python3 test_triton.py --model_name POL_FoN --data syraco_data.pickle --input_tensor_name input_1 --output_tensor_name image_predictions/Softmax --save toto.pckl --output_shape 2
import sys
import os
sys.path.insert(0, os.getenv("TRITON"))

from src.triton import utils
from tensorflow import keras
from argparse import Namespace
import concurrent
import time 

if __name__ == '__main__':

	FLAGS_tf = utils.get_flags()
	FLAGS_pt = Namespace(**vars(FLAGS_tf))
	FLAGS_pt.model_name = 'mnist_copy'
	
	infer_engine_tf = utils.setup(FLAGS_tf)
	infer_engine_pt = utils.setup(FLAGS_pt)


	fashion_mnist = keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	test_images_tf = test_images.reshape(test_images.shape[0], 28, 28, 1)
	print(len(test_images_tf))
	requests_tf = utils.prepare_requests(FLAGS_tf, test_images_tf)
	
	executor_list = []
	start = time.time()
	with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
	    tf_predictions = executor.submit(utils.async_infer, infer_engine_tf, requests_tf)
	    pt_predictions= executor.submit(utils.async_infer, infer_engine_pt, requests_tf)
	    
	    tf_predictions = tf_predictions.result()
	    pt_predictions = pt_predictions.result()
	end = time.time()

	print("(ensemble) time = ", end - start)

	start = time.time()
	utils.async_infer(infer_engine_tf, requests_tf)
	utils.async_infer(infer_engine_pt, requests_tf)

	end = time.time()

	print("(sequential) time = ", end - start)
	# for process in threads:
	#     process.join()
	# predictions = utils.async_infer(infer_engine_tf, requests)

	# print(predictions)

	# for i in range(len(predictions)):
	# 	print( "predicted: ",  predictions[i] ,". Actual : " , test_labels[i])