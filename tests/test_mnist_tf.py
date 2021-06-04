# python3 test_triton.py --model_name POL_FoN --data syraco_data.pickle --input_tensor_name input_1 --output_tensor_name image_predictions/Softmax --save toto.pckl --output_shape 2
import sys
sys.path.insert(0, '/home/amine/model_serving')

from src.triton import utils
from tensorflow import keras


if __name__ == '__main__':

	FLAGS = utils.get_flags()
	infer_engine = utils.setup(FLAGS)

	fashion_mnist = keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

	test_images = test_images[:10]

	requests = utils.prepare_requests(FLAGS, test_images)

	predictions = utils.async_infer(infer_engine, requests)

	print(predictions)

	for i in range(10):
		print( "predicted: ",  predictions[i] ,". Actual : " , test_labels[i])