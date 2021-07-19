# python3 test_triton.py --model_name POL_FoN --data syraco_data.pickle --input_tensor_name input_1 --output_tensor_name image_predictions/Softmax --save toto.pckl --output_shape 2
import sys
sys.path.insert(0, '/home/amine/model_serving')

from src.triton import utils
from torchvision import datasets, transforms
import torch

if __name__ == '__main__':

	FLAGS = utils.get_flags()
	infer_engine = utils.setup(FLAGS)
	print(type(infer_engine))
	test_kwargs = {'batch_size': 4}
	
	cuda_kwargs = {'num_workers': 1,
					'pin_memory': True,
					'shuffle': True}
	test_kwargs.update(cuda_kwargs)

	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
		])
	test = datasets.MNIST('../data', train=False, download=True,
					   transform=transform)

	test_loader = torch.utils.data.DataLoader(test, **test_kwargs)

	inputs = []
	actual = []
	for batch_idx, (data, target) in enumerate(test_loader):
		inputs.append(data.to("cuda"))
		actual.append(target)
	inputs = [i.cpu().numpy() for i in inputs]
	inputs = inputs[:10]
	predictions = utils.async_infer(infer_engine, inputs)

	print(predictions)

	for i in range(10):
		for j in range(test_kwargs['batch_size']):
			print( "predicted: ",  predictions[i*test_kwargs['batch_size']+j] ,". Actual : " , actual[i][j])