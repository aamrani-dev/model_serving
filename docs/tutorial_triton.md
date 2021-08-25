# Tutorial: depoly the MNIST fashion model  using Triton Inference Server

In this tutorial we will deploy the MNIST fashion model build in TensorFlow. Once the model is deployed, we will use the Python API to load data, prepare them and perform inferences. 


First of all, we create a folder named "mnist" that contains our model, as described in the [model repository](http://localhost:8888/notebooks/docs/model_repository.md). 

![pre_post_processing.png](img/pre_post_processing.png)

Inside the folder **1**, we put the model we want to deploy and all the needed files for that model.

The **config.pbtxt** contains the model configuration that we want to use for this model. Please refer to the [model configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md) documentation

![mnist_config.png](img/mnist_config.png)

Second, we create a file in the same folder named "mnist.py", where we will define two methods: preprocessing and postprocessing. 



```python
# mnist.py 

import numpy as np 

def preprocessing(data): 
	return data 

def postprocessing(predictions): 
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    res = [class_names[np.argmax(pred)] for pred in predictions]
    return res 

```

Well, the model is ready to be served. 

Now, we need to setup the Triton Inference Server and run it using the following command: 


```python
! source path_to_project/scripts/env_stack.sh
! triton start
```

Let's verify that the server is running: 


```python
! triton is_alive
```


As the Triton Inference Server is now running and the model is deployed , we can start performing inferences.

Let's create a file that you name as you like. We chose "test_mnist.py"



```python
import sys
import os
sys.path.insert(0, os.getenv("MODEL_SERVING"))
import requests
from tensorflow import keras
from tests.MNIST import MNIST
from src.ray.ray_inference_engine import Inference_engine
import ray 
from ray import serve
import time
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

	predictions = engine.run_inference(req)

	for i in range(len(predictions)):
		print( "predicted: ",  predictions[i] ,". Actual : " , test_labels[i])
```

![results.png](img/results.png)

That's it. Now you are ready to deploy your own model! 
