# Preprocessing and postprocessing methods

To serve the model, we need to provide two methods: preorpcessing and postprocessing. Indeed, the inference goes through three steps: preoprocessing, inference and postprocessing. 

These two methods must be defined in a file having as the model. This file must be saved inside the model folder in the model repository. 

Bellow is an example of a model's folder architecture: 

![pre_post_processing.png](img/pre_post_processing.png)

**mnist.py:**
    


```python
import numpy as np 


def preprocessing(data): 
	return data 

def postprocessing(predictions): 
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    res = [class_names[np.argmax(pred)] for pred in predictions]
    return res 

```
