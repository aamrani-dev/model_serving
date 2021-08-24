# Python API 

We offer a Python API to easily use Triton Inference Server and Ray Serve. By using this API, you can load your data from a pickle file, create a Triton inference engine, prepare your requests and perform asynchronous inferences.

* **load_data**:  read data from a pickle file. 
* **setup**: read and parse received FLAGS
* **prepare_requests**: split input data into chunks of size (batch_size)
* **async_infer**: perform asynchronous inferences
* **save_data**: save predictions to a pickle file 

