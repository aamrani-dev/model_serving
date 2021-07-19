import ray 
from ray import serve 

def launch_server(): 
	ray.shutdown()
	ray.init()
	client = serve.start()
	return client 

def register_model(client, model_name, model, *constructor_params):
	client.create_backend(model_name, model, constructor_params)
	client.create_endpoint(model_name, backend=model_name, route="/"+model_name)

def async_infer(model_name, params):
	@ray.remote
	def send_request(request):
		pred = requests.get("http://127.0.0.1:8000/"+model_name, params= request)
		return pred

	predictions = ray.get([send_request.remote(request) for request in requests])
def shutdown_serve():
	ray.shutdown()



