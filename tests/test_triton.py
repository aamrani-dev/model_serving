import sys
sys.path.insert(0, '/home/amine/model_serving')
import time 

import multiprocessing 
from src.triton import utils



def send_request(data):
    return infer_engine.run_inference(data) 

def toto(requests):
    with multiprocessing.Pool() as pool:
        predictions = pool.map(send_request, requests)    


if __name__ == '__main__':
    
    FLAGS = utils.get_flags()
    infer_engine = utils.setup(FLAGS)

    data = utils.load_data(FLAGS)
    print(data)
    requests = utils.prepare_requests(FLAGS, data)        

    start = time.time()
    fon_predictions = utils.async_infer(infer_engine, requests)
    print(fon_predictions)
    indices = [i for item,i in zip(fon_predictions, range(len(fon_predictions))) if item == "fossil"]

    data = [data[i] for i in indices]

    requests = utils.prepare_requests(FLAGS, data)

    FLAGS.model_name = "POL_Species"
    FLAGS.output_shape = 198
    infer_engine = utils.setup(FLAGS)

    species_predictions = utils.async_infer(infer_engine, requests)

    outputs = []
    j = 0
    for i in range(len(fon_predictions)):
        if i in indices:
            outputs.append(species_predictions[j])
            j += 1
        else:
            outputs.append({"is_fossil": False})
    if FLAGS.save != None:
        utils.save_data(FLAGS, outputs)



    