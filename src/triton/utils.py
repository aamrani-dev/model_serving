import sys
sys.path.insert(0, '/home/amine/model_serving')

import argparse
from src.triton import triton_inference_engine
import asyncio
import time 
import multiprocessing 
import pickle
from functools import partial
from multiprocessing.pool import ThreadPool



def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
        '--verbose',
        action="store_true",
        required=False,
        default=False,
        help='Enable verbose output')
    parser.add_argument('-u',
        '--url',
        type=str,
        required=False,
        default='localhost:8000',
        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-s',
        '--ssl',
        action="store_true",
        required=False,
        default=False,
        help='Enable encrypted link to the server using HTTPS')
    parser.add_argument(
        '-H',
        dest='http_headers',
        metavar="HTTP_HEADER",
        required=False,
        action='append',
        help='HTTP headers to add to inference server requests. ' +
        'Format is -H"Header:Value".')
    parser.add_argument(
        '--request-compression-algorithm',
        type=str,
        required=False,
        default=None,
        help=
        'The compression algorithm to be used when sending request body to server. Default is None.'
    )
    parser.add_argument(
        '--response-compression-algorithm',
        type=str,
        required=False,
        default=None,
        help=
        'The compression algorithm to be used when receiving response body from server. Default is None.'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        default=None,
        help=
        'model name'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        default=None,
        help=
        'Pickle file containing a list of inputs'
    )
    parser.add_argument(
        '--b',
        type=str,
        required=False,
        default=4,
        help=
        'batch size to use. Default: 4'
    )
    parser.add_argument(
        '--input_tensor_name',
        type=str,
        required=True,
        default=None,
        help=
        'input tensor name'
    )
    parser.add_argument(
        '--output_tensor_name',
        type=str,
        required=True,
        default=None,
        help=
        'output tensor name'
    )
    parser.add_argument(
        '--save',
        type=str,
        required=False,
        default=None,
        help=
        'Path where predictions will be saved'
    )
    parser.add_argument(
        '--output_shape',
        type=str,
        required=True,
        default=None,
        help=
        'Output tensor shape'
    )    
    FLAGS = parser.parse_args()

    return FLAGS

def setup(FLAGS):
    infer_engine = triton_inference_engine.Inference_engine(FLAGS)

    return infer_engine

def prepare_requests(FLAGS, data):
    batch_size = FLAGS.b
    requests_data = [data[i * batch_size:(i + 1) * batch_size] for i in range((len(data) + batch_size - 1) // batch_size )]
    return requests_data

def load_data(FLAGS):
    infile = open(FLAGS.data, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data

def save_data(FLAGS, data):
    outfile = open(FLAGS.save, "wb")
    pickle.dump(data, outfile)
    outfile.close()


infer_engine = None
def send_request(data):
    return infer_engine.run_inference(data) 

def async_infer(engine, requests):
    global infer_engine 
    infer_engine = engine
    
    # infer_engine = [infer_engine for i in range(len(requests))]
    with multiprocessing.Pool() as pool:
        predictions = pool.map(send_request, requests)
    predictions = [pred for batch in predictions for pred in batch]
    return predictions