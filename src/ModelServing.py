import numpy  as np
from abc import ABC, abstractmethod
import sys
sys.path.insert(0, '/home/amine/model_serving')
import argparse
import asyncio
import time 
import multiprocessing 
import pickle
from functools import partial

class ModelServing(ABC):
	def __init__(self):
		self.FLAGS = None
		self.infer_engine = None

	@abstractmethod
	def run_inference(self, data):
		pass

	@abstractmethod
	def get_flags(self):
		pass
		
	def prepare_requests(self, data):
	    '''
	        data: list of inputs

	        returns a list of list of inputs after spliting the inputs using specified batch size

	    '''
	    batch_size = int(self.FLAGS.b)
	    requests_data = [data[i * batch_size:(i + 1) * batch_size] for i in range((len(data) + batch_size - 1) // batch_size )]
	    return requests_data

	def load_data(self):
	    '''
	        reads data from pickle file 
	    '''
	    infile = open(self.FLAGS.data, 'rb')
	    data = pickle.load(infile)
	    infile.close()
	    return data

	def save_data(self, data):
	    '''
	        saves predictions in the specified path in picke format 
	    '''
	    outfile = open(self.FLAGS.save, "wb")
	    pickle.dump(data, outfile)
	    outfile.close()
	
   