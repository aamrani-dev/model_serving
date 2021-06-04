import pickle
import models.POL_FoN as POL_FoN
import numpy as np
def preprocessing(data): 
	return POL_FoN.preprocessing(data)


def postprocessing(predictions): 
	def format_output(out):
	    response_formated = np.array(out) * 100.
	    sorted_index = np.argsort(response_formated)[::-1]
	    probabilities_sorted = np.sort(response_formated)[::-1]
	    classes_sorted = [species_class_names[index] for index in sorted_index]
	    return probabilities_sorted, classes_sorted, classes_sorted[0]



	species_class_names_path = "/home/amine/Downloads/compress/ray_serving/syraco_classification_serving/file_path_accessible/models/2_model_POL_Species_class_name_20191014.npy" 
	infile = open(species_class_names_path, "rb")
	species_class_names =  pickle.load(infile)
	infile.close()
	
	res = []
	
	for pred in predictions:
		json = {} 
		probabilities_sorted, classes_sorted, predicted_class = format_output(pred)
		json["is_fossil"] = True
		json["predicted_class"] = predicted_class
		json["classes_sorted"] = "classes_sorted"
		json["classes_probabilities"] = "probabilities_sorted"
		res.append(json)
	return res
