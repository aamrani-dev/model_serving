import pickle 
import glob

files_names = glob.glob('/home/amine/Desktop/compress/ray_serving/syraco_classification_serving/file_path_accessible/data/Fossil/*')

file = open("syraco_data.pickle", 'wb')

pickle.dump(files_names, file)

file.close()
