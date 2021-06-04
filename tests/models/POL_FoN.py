from cv2 import imread,imwrite
from cv2 import cvtColor
from cv2 import COLOR_BGR2RGB
import cv2
import numpy as np 
import operator
import time
from scipy.ndimage import zoom


def preprocessing(data): 
	image_size = (224,224,3)
	
	def image_pad(old_image, new_width, new_height, channel=3):
	    # Redim image (w,h,c)

	    # get pixel border
	    def get_edge(image):
	        sizeX=image.shape[0]
	        sizeY=image.shape[1]
	        sizeC=1
	        if len(image.shape)==3:
	            if image.shape[2]==3:
	                sizeC=3
	        else:
	            sizeC = 1
	        border = np.zeros((sizeX * 2 + sizeY * 2,sizeC))
	        border[:sizeY] = image[0, :] # up
	        border[sizeY:sizeY+sizeX] = image[:, sizeY-1] # right
	        border[sizeY + sizeX:2*sizeY + sizeX] = image[sizeX-1, :] # bottom
	        border[2 * sizeY + sizeX:] = image[:, 0]  # left
	        return border
	    edge_old_image=get_edge(old_image)

	    # erase extremum values
	    quantile_delete=25
	    edge_border_intensity=edge_old_image[:,0].astype(np.float) +edge_old_image[:,1].astype(np.float) +edge_old_image[:,2].astype(np.float)
	    id_to_erase_greater=[edge_border_intensity < np.percentile(edge_border_intensity, 100-quantile_delete)]
	    id_to_erase_inf = [edge_border_intensity < np.percentile(edge_border_intensity, quantile_delete)]
	    id_to_erase=id_to_erase_greater + id_to_erase_inf # list of one array of bool
	    edge_old_image_clipped=edge_old_image[id_to_erase[0]]

	    # compute means. if to avoid division by 0 when image is white
	    if edge_old_image_clipped.shape[0]>3:
	        background_color=np.mean(edge_old_image_clipped,axis=0)
	    else:
	        background_color=np.mean(edge_old_image, axis=0)

	    # if too tall
	    if old_image.shape[0] > new_width or old_image.shape[1] > new_height:
	        ratio = min(float(new_width) / float(old_image.shape[0]), float(new_height) / float(old_image.shape[1]))
	        image=zoom(old_image,(ratio,ratio,1),order=1)
	    else:
	        image=old_image
	    shape = image.shape

	    # create black square
	    out_img=np.ones((new_width,new_height,channel))
	    out_img=out_img[:,:]*background_color

	    # put image in black square
	    before_x = int(np.ceil((new_width - shape[0]) / 2.))
	    after_x = before_x+shape[0]
	    before_y = int(np.ceil((new_height - shape[1]) / 2.))
	    after_y = before_y+shape[1]
	    out_img[before_x:after_x,before_y:after_y,:]=image


	    return out_img



	def preprocess(batch_image_path):
	    # pad images
	    batch_pad_images = np.zeros((len(batch_image_path), image_size[0], image_size[1], image_size[2]))
	    for i in range(len(batch_image_path)):
	        this_image_path = batch_image_path[i]
	        X = cv2.imread(this_image_path)
	        if X is not None:
	            X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)            
	            X = image_pad(X, image_size[0], image_size[1], channel=image_size[2]) 
	            batch_pad_images[i, :, :, :] = X
	        else:
	            print("Error this image cannot be read : " + str(this_image_path))
	            exit()
	    # normalise batch
	    batch_pad_images = (batch_pad_images / 255.) * 2. - 1
	    batch_preprocessed_images = batch_pad_images.reshape(
	        (batch_pad_images.shape[0], image_size[0], image_size[1], image_size[2]))
	    return batch_preprocessed_images
	return preprocess(data)


def postprocessing(predictions): 
	def format_output(out):
	    response_formated = np.array(out) * 100.
	    sorted_index = np.argsort(response_formated)[::-1]
	    probabilities_sorted = np.sort(response_formated)[::-1]
	    classes_sorted = [morpho_class_name[index] for index in sorted_index]
	    return probabilities_sorted, classes_sorted, classes_sorted[0]



	res = []

	for pred in predictions:
		print(pred)
		if np.argmax(pred): 
			res.append('fossil')
		else:
			res.append("not fossil")
	return res
