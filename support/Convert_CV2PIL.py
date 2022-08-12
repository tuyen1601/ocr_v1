import numpy as np
from PIL import Image

def npf32u8(np_arr):
    # intensity conversion
    if str(np_arr.dtype) != 'uint8':
        np_arr = np_arr.astype(np.float32)
        np_arr -= np.min(np_arr)
        np_arr /= np.max(np_arr)    # normalize the data to 0 - 1
        np_arr = 255 * np_arr       # Now scale by 255
        np_arr = np_arr.astype(np.uint8)
    return np_arr
def opencv2pil(opencv_image):

    opencv_image_rgb = npf32u8(opencv_image) # convert numpy array type float32 to uint8
    pil_image = Image.fromarray(opencv_image_rgb) # convert numpy array to Pillow Image Object
    return pil_image