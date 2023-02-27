import cv2
import numpy as np 
from tensorflow import keras
import matplotlib.pyplot as plt

def segment_image():
    img=cv2.imread('./Image/image.jpg', 0)
    img = cv2.GaussianBlur(img, (9, 9), 0)

    # ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    # ret,img=cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img=cv2.resize(img,(512,512))
    model = keras.models.load_model('./unet3.h5')
    img= np.expand_dims(img,axis=-1)
    img=np.expand_dims(img,axis=0)
    pred=model.predict(img)
    pred=np.squeeze(np.squeeze(pred,axis=0),axis=-1)
    plt.imsave('./Image/img_mask.JPG',pred)

