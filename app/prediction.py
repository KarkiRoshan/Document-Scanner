from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

import torch 
import cv2 
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten") 
model = VisionEncoderDecoderModel.from_pretrained("TrOCR")

def show_image(pathStr):
  img = Image.open(pathStr).convert("RGB")
  return img

def ocr_image(src_img):
  pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
  generated_ids = model.generate(pixel_values)
  return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def ocr_image(src_img):
  pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
  generated_ids = model.generate(pixel_values)
  return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def prediction():
    img = cv2.imread('./Image/img_mask.JPG',0) 
    cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
    ori_img=cv2.imread('./Image/image.jpg')
    ori_img=cv2.resize(ori_img,(512,512))
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    coordinates = []
    for c in contours:
        # get the bounding rect
        # print(c)
        if cv2.contourArea(c) < 100:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(ori_img, (x, y), (x+w,y+h), 255, 1)
        coordinates.append([x,y,(x+w),(y+h)])

    length_of_lines = len(coordinates)
    i_cop = ori_img.copy()
    image = []
    for i in range(length_of_lines):
    
        cropped_image = i_cop[coordinates[i][1]:coordinates[i][3],coordinates[i][0]:coordinates[i][2]]
        image.append(cropped_image)

    text = []
    for i in range(length_of_lines):
        cv2.imwrite('crop_img.png',image[i])
        hw_image = show_image('crop_img.png')
        each_line = ocr_image(hw_image)
        text.append(each_line)
    return text
