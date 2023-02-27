from app import app 
# from .unet import unet
from app import keras
from flask import request,redirect,jsonify,render_template
from .segment import segment_image
from .prediction import prediction
from PIL import Image
import numpy as np

@app.route("/", methods=["GET"])
def index():
     return render_template('index.html')

@app.route("/submit", methods=["POST"])
def upload_picture():

    file = request.files["image"]
    img = Image.open(file.stream)
    img.save('./app/static/image.jpg')
    img.save("./Image/image.jpg")
    return render_template("index.html", success="Image Posted sucessfully")

@app.route("/predict",methods=['GET'])
def segment():  
    predicted_array = [1,2,3,4] 
    segment_image()
    predicted_array = prediction()
    # array = np.array(predicted_array)
    # return(f"{predicted_array}")
    predicted_array.reverse()
    predicted_sentence = " ".join(predicted_array)
    return render_template("index.html", prediction=predicted_sentence)