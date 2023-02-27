from flask import Flask,render_template
from tensorflow import keras

# # model = VisionEncoderDecoderModel.from_pretrained("TrOCR")
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten") 
# model = VisionEncoderDecoderModel.from_pretrained("TrOCR")
app = Flask(__name__)


from app import views 