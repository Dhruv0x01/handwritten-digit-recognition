from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image
import numpy as np

from predict import predict_digit 
from utils.preprocess import preprocess_user_image 

app = Flask(__name__) 

@app.route("/")
def home():          
    return render_template("index.html")  

# Create the route
@app.route("/predict", methods=["POST"]) 
def predict():
    data = request.get_json()   
    image_data = data["image"] 

    # Ensure we remove header safely
    if image_data.startswith("data:image"):
        image_data = image_data.split(",")[1] 
       
    # Decode base64 safely
    image_bytes = base64.b64decode(image_data) 
    
    # Convert to image
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    
    # Preprocess
    processed_image = preprocess_user_image(image)

    prediction = predict_digit(processed_image)
    return jsonify({"prediction": int(prediction)})


if __name__ == "__main__":
    app.run(debug=True)            







