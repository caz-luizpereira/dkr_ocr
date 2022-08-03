from flask import Flask, request
import aplicacao as aplicacao
import tensorflow as tf
import base64
from PIL import Image
import io
import numpy as np
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)



###### Tunisia
#@tf.function
def detect_fn(image,detection_model):
        print("ola")
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

@app.route("/", methods=['GET'])
def helloWorld():
    return "pawinhaaa Application"

@app.route("/tunisia", methods=['POST'])
def imageTun():
    start1 = time.time()
    category_index,detection_model = aplicacao.load_modelsTun()
    body = request.get_json()
    while body["Server"] == "up":
        base64Data = body["img"]

        image_64_decode = base64.b64decode(base64Data)
        image = Image.open(io.BytesIO(image_64_decode))
        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        start = time.time()

        detections = detect_fn(input_tensor,detection_model)
   
        #print(base64Data)
        result = aplicacao.mainTun(base64Data,detections)#(base64Data,category_index,detection_model)#,category_index_desc,detection_model_desc)
        #print(result)
        a = str(result)
        #a = a.rstrip('\n')
      
        return str(a)
        
@app.route("/geral", methods=['POST'])
def imageGeral():
    
    category_index,detection_model = aplicacao.load_models()
    body = request.get_json()
    while body["Server"] == "up":
        base64Data = body["img"]
        
        image_64_decode = base64.b64decode(base64Data)
        image = Image.open(io.BytesIO(image_64_decode))
        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        start = time.time()
        
        detections = detect_fn(input_tensor,detection_model)
   
        
      
        result = aplicacao.mainGeral(base64Data,detections)#(base64Data,category_index,detection_model)#,category_index_desc,detection_model_desc)
      
        a = str(result)     
        
        
        return str(a)
      
# Run the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10002)