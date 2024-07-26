import numpy as np
import tensorflow as tf

from flask import Flask, request
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/kucing', methods=['POST'])
def kucing():

    # Ambil gambar yang dikirim pas request
    image_request = request.files['image']

    # konversi gambar menjadi array
    image_pil = Image.open(image_request)

    # ngerize gambarnya
    expected_size = (90, 90)
    resized_image_pil = image_pil.resize(expected_size)

    #generate array dengan numpy
    image_array = np.array(resized_image_pil)
    rescaled_image_array = image_array/255.
    batched_rescaled_image_array = np.array([rescaled_image_array])
    # print(batched_rescaled_image_array.shape)

    
    #load model
    loaded_model = tf.keras.models.load_model('kucing.h5')
    # print(loaded_model.get_config())
    result = loaded_model.predict(batched_rescaled_image_array)



    return get_formated_predict_result(result)

def get_formated_predict_result(predict_result) :
    class_indices = {'Abyssunian': 0, 'Bengal': 1, 'Birman': 2, 'Bombay': 3, 'British_shorthair': 4,}
    inverted_class_indices = {}

    for key in class_indices:
        class_indices_key = key
        class_indices_value = class_indices[key]

        inverted_class_indices[class_indices_value] = class_indices_key

    procesed_predict_result = predict_result[0]
    maxIndex = 0
    maxValue = 0

    for index in range(len(procesed_predict_result)):
        if procesed_predict_result[index] > maxValue:
            maxValue = procesed_predict_result[index]
            maxIndex = index

    return inverted_class_indices[maxIndex]

if __name__ == '__main__':
    app.run()
