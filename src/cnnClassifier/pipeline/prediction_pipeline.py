import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class Prediction:

    def __init__(self, model_path):
        self.model = load_model(model_path)

    
    def predict(self, filename: str):
        test_image = image.load_img(filename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(self.model.predict(test_image), axis=1)

        prediction = 'Adenocarcinoma Cancer'
        if result[0] == 1:
            prediction = 'Normal'
        
        return prediction
