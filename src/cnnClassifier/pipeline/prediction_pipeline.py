import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class Prediction:

    def __init__(self, model_path, filename: str):
        self.model = load_model(model_path)
        self.filename = filename

    
    def predict(self):
        test_image = image.load_img(self.filename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(self.model.predict(test_image), axis=1)

        prediction = 'Adenocarcinoma Cancer'
        if result[0] == 1:
            prediction = 'Normal'
        
        return [{"image": prediction}]
