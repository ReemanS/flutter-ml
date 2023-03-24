import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# Method to process the image and return the variable in a format that the model can understand
def import_to_array(img_path):
    img = Image.open(img_path).convert('L')
    img = ImageOps.exif_transpose(img)

    brightness = ImageEnhance.Brightness
    contrast = ImageEnhance.Contrast
    img = brightness(img).enhance(2.0)
    img = contrast(img).enhance(2.0)
    opened_img = img
    opened_img.save('1opened.png')
    img = img.resize((28, 28))
    cropped_img = img
    cropped_img.save('2cropped.png')
    img_array = np.array(img)
    img_array = np.invert(img_array)
    img_array = (img_array/255.0).astype('float32')
    img_array = img_array.reshape(1, 784)
    return img_array

# Load the model
interpreter = tf.lite.Interpreter(model_path="number_classification_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Apply the model to the image
def predict(img_path):
    img_array = import_to_array(img_path)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    return predicted_class

# Execute the model
# input_img = 'test7.png'
# print(predict(input_img))