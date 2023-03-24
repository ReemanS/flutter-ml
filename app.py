from flask import Flask, request, jsonify
from tensor import predict
import base64

app = Flask(__name__)

@app.route('/imgprocess', methods=['POST'])
def process_image():
    data = request.get_json(force=True)
    image_data = data['image']
    imgdata = base64.b64decode(image_data)

    decoded_image = 'image.png'
    with open(decoded_image, 'wb') as file:
        file.write(imgdata)

    output = str(predict(decoded_image))
    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4192)