import requests

resp = requests.post(
    # "http://localhost:5000/",
    # "https://pytorch-model-flask.herokuapp.com/predict",
    "https://mnist-pytorch-model.herokuapp.com/",
    # files={'image_file': open('app/test/images/eight.png', 'rb')}
    # files={'image_file': open('app/test/images/five.png', 'rb')}
    files={'image_file': open('app/test/images/four.png', 'rb')}
    # files={'image_file': open('app/test/images/nine.png', 'rb')}
)

print(resp.text)
