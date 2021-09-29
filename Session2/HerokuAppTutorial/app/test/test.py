import requests

resp = requests.post(
    # "http://localhost:5000/predict",
    "https://pytorch-model-flask.herokuapp.com/predict",
    # files={'file': open('HerokuAppTutorial/test/images/eight.png', 'rb')}
    files={'file': open('app/test/images/eight.png', 'rb')}
    # files={'file': open('app/test/images/five.png', 'rb')}
)

print(resp.text)
