# TSAI - Session 2

Rest APIs and Flask for ML Applications

## Project Live Link

<https://mnist-pytorch-model.herokuapp.com/>

Download Sample Images:

<img width="118" alt="five1" src="https://user-images.githubusercontent.com/15984084/135575122-94c65cf4-6ce7-4a08-92d5-f53fd3c447c6.png">
<img width="112" alt="nine" src="https://user-images.githubusercontent.com/15984084/135575123-ab4f966a-5bc2-44f5-b13e-d99471008358.png">
<img width="113" alt="zero" src="https://user-images.githubusercontent.com/15984084/135575124-e2a16715-1a9a-4328-b477-14aa0746aafb.png">
<img width="114" alt="four" src="https://user-images.githubusercontent.com/15984084/135575126-b91e8045-7811-46df-8bab-5d7aed96f24f.png">

---

## ASSIGNMENT

1. Visit the Links:
   - <https://youtu.be/bA7-DEtYCNM>
   - <https://www.python-engineer.com/posts/pytorch-model-deployment-with-flask/>
2. Work out the whole tutorial yourself.
3. Once done, change the model to any resnet18 model, and practice
4. Finally build a simple flask app(with Front-End and Back-End) that allows me to upload a jpeg/jpg file and see the predictions(prediction page- prediction + uploaded image).
   a. You can keep the UI as simple as you want.
5. Once done, share the link to your Heroku App.

---

## COMMANDS

### SETUP

- mkdir pytorch-deploy
- cd pytorch-deploy
- python3 -m venv venv

Activate for Non-Windows:

- . venv/bin/activate

Activate For Windows:

- venv\Scripts\activate

Install Flask and PyTorch

- pip install Flask
- pip install torch torchvision
- pip install matplotlib

### Deploy to Heroku

- pip install gunicorn
- pip freeze > requirements.txt

MAC Install:

- brew tap heroku/brew && brew install heroku

Linux Install:

- sudo snap install --classic heroku

Create App:

- heroku login -i
- heroku create your-app-name

Test your app locally:

- heroku local

Commit to GIT:

- git init
- heroku git:remote -a pytorch-model-flask
- git add .
- git commit -m "initial commit"
- git push heroku master

---

## REFERENCE

1. <https://www.python-engineer.com/posts/pytorch-model-deployment-with-flask/>
2. <https://www.python-engineer.com/courses/pytorchbeginner/13-feedforward-neural-network/>
3. <https://youtu.be/bA7-DEtYCNM>

---
