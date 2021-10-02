# TSAI - Session 1

MLOPs - Introduction and Version Control

## ASSIGNMENT

1. Change train.py file to work with PyTorch
2. Repeat whole tutorial and commit to GitHub
3. Mention accuracy of each model(trained twice, each time 10 epochs or more)
4. Publish metrics.csv file
5. Go to GitHUB ACTIONS Tab
   - click Python application
   - to the basic commands, add run: pytest -h
   - add test.py file that checks:
     - data.zip is not uploaded(check where it is expected)
     - models.h5 file is not uploaded
     - accuracy of the model is more than 70%
     - accuracy of your model for cat and dog is independently more than 70% (read from metrics.csv file)
6. Share the GitHub Link.

---

## DVC

Data Version Control, or DVC, is a data and ML experiment management tool that takes advantage of the existing engineering toolset that you're already familiar with (Git, CI/CD, etc.).

---

## COMMANDS

Get GIT Repository:

- git clone https://github.com/iterative/example-versioning.git
- cd example-versioning

### Setup ENVIRONMENT

- python3 -m venv .env
- source .env/bin/activate
- pip install -r requirements.txt

### COMMIT V1.0

- dvc get https://github.com/iterative/dataset-registry tutorials/versioning/data.zip
- unzip -q data.zip
- rm -f data.zip
- dvc add data
- python train.py
- dvc add model.h5
- git add data.dvc model.h5.dvc metrics.csv .gitignore
- git commit -m "First model, trained with 1000 images"
- git tag -a "v1.0" -m "model v1.0, 1000 images"

### COMMIT V2.0

- dvc get https://github.com/iterative/dataset-registry \
   tutorials/versioning/new-labels.zip
- unzip -q new-labels.zip
- rm -f new-labels.zip
- dvc add data
- python train.py
- dvc add model.h5
- git add data.dvc model.h5.dvc metrics.csv
- git commit -m "Second model, trained with 2000 images"
- git tag -a "v2.0" -m "model v2.0, 2000 images"

Switching between workspace versions:

- git checkout v1.0
- dvc checkout

Keep Current Code but previous dataset:

- git checkout v1.0 data.dvc
- dvc checkout data.dvc

### Automating capturing

RESET TO MASTER:

- git checkout master
- dvc checkout
- dvc remove model.h5.dvc

DVC RUN COMMAND:

- dvc run -n train -d train.py -d data \
  -o model.h5 -o bottleneck_features_train.npy \
  -o bottleneck_features_validation.npy -M metrics.csv \
  python train.py
- git add .
- git commit -m "DVC RUN COMMAND"

---
