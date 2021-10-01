from app.torch_utils import get_prediction, transform_image

tensor = transform_image('app/test/images/four.png')
prediction = get_prediction(tensor)
print(prediction)
