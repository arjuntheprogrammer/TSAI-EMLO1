import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from app.nn.neural_net import NeuralNet
from app.nn.constants import *

# Device configuration
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    DEVICE = 'cuda'

model = NeuralNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def transform_image(file_path):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    image = Image.open(file_path, mode='r')
    # image.show()
    return transform(image).unsqueeze(0)


def get_prediction(image_tensor):
    image = image_tensor.reshape(1, 1, 28, 28)
    logits, probas = model(image)
    _, predicted_labels = torch.max(probas, 1)
    return predicted_labels[0]
