import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from nn.neural_net import NeuralNet


input_size = 784  # 28x28
hidden_size = 500
num_classes = 10

PATH = "HerokuAppTutorial/nn/mnist_ffn.pth"
model = NeuralNet(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load(PATH))
model.eval()


def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    image = Image.open(io.BytesIO(image_bytes))
    # image.show()

    return transform(image).unsqueeze(0)


def get_prediction(image_tensor):
    images = image_tensor.reshape(-1, 28*28)
    outputs = model(images)
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted
