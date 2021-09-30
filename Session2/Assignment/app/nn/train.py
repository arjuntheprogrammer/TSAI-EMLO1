import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from neural_net import NeuralNet
from constants import *


# Device configuration
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    DEVICE = 'cuda'

transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]
)

# MNIST dataset
# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='data',
                               train=True,
                               transform=transform,
                               download=True)

test_dataset = datasets.MNIST(root='data',
                              train=False,
                              transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

torch.manual_seed(RANDOM_SEED)
model = NeuralNet()
model.to(DEVICE)


def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            # FORWARD AND BACK PROP
            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()

            cost.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            if not batch_idx % 50:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                      % (epoch+1, NUM_EPOCHS, batch_idx,
                         len(train_loader), cost))

        model.eval()
        with torch.set_grad_enabled(False):  # save memory during inference
            print('Epoch: %03d/%03d | Train: %.3f%%' % (
                epoch+1, NUM_EPOCHS,
                compute_accuracy(model, train_loader, device=DEVICE)))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    # Save Model
    torch.save(model.state_dict(), MODEL_PATH)


def evaluate():
    with torch.set_grad_enabled(False):  # save memory during inference
        print('Test accuracy: %.2f%%' %
              (compute_accuracy(model, test_loader, device=DEVICE)))

    for batch_idx, (features, targets) in enumerate(test_loader):
        features = features
        targets = targets
        break

    model.eval()
    logits, probas = model(features.to(DEVICE)[0, None])
    print('Probability 7 %.2f%%' % (probas[0][7]*100))


train()
evaluate()
