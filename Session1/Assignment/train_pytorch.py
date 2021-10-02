# https://towardsdatascience.com/how-to-code-a-simple-neural-network-in-pytorch-for-absolute-beginners-8f5209c50fdd
# https://www.pyimagesearch.com/2021/07/12/intro-to-pytorch-training-your-first-neural-network-using-pytorch/
# https://discuss.pytorch.org/t/pytorch-equivalent-of-keras/29412/2
from torchvision import models, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import pandas as pd

from tqdm import tqdm
import numpy as np
import sys
import os
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)

# dimensions of our images.
img_width, img_height = 150, 150
img_size = 150

top_model_weights_path = 'model.h5'
train_data_dir = os.path.join('data', 'train')
validation_data_dir = os.path.join('data', 'validation')

cats_train_path = os.path.join(train_data_dir, 'cats')
nb_train_samples = 2 * len([name for name in os.listdir(cats_train_path)
                            if os.path.isfile(
                                os.path.join(cats_train_path, name))])

nb_validation_samples = 800
epochs = 5
batch_size = 10

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def predict(model, data_loader):
    global device
    model.eval()
    partial_prediction = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print("output.shape = ", output.shape) # torch.Size([10, 512, 7, 7])
            partial_prediction.extend(output)

    output = torch.stack(partial_prediction, dim=0).cpu().numpy()
    print("output.shape = ", output.shape)
    # (1000, 512, 7, 7)
    # (800, 512, 7, 7)

    return output


def save_bottlebeck_features():
    global mean
    global std

    global use_cuda
    global device

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_images = datasets.ImageFolder(
        root=train_data_dir,
        transform=train_transforms)

    test_images = datasets.ImageFolder(
        root=validation_data_dir,
        transform=test_transforms)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = (dict(shuffle=False, batch_size=batch_size, num_workers=4, pin_memory=True)
                       if use_cuda
                       else dict(shuffle=False, batch_size=batch_size))

    train_loader = DataLoader(train_images, **dataloader_args)
    test_loader = DataLoader(test_images, **dataloader_args)

    train_labels = [list(x) for _, x in train_loader]
    valid_labels = [list(x) for _, x in test_loader]

    full_model = models.vgg16(pretrained=True)
    # print("full_model = ", full_model)

    model_without_head = nn.Sequential(
        *nn.ModuleList([full_model.features, full_model.avgpool]))
    # print("model_without_head = ", model_without_head)

    for param in model_without_head.parameters():
        param.requires_grad = False

    model_without_head.to(device)
    del full_model

    bottleneck_features_train = predict(model_without_head, train_loader)
    bottleneck_features_validation = predict(model_without_head, test_loader)

    np.save(open("bottleneck_features_train.npy", "wb"),
            bottleneck_features_train)

    np.save(
        open("bottleneck_features_validation.npy", "wb"),
        bottleneck_features_validation
    )

    np.save(open("train_labels.npy", "wb"), train_labels)
    np.save(open("valid_labels.npy", "wb"), valid_labels)


def class_wise_accuracy(model, validation_data):
    classes = ['cats', 'dogs']
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    with torch.no_grad():
        for images, labels in validation_data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            predicted = torch.round(outputs)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = int(labels[i])
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("class_correct", class_correct)
    print("class_total", class_total)

    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    return {
        'cat_acc': (class_correct[0]*100) / class_total[0],
        'dog_acc': (class_correct[1]*100) / class_total[1]
    }


def train_model(model, dataset, labels_data, optimizer, criterion):
    global epochs
    since = time.time()
    best_acc = 0.0
    model = model.to(device)

    metrics = {
        "epoch": [],
        "accuracy": [], "loss": [],
        "val_accuracy": [], "val_loss": [],
        "cat_acc": [], "dog_acc": []
    }

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-'*20)
        metrics['epoch'].append(epoch)

        for phase in ['train', 'validation']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels in tqdm(zip(dataset[phase], labels_data[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.round(outputs)
                    x = labels
                    x = x.unsqueeze(1).to(torch.float)

                    loss = criterion(outputs, x)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels)

            epoch_loss = (epoch_loss*100) / \
                (dataset[phase].shape[1] * dataset[phase].shape[0])
            epoch_acc = (epoch_corrects.double()) / \
                (dataset[phase].shape[1] * dataset[phase].shape[0])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                metrics['accuracy'].append(epoch_acc.item())
                metrics['loss'].append(epoch_loss)
            else:
                metrics['val_accuracy'].append(epoch_acc.item())
                metrics['val_loss'].append(epoch_loss)

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc

        class_wise_acc = class_wise_accuracy(model, zip(
            dataset['validation'], labels_data['validation']))

        metrics['cat_acc'].append(class_wise_acc['cat_acc'])
        metrics['dog_acc'].append(class_wise_acc['dog_acc'])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc))

    metrics = pd.DataFrame.from_dict(metrics)
    metrics.to_csv('metrics.csv', index=False)

    return model


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.load(open('train_labels.npy', 'rb'))

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.load(open('valid_labels.npy', 'rb'))

    train_data = torch.from_numpy(np.reshape(
        train_data, (batch_size, -1, 512, 7, 7)))
    train_labels = torch.from_numpy(np.reshape(train_labels, (batch_size, -1)))
    validation_data = torch.from_numpy(np.reshape(
        validation_data, (batch_size, -1, 512, 7, 7)))
    validation_labels = torch.from_numpy(
        np.reshape(validation_labels, (batch_size, -1)))

    print("train_data.shape = ", train_data.shape)
    print("train_labels.shape = ", train_labels.shape)

    print("validation_data.shape = ", validation_data.shape)
    print("validation_labels.shape = ", validation_labels.shape)
    print("\n\n")

    model = nn.Sequential(
        nn.Flatten(start_dim=1),

        nn.Linear(in_features=512*7*7, out_features=1024, bias=True),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(in_features=1024, out_features=256, bias=True),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(in_features=256, out_features=1, bias=True),
        nn.Sigmoid()
    )
    optimizer = optim.RMSprop(
        model.parameters(), lr=0.0001, momentum=0.0, weight_decay=1e-7, centered=True)
    criterion = nn.BCELoss()

    dataset = {'train': train_data, 'validation': validation_data}
    labels_data = {'train': train_labels, 'validation': validation_labels}

    train_model(model, dataset, labels_data, optimizer, criterion)
    torch.save(model, top_model_weights_path)


if __name__ == "__main__":
    global use_cuda
    global device

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    print("Using Cuda : ", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    SEED = 121
    torch.manual_seed(SEED)
    if use_cuda:
        torch.cuda.manual_seed(SEED)

    save_bottlebeck_features()
    train_top_model()
