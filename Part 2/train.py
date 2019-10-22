import argparse
import os
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
from collections import OrderedDict


parser = argparse.ArgumentParser(description='Examples: python train.py --data_dir flowers --save_dir save --gpu',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='data',
                    help='input data dir')
parser.add_argument('--save_dir', type=str, default='save/ckpt.pth',
                    help='checkpointed model dir')
parser.add_argument('--arch', type = str, default = 'vgg11',
                    help='specify the net architecture (VGG11,VGG16)')
parser.add_argument('--print_every', type=int, default=20,
                    help='print frequency')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=2,
                    help='number of epochs')
parser.add_argument('--hidden_units', type=int, default = 1024,
                    help = 'number of hidden units')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='train model with GPU')


args = parser.parse_args()

if not os.path.isdir(args.data_dir):
    raise OSError(f'Directory {args.data_dir} does not exist.')
    
if not os.path.isdir(args.save_dir):
    raise OSError(f'Directory {args.save_dir} does not exist.')

def load_data(path = args.data_dir):
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    data_transforms = {'train':transforms.Compose([transforms.RandomRotation(30),
                                   transforms.RandomResizedCrop(244),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                 'val_test':transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])}

    image_datasets = {'train':datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                'val': datasets.ImageFolder(valid_dir, transform=data_transforms['val_test']),
                'test':datasets.ImageFolder(test_dir, transform=data_transforms['val_test'])}


    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'],batch_size=64,shuffle = True),
               'val': torch.utils.data.DataLoader(image_datasets['val'],batch_size=32),
               'test': torch.utils.data.DataLoader(image_datasets['test'],batch_size=32)}
    return dataloaders,image_datasets
    print("data have been loaded!")
    
def train( dataloader, data_dir = args.data_dir,
                save_dir = args.save_dir,
                arch = args.arch,
                learning_rate = args.learning_rate,
                hidden_units = args.hidden_units,
                epochs=args.epochs,
                print_every=args.print_every,
                gpu =args.gpu):
    classifier = None
    
    if arch == 'vgg11' or 'VGG11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg16' or 'VGG16':
        model = models.vgg16(pretrained=True)
 
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    steps = 0

    device = 'cpu'
    if gpu:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
    # change to device
    model.to(device)
    print('The device is :'+ device)
    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders['train']):
            steps += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
  
            if steps % print_every == 0:
                model.eval()
                
                val_loss = 0
                accuracy = 0
                total = 0
                correct = 0
                for images, labels in dataloaders['val']:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model.forward(images)
                    val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    accuracy = correct/total

                print("Epoch: {}/{}... ".format(e+1, epochs),
                    "Train loss: {:.4f}".format(running_loss/print_every),
                     "Val loss: {:.4f}".format(val_loss/print_every),
                    "Test accuracy: {:.4f}".format(accuracy))

                running_loss = 0
                pass
            pass
        pass
    
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.to('cpu')
	

    checkpoint = {'name': arch,
              'input_size': 25088,
              'output_size': 102,
              'hidden_units': hidden_units,
              'state_dict': model.state_dict(),
              'class_to_idx':model.class_to_idx}
    torch.save(checkpoint,save_dir)
    
    print('Train finished')
    
if __name__ == '__main__':
    dataloaders,image_datasets = load_data()
    train(dataloaders)