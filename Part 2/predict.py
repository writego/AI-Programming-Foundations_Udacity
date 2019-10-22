import argparse
import os
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
import json
from collections import OrderedDict
from torchvision import datasets, transforms,models

parser = argparse.ArgumentParser(description='Examples: python predict.py --image test.jpg --K 5 --gpu',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--check_point', type=str, default='save/ckpt.pth',
                    help='checkpointed model')
parser.add_argument('--image', type=str, default='test.jpg',
                    help='input image')
parser.add_argument('--json_file', type=str, default='cat_to_name.json',
                    help='load a JSON file that maps the class values to other category names')
parser.add_argument('--K', type=int, default=5,
                    help=' The number of most likely classes ')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='predict image with GPU')

args = parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image).convert('RGB') 
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    return transform(img)

def predict(checkpoint = args.check_point,
           image = args.image,
           json_file = args.json_file,
           K = args.K,
           gpu = args.gpu):
    ckpt = torch.load(checkpoint)

    if ckpt['name'] == 'vgg11' or 'VGG11':
        model = models.vgg11(pretrained=True)
    elif ckpt['name'] == 'vgg16' or 'VGG16':
        model = models.vgg16(pretrained=True)

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(ckpt['input_size'], ckpt['hidden_units'])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(ckpt['hidden_units'], ckpt['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    model.load_state_dict(ckpt['state_dict'])
    model.class_to_idx = ckpt['class_to_idx']
    
    
    
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    test_image = process_image(image)[np.newaxis,:]
    
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        test_image = test_image.type(torch.cuda.FloatTensor)
        print('Predicting with :',device)
        
    output = model(test_image)
    out,predicted = output.topk(K)
    ps = torch.exp(out)


    ps, idx = list(ps.detach().cpu().numpy()[0]), list(predicted.cpu().numpy()[0]) 
    idx_to_class =dict(zip(model.class_to_idx.values(),model.class_to_idx.keys()))
    flowers_classes = [idx_to_class[cls] for cls in idx]
    flowers = [cat_to_name[cls] for cls in flowers_classes]
    print(dict(zip(flowers, ps)))

if __name__ == '__main__':
    predict()     