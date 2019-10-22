#!/usr/bin/env python
# coding: utf-8

# # 开发 AI 应用
# 
# 未来，AI 算法在日常生活中的应用将越来越广泛。例如，你可能想要在智能手机应用中包含图像分类器。为此，在整个应用架构中，你将使用一个用成百上千个图像训练过的深度学习模型。未来的软件开发很大一部分将是使用这些模型作为应用的常用部分。
# 
# 在此项目中，你将训练一个图像分类器来识别不同的花卉品种。可以想象有这么一款手机应用，当你对着花卉拍摄时，它能够告诉你这朵花的名称。在实际操作中，你会训练此分类器，然后导出它以用在你的应用中。我们将使用[此数据集](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)，其中包含 102 个花卉类别。你可以在下面查看几个示例。 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# 该项目分为多个步骤：
# 
# * 加载和预处理图像数据集
# * 用数据集训练图像分类器
# * 使用训练的分类器预测图像内容
# 
# 我们将指导你完成每一步，你将用 Python 实现这些步骤。
# 
# 完成此项目后，你将拥有一个可以用任何带标签图像的数据集进行训练的应用。你的网络将学习花卉，并成为一个命令行应用。但是，你对新技能的应用取决于你的想象力和构建数据集的精力。例如，想象有一款应用能够拍摄汽车，告诉你汽车的制造商和型号，然后查询关于该汽车的信息。构建你自己的数据集并开发一款新型应用吧。
# 
# 首先，导入你所需的软件包。建议在代码开头导入所有软件包。当你创建此 notebook 时，如果发现你需要导入某个软件包，确保在开头导入该软件包。

# In[1]:


# Imports here
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms,models


# ## 加载数据
# 
# 在此项目中，你将使用 `torchvision` 加载数据（[文档](http://pytorch.org/docs/master/torchvision/transforms.html#)）。数据应该和此 notebook 一起包含在内，否则你可以[在此处下载数据](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz)。数据集分成了三部分：训练集、验证集和测试集。对于训练集，你需要变换数据，例如随机缩放、剪裁和翻转。这样有助于网络泛化，并带来更好的效果。你还需要确保将输入数据的大小调整为 224x224 像素，因为预训练的网络需要这么做。
# 
# 验证集和测试集用于衡量模型对尚未见过的数据的预测效果。对此步骤，你不需要进行任何缩放或旋转变换，但是需要将图像剪裁到合适的大小。
# 
# 对于所有三个数据集，你都需要将均值和标准差标准化到网络期望的结果。均值为 `[0.485, 0.456, 0.406]`，标准差为 `[0.229, 0.224, 0.225]`。这样使得每个颜色通道的值位于 -1 到 1 之间，而不是 0 到 1 之间。

# In[2]:


train_dir = 'flowers/train'
valid_dir = 'flowers/valid'
test_dir = 'flowers/test'


# In[3]:


#定义训练集数据转换器
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

#定义验证集和测试集数据转换器
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

#加载训练集，验证集，测试集图片，并分别按照定义好的转换器处理图片
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

#懒加载生成器，将数据打乱顺序，分批
trainloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True)


# In[4]:


images, labels = next(iter(trainloader))
print(images[0])
print(labels[0])


# In[5]:


def de_process_image(torch_image):
    image = torch_image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing #后面有个分布细节步骤
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    return image

def imshow(torch_image):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()    
    image = de_process_image(torch_image)    
    ax.imshow(image)

imshow(images[0])


# ### 标签映射
# 
# 你还需要加载从类别标签到类别名称的映射。你可以在文件 `cat_to_name.json` 中找到此映射。它是一个 JSON 对象，可以使用 [`json` 模块](https://docs.python.org/2/library/json.html)读取它。这样可以获得一个从整数编码的类别到实际花卉名称的映射字典。

# In[6]:


print(train_data.class_to_idx) 


# In[7]:


import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

print(cat_to_name['62'])


# # 构建和训练分类器
# 
# 数据准备好后，就开始构建和训练分类器了。和往常一样，你应该使用 `torchvision.models` 中的某个预训练模型获取图像特征。使用这些特征构建和训练新的前馈分类器。
# 
# 这部分将由你来完成。如果你想与他人讨论这部分，欢迎与你的同学讨论！你还可以在论坛上提问或在工作时间内咨询我们的课程经理和助教导师。
# 
# 请参阅[审阅标准](https://review.udacity.com/#!/rubrics/1663/view)，了解如何成功地完成此部分。你需要执行以下操作：
# 
# * 加载[预训练的网络](http://pytorch.org/docs/master/torchvision/models.html)（如果你需要一个起点，推荐使用 VGG 网络，它简单易用）
# * 使用 ReLU 激活函数和丢弃定义新的未训练前馈网络作为分类器
# * 使用反向传播训练分类器层，并使用预训练的网络获取特征
# * 跟踪验证集的损失和准确率，以确定最佳超参数
# 
# 我们在下面为你留了一个空的单元格，但是你可以使用多个单元格。建议将问题拆分为更小的部分，并单独运行。检查确保每部分都达到预期效果，然后再完成下个部分。你可能会发现，当你实现每部分时，可能需要回去修改之前的代码，这很正常！
# 
# 训练时，确保仅更新前馈网络的权重。如果一切构建正确的话，验证准确率应该能够超过 70%。确保尝试不同的超参数（学习速率、分类器中的单元、周期等），寻找最佳模型。保存这些超参数并用作项目下个部分的默认值。

# In[8]:


# TODO: Build and train your network
#将创建模型的过程封装
def get_model(n_number):
    model = models.densenet121(pretrained=True) #用pytorch自带的desnet121初始化，pretrained=True表示使用预训练好的参数
    
    for param in model.parameters():
        param.requires_grad = False #将前面所有的参数都锁定
    
    #Imagenet densenet原来是1000分类器，我们创建适合本项目的分类器
    #512是个中间数，你可以自己设置，属于可以调的参数
    #102是你要分类的类别数，比如你如果只是要分猫和狗，就设为2，如果要分10类，就设置为10，你如果要分100类，就设置为100
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, n_number)), 
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(n_number, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    
    return model


model = get_model(512)
criterion = nn.NLLLoss() #损失函数
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0003) #优化方法

model.classifier  #可以先看看你自己搭的最后的分类器是什么样的结构，是不是你想要的


# In[9]:


# TODO: Do validation on the test set
epochs = 2  #需要你自己调的参数
print_every = 40  #这个参数也可以改
steps = 0

# change to cuda
model.to('cuda') #如果本机没有GPU就把这句话注释掉，如果在Lab运行就需要

for e in range(epochs):
    model.train() #训练模式，dropout会打开
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        inputs, labels = inputs.to('cuda'), labels.to('cuda') #如果本机没有GPU就注释掉
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #print("loss:{}".format(loss))
        #print("loss.item():{}".format(loss.item()))
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval() #评估模式，dropout
            valid_loss = 0            
            correct = 0
            total = 0

            with torch.no_grad():
                for valid_images, valid_labels in validloader:
                    valid_images, valid_labels = valid_images.to('cuda'),valid_labels.to('cuda') 
                    
                    valid_outputs = model.forward(valid_images)
                    valid_loss += criterion(valid_outputs, valid_labels).item()

                    valid_outputs = model(valid_images)
                    _, predicted = torch.max(valid_outputs.data, 1)
                    total += valid_labels.size(0)
                    correct += (predicted == valid_labels).sum().item()

            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Valid Accuracy: {:.3f}".format(correct/total))
            
            running_loss = 0
            model.train()


print("finished")


# ## 测试网络
# 
# 建议使用网络在训练或验证过程中从未见过的测试数据测试训练的网络。这样，可以很好地判断模型预测全新图像的效果。用网络预测测试图像，并测量准确率，就像验证过程一样。如果模型训练良好的话，你应该能够达到大约 70% 的准确率。

# In[10]:


# TODO: Do validation on the test set
#用真正的测试集测试模型效果
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to('cuda'),labels.to('cuda') 

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


# ## 保存检查点
# 
# 训练好网络后，保存模型，以便稍后加载它并进行预测。你可能还需要保存其他内容，例如从类别到索引的映射，索引是从某个图像数据集中获取的：`image_datasets['train'].class_to_idx`。你可以将其作为属性附加到模型上，这样稍后推理会更轻松。

# In[18]:


# TODO: Save the checkpoint 
#保存可以调节的参数n_number,和模型的参数状态字典
checkpoint = {'n_number': 512,
              'state_dict': model.state_dict(),
              'cat_to_name':cat_to_name,
              'class_to_idx':train_data.class_to_idx
             }
torch.save(checkpoint, 'checkpoint.pth')


# ## 加载检查点
# 
# 此刻，建议写一个可以加载检查点并重新构建模型的函数。这样的话，你可以回到此项目并继续完善它，而不用重新训练网络。

# In[19]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
#可以将复原过程封装成一个函数
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = get_model(checkpoint['n_number']) #用原来的参数初始化模型
    model.load_state_dict(checkpoint['state_dict']) #将参数状态字典加载到模型中
    model.cat_to_name = checkpoint['cat_to_name']  #目录（也就是文件夹名）到真实的花的类别名的映射
    model.class_to_idx = checkpoint['class_to_idx'] #class名（也就是目录名）到Id的映射

    return model

#恢复检查点
#恢复模型，不仅恢复了模型结构，也包括各种状态参数
model= load_checkpoint('checkpoint.pth')
print(model.class_to_idx)


# # 类别推理
# 
# 现在，你需要写一个使用训练的网络进行推理的函数。即你将向网络中传入一个图像，并预测图像中的花卉类别。写一个叫做 `predict` 的函数，该函数会接受图像和模型，然后返回概率在前 $K$ 的类别及其概率。应该如下所示：

# In[ ]:





# 首先，你需要处理输入图像，使其可以用于你的网络。
# 
# ## 图像处理
# 
# 你需要使用 `PIL` 加载图像（[文档](https://pillow.readthedocs.io/en/latest/reference/Image.html)）。建议写一个函数来处理图像，使图像可以作为模型的输入。该函数应该按照训练的相同方式处理图像。
# 
# 首先，调整图像大小，使最小的边为 256 像素，并保持宽高比。为此，可以使用 [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) 或 [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) 方法。然后，你需要从图像的中心裁剪出 224x224 的部分。
# 
# 图像的颜色通道通常编码为整数 0-255，但是该模型要求值为浮点数 0-1。你需要变换值。使用 Numpy 数组最简单，你可以从 PIL 图像中获取，例如 `np_image = np.array(pil_image)`。
# 
# 和之前一样，网络要求图像按照特定的方式标准化。均值应标准化为 `[0.485, 0.456, 0.406]`，标准差应标准化为 `[0.229, 0.224, 0.225]`。你需要用每个颜色通道减去均值，然后除以标准差。
# 
# 最后，PyTorch 要求颜色通道为第一个维度，但是在 PIL 图像和 Numpy 数组中是第三个维度。你可以使用 [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html)对维度重新排序。颜色通道必须是第一个维度，并保持另外两个维度的顺序。

# In[20]:


# 先封装几个帮助函数
# 首先是图片处理，前面直接用Pytorch框架处理了整个文件夹的图片，现在你想随便预测一张图片，也需要对图片进行相应的处理
#可以参考这篇文章，通俗易懂：https://www.cnblogs.com/way_testlife/archive/2011/04/17/2019013.html

#预处理图片，前面讲过Pytorch中如何预处理图片，增加的处理的步骤，最好是跟前面保持一致
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #下面的步骤都加了注释，理解不了的同学可以把中间结果打印出来看看是啥
    
    from PIL import Image
    #load图片
    im = Image.open(image_path)
    #im.show()
    
    #确定resize的图片大小
    row,col = im.size
    if row<col:
        row,col = 256,256*col/row
    else:
        row,col = 256*row/col,256

    im.thumbnail((row,col))
    
    #裁剪为224*224
    box = ((row-224)/2, (col-224)/2, (row+224)/2, (col+224)/2)
    im = im.crop(box)
    #im.show()
    
    #将图片变为numpy array，并将整型数变为0~1之间浮点数
    np_im = np.array(im)
    #print(np_im)
    #print(np_im.shape)
    np_im.astype(np.float32)
    np_im = np_im/255
    #print(np_im)
    
    #按照前面训练的方式标准化 均值应标准化为 [0.485, 0.456, 0.406]，标准差应标准化为 [0.229, 0.224, 0.225]
    #你需要用每个颜色通道减去均值，然后除以标准差。
    #也可以直接用广播运算，以下是为了方便理解，更具体的步骤
    np_im[:,:,0]=(np_im[:,:,0]-0.485)/0.229
    np_im[:,:,1]=(np_im[:,:,1]-0.456)/0.224
    np_im[:,:,2]=(np_im[:,:,2]-0.406)/0.225
    
    #PyTorch 要求颜色通道为第一个维度，但是在 PIL 图像和 Numpy 数组中是第三个维度。
    #你可以使用 ndarray.transpose对维度重新排序。颜色通道必须是第一个维度，并保持另外两个维度的顺序。
    np_im = np_im.transpose((2,0,1))
    torch_im = torch.from_numpy(np_im)
    return torch_im

    # TODO: Process a PIL image for use in a PyTorch model

torch_im = process_image('flowers/train/1/image_06734.jpg')


# 要检查你的项目，可以使用以下函数来转换 PyTorch 张量并将其显示在  notebook 中。如果 `process_image` 函数可行，用该函数运行输出应该会返回原始图像（但是剪裁掉的部分除外）。

# In[21]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[22]:


#用前面封装的方法来看看图片处理是否正确
#其实就是吧处理的步骤反过来了
imshow(torch_im)


# ## 类别预测
# 
# 可以获得格式正确的图像后 
# 
# 要获得前 $K$ 个值，在张量中使用 [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk)。该函数会返回前 `k` 个概率和对应的类别索引。你需要使用  `class_to_idx`（希望你将其添加到了模型中）将这些索引转换为实际类别标签，或者从用来加载数据的[ `ImageFolder`](https://pytorch.org/docs/master/torchvision/datasets.html?highlight=imagefolder#torchvision.datasets.ImageFolder)进行转换。确保颠倒字典
# 
# 同样，此方法应该接受图像路径和模型检查点，并返回概率和类别。

# In[23]:


model.to('cpu')
model.eval()
model.double()
def predict(torch_im, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    torch_im.resize_(1,3,224,224) #resize成模型能用的结构
    
    output = model(torch_im)
    output = torch.exp(output) 

    probs, idx = output[0].topk(topk)
            
    classes = [key for id in idx for key,value in model.class_to_idx.items() if id==value ]
    
    return probs,classes    
    
probs,classes = predict(process_image('flowers/train/1/image_06734.jpg'),model, topk=5)
print(probs)
print(classes)


# ## 检查运行状况
# 
# 你已经可以使用训练的模型做出预测，现在检查模型的性能如何。即使测试准确率很高，始终有必要检查是否存在明显的错误。使用 `matplotlib` 将前 5 个类别的概率以及输入图像绘制为条形图，应该如下所示：
# 
# <img src='assets/inference_example.png' width=300px>
# 
# 你可以使用 `cat_to_name.json` 文件（应该之前已经在 notebook 中加载该文件）将类别整数编码转换为实际花卉名称。要将 PyTorch 张量显示为图像，请使用定义如下的 `imshow` 函数。

# In[ ]:





# In[24]:


from matplotlib.ticker import FormatStrFormatter

def de_process_image(torch_image):
    image = torch_image.numpy().transpose((1,2,0))    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    return image

def imshow2(image_path):
    torch_im = process_image(image_path)
    img = de_process_image(torch_im)
    
    probs,classes = predict(torch_im, model, topk=5)

    fig, (ax1,ax2) = plt.subplots(figsize=(9,9),nrows=2)
    
    list = [cat_to_name[x] for x in classes]
    
    ax1.imshow(img)
    ax2.set_title(classes[0])

    ax1.axis('off')
    
    ax2.barh(np.arange(5), probs.detach().numpy())
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(list)

    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    
imshow2('flowers/train/1/image_06734.jpg')


# In[ ]:





# In[ ]:




