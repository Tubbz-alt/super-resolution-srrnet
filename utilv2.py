import torch
import numpy as np 
import gc 
import PIL.Image as Image
import torchvision
import os
import math
import matplotlib.pyplot as plt
import random
import datetime

def conver_to_pilimage(X):
    return torchvision.transforms.ToPILImage()(X)

def conver_to_tensor(X):
    return torchvision.transforms.ToTensor()(X)

def display_in_plot(original, downscaled, upscaled, size=18):
    original_array = np.array(conver_to_pilimage(original[0]))
    downscaled_array = np.array(conver_to_pilimage(downscaled[0]))
    upscaled_array = np.array(conver_to_pilimage(upscaled[0]))

    _, axees = plt.subplots(1, 3, figsize=(size, 3 * size))
    axees[0].set_title('Original Image')
    axees[1].set_title('Downscaled Image')
    axees[2].set_title('Upscaled Image (by SRResNet)')

    '''
    for i in range(3):
        axees[i].set_xlim([0, 1]) 
        axees[i].set_ylim([0, 1])
    '''

    axees[0].imshow(original_array)
    axees[1].imshow(downscaled_array)
    axees[2].imshow(upscaled_array)

    del original_array
    del downscaled_array
    del upscaled_array

def demonstrate(image_path, model):
    original, downscaled = get_tensor(image_path)

    with torch.no_grad():
        upscaled = torch.tensor(model(downscaled))
    
    display_in_plot(original, downscaled, upscaled)

class DataIter():
    def __init__(self, folder, sc1=2, sc2=4):
        self.__folder = folder
        self.__names = list(os.listdir(folder))
        self.__i = 0
        self.__sc1 = sc1
        self.__sc2 = sc2

        random.shuffle(self.__names)

    def __len__(self):
        return len(self.__names)

    def __iter__(self):
        self.__i = 0
        return self

    def __next__(self):
        if self.__i >= len(self): raise StopIteration
        self.__i += 1
        return get_tensor(
                    os.path.join(self.__folder, self.__names[self.__i - 1]),
                    self.__sc1,
                    self.__sc2)

def get_tensor(image_name, sc1=2, sc2=2):
    with Image.open(image_name) as image:
        tensor = conver_to_tensor(image).float().unsqueeze(0)
        tensor = torch.as_tensor(tensor[:,:,::sc1,::sc1])
        del image

    return_tuple = (tensor, torch.tensor(tensor[:,:,::sc2,::sc2]))
    return return_tuple

def save_instance(model, loss_list, depth, block):
    current_dtime = str(datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S"))

    model_name = 'checkpoints/model_{}_d{}_b{}.model'.format(current_dtime, depth, block)
    txt_name = 'checkpoints/logs/log_{}.txt'.format(current_dtime)

    torch.save(model, model_name)

    with open(txt_name, 'a+') as txt:
        txt.write(str(loss_list))
    