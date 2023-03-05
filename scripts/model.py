import yaml
import pickle
import os

import torchvision
from torchvision import datasets
import torch
from torch import nn
from torch.utils.data import DataLoader

params = yaml.safe_load(open("params.yaml"))["model"]

batch_size = params["batch_size"]
seed = params["seed"]

#импорт модели и предобученных на ImageNet весов
weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
model = torchvision.models.mobilenet_v2(weights=weights)

#преобразования изображений, использовавшиеся для обучения сети
auto_transforms = weights.transforms()

# импорт путей к директориям датасета из метафайлов
train_dir = 'train_dir.pkl'
test_dir= 'test_dir.pkl'

with open(train_dir, 'rb') as f:
    train = pickle.load(f)

with open(train_dir, 'rb') as f:
    test = pickle.load(f)

#преобразование изображений из папок в набор данных
train_data = datasets.ImageFolder(train, transform=auto_transforms)
test_data = datasets.ImageFolder(test, transform=auto_transforms)

#cписок имен классов
class_names = train_data.classes

#делает данные итерируемыми, чтобы модель могла изучить отношения между объектами и метками
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=batch_size, # число выборок
                              num_workers=1, #число подпроцессов, используемых для загрузки данных
                              shuffle=True, #перемешивание данных в каждую итерацию обучения
                              pin_memory=True)  #копирование тензоров в выбранный девайс

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=batch_size,
                             num_workers=1,
                             shuffle=False,
                             pin_memory=True)

# cохранение путей к данным
train_data_loader = 'train_data_loader.pkl'
test_data_loader = 'test_data_loader.pkl'

with open(train_data_loader, 'wb') as f:
    pickle.dump(train_dataloader, f)

with open(test_data_loader , 'wb') as f:
    pickle.dump(test_dataloader, f)

#заморозка всех параметров слоев, кроме выходного,
for param in model.features.parameters():
    param.requires_grad = False

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# изменение выходного слоя
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=False), #удаляет связи между слоями нейросети с вероятностью p(чтоб не было переоснащения)
    torch.nn.Linear(in_features=1280, #входная форма не изменяется
                    out_features=len(class_names),
                    bias=True))
# сохранение модели
model.save(os.path.join("drive/My Drive/DVC/", 'model.h5'))
