import yaml
import pickle
import json
from tqdm.auto import tqdm  # отображает индикатор прогресса
from timeit import default_timer as timer
import torch
from torch import nn
from typing import Dict, List, Tuple
import os
import pandas as pd
import numpy as np
import onnx 
import onnxruntime as ort

# настройка среды выполнения
device = "cuda" if torch.cuda.is_available() else "cpu"

# загрузка параметров
params = yaml.safe_load(open("params.yaml"))["train_and_eval"]

learning_rate = params["learning_rate"]
epochs = params["epochs"]
seed = params["seed"]

#установка начального значения для воспроизводимости
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

    #замер времени выполнения эпох
    train_time_per_epoch = 0 
    # режим обучения
    model.train()

    # Начальные значения для функции потерь и точности
    train_loss, train_acc = 0, 0

    # Цикл по всем пакетам(выборкам)
    for batch, (X, y) in enumerate(dataloader):
        # Перемещение данных на  девайс
        X, y = X.to(device), y.to(device)

        start_time = timer()

        y_pred = model(X)
        # Подсчет потерь
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        # Обнуление всех градиенты для переменных для обновления(веса модели)
        optimizer.zero_grad()
        # Вычисление градиентов потерь относительно параметров модели
        loss.backward()
        # функция оптимизатора для обновления параметров
        optimizer.step()

        end_time = timer()
        train_time_per_epoch += (end_time - start_time)

        # Расчет и сложение метрик
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Получение средних значений метрик
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc, train_time_per_epoch

# загрузка модели
model = torch.load('model.pt')
# перемещение на девайс
model.to(device)

#функция потерь и оптимизатор
loss_fn = nn.CrossEntropyLoss() #кросс-энтропия, с отрицательными логарифмами вероятности
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #метод стохастической оптимизации, lr-скорость обучения(по умолчанию)

# для подсчета времени
train_time = 0

epochs_list = []
train_acc_list = []
train_loss_list = []
epoch_num = 0

train_data_loader = 'train_data_loader.pkl'
test_data_loader = 'test_data_loader.pkl'
# для сохранения метрик
metrics = 'metrics.json'

# получение путей к данным
with open(train_data_loader, 'rb') as f:
    train_dataloader = pickle.load(f)

with open(test_data_loader, 'rb') as f:
    test_dataloader = pickle.load(f)


# обучение
for epoch in tqdm(range(epochs)):
    train_loss, train_acc, train_time_per_epoch = train_step(model=model,
                                                  dataloader=train_dataloader,
                                                  loss_fn=loss_fn,
                                                  optimizer=optimizer,
                                                  device=device)
    train_time += train_time_per_epoch

    epoch_num +=1
    # Вывод данных об обучении
    print(f"Epoch: {epoch + 1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"train_time: {train_time_per_epoch:.3f} | ")
  
    # Добавление в результаты
    epochs_list.append(epoch_num)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    
train_time_per_epoch_av = train_time/epochs

# вывод информации  времени обучения
print(f"[INFO] Total training time: {train_time:.3f} seconds")

sample_input = torch.randn(32, 3, 224, 224).to(device)

model.eval()

torch.onnx.export(
    model,                  
    sample_input,                    
    "onnx_model.onnx", 
    verbose = False, 
    export_params=True,        
    opset_version=12,       
    input_names = ['input'],    
    output_names = ['output'])

onnx_model = onnx.load("onnx_model.onnx")
ort_session = ort.InferenceSession("onnx_model.onnx")

correct_preds = 0
total = 0
inference_time = 0
inference_time_per_batch = 0
for data in test_dataloader:
    images, labels = data[0], np.array(data[1]) 
    start_time = timer()
    outputs = ort_session.run(
        None,
        {'input': images.numpy()}
    )
    end_time = timer()
    inference_time += (end_time - start_time)
    res = np.array(np.argmax(outputs, axis=2))[0]
    total += labels.shape[0]
    correct_preds += (res == labels).sum().item()

inference_time_per_batch = inference_time / len(test_dataloader)
inference_accuracy = correct_preds / total

print(f"inference_time: {inference_time:.4f} | "
      f"inference_time_per_batch : {inference_time_per_batch :.4f} | "
      f"inference_accuracy: {inference_accuracy:.3f} | ")

results_dict = {'Epoch': epochs_list, 'train_loss': train_loss_list,
                'train_accuracy': train_acc_list, 'inference accuracy': inference_accuracy } 

df = pd.DataFrame(results_dict)

df.to_csv ('results.csv', index=False )

results = {"train_loss": train_loss, 
           "train_acc":  train_acc, 
           "train_time": round(train_time, 3),
           "train_time_per_epoch": round(train_time_per_epoch_av, 3),
           "inference_accuracy": inference_accuracy,
           "inference time": round(inference_time, 3),
           "inference_time_per_batch": round(inference_time_per_batch, 3)}

with open(metrics, 'w') as f:
    json_metrics = json.dump(results, f, indent=4)
