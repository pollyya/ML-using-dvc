import yaml
import pickle
import json
from tqdm.auto import tqdm  # отображает индикатор прогресса
from timeit import default_timer as timer
import torch
from torch import nn
from typing import Dict, List, Tuple
import os

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
    # режим обучения
    model.train()

    # Начальные значения для функции потерь и точности
    train_loss, train_acc = 0, 0

    # Цикл по всем пакетам(выборкам)
    for batch, (X, y) in enumerate(dataloader):
        # Перемещение данных на  девайс
        X, y = X.to(device), y.to(device)

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

        # Расчет и сложение метрик
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Получение средних значений метрик
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    # режим оценки
    model.eval()

    # Начальные значения для функции потерь и точности
    test_loss, test_acc = 0, 0

    # включение режима вывода
    with torch.inference_mode():
        # Цикл по всем пакетам(выборкам)
        for batch, (X, y) in enumerate(dataloader):
            # Перенос данных на выбранное устройство
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            # Подсчет потерь
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Расчет и сложение метрик
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Расчет средних значений метрик
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# загрузка модели
model = torch.load(os.path.join("drive/My Drive/DVC/", 'model.h5'))
# перемещение на девайс
model.to(device)

#функция потерь и оптимизатор
loss_fn = nn.CrossEntropyLoss() #кросс-энтропия, с отрицательными логарифмами вероятности
optimizer = torch.optim.Adam(model.parameters(), LR=learning_rate) #метод стохастической оптимизации, lr-скорость обучения(по умолчанию)

# создание пустого словаря результатов
results = {"train_loss": [], "train_acc": [], "train_time":[], "train_time_per_epoch":[],
    "test_loss": [], "test_acc": [], "test_time":[], "test_time_per_epoch":[]}

# для подсчета времени
train_time = 0
test_time = 0

train_data_loader = 'train_data_loader.pkl'
test_data_loader = 'test_data_loader.pkl'
# для сохранения метрик
metrics = 'metrics.json'

# получение путей к данным
with open(train_data_loader, 'rb') as f:
    train_dataloader = pickle.load(f)

with open(test_data_loader, 'rb') as f:
    test_dataloader = pickle.load(f)

# цикл по этапам обучения и тестирования для заданного числа эпох
for epoch in tqdm(range(epochs)):
    start_time = timer()
    train_loss, train_acc = train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device)
    end_time = timer()
    train_time_per_epoch = end_time - start_time
    train_time += train_time_per_epoch

    start_time = timer()
    test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)

    end_time = timer()
    test_time_per_epoch = end_time - start_time
    test_time += test_time_per_epoch
    # Вывод данных об обучении
    print(f"Epoch: {epoch + 1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"train_time: {train_time_per_epoch:.3f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f} | "
          f"test_time: {test_time_per_epoch:.3f}")


    # Добавление в результаты
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["train_time_per_epoch"].append(train_time_per_epoch)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
    results["test_time_per_epoch"].append(test_time_per_epoch)

results["train_time"].append(train_time)
results["test_time"].append(test_time)

# вывод информации  времени обучения
time_total = results["train_time"][0] + results["test_time"][0]
print(f"[INFO] Total training time: {time_total:.3f} seconds")

with open(metrics, 'w') as f:
    json_metrics = json.dump(results, f, indent=4)