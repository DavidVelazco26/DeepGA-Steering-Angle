import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt

def entrenamiento(device, cnn, max_epochs, loss_func, optimizer, train_dl):
  cnn.train() #Configurando la red en modo de entrenamiento
  '''Iterando sobre las épocas de entrenamiento (Linea 1)'''
  #torch.set_default_tensor_type('torch.DoubleTensor')
  torch.set_default_dtype(torch.float32)
  #torch.set_default_dtype(torch.float64)
  for epoch in range(max_epochs):

    '''Iterando sobre los batches (si el tamaño de batch es de 100, solo habrá un batch) (Linea 2)'''
    for i, data in enumerate(train_dl, 0):

      '''Leyendo datos del batch (Linea 3 y Linea 4)'''
      xb, yb = data['image'], data['label'] #Leyendo entrada y etiqueta
      #xb = xb.to(torch.float32)
      xb = xb.type(torch.DoubleTensor).to(device, dtype = torch.float32) #Moviendo entrada a GPU
      yb = yb.to(device, dtype = torch.float32) #Moviendo etiqueta a GPU

      '''Obteniendo salida de la CNN (Linea 5)'''
      y_pred = cnn(xb)

      '''Obteniendo valor de la función de costo (RMSE) (Linea 6)'''
      loss = loss_func(torch.transpose(y_pred, 0, 1), yb.unsqueeze(0))

      '''Obteniendo gradiente y hacer paso hacia atras (Linea 7)'''
      loss.backward() #Paso hacia atrás (derivadas)
      optimizer.step() #Ajustando pasos con optimizador
      optimizer.zero_grad(set_to_none = True) #Reiniciando gradientes a 0

  return loss.item(), cnn #Devuelve la perdida final y la cnn entrenada

def validacion(device, cnn, test_dl):
  '''Inicializar perdida (Linea 1)'''
  cnn.eval() #Configurando la red en modo evaluacion

  '''Iterando sobre los batches (si el tamaño de batch es de 100, solo habrá un batch) (Linea 2)'''
  for i, data in enumerate(test_dl, 0):
    '''Leyendo datos del batch (Linea 3 y Linea 4)'''
    xb, yb = data['image'], data['label'] #Leyendo entrada y etiqueta
    xb = xb.type(torch.DoubleTensor).to(device, dtype = torch.float32) #Moviendo entrada a GPU
    yb = yb.to(device, dtype = torch.float32) #Moviendo etiqueta a GPU

    '''Obteniendo salida de la CNN (Linea 5)'''
    y_pred = cnn(xb).detach().cpu().numpy().reshape(10,)
    y = yb.detach().cpu().numpy()
    print('y = ',y)
    print('y_pred = ',y_pred)
  SMAPE = np.mean(np.abs((y_pred+1) - (y + 1))/(np.abs(y_pred+1) + np.abs(y+1)))
  MAPE = np.mean(np.abs((y_pred+1) - (y + 1))/(y+1))
  return SMAPE, MAPE
def training(device, model, n_epochs, loss_func, train_dl, test_dl, lr, w, max_params):
    #Number of parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.to(device)

    #Optimizer
    opt = optim.Adam(model.parameters(), lr = lr)

    #Obtaining training accuracy
    mse, cnn = entrenamiento(device, model, n_epochs, loss_func, opt, train_dl)

    SMAPE, MAPE = validacion(device, cnn, test_dl)

    #Fitness function based on accuracy and No. of parameters
    f = (1 - w)*(MAPE) - w*((max_params - params)/max_params)

    #Append results to multiprocessing list
    return MAPE, SMAPE, params, mse, f