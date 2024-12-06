import tqdm
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from resnet50 import ResNetModel
from dataset import SoundDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference(model, test_dl, criterion):
    test_loss = 0
    correct_prediction = 0
    total_prediction = 0
    
    model.eval()
    for i, data in tqdm(enumerate(test_dl)):
        inputs, labels = data[0].to(device), data[1].to(device)
         # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s
        
        outputs = model(inputs)
        _, prediction = torch.max(outputs,1)
        
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        test_loss += loss.item()
    
    num_batches = len(test_dl)
    avg_loss = test_loss / num_batches
    acc = correct_prediction/total_prediction
    
    return acc, avg_loss


def training(model, train_dl, test_dl, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        corret_prediction = 0
        total_prediction = 0
        
        model.train()
        for i, data in tqdm(enumerate(train_dl)):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, prediction = torch.max(outputs, 1)
            
            corret_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
        
        num_batches = len(train_dl)
        avg_loss = train_loss / num_batches
        avg_acc = corret_prediction/total_prediction
        train_accuracies.append(avg_acc)
        train_losses.append(avg_loss)

        test_acc, test_loss = inference(model, test_dl, criterion)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        
    return train_accuracies, train_losses, test_accuracies, test_losses

if __name__ == "__main__":
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("train.csv")  
    data_path = ""
    
    model = ResNetModel("resnet50", "mlp")
    train_ds = SoundDataset(df_train, data_path)
    test_ds = SoundDataset(df_test, data_path)
    
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    training(model, train_dl, test_dl, 100)