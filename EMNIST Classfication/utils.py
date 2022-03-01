import torch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def acc(model,dataloader,device):
    size = len(dataloader.dataset) # dataset size
    model.eval() 
    correct = 0 
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y .to(device)
            pred = model(X)
            pred = pred.argmax(1)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    acc = correct/size 
    return 100.*acc


def train(dataloader, device, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, device, model, loss_fn):
    loss,acc = get_loss_acc(dataloader,device,model,loss_fn)
    print(f"Test Error: \n Accuracy: {acc:>0.1f}%, Avg loss: {loss:>8f} \n")

    
def get_loss_acc(dataloader, device, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    acc = 100.* correct/size
    return loss,acc

def write_loss_acc(writer,device,epoch,model,loss_fn,**dataloader):
    # print(dataloader)
    loss_dict = {}
    acc_dict = {}
    for catagory in dataloader:
        loss,acc = get_loss_acc(dataloader[catagory],device,model,loss_fn)
        loss_dict[catagory] = loss
        acc_dict[catagory] = acc
    writer.add_scalars('Loss',loss_dict,global_step=epoch)
    writer.add_scalars('Acc',acc_dict,global_step=epoch)
    
def compute_f1(dataloader,device,model):
    model.eval()
    y_pred = []
    y_true = dataloader.dataset.targets.numpy()
    with torch.no_grad():
        for X,y in dataloader:
            pred = model(X.to(device))
            # print(pred)
            pred = torch.max(pred,1)[1]
            y_pred.extend(pred.cpu().numpy())
    print(f"f1 score in micro : {f1_score(y_true,y_pred,average='micro')}")
    # print(f"f1 score in macro : {f1_score(y_true,y_pred,average='macro')}")
    print(f"f1 score in weighted : {f1_score(y_true,y_pred,average='weighted')}")
    return y_true,y_pred

def get_CM(dataloader,device,model,classes,savefile=None):
    y_true,y_pred = compute_f1(dataloader,device,model)
    
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix,index = [i for i in classes],
                         columns = [i for i in classes])
    plt.figure(figsize = (128,128))
    sn.heatmap(df_cm, annot=True)
    if savefile :
        plt.savefig(savefile)
    
    
