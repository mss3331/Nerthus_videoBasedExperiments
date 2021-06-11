from __future__ import print_function
from __future__ import division
import torch
import pandas
import helpers
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sklearn.metrics as sk
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import copy
import time
from tqdm import tqdm

def train_model(model, dataloaders, criterion, optimizer,device,model_name,colab_dir,
                num_epochs=25, is_inception=False):
    since = time.time()

    results_dic= {"val_acc_history":[],"train_acc_history":[],
                  "val_loss_history":[],"train_loss_history":[]}
    best_resuls_dic= {"Epoch":[],"val_pred":[],"val_target":[],
                      "val_images":[],"precision_recall_fscore_support":[]}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10,flush=True)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            prediction_list = []
            target_list = []
            image_name_list = []
            count = 0
            # Iterate over data.
            pbar = tqdm(dataloaders[phase], total=len(dataloaders[phase]))
            for inputs, labels, filenames in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                target_list = np.concatenate((target_list,labels.clone().detach().cpu().numpy()))
                image_name_list = np.concatenate((image_name_list,filenames))
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 *loss2
                    else:
                        outputs = model(inputs)
                        # print(outputs.shape, labels.shape)
                        # print(labels)
                        # exit(0)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    prediction_list = np.concatenate((prediction_list, preds.clone().detach().cpu().numpy()))


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix({phase+' Epoch': str(epoch)+"/"+str(num_epochs-1),
                                  'running Loss': running_loss / len(dataloaders[phase].dataset),
                                  'running acc': np.mean(prediction_list==target_list).round(5)#torch.sum(preds == labels.data).item()/inputs.size(0),
                                  })

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
                storeBestResults(best_resuls_dic,epoch,prediction_list,target_list,image_name_list,colab_dir)
                print("Saving best Checkpoint")
                torch.save({
                    'best_epoch_num': epoch,
                    'best_model_wts': best_model_wts,
                    'best_optimizer_wts': best_optimizer_wts,
                    'best_val_acc': best_acc},
                    colab_dir+'/checkpoints/' +model_name+ '.pth')

            storeResults(phase,results_dic,epoch_acc,epoch_loss)
            if phase == 'val':
                print('Best So far {} Acc: {:.4f}'.format(phase, best_acc))

        print()

        helpers.plot_result(num_epochs=epoch+1,results_dic=results_dic, model_name=model_name, colab_dir=colab_dir)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, results_dic

def storeResults(phase,results_dic,epoch_acc,epoch_loss):
    results_dic[phase+"_acc_history"].append(epoch_acc)
    results_dic[phase+"_loss_history"].append(epoch_loss)

def storeBestResults(best_resuls_dic,epoch,prediction_list,target_list,image_name_list,colab_dir):
    best_resuls_dic["Epoch"] = epoch
    best_resuls_dic["val_pred"] = prediction_list
    best_resuls_dic["val_target"] = target_list
    best_resuls_dic["val_images"] = image_name_list
    best_resuls_dic["precision_recall_fscore_support"] = sk.precision_recall_fscore_support(target_list, prediction_list, average='micro')
    target_names = ['class 0', 'class 1', 'class 2', 'class 3']
    best_resuls_dic["summary"] = sk.classification_report(target_list,prediction_list, target_names=target_names,output_dict=True)
    pandas.DataFrame(best_resuls_dic["summary"]).transpose().to_excel(colab_dir+"/results/summary_report.xlsx")
    rows = "prediction_list,target_list,correct,image_name_list".split(",")
    pandas.DataFrame((prediction_list,target_list,prediction_list==target_list,image_name_list),index=rows).transpose().to_excel(colab_dir+"/results/results.xlsx")
    # pandas.DataFrame((best_resuls_dic["precision_recall_fscore_support"])).transpose().to_excel("./results/precision_recall_fscore_support.xlsx")


