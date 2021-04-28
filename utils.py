
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np


import time
import datetime
import os


# In[ ]:


def Train(net, lr, decay, momentum, device, trainloader, testloader, epoch):
    
    net = net.to(device)

    INITIAL_LR = lr
    DECAY = decay
    MOMENTUM = momentum

    #save_check_point
    CHECKPOINT_PATH = "./saved_model"
    # FLAG for loading the pretrained model
    TRAIN_FROM_SCRATCH = True
    # Code for loading checkpoint and recover epoch id.
    CKPT_PATH = "./saved_model/model-nas.h5"
    def get_checkpoint(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path)
        except Exception as e:
            print(e)
            return None
        return ckpt

    ckpt = get_checkpoint(CKPT_PATH)
    if ckpt is None or TRAIN_FROM_SCRATCH:
        if not TRAIN_FROM_SCRATCH:
            print("Checkpoint not found.")
        print("Training from scratch ...")
        start_epoch = 0
        current_learning_rate = INITIAL_LR
        best_acc = 0
    else:
        print("Successfully loaded checkpoint: %s" %CKPT_PATH)
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch'] + 1
        current_learning_rate = ckpt['lr']
        best_acc = ckpt['best_acc']
        print("Starting from epoch %d " %start_epoch)

    print("Starting from learning rate %f:" %current_learning_rate)
    
    
    START = start_epoch
    EPOCHS = epoch

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.RMSprop(net.parameters(), lr=current_learning_rate, alpha=0.99, eps=1e-08, weight_decay=DECAY, momentum=MOMENTUM)
    optimizer = optim.SGD(net.parameters(), lr=current_learning_rate, weight_decay=DECAY, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS, eta_min=0)
    #Training and Test
    best_val_acc = best_acc
    acc_train = []
    acc_tests = []
    global_step = 0
    
    
    
    for i in range(START, EPOCHS):
        print(datetime.datetime.now())
        # Switch to train mode
        net.train()
        print('Start Training...')
        print("Epoch %d:" %i)

        total_examples = 0
        correct_examples = 0

        train_loss = 0
        train_acc = 0
        
        # Train the training dataset for 1 epoch.
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device)
            targets = targets.long()
            targets = targets.to(device)
         
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs,targets)
            loss.backward()
        
            optimizer.step()

            # Calculate predicted labels
            _, predicted = outputs.max(1)
            total_examples += predicted.size(0)
            correct_examples += predicted.eq(targets).sum().item()
            train_loss += loss
            global_step += 1
                
        avg_loss = train_loss / (batch_idx + 1)
        avg_acc = correct_examples / total_examples
        print("Training loss: %.4f, Training accuracy: %.4f" %(avg_loss, avg_acc))
        print(datetime.datetime.now())
    
        # Validate on the validation dataset
        print("Start Validation...")
        total_examples = 0
        correct_examples = 0
        acc_train.append(avg_acc)
    
        net.eval()

        val_loss = 0
        val_acc = 0
        # Disable gradient during validation
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                # Copy inputs to device
                inputs = inputs.to(device)
                targets = targets.long().to(device)
                # Zero the gradient
                optimizer.zero_grad()
                # Generate output from the DNN.
                outputs = net(inputs)
                loss = criterion(outputs, targets)            
                # Calculate predicted labels
                _, predicted = outputs.max(1)
                total_examples += predicted.size(0)
                correct_examples += predicted.eq(targets).sum().item()
                val_loss += loss

        avg_loss = val_loss / len(testloader)
        avg_acc = correct_examples / total_examples
        acc_tests.append(avg_acc)
        
        print("Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc))
        #schduler step
        scheduler.step()
        for param_group in optimizer.param_groups:
            current_learning_rate = param_group['lr'] 
        print("Current learning rate has decayed to %f" %current_learning_rate)
        
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            print("Saving...")
            torch.save(net.state_dict(), "model_nas.pt")
        
            if not os.path.exists(CHECKPOINT_PATH):
                os.makedirs(CHECKPOINT_PATH)
            print("Saving ...")
            state = {'net': net.state_dict(),
                     'epoch': i,
                     'lr': current_learning_rate,
                     'bst_acc':best_val_acc}
            torch.save(state, os.path.join(CHECKPOINT_PATH, 'model-nas.h5'))
            
    print("Optimization finished.")
    
    return acc_train, acc_tests

