# 
'''
MLP REPRODUCIBILITY EXPERIMENT

OREGON STATE UNIVERSITY: AI535 PROJECT SPRING 2022

# Run this file to perform the neural net experiments and comparisons automatically.

# See this:
https://pytorch.org/docs/stable/notes/randomness.html


'''
from copy import deepcopy
import glob
import os
import shutil


import itertools, random

import json
  
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, softmax
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda

from scipy.stats import wilcoxon

# import autosgd as metaopt


# import pytorch_lightning as pl
  
# import intel_extension_for_pytorch as ipex

import dashplots

#------global config.----------------
# read/write to config file


# rm -r ./MLP_EXP_*/* 

# read
# with open('config.json', 'r') as cfglist:
#   cfgs = json.load(cfglist)
  
# edit
# cfgs['dataset'] = "fmnist", "mnist", "cifar10", "svhn", "cifar100"
# cfgs['optim'] = 'sgd', 'sgdmom', 'adam',
# cfgs['norms'] = 'layer', 'batch', 'false'
# cfgs['noise'] = 'dropout', 'false'
# cfgs['lnoise'] = 'wdecay', 'false'
# cfgs['residual'] = 'true', 'false'
# cfgs['seed'] = 0

# Compare Optims alone.
# cfgs["dataset"] = "cifar10" #imagenet not used here.
# # cfgs['optim'] = "sgd" # swa not used here.
# cfgs["num_layers"] = 4 # should be even
# cfgs['norms'] = 'layer'
# cfgs['noise'] = 'false'
# cfgs['residual'] = 'true'
# cfgs['runs'] = 1 # should be > 1
# cfgs['epochs'] = 5

# write
# with open('config.json', 'w') as cfglist:
  # json.dump(cfgs, cfglist)

# ---------------------

# Download Experimental Dataset
def exp_dataset(cfgs):
  # Download Dataset
  data_folder = "data"

  if cfgs["dataset"] == "fmnist":
    training_data = datasets.FashionMNIST(
      root=data_folder,
      train=True,
      download=True,
      transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
      root=data_folder,
      train=False,
      download= True,
      transform=ToTensor()
    )
    indim = [28,28]
    class_num = 10
    channels = 1
  elif cfgs["dataset"] == "mnist":
    training_data = datasets.MNIST(
      root=data_folder,
      train=True,
      download=True,
      transform=ToTensor(),
    )
    test_data = datasets.MNIST(
      root=data_folder,
      train=False,
      download= True,
      transform=ToTensor(),
    )
    indim = [28,28]
    class_num = 10
    channels = 1
  elif cfgs["dataset"] == "cifar10":
    training_data = datasets.CIFAR10(
      root=data_folder+"/CIFAR10",
      train=True,
      download=True,
      transform=transforms.Compose([ToTensor()]),
    )
    test_data = datasets.CIFAR10(
      root=data_folder+"/CIFAR10",
      train=False,
      download= True,
      transform=transforms.Compose([ToTensor()])
    )
    indim = [3,32,32]
    class_num = 10
    channels = indim[0]
  elif cfgs["dataset"] == "svhn":
    training_data = datasets.SVHN(
      root=data_folder+"/SVHN",
      split="train",
      download=True,
      transform=ToTensor(),
    )
    test_data = datasets.SVHN(
      root=data_folder+"/SVHN",
      split="test",
      download= True,
      transform=ToTensor()
    )
    indim = [3,32,32]
    class_num = 10
    channels = indim[0]
  elif cfgs["dataset"] == "cifar100":
    training_data = datasets.CIFAR100(
      root=data_folder+"/CIFAR100",
      train=True,
      download=True,
      transform=ToTensor(),
    )
    test_data = datasets.CIFAR100(
      root=data_folder+"/CIFAR100",
      train=False,
      download= True,
      transform=ToTensor()
    )
    indim = [3,32,32]
    class_num = 100
    channels =indim[0]
  else:
    pass
  
  return training_data, test_data, indim, class_num, channels 

# Models
 
# class RemoveChannel(nn.Module):
#   def __init__(self,channel) -> None:
#       super().__init__()
#       # weight for collapsing channel
#       self.skip = False
#       if channel == 1:
#         self.skip = True
#       else:
#         W = torch.empty((channel,1), requires_grad=True) 
#         self.weights = nn.Parameter(W)
        
#         # init weights
#         nn.init.kaiming_normal_(self.weights)
      
#   def forward(self,x):
#     if not self.skip:
#       x = x.reshape(x.shape[0],x.shape[2],x.shape[3], x.shape[1])
#       x = torch.matmul(x, self.weights.squeeze())
#     return x

# submodule
def normlayer(norm_type, norm_dim):
  '''
  Optional Normalization Layer
  norm_type: "layer", "batch", "false"
  norm_dim: integer dim of layer output
  '''
  if norm_type == "layer":
    normblk = nn.LayerNorm(norm_dim)
  elif norm_type == "batch":
    normblk = nn.BatchNorm1d(norm_dim)
  else: # we expect this to be false
    normblk = nn.Identity()
  return normblk
normstruct = lambda norm_type, norm_dim : normlayer(norm_type,norm_dim)

# submodule
def noiselayer(noise_type):
  '''
  Optional Noise Layer
  noise_type: "wdecay", "dropout", "false"
  '''
  if noise_type == "dropout":
    noiseblk = nn.Dropout(p=0.5)
  else: # we expect this to be false or wdecay
    noiseblk = nn.Identity()
  return noiseblk
noisestruct = lambda noise_type : noiselayer(noise_type)

# submodule
class BlkTwoLayerRes(nn.Module):
  '''
  Fully connected: Two-Layer Residual Block 
  '''
  def __init__(self, cfgs, in_dim: int=128, hid_dim: int=256, out_dim: int=128) -> None:
    super().__init__()
    
    norm_type = cfgs['norms']
    noise_type= cfgs['noise']
        
    linear_bias = True
    # set bias of linear layer to False 
    # if a norm_layer is used 
    if norm_type != "false":
      linear_bias = False 
    
    self.twolayer_res = nn.Sequential(
      nn.Linear(in_dim,hid_dim, bias=linear_bias),
      # fcns: linear layer normalization, noise and nonlinearity.
      normstruct(norm_type,hid_dim),
      noisestruct(noise_type),
      nn.ReLU(),
      nn.Linear(hid_dim,out_dim, bias=linear_bias),
    )
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    
    out = self.twolayer_res(x)
    out += x
    return out

# FullyConnected FNN Model
class FNN(nn.Module):
  
  def __init__(self, cfgs, flat_indim, channels=1, hid_dim=128, class_dim=10):
    super(FNN,self).__init__()
    
    in_dim, out_dim = hid_dim, hid_dim
    num_layers = cfgs["num_layers"]
    
    norm_type = cfgs['norms']
    noise_type= cfgs['noise']
    use_residual = cfgs['residual']
    
    linear_bias = True
    # set bias of linear layer to False 
    # if a norm_layer is used 
    if norm_type != "false":
      linear_bias = False 
    
    # self.remchan = RemoveChannel(channels)
    self.flatten = nn.Flatten()
    self.linear_one = nn.Sequential(
      # -- input normalization
      normstruct(norm_type,flat_indim),
      nn.Linear(flat_indim,out_dim, bias=linear_bias),
      # -- add fcns: norms and noise-injection
      # linear layer normalization
      normstruct(norm_type,out_dim),
      noisestruct(noise_type)
    )
    self.linear_others = nn.Sequential(
      nn.Linear(in_dim,out_dim, bias=linear_bias),
      # -- add fcns: norms and noise-injection
      # linear layer normalization
      normstruct(norm_type,out_dim),
      noisestruct(noise_type)
    )
    
    # define the fully connected network
    self.LinearReLU_stack = nn.Sequential()
    
    for id in range(num_layers):
      if id == 0:
        # first layer
        self.LinearReLU_stack.append(self.linear_one)
      elif id == num_layers-1:
        # last layer
        self.LinearReLU_stack.append(nn.ReLU())
        self.LinearReLU_stack.append(nn.Linear(hid_dim,class_dim))
      elif id > 0 and id < num_layers-1:
        # in-between layers
        self.LinearReLU_stack.append(nn.ReLU())
        if use_residual == "false":
          self.LinearReLU_stack.append(self.linear_others)
        else:
          self.LinearReLU_stack.append(BlkTwoLayerRes(cfgs,hid_dim,hid_dim,hid_dim))
      else:
        pass
      
  def forward(self,x):
    # x = self.remchan(x)
    x = self.flatten(x)
    logits = self.LinearReLU_stack(x)
    return logits


# Reset Weights/Parameters in Model
def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    model.apply(fn=weight_reset)

def reset_all_linear_layer_weights(model: nn.Module) -> nn.Module:
    """
    Resets all weights recursively for linear layers.
    """

    @torch.no_grad()
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.fill_(1.0)

    model.apply(init_weights)

def reset_all_weights_with_specific_layer_type(model: nn.Module, modules_type2reset) -> nn.Module:
    """
    Resets all weights recursively for linear layers.
    """

    @torch.no_grad()
    def init_weights(m):
        if type(m) == modules_type2reset:
            # if type(m) == torch.nn.BatchNorm2d:
            #     m.weight.fill_(1.0)
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(init_weights)

# ---------seed worker----------------------------------
def seed_worker(worker_id):
    worker_seed = 0 #torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# define a train and test loop
def train_loop(dataloader, model, loss_fcn, optimizer):
  model.train() # set model in train mode.
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  train_loss, correct = 0, 0
  
  for batch, (X,Y) in enumerate(dataloader):
    X = X.to(device, non_blocking=True)
    Y = Y.to(device, non_blocking=True)
    logit_pred = model(X)
    Yhat = logit_pred.argmax(1)
    loss = loss_fcn(logit_pred,Y)
    train_loss += loss.item()
    correct += (Yhat==Y).type(torch.float).sum().item()
    
    # optimizer.zero_grad(set_to_none=True)
    for param in model.parameters():
      param.grad = None
      
    loss.backward()
    optimizer.step()
    
    # if np.remainder(batch+1,num_batches) == 0:
      # loss, current = loss.item(), batch*len(X)
      # print(f"Step: {batch:>5d}, Loss:{loss:>7f} [{current:>5d}/{size:>5d}]")
      
  print(f"Batches/Steps/Iterations per Epoch: {num_batches:>5d}")
  print(f"Train:\t[ Avg Loss: {train_loss/num_batches:>7f}, Accuracy: {(100*correct/size):>0.1f}% ]", end=' || ')
  
  return train_loss/num_batches, correct/size
      
def test_loop(dataloader, model, loss_fcn):
  model.eval() # to ensure components like dropout is not used in inference.
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0
  preds = []

  with torch.no_grad():
    for X,Y in dataloader:
      X = X.to(device,non_blocking=True)
      Y = Y.to(device,non_blocking=True)
      logit_pred = model(X)
      
      Yhat = logit_pred.argmax(1)
      
      # class_prob = softmax(logit_pred)
      # class = class_prob.argmax(1)
      
      test_loss += loss_fcn(logit_pred,Y).item()
      correct += (Yhat==Y).type(torch.float).sum().item()
      
      preds.append(Yhat.tolist())
    
    # num_of_batches is used here, since, the loss is computed at each batch
    print(f"Test:  [ Avg loss: {test_loss/num_batches:>7f}, Accuracy: {(100*correct/size):>0.1f}% ]")
    
  preds = list(itertools.chain.from_iterable(preds))  
  return test_loss/num_batches, correct/size, preds
    

#--RUN-------------------------------------------------------------------------
# run Main 
def dlnn_main(cfgs, runs:int=2, epochs:int=10, numworkers:int=0):
  
  cfgs['runs'] = runs # should be > 1
  cfgs['epochs'] = epochs # should be > 1
  cfgs["numworkers"] = numworkers # noworkers, set number of workers to 0, irrespective of device
  
  # Experiment Group counter
  exp_cnter = 0
  
  # set seed for technical reproducibility
  # if you change the seed in cfg,
  # also change the worker seed in the fcn: seed_worker() defined above
  seed = cfgs['seed']
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  # pl.seed_everything(seed)

  # torch.use_deterministic_algorithms(True)
  if cfgs["device"] == "cuda":
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    
  # fix seed for batch generation
  gen_seed = torch.Generator()
  gen_seed.manual_seed(seed)
  
  # set dataset to use according to assigned humans running the experiment
  if cfgs["human"] == "oluwasegun":
    data_fullname = ["CIFAR10","CIFAR100"]
    datalist = ["cifar10", "cifar100"] #- oluwasegun
  elif cfgs["human"] == "apurva":
    data_fullname = ["MNIST", "FMNIST"]
    datalist = ["mnist","fmnist"] #- apurva
  elif cfgs["human"] == "nischal":
    data_fullname = ["SVHN"]
    datalist = ["svhn"] # - nischal
  else:
    data_fullname = ["MNIST", "FMNIST", "CIFAR10", "SVHN", "CIFAR100"]
    datalist = ["mnist", "fmnist", "cifar10", "svhn", "cifar100"]
  
  optim_fullname = ["SGD","SGD+Momentum", "Adam"]
  optimlist = ["sgd","sgdmom","adam"]
   
  layerlist = [2,6,20]
  residuallist = ["false", "true"]
  normlist = ["false","layer","batch"] 
  lnoiselist = ["false","wdecay"]

  cfgs["noise"] = "false" # noiselist = ["false","dropout"]
 
   
  # -Specify main hyperparameters:
  # number of epochs, batch size, learning rate.
  batch_size = 128
  
  for ood, data_name in enumerate(datalist):
    cfgs["dataset"] = data_name
    training_data, test_data, indim, class_num, channels = exp_dataset(cfgs)
  
    # sgdlr = 1e-3 #general
    if cfgs["dataset"] == "mnist":
      sgdlr = 1e-1 # mnist
      sgdmlr = 1e-2 #mnist
      adamlr = 3e-4 # mnist
      epochs = cfgs['epochs']
    if cfgs["dataset"] == "fmnist":
      sgdlr = 1e-1 # fmnist
      sgdmlr = 1e-2 #fmnist
      adamlr = 1e-3 # fmnist
      epochs = cfgs['epochs']
    if cfgs["dataset"] == "svhn":
      sgdlr = 1e-1 # svhn
      sgdmlr = 1e-2 #svhn
      adamlr = 3e-4 # svhn
      epochs = cfgs['epochs']
    if cfgs["dataset"] == "cifar10":
      sgdlr = 1e-1 # cifar10
      sgdmlr = 1e-2 #cifar10
      adamlr = 3e-4 # cifar10
      epochs = cfgs['epochs']
    if cfgs["dataset"] == "cifar100":
      sgdlr = 1e-1 # cifar100
      sgdmlr = 1e-2 #cifar100
      adamlr = 3e-4 # cifar100
      epochs = cfgs['epochs']
      
    for nod, layer_num in enumerate(layerlist):
      cfgs["num_layers"] = layer_num # should be even
      
      for mod, lnoise_opt in enumerate(lnoiselist):
        cfgs["lnoise"] = lnoise_opt
        
        # for lod, noise_opt in enumerate(noiselist):
        #   cfgs["noise"] = noise_opt
        
        for kod, norm_opt in enumerate(normlist):
          cfgs['norms'] = norm_opt
        
          for jod, res_opt in enumerate(residuallist):
            cfgs["residual"] = res_opt # should be even
          
            # Experiments-Compasrisons
            
            # increment experiment. count
            exp_cnter +=1
            curdir = os.getcwd()
            exp_dir = f"{curdir}/MLP_EXP_{exp_cnter}"     
            os.makedirs(exp_dir, exist_ok=True)

            for iod, optim_name in enumerate(optimlist):
              
              cfgs['optim'] = optim_name
              print(f"{optim_fullname[iod]}\n----------")
              
              # -- RUNS for each OPTIMs under current cfg
              # train_dl, test_dl, ground_truth, steps_per_epoch, mdl, loss_fcn, optimizer, train_losses, dev_losses, train_accs, dev_accs, preds_list,PathStr = 
              main_opt_runners(cfgs, runs, epochs, gen_seed, batch_size, training_data, test_data, indim, class_num, channels, sgdlr, sgdmlr, adamlr, exp_dir)
                          
            # -- COMPARE OPTIMs under current cfg:      
            runner_cmps(cfgs, runs, epochs, optim_fullname, optimlist, exp_dir)
  
  print(f"SUCCESS: End of Experiments!")

# main runs for each optims.
def main_opt_runners(cfgs, runs, epochs, gen_seed, batch_size, training_data, test_data, indim, class_num, channels, sgdlr, sgdmlr, adamlr, expdir):

  # -Load Data
  
  if cfgs["device"] == "cpu" or not cfgs["numworkers"]:
      train_dl = DataLoader(training_data,batch_size=batch_size,shuffle=True,generator=gen_seed)
      test_dl = DataLoader(test_data,batch_size=batch_size,generator=gen_seed)
  else:
      train_dl = DataLoader(training_data,batch_size=batch_size, num_workers=cfgs["numworkers"], shuffle=True,persistent_workers=True,pin_memory=True,worker_init_fn=seed_worker,generator=gen_seed)
      test_dl = DataLoader(test_data,batch_size=batch_size,num_workers=2,worker_init_fn=seed_worker,persistent_workers=True,generator=gen_seed)
  
  try:
    ground_truth = test_dl.dataset.targets.tolist()
  except:
    try:
      ground_truth = test_dl.dataset.targets
    except:
      try:
        ground_truth = test_dl.dataset.labels.tolist()
      except:
          try:
            ground_truth = test_dl.dataset.labels
          except:
            pass
  
  # number of batches for given batch size, and data size
  steps_per_epoch = len(train_dl)

  # -Load Model
  if cfgs["model"] == "mlp":
    flat_indim = 1
    for d in indim:
      flat_indim *= d
      # flat_indim  = int(flat_indim/channels)
      
  mdl = FNN(cfgs, flat_indim, channels, hid_dim=128, class_dim=class_num)
    
  mdl.to(device,non_blocking=True)

  # -Load Loss function and Optimizer
  # define loss fcn and optimizer
  loss_fcn = nn.CrossEntropyLoss()
  softmax = nn.Softmax(dim=1)
  
  lnoise_type= cfgs['lnoise']
  wdecay = 1e-4 if lnoise_type == "wdecay" else 0
  
  if cfgs["optim"] == "sgd":
    optimizer = torch.optim.SGD(mdl.parameters(),lr=sgdlr, weight_decay=wdecay)
  elif cfgs["optim"] == "sgdmom":
    optimizer = torch.optim.SGD(mdl.parameters(),lr=sgdmlr,momentum=0.9,weight_decay=wdecay)
  elif cfgs["optim"] == "adam":
    optimizer = torch.optim.Adam(mdl.parameters(), lr=adamlr, weight_decay=wdecay)
  else:
    pass

  # metrics: lists
  train_losses = []
  dev_losses = []
  train_accs = []
  dev_accs = []
  preds_list = []
  
  # use the defined train and test loop
  for r in range(runs):  
    
    # -Train and Evaluate (test)

    reset_all_weights(mdl)
    
    # use the defined train and test loop
    for t in range(epochs):
      print(f"Epoch {t+1}\n----------")
      loss,acc = train_loop(train_dl,mdl,loss_fcn,optimizer)
      train_losses.append(loss)
      train_accs.append(acc)
      #
      loss,acc,test_preds = test_loop(test_dl, mdl, loss_fcn)
      dev_losses.append(loss)
      dev_accs.append(acc)
      
    #
    preds_list.append(test_preds)
    optimizer.state.clear()
    print(f"Run: {r+1} => Done!")
    
  preds_list.append(ground_truth)
  optimizer.state.clear()
  
  
 
  # add batch_size to this
  PathStr = cfgs["model"]+str(cfgs["num_layers"])+"_"+cfgs["dataset"]+"_"+cfgs["optim"]+cfgs["norms"]+cfgs["noise"]+cfgs["lnoise"]+cfgs["residual"]+str(runs)+"_"+str(epochs)

  # -- POST-RUNS: Computation ...
  
  # - Save Predictions for the just completed runs
  df = pd.DataFrame(preds_list)
  df = df.T
  # print(df.head())
  # print(df.tail())
  # df.style 
  
  
  PATHpreds = f"{expdir}/stores/preds"
  os.makedirs(PATHpreds, exist_ok=True)
  PATHpreds = f"{PATHpreds}/preds_{PathStr}"
  df.to_csv(PATHpreds+".csv")

  # - Save the Model (optional)
  PATHmdl = f"{expdir}/stores/mdls"
  os.makedirs(PATHmdl, exist_ok=True)
  PATHmdl = f"{PATHmdl}/{PathStr}"
  torch.save({'mdl_state_dict': mdl.state_dict(),}, PATHmdl+".pt")
  # load saved model
  # chkpt = torch.load(PATH)
  # mdl.load_state_dict(chkpt['mdl_state_dict'])
  # mdl.eval() or mdl.train()

  # - Plot and Individual Test-Train Loss Metrics (Optional)
  PATHplots = f"{expdir}/stores/plots"
  os.makedirs(PATHplots, exist_ok=True)
  PATHplots = f"{PATHplots}/{PathStr}"
  dashplots.traintest(train_losses, train_accs, dev_losses, dev_accs,epochs,steps_per_epoch,figname=PATHplots,live=False)
  
  # Compute Metrics

  # - Recompute Accuracy: Consistency with Ground-Truth for each of the runs
  acc_mets = []
  lastid = len(preds_list)-1
  total = len(preds_list[lastid])
  for id in range(lastid):
    zero_one_acc = sum(np.array(preds_list[id]) == np.array(preds_list[lastid]))/total
    acc_mets.append(zero_one_acc)
    
  # - Save Accs  
  df = pd.DataFrame(acc_mets)
  PATHacc = f"{expdir}/stores/acc"
  os.makedirs(PATHacc, exist_ok=True)
  PATHacc = f"{PATHacc}/accs_{PathStr}"
  df.to_csv(PATHacc+".csv", index=True) # index is true by default, can change to false to remove row ids
  
  # - Compute Prediction Consistency across Runs
  pdiff_mets = []
  act_pdiff_mets=[]
  # pval_mets = []
  lastid = len(preds_list)-1
  total = len(preds_list[lastid])
  for id in range(lastid-1):
    for jd in range(id+1, lastid):
      
      # Pred. Diff
      act_pred_diff = sum(np.array(preds_list[id]) != np.array(preds_list[jd]))
      act_pdiff_mets.append(act_pred_diff)
      pred_diff = act_pred_diff/total
      pdiff_mets.append(pred_diff)
      
      
      # Pval
      # _,this_pval = wilcoxon(preds_list[id],preds_list[jd])
      # pval_mets.append(this_pval)
  
  #TODO: Effective Test-Accuracy:
  # mean accuracy - mean pred.difference
  eff_test_acc = (np.mean(np.array(acc_mets)*total)-np.mean(np.array(act_pdiff_mets)))/total
      
  # Statistical Test on the runs Combination 2 Pdiffs
  # null: paired pred. diffs come from the same distribution, i.e not significant
  # alt: paired pred. diffs don't come from the same distribution, i.e significant
   
  # The Wilcoxon T-test. Given n independent samples (xi, yi) from a bivariate distribution (i.e. paired samples), it computes differences di = xi - yi. (OR. Skip this: by supplying with the paired differences, di). One assumption of the test is that the differences are symmetric. 
  # The two-sided test has the null hypothesis that the median of the differences is zero against the alternative that it is different from zero.
  try:
    med_stat, pval = wilcoxon(pdiff_mets)
  except ValueError:
    # ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
    med_stat, pval = -1, 0.5 # indicative of when the predictions are the same 
    
  # if pval isn't 0.5, we would reject the null hypothesis at a confidence level of 5%, concluding that the pred. diff across runs is significant.
  
  # e.g: WilcoxonResult(statistic=0.0, pvalue=0.001953125)
  # Our p-value, 0.001953125, is less than 0.05, so we have sufficient evidence to reject the null hypothesis that median difference is zero. This means the median difference is significantly different from zero.
  
  # The p-value of less than 0.05 indicates that this test rejects the null hypothesis at the 5% significance level. This means that the data distribution falls within the range of what would happen 95% of the time, described by the alternate-hypothesis.
  
  # If the calculated p-value exceeds .05, this means that the data distribution falls within the range of what would happen 95% of the time, described by the null-hypothesis. Hence, the null hypothesis is not rejected at the .05 level.

  # Write current cfg to this stores folder
  # for book-keeping
  run_cfgs = deepcopy(cfgs)
  run_cfgs['eff_test_accuracy'] = eff_test_acc
  run_cfgs['med_stat'] = med_stat
  run_cfgs['pval'] = pval
  
  PATHruncfg = f"{expdir}/stores/exp_cfg"
  os.makedirs(PATHruncfg, exist_ok=True)
  PATHruncfg = f"{PATHruncfg}/cfgs_{PathStr}.json"
  with open(PATHruncfg, 'w') as cfglist:
    json.dump(run_cfgs, cfglist)
    
  # - Save Fractional PDiff  
  df = pd.DataFrame(pdiff_mets)
  
  PATHpd = f"{expdir}/stores/pdiff"
  os.makedirs(PATHpd, exist_ok=True)
  PATHpd = f"{PATHpd}/pdiff_{PathStr}"
  df.to_csv(PATHpd+".csv")
  
    # - Save Actual PDiff  
  df = pd.DataFrame(act_pdiff_mets)
  
  PATHpd = f"{expdir}/stores/pdiff"
  os.makedirs(PATHpd, exist_ok=True)
  PATHpd = f"{PATHpd}/actpdiff_{PathStr}"
  df.to_csv(PATHpd+".csv")
  
  # # - Save Preds Pval  
  # df = pd.DataFrame(pval_mets)
  
  # PATHpd = f"{expdir}/stores/pdiff"
  # os.makedirs(PATHpd, exist_ok=True)
  # PATHpd = f"{PATHpd}/pval_{PathStr}"
  # df.to_csv(PATHpd+".csv")
  
  return train_dl,test_dl,ground_truth,steps_per_epoch,mdl,loss_fcn,optimizer,train_losses,dev_losses,train_accs,dev_accs,preds_list,PathStr


# compare each optims, based on setting:
def runner_cmps(cfgs, runs, epochs, optim_fullname, optimlist,expdir):
              # Based on cfg settings.
              
              # - Read PDIFF and ACC of each Optimizer.
              namex = optim_fullname
              datay_pd = []
              datay_actpd = []
              # datay_pval = []
              datay_acc = []
              eff_test_accs = []
              pdiff_pvals = []
              for iod, optim_name in enumerate(optimlist):
                cfgs['optim'] = optim_name
                
                PathStr = cfgs["model"]+str(cfgs["num_layers"])+"_"+cfgs["dataset"]+"_"+cfgs["optim"]+cfgs["norms"]+cfgs["noise"]+cfgs["lnoise"]+cfgs["residual"]+str(runs)+"_"+str(epochs)
                
                #
                PATHpd = f"{expdir}/stores/pdiff/pdiff_{PathStr}"
                dfpd = pd.read_csv(PATHpd+".csv",)
                datay_pd.append((dfpd.iloc[0:,1].to_list())) # if 1 row: change to [0,1:]
                
                #
                PATHpdact = f"{expdir}/stores/pdiff/actpdiff_{PathStr}"
                dfpd = pd.read_csv(PATHpdact+".csv",)
                datay_actpd.append((dfpd.iloc[0:,1].to_list())) 
                
                #
                PATHacc = f"{expdir}/stores/acc/accs_{PathStr}"
                dfa = pd.read_csv(PATHacc+".csv")
                datay_acc.append((dfa.iloc[0:,1].to_list()))
                
                #
                # PATHpdpval = f"{expdir}/stores/pdiff/pval_{PathStr}"
                # dfpd = pd.read_csv(PATHpdpval+".csv",)
                # datay_pval.append((dfpd.iloc[0:,1].to_list()))               
                
                #
                PATHruncfg = f"{expdir}/stores/exp_cfg/cfgs_{PathStr}.json"
                with open(PATHruncfg, 'r') as cfglist:
                  run_cfgs = json.load(cfglist)
                eff_test_accs.append(run_cfgs['eff_test_accuracy'])
                pdiff_pvals.append(run_cfgs['pval'])
              
              
              PathStr = cfgs["model"]+str(cfgs["num_layers"])+"_"+cfgs["dataset"]+"_optims_"+cfgs["norms"]+cfgs["noise"]+cfgs["lnoise"]+cfgs["residual"]+str(runs)+"_"+str(epochs)
              
              # Tstat Comparison
              tstat_mets = {'Optimizer': optim_fullname, 'p-value':pdiff_pvals}
              dftstat = pd.DataFrame(tstat_mets)
              PATHcmp = f"{expdir}/stores/exp_cfg/tstat_cmps_{PathStr}"
              dashplots.wilcxstatplot(namex,dftstat,runs,figname=PATHcmp,live=False)
              
              # Box-Plot Comparison
              PATHcmp = f"{expdir}/stores/pdiff/pdiff_cmps_{PathStr}"
              dashplots.pdiffplot(namex,datay_pd,runs,figname=PATHcmp,live=False)
              
              # Box-Plot Comparison
              PATHcmp = f"{expdir}/stores/pdiff/actpdiff_cmps_{PathStr}"
              dashplots.actpdiffplot(namex,datay_actpd,runs,figname=PATHcmp,live=False)
              
              # # Box-Plot Comparison
              # PATHcmp = f"{expdir}/stores/pdiff/pval_cmps_{PathStr}"
              # dashplots.pvalplot(namex,datay_pval,runs,figname=PATHcmp,live=False)
              
              # Box-Plot Comparison
              PATHcmp = f"{expdir}/stores/acc/accs_cmps_{PathStr}"
              dashplots.paccplot(namex,datay_acc,runs,figname=PATHcmp,live=False)
              
              # Box-Plot Comparison
              details = {'Optimizer': optim_fullname, 'Effective Test-Accuracy':eff_test_accs}
              df = pd.DataFrame(details)
              PATHcmp = f"{expdir}/stores/acc/effaccs_cmps_{PathStr}"
              dashplots.effpaccplot(namex,df,runs,figname=PATHcmp,live=False)
              
              # Write current cfg to this stores folder
              # for book-keeping
              cmp_cfgs = deepcopy(cfgs)
              cmp_cfgs['optims'] = "all"
              PATHcmpcfg = f"{expdir}/cmp_config.json"
              with open(PATHcmpcfg, 'w') as cfglist:
                json.dump(cmp_cfgs, cfglist)
          
        
      
if __name__=='__main__':
  
  # clear cached modules if its folder exists
  shutil.rmtree("__pycache__",ignore_errors=True)
  
  # 
  # clear_old_expdir = False : clear experiment folders if exist,
  # clear_old_expdir = True : or otherwise: archive in oldbins folder.
  clear_old_expdir = False # leave at False to archive old experiments.
  oldexpdir = glob.glob("./MLP_EXP_*")
  if clear_old_expdir:
    pass
    # for edir in oldexpdir:
    #   shutil.rmtree(edir,ignore_errors=True)
  else:
    # get a unique number for naming the archive dir.
    num_cnt = random.randint(0,9999)
    while os.path.exists(f"oldbins/old_{num_cnt}"):
      num_cnt = random.randint(0,9999)
      
    # move into the archive dir.
    for id,edir in enumerate(oldexpdir):
      if id == 0:
        shutil.move(edir, f"oldbins/old_{num_cnt}/{edir}")
      else:
        shutil.move(edir, f"oldbins/old_{num_cnt}")
        
  # empty cuda cahe.  
  device = "cuda" if torch.cuda.is_available() else "cpu"
  
  # --------------------------------------
  # load initial cfg.
  with open('config.json', 'r') as cfglist:
    cfgs = json.load(cfglist)
  
  # set main compute device: cuda or cpu
  if device == "cuda":
    cfgs["device"] = "cuda"
    torch.cuda.empty_cache() 
  else:
    cfgs["device"] = "cpu"
  
  # configure experiments
  cfgs["human"] = "oluwasegun" # options: olwasegun, apurva, nischal, any
  
  # > 1, set to 5 to reduce time spent on experiments
  runs = 5 
  
  # >= 1, set to 100 or 50 or 200 for sensible results, at which overfitting might occur
  epochs = 10 # cifar10 and cifar100
  
  # >= 0, max setting: 8 , recommended: set to 2 or 4
  numworkers = 4
  
  # run experiments wrt cfgs.
  dlnn_main(cfgs,runs,epochs,numworkers)
  
  # --------------------------------------
  
  # empty cuda cahe.  
  if device == "cuda":
    torch.cuda.empty_cache() 
  
  
  