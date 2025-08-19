"""
Training loop class
"""
import numpy as np
import torch
import torch.nn as nn
import os

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from utils.Losses import diceLoss
from utils.EarlyStopping import EarlyStopping

from segmentation_models_pytorch.losses import DiceLoss


class segmenter():
    def __init__(self, model, optimizer, train_loader,
     val_loader, writer, num_epochs=1, num_outputs=1, device="cuda:0",
      output_path='./logs', test_fold=None):
        self.device = torch.device(device)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', verbose=True)
        self.test_fold = test_fold
        self.weights = (1, 1) #* X-ent weight vs DSC weight
        if num_outputs == 1:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1])).to(device)
            self.dsc = diceLoss().to(device)
            self.target_dtype = torch.float32
        else:
            #class_weights = torch.tensor((1), dtype=torch.float) # Background, muscle, subcut, visceral
            self.bce = nn.CrossEntropyLoss().to(device)
            self.dsc = DiceLoss(mode='multiclass', log_loss=True, from_logits=True).to(device)
            self.target_dtype = torch.long

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.scaler = GradScaler()
        self.stopper = EarlyStopping(patience=75)
        #~ Params
        self.num_epochs = num_epochs
        self.num_outputs = num_outputs
        self.best_loss = 10000
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        torch.autograd.set_detect_anomaly(True)

    @staticmethod
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    def forward(self):
        self.stop = False
        for epoch in range(self.num_epochs+1):
            print(f'Epoch {epoch} of {self.num_epochs}')
            self.writer.epoch = epoch
            self.training(epoch)
            self.validation(epoch)
            self.save_best_model()
            #~ Early stopping
            if self.stop:
                print('==== Early Stopping ====')
                break
    
    def training(self, epoch, writer_step=25):
        self.model.train()
        self.writer.reset_losses()
        for idx, data in enumerate(tqdm(self.train_loader)):
            inputs = data['inputs'].to(self.device, dtype=torch.float32)
            targets = data['targets'].to(self.device, dtype=self.target_dtype)
            if self.num_outputs == 1 and targets.shape[1] != 1:
                targets = targets[:, None]
                print(f'Updating target shape: {targets.shape}')
                
            #* Zero parameter gradient
            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(inputs)                
                bce_loss = self.bce(outputs, targets)
                dice_loss = self.dsc(outputs, targets)
                train_loss = self.weights[0]*bce_loss + self.weights[1]*dice_loss

            if epoch % writer_step == 0 and idx == 0: # plot before back-pass in case backwards returns NAN
                print('Plotting inputs...')
                self.writer.plot_inputs('Inputs', inputs)
                self.writer.plot_targets(
                    'Ground-truth', inputs, targets[:, None])

            #* Compute backward pass on scaled loss
            self.scaler.scale(train_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.writer.metrics['train_loss'].append(train_loss.item())
            
                

        print('Train Loss:', np.mean(self.writer.metrics['train_loss']))
        self.writer.add_scalar('Training_loss', np.mean(self.writer.metrics['train_loss']), epoch)
        self.writer.add_scalar(
            'Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)

    def validation(self, epoch, writer_step=5):
        self.model.eval()
        with torch.set_grad_enabled(False):
            print('VALIDATION')
            for batch_idx, data in enumerate(self.val_loader):
                inputs = data['inputs'].to(self.device, dtype=torch.float32)
                targets = data['targets'].to(self.device, dtype=self.target_dtype)
                if self.num_outputs == 1 and targets.shape[1] != 1:
                    targets = targets[:, None]
                    print(f'Updating target shape: {targets.shape}')
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                bce_loss = self.bce(outputs, targets)
                dice_loss = self.dsc(outputs, targets)
                
                valid_loss = self.weights[0]*bce_loss + self.weights[1]*dice_loss
                #* Writer
                self.writer.metrics['BCE'].append(self.weights[0]*bce_loss.item())
                self.writer.metrics['DSC'].append(self.weights[1]*dice_loss.item())
                self.writer.metrics['val_loss'].append(valid_loss.item())
                #* --- PLOT TENSORBOARD ---#
                if epoch % writer_step == 0 and batch_idx == 0:
                    self.writer.plot_segmentation('Predictions', inputs, outputs, targets=targets)


        print('Validation Loss:', np.mean(self.writer.metrics['val_loss']))
        self.writer.add_scalar('Validation Loss', np.mean(
            self.writer.metrics['val_loss']), epoch)
        self.writer.add_scalar('BCE', np.mean(self.writer.metrics['BCE']), epoch)
        self.writer.add_scalar('DSC', np.mean(self.writer.metrics['DSC']), epoch)
        self.scheduler.step(np.mean(self.writer.metrics['val_loss']))

    def save_best_model(self, model_name='best_model.pt'):
        if self.test_fold is None:
            save_path = f'{self.output_path}/{model_name}'
        if self.test_fold is not None:
            save_path = f'{self.output_path}fold_{self.test_fold}/{model_name}'
        loss1 = np.mean(self.writer.metrics['val_loss'])
        is_best = loss1 < self.best_loss
        self.best_loss = min(loss1, self.best_loss)
        if is_best:
            print('Saving best model')
            print(f'Saving model to {save_path}')
            torch.save(self.model.state_dict(), save_path)
                          
        #~ Check stopping criterion
        if self.stopper.step(torch.tensor([loss1])):
            self.stop = True
