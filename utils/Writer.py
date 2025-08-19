"""
Tensorboard writer
"""
import matplotlib
matplotlib.use('Agg')
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

class customWriter(SummaryWriter):
    def __init__(self, logdir, batch_size):
        super().__init__()
        self.metrics = {'train_loss': [], 'val_loss': [], 
                        'BCE': [], 'DSC': []}
        self.epoch = 0
        self.batch_size = batch_size

    def reset_losses(self):
        self.metrics = {key: [] for key in self.metrics.keys()}

    @staticmethod
    def norm_img(img):
        return (img-img.min())/(img.max()-img.min())
    
    @staticmethod
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    def plot_inputs(self, title, img):
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        
        length = self.batch_size if self.batch_size < img.shape[0]  else img.shape[0]

        for idx in np.arange(length):
            if length % 2 == 0:
                # IF even batch size
                ax = fig.add_subplot(length // 2, length // 2,
                                    idx+1, label='Inputs')
            else:
                ax = fig.add_subplot((length+1) // 2, (length+1) // 2,
                                    idx+1, label='Inputs')
            ax.axis('off')
            plt_img = np.moveaxis(img[idx].cpu().numpy(), 0, -1)
            plt_img = self.norm_img(plt_img)
            ax.imshow(plt_img, cmap='gray')
            ax.set_title(f'Input @ epoch: {self.epoch} - idx: {idx}')
        self.add_figure(title, fig)
        self.flush()

    def plot_segmentation(self, title, img, prediction, targets=None):
        fig = plt.figure(figsize=(20, 20))
        plt.tight_layout()

        length = self.batch_size if self.batch_size < img.shape[0]  else img.shape[0]

        for idx in np.arange(length):
            if length % 2 == 0:
                # IF even batch size
                ax = fig.add_subplot(length // 2, length // 2,
                                    idx+1, label='Predictions')
            else:
                ax = fig.add_subplot((length+1) // 2, (length+1) // 2,
                                    idx+1, label='Predictions')

            plt_img = np.moveaxis(img[idx].cpu().numpy(), 0, -1)
            plt_img = self.norm_img(plt_img)
            ax.axis('off')
            ax.imshow(plt_img, cmap='gray')
            pred = prediction[idx]
            if pred.shape[0] == 1:
                pred = np.round(self.sigmoid(pred).detach().cpu().numpy())
            else:
                pred = torch.argmax(pred, dim=0).detach().cpu().numpy()
            pred = np.squeeze(pred)
            #ax.imshow(pred, alpha=0.5, cmap='viridis')
            ax.imshow(np.where(pred==0, np.nan, pred), alpha=0.5, cmap='viridis')
            if targets is not None:
                tgt = np.squeeze(targets[idx].cpu().numpy())
                #ax.imshow(np.where(tgt==0, np.nan, tgt), alpha=0.5, cmap='viridis')
            #ax.set_title(f'Input @ epoch: {self.epoch} - idx: {idx}')

        self.add_figure(title, fig, global_step=self.epoch)
        self.flush()

    def plot_targets(self, title, img, targets):
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 1, self.batch_size // 1,
                                 idx+1, label='Targets')
            ax.axis('off')
            plt_img = np.moveaxis(img[idx].cpu().numpy(), 0, -1)
            plt_img = self.norm_img(plt_img)
            ax.imshow(plt_img, cmap='gray')
            tgt = targets[idx].cpu().numpy()
            tgt = np.squeeze(tgt)
            ax.imshow(np.where(tgt==0, np.nan, tgt), alpha=0.5, cmap='viridis')

        self.add_figure(title, fig, global_step=self.epoch)
        self.flush()