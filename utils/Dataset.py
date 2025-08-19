"""
Custom dataset class for building seg. models
"""
import os
import numpy as np
from torch.utils.data import Dataset
import skimage
from scipy.ndimage import binary_fill_holes
from utils.ObserverSim import ObserverSimulator

class customDataset(Dataset):
    def __init__(self, image_path, folds, transforms, 
                 read_masks=False, normalise=True, window=400,
                   level=1074, worldmatch=False,
                   input_channels=1, preprocessing=None):
        super().__init__()
        # If worldmatch=True, apply WM correction (-1024 HU)
        self.images = self.load_data(image_path, folds, 'slices', worldmatch)     #~ Path to directory containing images
        self.transforms = transforms #~ Minimum ToTensor()
        self.ids = self.images['id']
        self.normalise = normalise
        self.WL_norm = self.WL_norm(self.images, window=window, level=level)
        self.read_masks = read_masks#
        self.input_channels = input_channels
        self.simulator = ObserverSimulator()
        self.preprocessing_fn = preprocessing
        
        if self.read_masks:
            self.masks = self.load_data(image_path, folds, 'masks', worldmatch)
    
    @staticmethod
    def load_data(path, folds, image_type, worldmatch=False):
        #* Expects contents of directory to be .npy (ID.npy)
        data_dict = {'slices': [], 'id': []}
        for fold in folds:
            fold_dir = os.path.join(path, f'fold_{fold}/{image_type}/')
            for file in os.listdir(fold_dir):
                if file.endswith('.npy'): #!!
                    name = file.split('.')[0]
                    data_dict['id'].append(name)
                    slice_ = np.load(fold_dir + file)
                    
                    slice_ = np.squeeze(slice_)
                    if worldmatch:
                        slice_ -= 1024
                    data_dict['slices'].append(slice_)
        data_dict['slices'] = np.array(data_dict['slices'])
        return data_dict

    @staticmethod
    def norm_inputs(data):
        #*Normalise inputs between [0, 1]
        return (data['slices'] - data['slices'].min())/(data['slices'].max()-data['slices'].min())

    @staticmethod
    def WL_norm(data, window, level):
        if window is None:
            # Set to range of training set
            window=data['slices'].max()-data['slices'].min()
        if level is None:
            # Set to mean of dataset
            level=data['slices'].mean()
        print(f'Applying window/level: {window}/{level}')
        minval = level - window/2
        maxval = level + window/2
        wld = np.clip(data['slices'], minval, maxval)
        wld -= minval
        wld /= window
        return wld

    @staticmethod
    def convert_threeChannel(img):
        #~ SHAPE: H x W x C
        return np.repeat(img, 3, axis=-1)

    @staticmethod
    def generate_body_mask(image, threshold=834): # threshold = [1024-190]
        blur_im = skimage.filters.gaussian(image, sigma=3, preserve_range=True)
        bin_im = blur_im > threshold # Apply threshold
        lab_im = skimage.measure.label(bin_im, connectivity=2, return_num=False) # connected components
        body = binary_fill_holes( np.where(lab_im == 1, 1, 0) ).astype(int)
        #body = skimage.morphology.convex_hull_image( np.where(lab_im == 1, 1, 0) ).astype(int) # convex hull
        return body


    def __len__(self):
        return len(self.images['id'])
    
    def __getitem__(self, index: int):
        pid = self.ids[index]
        if self.read_masks:

            if len(self.masks['slices'].shape) != 4:
                # If only skeletal muscle in mask
                sm = self.masks['slices'][index]
                # Simulate observer variation
                #sm = self.simulator(sm)
                #Add body mask 
                body = self.generate_body_mask(self.images['slices'][index])
                body[(sm == 1)] = 0

                masks = np.stack([sm, body], axis=0)
                background = np.where(np.sum(masks, axis=0) == 0, 1, 0)
                masks = np.concatenate([background[None], masks], axis=0)
                mask = np.argmax(masks, axis=0)
            else:
                mask = self.masks['slices'][index]
                if mask.shape[0] == self.input_channels:
                    dim=0
                elif mask.shape[-1] == self.input_channels:
                    dim=-1
                else:
                    raise ValueError(
                        f"Provide channels in first/last dimension: {mask.shape}")
                mask = np.argmax(mask, axis=dim)

        #* Apply W/L normalisation
        if self.normalise:
            img = self.WL_norm[index, ..., np.newaxis]
        else:
            img = self.images['slices'][index, ..., np.newaxis]

        #* Convert to three channels if needed
        out_img = self.convert_threeChannel(img)

        if self.preprocessing_fn is not None:
            out_img = self.preprocessing_fn(out_img)
        if self.transforms:
            if self.read_masks:
                augmented = self.transforms(image=out_img, mask=mask)
                sample = {'inputs': augmented['image'], 
                        'targets': augmented['mask'],
                        'id': pid}
                return sample
            else:
                augmented = self.transforms(image=out_img)
                sample = {'inputs': augmented['image'],
                        'id': pid}
                return sample
        else:
            print('Need some transforms - minimum ToTensor().')
            raise

