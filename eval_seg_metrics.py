import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seg_metrics.seg_metrics as sg
from tqdm import tqdm



def check_BB():
    BB = False
    response = input('Do you want to include BB in the analysis? (y/n): ')

    if response == 'y':
        return True
    elif response == 'n':
        return False
    
BB = check_BB()

rois = ['Muscle', 'SC-Fat'] #,  'BB'] #'IMAT',
file_dir = 'BB'

if not BB:
    rois = ['Muscle', 'SC-Fat']#, 'IMAT']
    file_dir = 'noBB'

pred_dir = f'./cv_{file_dir}/output'
gt_dir = f'./cv_{file_dir}/data/all/masks'
out_dir = f'./cv_results/{file_dir}'

metrics = ['dice', 'msd', 'hd95']
spacing = [0.833, 0.833]

def threshold_muscle_mask(muscle_mask, image):
    '''
    Threshold mask base on mean and std of muscle mask
    mask: np.array, mask to threshold
    image: np.array, image to threshold
    threshold: float, threshold value
    return: np.array, thresholded mask,
            np.array, IMAT mask
    '''
    muscle_pixels = np.where(muscle_mask == 1, image, np.nan)

    mean = np.nanmean(muscle_pixels)
    std = np.nanstd(muscle_pixels)

    threshold = mean + std

    #imat = np.where(muscle_pixels >= threshold, 1, np.nan)
    clean_muscle_mask = np.where(muscle_pixels < threshold, 1, 0)
    imat = np.where(muscle_pixels >= threshold, 1, 0)
    
    return clean_muscle_mask, imat



def process_preds(mask, img, BB=False):
    mask = mask[1:, :, :] # remove background

    mask = np.where(mask > 1, 1, np.nan)
    mask_all = np.nan * np.ones_like(mask[0])

    muscle_mask = mask[0]
    #muscle_mask, imat_mask = threshold_muscle_mask(muscle_mask, img)
    sc_fat_mask = mask[1]
    
    mask_all = np.where(muscle_mask > 0, 1, mask_all)
    mask_all = np.where(sc_fat_mask > 0, 2, mask_all)
    #mask_all = np.where(imat_mask > 0, 3, mask_all)
    
    if BB:
        bb_mask = mask[2]
        mask_all = np.where(bb_mask > 0, 4, mask_all)
    
    return mask_all

def process_gts(mask, img, BB=False):
    mask_muscle = mask[:, :, 1]
    mask_all = np.nan * np.ones_like(mask_muscle)
    
    # mask_muscle_clean, mask_imat = threshold_muscle_mask(mask_muscle, img)
    
    #mask_all[mask_muscle_clean == 1] = 1 # muscle
    mask_all[mask[:, :, 1] == 1] = 1
    mask_all[mask[:, :, 2] == 1] = 2 # sc fat
    # mask_all[mask_imat == 1] = 3 # imat

    if BB:
        mask_all[mask[:, :, 3] == 1] = 4
    
    return mask_all

def main():
    col_order = ['PatID', 'Fold', 'label', 'dice', 'msd', 'hd95']
    df_metrics_all = pd.DataFrame()

    print(f'Using {file_dir} data')
    
    for k in range(5):
        print('='*50)
        print(f'Processing fold {k}')

        fold_dir = f'{pred_dir}/fold_{k}'
        arrays = np.load(f'{fold_dir}/predictions.npz')

        ids = arrays['id']
        list_metrics = []

        for i in tqdm(range(len(ids))):
            pat_id = ids[i][0]
            img = arrays['slices'][i][2, :, :]
            mask_pred = arrays['masks'][i]
            mask_gt = np.load(f'{gt_dir}/{pat_id}.npy')
        
            mask_pred_all = process_preds(mask_pred, img, BB)
            mask_gt_all = process_gts(mask_gt, img, BB)
      
            # calculate metrics
            for j, label in enumerate(rois):
                
                results = sg.write_metrics(
                    labels=[j+1],
                    gdth_img=mask_gt_all,
                    pred_img=mask_pred_all,
                    metrics=metrics,
                    spacing=spacing)
                
                df_pat = pd.DataFrame(results[0])
                df_pat['PatID'] = pat_id
                df_pat['label'] = label
                df_pat['Fold'] = k
                list_metrics.append(df_pat)
            
        df_metrics = pd.concat(list_metrics, axis=0)
        
        df_metrics['dice'] = df_metrics['dice'].round(3)
        df_metrics['msd'] = df_metrics['msd'].round(3)
        df_metrics['hd95'] = df_metrics['hd95'].round(3)

        df_metrics = df_metrics[col_order]
        df_metrics.to_csv(f'{out_dir}/metrics_fold_{k}-noIMAT.csv', index=False)

        df_metrics_all = pd.concat([df_metrics_all, df_metrics], axis=0)
        print(f'Fold {k} done!')

    df_metrics_all = df_metrics_all[col_order]
    df_metrics_all.to_csv(f'{out_dir}/metrics_all_folds-noIMAT.csv', index=False)
    print('All folds done!')

if __name__ == '__main__':
    main()
            