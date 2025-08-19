'''
Split in to 5 fold for cross validation
'''

import os
import numpy as np

def get_fold_ids(data_dir, fold_num=5):
    '''
    Get the fold ids for the cross validation
    '''
    # Get the list of files
    file_list = os.listdir(f'{data_dir}/masks')
    file_list = [f for f in file_list if f.endswith('.npy')]

    # shuffle the file list
    np.random.seed(0)
    np.random.shuffle(file_list)

    fold_ids = {}
    for i in range(fold_num):
        fold_ids[i] = file_list[i::fold_num]
    
    return fold_ids

def copy_files(fold_id, pat_ids, data_dir):
    '''
    Copy the files to the fold directories
    '''
    
    fold_dir = f'{data_dir}/fold_{fold_id}'
    print(f'Creating fold directory: {fold_dir}')
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(f'{fold_dir}/slices', exist_ok=True)
    os.makedirs(f'{fold_dir}/masks', exist_ok=True)

    for f in pat_ids:
        print(f)
        os.system(f'cp {data_dir}/data_all/slices/{f} {fold_dir}/slices/')
        os.system(f'cp {data_dir}/data_all/masks/{f} {fold_dir}/masks/')

def main():
    '''
    Main function
    '''
    data_dir = './cross_validation'
    fold_num = 5

    fold_ids = get_fold_ids(f'{data_dir}/data_all', fold_num)
    
    for i in range(fold_num):
        print(f'Fold {i} | IDs: {fold_ids[i]}')
        copy_files(i, fold_ids[i], f'{data_dir}')


if __name__ == '__main__':
    main()

