import os
import numpy as np
from glob import glob

mask_files = glob("./data/*/masks/*.npy")

print(f'Found {len(mask_files)}')

test_file = np.load(mask_files[0])

print("Starting Script")
for f_name in mask_files:
    mask = np.load(f_name)


    if mask.shape[-1] == 4:

        mask_new = mask[:, :, :-1]
        np.save(f_name, mask_new)

print("Script Complete")