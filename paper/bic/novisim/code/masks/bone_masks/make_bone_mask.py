"""Make new bone mask from other masks.

This is a temporary hack to get a proper bone mask.
"""

import meshlib.mrmeshpy as mr
import meshlib.mrmeshnumpy as mrn
import numpy as np
import h5py

def generate_new_h5():
    # load other masks
    with h5py.File("../blood_masks/770c_pag_8x.h5", "r") as hf:
        bone_mask = hf["bone_region/mask"][:]
        implant_mask = hf["implant/mask"][:]
        blood_mask = hf["blood/mask"][:]

    # to get bone mask, we remove implant and blood to get only bone
    # this also means there are no overlapping regions
    bone_mask[implant_mask.astype(bool)] = 0
    bone_mask[blood_mask.astype(bool)] = 0

    # create for future use
    with h5py.File("770c_pag_8x_bone.h5", "w") as hf:
        hf.create_dataset("bone/mask", data=bone_mask)
    return

if __name__ == "__main__":
    generate_new_h5()
