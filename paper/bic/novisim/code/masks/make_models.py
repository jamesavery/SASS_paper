"""make_models: Generate STL model from numpy arrays.

Pipeline is: ndarray -> mesh -> STL
 - https://stackoverflow.com/a/74118104
 - https://github.com/MeshInspector/MeshLib
"""

import meshlib.mrmeshpy as mr
import meshlib.mrmeshnumpy as mrn
import numpy as np
import h5py

def load_masks_from_h5(fpath, dset):
    # convenience function
    with h5py.File(fpath, "r") as hf:
        mask = hf[dset][:].astype(np.uint8)
    return mask

def generate_stl(mask, outputname):
    # convert boolean values to numbers, float32 or 64 is required by meshlib
    inputData = mask.astype(np.float32)
    # convert 3D array to SimpleVolume data
    simpleVolume = mrn.simpleVolumeFrom3Darray(inputData)
    # convert SimpleVolume to FloatGrid data
    floatGrid = mr.simpleVolumeToDenseGrid(simpleVolume )
    # make mesh by iso-value = 0.5 and voxel size = (0.1, 0.1, 0.1)
    mesh = mr.gridToMesh(floatGrid , mr.Vector3f(0.1, 0.1, 0.1), 0.5)
    # save mesh
    mr.saveMesh(mesh, outputname+".stl")
    return

def overlay_at_center(bkg,overlay):
    # assumes that bkg is larger than overlay
    h1,w1,d1 = bkg.shape
    h2,w2,d2 = overlay.shape
    assert h1 >= h2 and w1 >= w2 and d1 >= d2, 'requires: dim(bkg) > dim(overlay)'
    # find new "origo" offset from half of remainder in each dimension
    ch,cw,cd = (h1-h2)//2, (w1-w2)//2, (d1-d2)//2
    # place overlay in center of bkg
    bkg[ch:ch+h2, cw:cw+w2, cd:cd+d2] += overlay
    return bkg

def create_ground_truth(scale=(409,432,432), save_stl=False, saveh5=False):
    """
    Create single 3d LUT with ground truth labeling from STL models containing:
    implant, bone and blood.
    """

    #####################
    ### Load h5 masks ###
    #####################

    implant_mask = load_masks_from_h5(fpath="implant_masks/770c_pag_8x.h5",
                                          dset="implant/mask")
    blood_mask = load_masks_from_h5(fpath="blood_masks/770c_pag_8x.h5",
                                        dset="blood/mask")
    cut_cylinder_bone = load_masks_from_h5(fpath="bone_masks/770c_pag_8x.h5",
                                        dset="cut_cylinder_bone/mask")
    bone_mask = load_masks_from_h5(fpath="bone_masks/770c_pag_8x_bone.h5",
                                        dset="bone/mask")

    # make all uint8
    implant_mask = implant_mask.astype(np.uint8, copy=False)
    blood_mask = blood_mask.astype(np.uint8, copy=False)
    cut_cylinder_bone = cut_cylinder_bone.astype(np.uint8, copy=False)
    bone_mask = bone_mask.astype(np.uint8, copy=False)

    ####################
    ### implant == 1 ###
    ####################

    assert np.max(implant_mask) == 1

    #################
    ### bone == 2 ###
    #################

    implant_boolmask = implant_mask.astype(bool)
    blood_boolmask = blood_mask.astype(bool)

    # to get only bone mask, we ensure there is no overlap with implant and blood
    bone_mask[implant_boolmask] = 0
    bone_mask[blood_boolmask] = 0

    bone_mask <<= 1 # shift 1 to 2 (0b0010)

    assert np.max(bone_mask) == 2

    ##################
    ### blood == 4 ###
    ##################

    # note that this depends on the bone_mask already being generated
    cut_cylinder_bone[bone_mask==2] = 0
    cut_cylinder_bone[implant_boolmask] = 0

    blood_mask = cut_cylinder_bone

    # add resin shell outside radius
    # this reflects how the real samples are, but also helps raise the counts
    # and thereby height on "y-axis" in histogram
    vmin = 0
    r = blood_mask.shape[1] // 2
    for y in range(blood_mask.shape[2]):
        for x in range(blood_mask.shape[1]):
            if (x - r) ** 2 + (y - r) ** 2 >= r ** 2:
                blood_mask[:, y, x] = vmin

    #blood_mask = cut_cylinder_bone
    blood_mask <<= 2 # shift 1 to 4 (0b0100)

    assert np.max(blood_mask) == 4

    ####################
    ### ground truth ###
    ####################

    # initialize matrix of required size
    ground_truth = np.zeros(scale, dtype=np.uint8)

    # add masks with overlaps being uniquely assigned
    ground_truth = overlay_at_center(bkg=ground_truth,overlay=implant_mask) # 1
    ground_truth = overlay_at_center(bkg=ground_truth,overlay=bone_mask)    # 2
    ground_truth = overlay_at_center(bkg=ground_truth,overlay=blood_mask)   # 4

    # this gives the following table
    # 0 = background/resin
    # 1 = implant
    # 2 = bone
    # 3 = implant+bone * 
    # 4 = blood
    # 5 = implant+blood *
    # 6 = bone+blood *
    # 7 = implant+bone+blood *

    # All values with * need to be resolved since the overlap is unphysical.
    # We want to correct the labels so the individual materials (implant, bone
    # and blood) each have their unique non-overlapping label.  One possible
    # heuristic can be to use Nearest Neighbor. Note that for now this is avoided by
    # how the bone_mask is constructed.
    
    if save_stl:
        print("... Saving STL models.")
        generate_stl(implant_mask, "STL/implant_mesh_8x")
        generate_stl(bone_mask, "STL/bone_mesh_8x")
        generate_stl(blood_mask, "STL/blood_mesh_8x")

    if saveh5:
        print("... Saving ground truth as h5 file.")
        with h5py.File("gtruth.h5", "w") as hf:
            hf.create_dataset("data", data=ground_truth)

    return ground_truth

if __name__ == "__main__":

    # ground truth
    #x1 = (3272,3456,3456)
    #x2 = (1636,1728,1728)
    #x4 = (818,864,864)
    x8 = (409,432,432)
    gtruth = create_ground_truth(scale=x8,
                                 save_stl=True,
                                 saveh5=True)
