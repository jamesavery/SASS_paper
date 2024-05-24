"""
Generate STL model from numpy array.
Pipeline is: ndarray -> mesh -> STL
 - https://stackoverflow.com/a/74118104
 - https://github.com/MeshInspector/MeshLib
"""

import meshlib.mrmeshpy as mr
import meshlib.mrmeshnumpy as mrn
import numpy as np
import h5py

def load_masks_from_h5(fpath, dset):
    # load boolean mask for titanium implant
    with h5py.File(fpath, "r") as hf:
        mask = hf[dset][:]
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
    # note that this assumes that 
    h1,w1,d1 = bkg.shape
    h2,w2,d2 = overlay.shape
    assert h1 >= h2 and w1 >= w2 and d1 >= d2, 'requires: dim(bkg) > dim(overlay)'
    # find center coordinates
    ch,cw,cd = (h1-h2)//2, (w1-w2)//2, (d1-d2)//2
    # place overlay in center of bkg
    bkg[ch:ch+h2, cw:cw+w2, cd:cd+d2] += overlay
    return bkg

def create_ground_truth(implant, bone, blood, match_tomo_dim, saveh5=False):
    """
    Create single 3d LUT with ground truth labeling from STL models containing:
    implant, bone and blood.
    """

    # implant
    implant_mask = implant.astype(np.uint8)
    np.place(implant_mask, implant_mask==True, 0b0001)

    # bone
    bone_mask = bone.astype(np.uint8)
    implant_boolmask = implant.astype(bool)
    blood_boolmask = blood.astype(bool)
    # to get bone mask, we remove implant and blood to get only bone
    # this also means there are no overlapping regions
    bone_mask[implant_boolmask] = 0
    bone_mask[blood_boolmask] = 0
    np.place(bone_mask, bone_mask==True, 0b0010)

    # blood
    blood_mask = blood.astype(np.uint8)
    np.place(blood_mask, blood_mask==True, 0b0100)

    # initialize matrix with given size
    # example: 4000x4000x4000 in uint8 -> 64 GB
    ground_truth = np.zeros(match_tomo_dim, dtype=np.uint8)

    # add masks with overlaps being uniquely assigned
    ground_truth = overlay_at_center(ground_truth,implant_mask) # 1
    ground_truth = overlay_at_center(ground_truth,bone_mask)    # 2
    ground_truth = overlay_at_center(ground_truth,blood_mask)   # 4

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
    # heuristic can be to use Nearest Neighbor. NOTE: for now this is avoided by
    # how the bone_mask is constructed.
    
    #print("unique values:", np.unique(ground_truth))

    if saveh5:
        with h5py.File("gtruth.h5", "w") as hf:
            hf.create_dataset("data", data=ground_truth)

    return

if __name__ == "__main__":
    # implant
    implant_mask = load_masks_from_h5(fpath="implant_masks/770c_pag_8x.h5",
                                          dset="implant/mask")
    generate_stl(implant_mask, "implant_mesh_8x")
    # blood
    blood_mask = load_masks_from_h5(fpath="blood_masks/770c_pag_8x.h5",
                                        dset="blood/mask")
    generate_stl(blood_mask, "blood_mesh_8x")
    # bone
    bone_mask = load_masks_from_h5(fpath="bone_masks/770c_pag_8x_bone.h5",
                                        dset="bone/mask")
    generate_stl(bone_mask, "bone_mesh_8x")

    # ground truth
    x8 = (409,432,432)
    create_ground_truth(implant=implant_mask,
                        bone=bone_mask, # composite
                        blood=blood_mask,
                        match_tomo_dim=x8,
                        saveh5=True)
