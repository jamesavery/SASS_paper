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

def generate_masks_from_h5():
    # load boolean mask for titanium implant
    with h5py.File("770c_pag_8x.h5", "r") as hf:
        mask = hf["implant/mask"][:]

    # now we create masks for bone and blood,
    # positioned relative to implant with same coordinate system
    # the bone and blood are spatially inverse in position
    print(mask.shape)
    #mask = np.zeros((500,500,500))
    #padding = 100 # offset from dummy_mask_1
    #tn = 20 # thickness
    return mask

def generate_stl(mask, outputname):
    # convert boolean values to numbers, float32 or 64 is required by meslib
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

if __name__ == "__main__":
    c = generate_masks_from_h5() # titanium implant
    generate_stl(c, "implant_mesh_8x")
