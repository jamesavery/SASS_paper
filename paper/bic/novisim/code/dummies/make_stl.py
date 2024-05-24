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

def generate_dummy_mask_1():
    # cubic shell with solid cube inside
    mask = np.zeros((500,500,500))
    # shell
    tn = 5 # thickness of shell
    # z
    mask[:,:,0:tn] = 1
    mask[:,:,-tn:] = 1
    # y
    mask[:,0:tn,:] = 1
    mask[:,-tn:,:] = 1
    # x
    mask[0:tn,:,:] = 1
    mask[-tn:,:,:] = 1
    # solid center cube
    mask[200:300,200:300,200:300] = 1
    return mask

def generate_dummy_mask_2():
    # skeleton/frame of cube between shell and solid cube
    mask = np.zeros((500,500,500))
    padding = 100 # offset from dummy_mask_1
    tn = 20 # thickness
    # x
    mask[padding:-padding,padding:padding+tn,padding:padding+tn] = 1
    mask[-padding:padding,padding:padding+tn,padding:padding+tn] = 1
    mask[padding:-padding,-padding-tn:-padding,padding:padding+tn] = 1
    mask[padding:-padding,padding:padding+tn,-padding-tn:-padding] = 1
    mask[padding:-padding,-padding-tn:-padding,-padding-tn:-padding] = 1
    mask[-padding:padding,padding:padding+tn,-padding-tn:-padding] = 1

    mask[padding:padding+tn,padding:-padding,padding:padding+tn] = 1
    mask[-padding-tn:-padding,padding:-padding,padding:padding+tn] = 1
    mask[padding:padding+tn,-padding:padding,padding:padding+tn] = 1
    mask[padding:padding+tn,padding:-padding,-padding-tn:-padding] = 1
    mask[padding:padding+tn,-padding:padding,-padding-tn:-padding] = 1
    mask[-padding-tn:-padding,padding:-padding,-padding-tn:-padding] = 1

    mask[padding:padding+tn,padding:padding+tn,padding:-padding] = 1
    mask[-padding-tn:-padding,padding:padding+tn,padding:-padding] = 1
    mask[padding:padding+tn,-padding-tn:-padding,padding:-padding] = 1
    mask[-padding-tn:-padding,-padding-tn:-padding,padding:-padding] = 1
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

if __name__ == "__main__":
    a = generate_dummy_mask_1() # cubic shell with solid cube inside
    b = generate_dummy_mask_2()  # non-solid cubic frame
    generate_stl(a, "dummy_mesh_1")
    generate_stl(b, "dummy_mesh_2")
