"""mapping.py:

    Purpose
    -------

    This maps a voxel with its corresponding label from "input", meaning
    the model we feed to Novi-sim, and the "output", meaning the resulting
    tomogram that is produced by Novi-sim.

    Index ordering
    --------------

    We want all tomograms to have ordering tomo[a,b,c], where
    a : through depth of screw
    b : along short side of screw
    c : along wide side of screw

    Pipeline
    --------
    
    ----------------------------------------------------
    3 STL models ->  Novi-sim -> Novi-tomogram output
           |
           |----------------------> Ground truth
                                          ^
                                          I
                                          v
    Osteomorph(Novi-sim tomogram) -> Segmentation output 
    ----------------------------------------------------

    We then want to compare segmentation output and ground truth

    Ground truth matrix
    -------------------

    Grounth truth is a matrix containing one byte per voxel.
    It determines what the label is, and we should incorporate multiple
    heuristics for multiple ground truths. Generally a mapping from
    mm-space to voxel-space will introduce ambiguities such as finding
    a voxel at position [329.40, 220.8, 123.5] - there are multiple ways
    to handle these (k-nearest neighbours, 6-neighbourhood region from
    + and - one in each dimension, or (27-1)-neighbourhood region from including
    diagonal neighbours). Each can be stored in a smart way in a single byte.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cv2
import math
import matplotlib
matplotlib.use("TKAgg")
from sklearn.cluster import KMeans
from typing import Tuple, Union, List
# importing tuple, and list is technically only required for Python <= 3.8, note capital letters
# In Python 3.10+ Union is not needed but is replaced by a bar | between multiple types
# Also Optional[X] is the same as X | None or Union[X, None]


def mmap(fpath:str, tshape:Tuple[int,int,int], tslice:Union[None,str], axis:Union[None,int]) -> np.ndarray:
    """Efficient memory mapped partial volumes.
    Note that axes are swapped to get correct ordering.
    Tomograms comes from novi-sim in (z,x,y), we swap to get (x,y,z).

    Input:
       fpath:  path to filename containing raw volumetric data
       tshape: 3-tuple of int describing volumetric dimensions
       tslice: None or int, depending on if only a slice is requested
       axis:   None or int {0,1,2} that sets from which axis slice is taken.
    Return: 
       data:   Numpy array containing either 2d or 3d data.
    """
    assert axis == None or axis <= 2
    mread = np.memmap(fpath, dtype='float32', mode='r', shape=tshape, order="F")
    mread = mread.swapaxes(0,2)
    if tslice:
        # extract slice along chosen axis
        sliceidx = tuple(np.s_[tslice] if s==axis else np.s_[:] for s in range(3))
        data = np.array(mread[sliceidx])
    else:
        # read full volume
        data = mread
    return data

def make_label_lut(maxval:int) -> List[str]:
    """
    Input:
       maxval: largest numerical value in ground truth matrix
    Return: 
       labels: list of strings, with format [0,c1,c2,0,c4,...,c5,0,...]
               with values at 2^n where n is number of classes.
               Note that n will always be small (for example 4)

    maxval is simply the highest value, so if label n has value 2^n
    The sample background has value zero and is not counted as an actual label.
    The zeroth label is thus never used, also because it would gives ambiguous
    combinations in overlapping regions.
    """

    labels = [str(0) for i in range(maxval+1)]
    labels[0b0001] = "implant"
    labels[0b0010] = "bone"
    labels[0b0100] = "blood"
    return labels

def accuracy_score(segm:np.ndarray, gtruth:np.ndarray) -> dict[str,float]:
    """Compute individual class accuracy, corresponding to Rand index.
    
    Input:
       segm:   3d numpy array representing output from a segmentation
       gtruth: 3d numpy array containing actual ground truth
    Return: 
       score:  Dictionary containing scores in range [0-1] for each label
    
    Please remember to adapt gtruth labeling to match that used by segm. Each
    algorithm will have some arbitrary mapping, and this will affect the score.
    """
    assert segm.shape == gtruth.shape, "segmentation matrix and gtruth must have same dimensions"

    if len(segm.shape) > 2:
        # since shapes are identical, same conditions is fulfilled for gtruth
        segm = segm.flatten()
        gtruth = gtruth.flatten()

    # generate label LUT and map identified values as a labels
    label_lut = make_label_lut(maxval=np.max(gtruth))
    score = {}
    # skip zeroth value (since it's not used)
    for c in np.unique(gtruth)[1:]:
        class_acc = np.mean(segm[gtruth == c] == c)
        score.update({f"{label_lut[int(c)]}": round(class_acc,5)})
    return score

def crop_center(arr:np.ndarray, crop_size:Tuple[int,int,int]) -> np.ndarray:
    """
    Crop 3D-array to the specified crop_size, keeping the center voxel fixed.
    
    Input:
       tomo: 3d numpy array to be cropped
       crop_size: tuple containing size of the crop (depth, height, width)
    Return: 
       cropped_array: 3d numpy array which has been cropped around center.
    """

    d, h, w = arr.shape
    cd, ch, cw = crop_size
    
    start_d = max((d - cd) // 2, 0)
    start_h = max((h - ch) // 2, 0)
    start_w = max((w - cw) // 2, 0)
    
    end_d = start_d + cd
    end_h = start_h + ch
    end_w = start_w + cw
    
    cropped_array = arr[start_d:end_d, start_h:end_h, start_w:end_w]
    
    return cropped_array

def evaluate_segmentation(tomo:np.ndarray, tomo_index:Tuple[int,int,int], gtruth:np.ndarray, pixel_pitch:Tuple[float,float]) -> dict[str,float]:
    """
    Input:
       tomo:        3d numpy array
       tomo_index:  tuple(i,j,k)
       gtruth:      3d numpy array (same size as tomo)
       pixel_pitch: float
       sensor_dim:  tuple(float,float) -> (hor,ver)
    Return:
       score:       Dictionary containing scores in range [0-1] for each label
    
    Idea: The problem of mapping between pixels in ground truth and tomogram is
    made more complicated due to the non-trivial geometry of the acquisition
    setup. The issue is that resulting size of the sample within the ground
    truth matrix and the 3D-ROI from the Novi-sim simulation do not match,
    because of the geometrical scaling effect from having a non-zero distance
    between sample and detector. So technically a voxel in the gtruth matrix
    could potentially correspond to a voxel that is outside the exposed ROI in
    the simulated tomogram, because sample-edges were cut-off in the view.
    Luckily we do not have to worry about this too much, since we will always be
    mapping voxels from the simulated tomogram to the ground truth and never the
    other way around.  Note: Mapping between Osteomorph segmentation and ground
    truth is obviously trivial, since ground truth is built from segmentation
    masks made with Osteo.
    """

    # For now it is a requirement that the tomogram and the ground truth matrix
    # have same shapes, but making a new gtruth matrix of a specific size is
    # fast and easy
    tomo_dim = tomo.shape # also equal to sensor_dim
    gtruth_dim = gtruth.shape
    assert tomo_dim == gtruth_dim

    # find magnification - choice of units is irrelevant, only relative scaling is needed
    # note that sample2detector_dist is not needed
    source2sample_dist = 145 # meter
    source2detector_dist = 146 # meter

    # magnification can now be calculated as
    # magnification = source2detector_dist / source2sample_dist
    # magnification = image_size / sample_size
    # => image_size = sample_size * (source2detector_dist / source2sample_dist)
    # where the image size is the enlarged projection unto the detector
    # sensor dimension (in image plane) is always larger than actual x-rayed area in sample-plane
    # sanity check: yes numerator is always larger than denominator, so sample_dim < sensor_dim
    mag = source2detector_dist / source2sample_dist
    image_dim = tuple(round(s*mag) for s in tomo_dim)

    """ lazy and slow method """

    print("magnification factor:", mag)

    #scaled_gtruth = zoom(gtruth, mag)
    #scaled_gtruth = crop_center(scaled_gtruth, tomo.shape)
    scaled_gtruth = gtruth # ignore while debugging

    # sanity check
    #print("accuracy: ", accuracy_score(segm=gtruth, gtruth=gtruth))

    # calculate accuracy scores for all classes
    # FIXME: since we don't have the Osteopmorph segmentation on Novi-sim data yet,
    # we simply make random segmentation matrix, and expect an accuracy score of ~ 0.33 each
    segmat = np.random.randint(1,4,(gtruth.shape)) # creates array containing [1,2,3]
    segmat[segmat==3] = 4 # convert 3 to blood (4)
    score = accuracy_score(segm=segmat, gtruth=scaled_gtruth)

    """ explicit method """

    ## currently implemented for one index at a time (so forces user to loop through)
    ## but can be vectorized by generating scaled grid with identified voxel offset

    ## redefine origo in scaled voxel-space
    ## both the tomogram and the ground truth matrix agrees on center voxel, and
    ## only differ in their relative scaling we can use this to find new
    ## corrected corrected origo, i.e. (0,0,0)-voxel
    #center_voxel = tuple(round(i/2) for i in gtruth.shape)
    #scaled_center_voxel = tuple(round(i/mag_inv) for i in center_voxel)
    ## new origo corresponding to (0,0,0)
    #zero_offset = tuple(a-b for a,b in zip(center_voxel,scaled_center_voxel)) 

    ## find corrected voxel position in gtruth matrix
    #corrected_index = tuple(z+i for z,i in zip(zero_offset, tomo_index))

    #print("input index", tomo_index)
    #print("corrected index", corrected_index)

    ## look up identified voxel value in ground truth matrix
    #voxel_value = gtruth[corrected_index]

    ## generate label LUT and map identified voxel as a label
    #label_lut = make_label_lut(maxval=np.max(gtruth))

    ## convert voxel numerical value to label
    #voxel_label = label_lut[voxel_value]

    return score

def plot_all_axes(tomo:np.ndarray, idxs:Union[None, Tuple[int,int,int]]=None, hist:bool=False) -> None:
    """Convenience function to plot along 3 dimensions.
    
    Input:
       tomo: 3d numpy array to be plotted
       idxs: If None half of dimension-size is used, otherwise uses provided tuple
       hist: Boolean that determines whether to show histograms or images
    """

    if not idxs:
        cx, cy, cz = (i//2 for i in tomo.shape)
    else:
        cx, cy, cz = idxs
    # [z,y,x] -> e.g. [ver-axis, 200, hor-axis]
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4))
    if hist:
        ax[0].hist(tomo[cx,:,:].flatten(), bins=500);
        ax[0].set_xlabel("count"); ax[0].set_ylabel("intensity"); ax[0].set_title(f"histogram");
        ax[1].hist(tomo[:,cy,:].flatten(), bins=500);
        ax[1].set_xlabel("count"); ax[1].set_ylabel("intensity"); ax[1].set_title(f"histogram");
        ax[2].hist(tomo[:,:,cz].flatten(), bins=500);
        ax[2].set_xlabel("count"); ax[2].set_ylabel("intensity"); ax[2].set_title(f"histogram");
    else:
        ax[0].imshow(tomo[cx,:,:], cmap="viridis");
        ax[0].set_xlabel("x"); ax[0].set_ylabel("y"); ax[0].set_title(f"z={cz}");
        ax[1].imshow(tomo[:,cy,:], cmap="viridis");
        ax[1].set_xlabel("x"); ax[1].set_ylabel("z"); ax[1].set_title(f"y={cy}");
        ax[2].imshow(tomo[:,:,cz], cmap="viridis");
        ax[2].set_xlabel("y"); ax[2].set_ylabel("z"); ax[2].set_title(f"x={cx}");
    plt.tight_layout()
    plt.show()
    return

def minmaxnorm(arr:np.ndarray) -> np.ndarray:
    """Convert from arbitrary float range to uint16.
    
    Input:
       arr: Numpy array to be normalized
    Return:
       I:   Numpy array in range 0-65535 in uint16.
    """
    # first convert to 0-1 range
    I = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
    # multiply by max value in wanted uint16 range
    I *= int(2**16-1)
    I = I.astype(np.uint16)
    return I

def kmeans(vol:np.ndarray, nclasses:int) -> np.ndarray:
    """K-Means clustering.

    Input:
       vol:      3d or 2d numpy array containing either full tomogram or single slice
       nclasses: Determines how many classes to look for
    Return:
       tomo:     labelled tomogram
    
    From: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans

    Note that the algorithm is highly unpredictable and can vary wildly for
    different seeds of random_state.
    """
    print("... running KMeans segmentation")

    original_shape = vol.shape
    all_pixels = vol.reshape((-1,1))

    km = KMeans(random_state=1, n_clusters=nclasses)
    km.fit(all_pixels)

    centers = km.cluster_centers_

    # Convert to Integer format
    centers = np.array(centers,dtype='uint8')

    # Storing info in color array
    colors = []
    for each_col in centers:
        colors.append(each_col[0])

    new_img = np.zeros(original_shape,dtype='uint8').flatten()

    # Iterate over the image
    for ix in range(new_img.shape[0]):
        new_img[ix] = colors[km.labels_[ix]]

    new_img = new_img.reshape((original_shape))

    if nclasses == 4:
        new_img[new_img==1]=0
        new_img[new_img==4]=1
        new_img[new_img==2]=4
        new_img[new_img==3]=2
        # bone: 190.0
        # blood: 153.0
        # air: 113.0
        # implant: 23.0

    return new_img

def multiclass_otsu(vol:np.ndarray, nclasses:int, plot:bool) -> np.ndarray:
    """Multi-class Otsu thresholding.

    Input:
       vol:      3d or 2d numpy array containing either full tomogram or single slice
       nclasses: Determines how many classes to look for
    Return:
       tomo:     labelled tomogram

    Source paper: https://people.ece.cornell.edu/acharya/papers/mlt_thr_img.pdf
    Implemenation used: https://stackoverflow.com/a/53883887
                        developed by: Sujoy Kumar Goswami

    Note:
    It seems to work OK for most slices along the z-dimension, but not all. It has a hard time
    differentiating between blood and bone - which makes sense and is also good for us. Also it
    does not correlate any classes between slices in the z-direction either.
    """
    print("... running multi-class Otsu segmentation")
    # FIXME: screws up for nclasses<4 ??

    a = 0 # minimum value
    b = (2**16)-1 # maximum value
    n = nclasses # number of thresholds (better choose even value)
    k = 0.7 # free variable to take any positive value
    thresholds = [] # list which will contain 'n' thresholds

    def sujoy(image, a, b):
        if a>b:
            s=-1
            m=-1
            return m,s

        image = np.array(image)
        t1 = (image>=a)
        t2 = (image<=b)
        X = np.multiply(t1,t2)
        Y = np.multiply(image,X)
        s = np.sum(X)
        m = np.sum(Y)/s
        return m,s

    for i in range(int(n/2-1)):
        vol = np.array(vol)
        t1 = (vol>=a)
        t2 = (vol<=b)
        X = np.multiply(t1,t2)
        Y = np.multiply(vol,X)
        mu = np.sum(Y)/np.sum(X)

        Z = Y - mu
        Z = np.multiply(Z,X)
        W = np.multiply(Z,Z)
        sigma = math.sqrt(np.sum(W)/np.sum(X))

        T1 = mu - k*sigma
        T2 = mu + k*sigma

        x, y = sujoy(vol, a, T1)
        w, z = sujoy(vol, T2, b)

        thresholds.append(x)
        thresholds.append(w)

        a = T1+1
        b = T2-1
        k = k*(i+1)

    T1 = mu
    T2 = mu+1
    x, y = sujoy(vol, a, T1)
    w, z = sujoy(vol, T2, b)    
    thresholds.append(x)
    thresholds.append(w)
    thresholds.sort()

    #print(f"Multi-scale Otsu found {len(thresholds)} thresholds: {thresholds}.")

    # plotting part is copied from skimage multi-otsu example:
    # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html 


    # Using the threshold values, we generate the three regions.
    regions = np.digitize(vol, bins=thresholds)

    if plot:
        # make plot of: original image + histogram with obtained thresholds + segmentation map
        # FIXME: add legend to imshow with class 1,2,3 etc
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

        ax[0].imshow(vol, 'viridis')
        ax[0].set_title('Tomogram')

        ax[1].hist(vol.ravel(), bins=255, range=(1,vol.max()))
        ax[1].set_title('Histogram')
        for thresh in thresholds:
            ax[1].axvline(thresh, color='r', linestyle='--', alpha=0.5)

        ax[2].imshow(regions, cmap='jet')
        ax[2].set_title('Multi-Otsu segmentation')

        plt.subplots_adjust()
        plt.tight_layout()
        #plt.savefig("multi_otsu.png", facecolor="w", pad_inches=0, dpi=200)
        plt.show()

    # Other models don't know if label 1 should be bone or blood or ...
    # so we manually map their labels to match those arbitrarily chosen for the ground truth
    # such that they agree on what each label is called
    if nclasses == 4:
        regions[regions==1]=0
        regions[regions==4]=1
        regions[regions==2]=4
        regions[regions==3]=2

    return regions

def simplify_tomo(tomo:np.ndarray, level:int) -> np.ndarray:
    """Mask away components of tomogram, based on chosen level.

    Input:
       tomo:  Numpy array containig 3d tomogram
       level: Integer in range {0,1,2,3} with increasing numbers removing more
              and more components.
    Return:
       tomo:  Masked numpy array containing 3d tomogram with fewer components.
              All masked components have been mapped to zero.
    
    When comparing two tomograms, we sometimes want to remove certain
    components, to make the comparison more fair. For example when comparing
    with other segmentation methods, we introduce multiple levels of masking, to
    verify what difference it makes to keep air, implant and resin. This helps
    to illustrate how robust a method is, by making the segmentation problem
    harder/easier dependent on the number of components.
    
    Note that when masking away regions, we do not mask any effects/artifacts
    resulting from these regions. For example, when masking away the implant, we
    still see    the bleeding edges that have altered intensity values from the
    implant.
    """ 

    with h5py.File("masks/bone_masks/770c_pag_8x.h5", "r") as hf:
        #cut_cylinder_bone = hf["cut_cylinder_bone/mask"][:].astype(np.uint8)
        cut_cylinder_air = hf["cut_cylinder_air/mask"][:].astype(np.uint8)
        bone_region = hf["bone_region/mask"][:].astype(np.uint8)
        implant_region = hf["implant/mask"][:].astype(np.uint8)

    if level == 0:
        # do not remove anything
        pass
    elif level == 1:
        # remove back mask
        # classes remaining: blood + bone + implant + air
        tomo[cut_cylinder_air.astype(bool)] = 0
    elif level == 2:
        # as above + remove edges in opposite end
        # classes remaining: blood + bone + implant
        new_mask = bone_region + implant_region
        new_mask[new_mask==2] = 0 # remove small overlap (unphysical)
        tomo[~new_mask.astype(bool)] = 0
    elif level == 3:
        # as above + remove implant
        # classes remaining: blood + bone
        tomo[~bone_region.astype(bool)] = 0
    return tomo

def compare_tomograms(segm, gtruth, zslice):
    zslice = 216
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
    ax[0].imshow(segm[zslice,:,:]); ax[0].set_title("segmentation")
    ax[1].imshow(gtruth[zslice,:,:]); ax[1].set_title("gtruth")
    plt.suptitle("Visual comparison of segmentation and ground truth")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    
    # load tomogram
    tomo = mmap(fpath="tomograms/rec_432x432x409.raw",
                tshape=(432,432,409),
                tslice=None,
                axis=None)

    # map to uint16
    tomo = minmaxnorm(tomo)

    # verify tomogram
    plot_all_axes(tomo)
    #plot_all_axes(tomo, hist=True)

    """ load ground truth matrix """

    # load corresponding ground truth matrix
    with h5py.File("masks/gtruth.h5", "r") as hf:
        gtruth = hf["data"][:]

    # verify ground truth matrix
    plot_all_axes(gtruth)

    """ Alternative segmentation methods """

    # prepare volumes
    level = 0
    nclasses = 4
    # note that for:
    # level=0 there are maximally n=4 classes
    # level=1 there are maximally n=4 classes
    # level=2 there are maximally n=3 classes
    # level=3 there are maximally n=2 classes

    tomo = simplify_tomo(tomo=tomo, level=level)
    gtruth = simplify_tomo(tomo=gtruth, level=level)

    # Multi-scale Otsu
    segm = multiclass_otsu(vol=tomo, nclasses=nclasses, plot=False)

    # KMeans
    #segm = kmeans(vol=tomo, nclasses=nclasses)

    """ output score """

    score = accuracy_score(segm=segm, gtruth=gtruth)
    print("score: ", score)
    
    # visual comparison
    compare_tomograms(segm, gtruth, zslice=216)
