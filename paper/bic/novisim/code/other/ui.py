import cv2
import numpy as np
import matplotlib.pyplot as plt

def mmap(fpath, tshape, tslice, axis):
    """Efficient memory mapped partial volumes."""
    assert axis <= 2
    mread = np.memmap(fpath, dtype='float32', mode='r', shape=tshape, order="F")
    mread = mread.swapaxes(0,1)
    mread = mread.swapaxes(1,2)
    sliceidx = tuple(np.s_[tslice] if s==axis else np.s_[:] for s in range(3))
    data = np.array(mread[sliceidx])
    return data

global s
global d

# define callbacks
def change_slice(val):
    pass
def change_dim(val):
    pass

if __name__ == "__main__":
    """ setup window parameters """
    # load single slice to get dimensions
    tdim = (800,800,800)
    s = 0
    d = 0

    """ create window """
    title_window = "Novi-sim output tomogram sweep"
    title_s1 = "Tomogram slice"
    title_s2 = "Tomogram dim"
    cv2.namedWindow(title_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_window, 500,500)
    # FIXME: check which dim was chosen in s2 slider, and use its max value for s1
    cv2.createTrackbar(title_s1, title_window, 0, 250-1, change_slice)
    cv2.createTrackbar(title_s2, title_window, 0, 2, change_slice)

    """ main loop """
    while True:
        s = cv2.getTrackbarPos(title_s1, title_window)
        d = cv2.getTrackbarPos(title_s2, title_window)

        # update image
        tomo = mmap(fpath="tomograms/latest_22_may/green_numbers/rec_800x800x800.raw", tshape=tdim, tslice=s, axis=d)
        # convert to uint8
        tomo = cv2.normalize(src=tomo, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        tomo = cv2.applyColorMap(tomo, cv2.COLORMAP_JET) # requires uint8, currently float32
        cv2.imshow(title_window, tomo)

        # close properly
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
