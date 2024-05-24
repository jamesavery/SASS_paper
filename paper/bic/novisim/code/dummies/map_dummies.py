import h5py
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

def plot_3_axes():
    tomo_0 = mmap(fpath="rec_800x800x800_centered.raw",
                tshape=(800,800,800),
                tslice=400,
                axis=0)
    tomo_1 = mmap(fpath="rec_800x800x800_centered.raw",
                tshape=(800,800,800),
                tslice=400,
                axis=1)
    tomo_2 = mmap(fpath="rec_800x800x800_centered.raw",
                tshape=(800,800,800),
                tslice=400,
                axis=2)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4))
    ax[0].imshow(tomo_0); ax[0].set_title("axis 0");
    ax[1].imshow(tomo_1); ax[1].set_title("axis 1");
    ax[2].imshow(tomo_2); ax[2].set_title("axis 2");
    plt.show()
    return

if __name__ == "__main__":
    plot_3_axes()

