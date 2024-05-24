import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

def main():

    # data comes from novi-sim in (z,x,y), we swap to get (x,y,z)

    # read full tomogram
    tomo = np.fromfile("tomograms/8x_tomo_novisim.raw", dtype=np.float32).reshape((432,432,409), order="F")
    tomo = tomo.swapaxes(0,1)
    tomo = tomo.swapaxes(1,2)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4))
    ax[0].imshow(tomo[175,:,:], cmap="gray"); ax[0].set_title("plane through index 0\nalong depth of implant");
    ax[1].imshow(tomo[:,125,:], cmap="gray"); ax[1].set_title("plane through index 1\nalong short side of implant");
    ax[2].imshow(tomo[:,:,125], cmap="gray"); ax[2].set_title("plane through index 2\nalong wide side of implant");
    plt.tight_layout(); plt.show()


    #import h5py
    #with h5py.File("carl.h5", "w") as hf:
    #    hf.create_dataset('tomo', data=tomo)

    return

if __name__ == "__main__":
    main()
