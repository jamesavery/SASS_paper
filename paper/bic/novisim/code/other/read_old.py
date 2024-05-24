import numpy as np

def read_full(fpath, tsize):
    """
    Requires that full file can fit in memory.

    # format contains no header
    # for file of size 108 MB, with tomogram of size 300x300x300:
    # 108 MB / (300*300*300) = 4 bytes = 32 bits per voxel
    """
    raw = np.fromfile(fpath, dtype=np.float32)
    tomogram = raw.reshape(tsize) # tsize=(x,y,z)
    return tomogram

def read_partial(fpath, tsize):
    """
    Explicit reading of partial slice. Mostly done for fun, as it highly
    depends on ordering of elements.
    """
    with open(fpath, mode='rb') as infile:
        slice_num = tsize[2] # being explicit
        start_index = slice_num*tsize[0]*tsize[1]*4 # 4 bytes -> float32
        infile.seek(start_index, 0) # from beginning of file
        # notice count already takes into account that data is f32, so no multiply by 4
        data = np.fromfile(infile, count=tsize[0]*tsize[1], dtype=np.float32)
    tomogram= data.reshape(tsize[0], tsize[1])
    return tomogram
