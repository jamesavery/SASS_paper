import matplotlib
matplotlib.use('Agg')
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal
from skimage.filters import threshold_otsu
from tqdm import tqdm
import vedo

base_dir = '/data_fast/MAXIBONE/Goats/tomograms'
output_dir = '../generated'
sample = '770c_pag'
plane_cmap = 'RdYlBu'
histogram_cmap = 'coolwarm'

# sample : [[ystart, yend, xstart, xend], [ystart, yend, zstart, zend], [xstart, xend, zstart, zend]]
crop_map = {
    '770_pag' : [ np.array((3000, 4000, 3000, 4000)), np.array((0, 2000, 1000, 3000)), np.array((0, 2000, 3000, 4000))],
}
default_crop = [ np.array((3000, 4000, 1000, 3000)), np.array((3000, 4000, 1500, 2500)), np.array((3000, 4500, 3500, 5000)) ]

# Colors
red = np.array([255, 0, 0])
orange = np.array([255, 128, 0])
blue = np.array([64, 128, 255])
yellow = np.array([255, 255, 0])
gray = np.array([32, 32, 32])
green = np.array([0, 255, 0])

# Segmentation color parameters
blood_base = .7
blood_boost = 1
bone_base = .7
bone_boost = 1

os.makedirs(output_dir, exist_ok=True)

def full_planes(voxels, voxel_size):
    nz, ny, nx = voxels.shape
    names = ['yx', 'zx', 'zy']
    planes = [ voxels[nz//2,:,:], voxels[:,ny//2,:], voxels[:,:,nx//2] ]
    colorbar_scales = [ .8, .75, .75 ]

    for name, plane, colorbar_scale in zip(names, planes, colorbar_scales):
        ax0, ax1 = name
        plt.figure(figsize=(10, 10))
        plt.imshow(plane, cmap=plane_cmap)
        plt.ylabel(f'{ax0}/µm')
        plt.xlabel(f'{ax1}/µm')
        xticks = np.array(plt.xticks()[0][1:-1])
        xticks_labels = np.round(xticks * voxel_size).astype(int)
        plt.xticks(xticks, xticks_labels)
        yticks = np.array(plt.yticks()[0][1:-1])
        yticks_labels = np.round(yticks * voxel_size).astype(int)
        plt.yticks(yticks, yticks_labels)
        plt.colorbar(shrink=colorbar_scale, aspect=20*colorbar_scale)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{sample}_full_{name}.pdf', bbox_inches='tight')
        plt.clf()

def roi_planes(voxels, voxel_size):
    nz, ny, nx = voxels.shape
    names = ['yx', 'zx', 'zy']
    planes = [ voxels[nz//2,:,:], voxels[:,ny//2,:], voxels[:,:,nx//2] ]
    if sample in crop_map:
        crops = crop_map[sample]
    else:
        crops = default_crop
    crops = [ np.floor(crop // voxel_size).astype(int) for crop in crops ]
    colorbar_scales = [ .4, .8, .8 ]

    for name, plane, crop, colorbar_scale in zip(names, planes, crops, colorbar_scales):
        ax0, ax1 = name
        plt.figure(figsize=(10, 10))
        plt.imshow(plane[crop[0]:crop[1], crop[2]:crop[3]], cmap=plane_cmap)
        plt.ylabel(f'{ax0}/µm')
        plt.xlabel(f'{ax1}/µm')
        xticks = np.array(plt.xticks()[0][1:-1])
        xticks_labels = np.round((xticks + crop[2]) * voxel_size).astype(int)
        plt.xticks(xticks, xticks_labels)
        yticks = np.array(plt.yticks()[0][1:-1])
        yticks_labels = np.round((yticks + crop[0]) * voxel_size).astype(int)
        plt.yticks(yticks, yticks_labels)
        plt.colorbar(shrink=colorbar_scale, aspect=20*colorbar_scale)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{sample}_roi_{name}.pdf', bbox_inches='tight')
        plt.clf()

def hist_1d(voxels):
    vmin, vmax = np.min(voxels), np.max(voxels)

    bins, hist = np.histogram(voxels, bins=256)
    # Make bins percentages
    bins[0] = 0
    bins = (bins / np.sum(bins)) * 100

    # sample : air, blood, bone, implant
    peak_map = {
        '770_pag' : [0, 1, 2, 3],
        '770c_pag' : [(0,1), 3, 4, 5],
        '771c_pag' : [(0, 1), 2, 3, 4],
        '772_pag' : [None, 0, 1, 2],
        '775c_pag' : [(0, 1), 2, 3, 4],
        '810c_pag' : [(0, 1), 2, 3, (4,5)],
        '811_pag' : [(0, 1), (2,3), 5, 6],
    }

    # Find the peaks
    peaks, info = scipy.signal.find_peaks(bins, height=0.1)
    scaled_peaks = np.array(peaks) * (vmax - vmin) / 256 + vmin
    #air = (scaled_peaks[0] + scaled_peaks[1]) / 2
    print (peaks)
    if len(scaled_peaks) == 4:
        peak_air, peak_blood, peak_bone, peak_implant = 0, 1, 2, 3
    else:
        peak_air, peak_blood, peak_bone, peak_implant = peak_map[sample]
    if type(peak_air) is tuple:
        air = np.mean([scaled_peaks[i] for i in peak_air])
    elif peak_air is None:
        air = 0
    else:
        air = scaled_peaks[peak_air]
    if type(peak_blood) is tuple:
        blood = np.mean([scaled_peaks[i] for i in peak_blood])
    else:
        blood = scaled_peaks[peak_blood]
    if type(peak_bone) is tuple:
        bone = np.mean([scaled_peaks[i] for i in peak_bone])
    else:
        bone = scaled_peaks[peak_bone]
    if type(peak_implant) is tuple:
        implant = np.mean([scaled_peaks[i] for i in peak_implant])
    else:
        implant = scaled_peaks[peak_implant]
    xmin = hist[0]
    xmax = hist[-1]
    ymin = -0.1
    ymax = np.max(bins) * 1.05

    plt.figure(figsize=(8, 4))
    plt.vlines([air], ymin, ymax, colors='blue')
    plt.vlines([blood], ymin, ymax, colors='red')
    plt.vlines([bone], ymin, ymax, colors='orange')
    plt.vlines([implant], ymin, ymax, colors='green')
    plt.plot(hist[:-1], bins, color='black')
    plt.xlabel('Voxel intensity')
    plt.ylabel('Count (%)')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{sample}_hist_1d.pdf', bbox_inches='tight')
    plt.clf()

def row_normalize(A):
    return A/(1+np.max(A,axis=1))[:,np.newaxis]

def hist_2d(sample, mask):
    histograms = np.load(f'{base_dir}/processed/histograms/{sample}/bins{mask}.npz')

    axes = histograms['axis_names']
    axes_labels = [f'{axis} plane coordinate' for axis in axes]
    fields = histograms['field_names']
    fields_labels = [f'Distance from implant ({field})' for field in fields]

    hists = [(axis, axes_labels[i], histograms[f'{axis}_bins']) for i, axis in enumerate(axes)] + [(field, fields_labels[i], histograms[f'field_bins'][i][::-1] if 'gauss' in field else histograms[f'field_bins'][i]) for i, field in enumerate(fields)]

    colorbar_scale = 1

    for axis, name, hist in hists:
        hist = row_normalize(hist)
        plt.figure(figsize=(10, 10))
        plt.imshow(hist, cmap=histogram_cmap)
        plt.xlabel('Voxel intensity')
        plt.ylabel(f'{name}')
        plt.colorbar(shrink=colorbar_scale, aspect=20*colorbar_scale)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{sample}_hist_2d_{axis}.pdf', bbox_inches='tight')
        plt.clf()

def implant_mask(sample, implant_scale):
    voxel_size = 1.825 * implant_scale
    masks_file = h5py.File(f'{base_dir}/masks/{implant_scale}x/{sample}.h5', 'r')

    names = [ 'implant', 'cut_cylinder_air', 'bone_region' ]
    masks = [ masks_file[name]['mask'] for name in names ]

    nz, ny, nx = masks[0].shape

    for name, mask in zip(names, masks):
        plt.figure(figsize=(10, 10))
        plt.imshow(mask[:,:,nx//2], cmap='gray')
        plt.ylabel(f'z/µm')
        plt.xlabel(f'y/µm')
        xticks = np.array(plt.xticks()[0][1:-1])
        xticks_labels = np.round(xticks * voxel_size).astype(int)
        plt.xticks(xticks, xticks_labels)
        yticks = np.array(plt.yticks()[0][1:-1])
        yticks_labels = np.round(yticks * voxel_size).astype(int)
        plt.yticks(yticks, yticks_labels)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{sample}_mask_{name}_zy.pdf', bbox_inches='tight')
        plt.clf()

def diffusion_slice(scale):
    voxel_size = 1.825 * scale
    field = np.load(f'{base_dir}/binary/fields/implant-gauss/{scale}x/{sample}.npy', mmap_mode='r')
    nz, ny, nx = field.shape
    plane = field[:,:,nx//2]
    plane = plane / np.max(plane)
    colorbar_scale = .75

    plt.figure(figsize=(10, 10))
    plt.imshow(plane, cmap=histogram_cmap)
    plt.xlabel('z/µm')
    plt.ylabel('y/µm')
    xticks = np.array(plt.xticks()[0][1:-1])
    xticks_labels = np.round(xticks * voxel_size).astype(int)
    plt.xticks(xticks, xticks_labels)
    yticks = np.array(plt.yticks()[0][1:-1])
    yticks_labels = np.round(yticks * voxel_size).astype(int)
    plt.yticks(yticks, yticks_labels)
    plt.colorbar(shrink=colorbar_scale, aspect=20*colorbar_scale)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{sample}_diffusion_zy.pdf', bbox_inches='tight')

def otsu():
    with h5py.File(f'{base_dir}/processed/probabilities/{sample}.h5', 'r') as f:
        group = f['otsu_separation']['bone_region']['field_bins_gauss+edt']
        threshes = group['threshes']
        fitted_threshes = group['new_threshes']
        histograms = np.load(f'{base_dir}/processed/histograms/{sample}/bins-bone_region.npz')
        hist = histograms['field_bins'][2] # 2 is gauss+edt
        hist = row_normalize(hist)
        display = np.zeros(hist.shape + (3,), dtype=np.uint8)
        display[:,:] = ((hist / (hist.max(axis=1)[:,np.newaxis]+1)) * 255)[:,:,np.newaxis]
        for i, thresh in enumerate(threshes):
            display[i,int(thresh)-2:int(thresh)+2] = orange # otsu is orange
            display[i,int(fitted_threshes[i])-2:int(fitted_threshes[i])+2] = blue # constant is blue

        plt.figure(figsize=(10, 10))
        plt.imshow(display[::-1])
        plt.xlabel('Voxel intensity')
        plt.ylabel('Distance from implant (Gauss+EDT)')
        #plt.colorbar(shrink=0.8, aspect=20*0.8)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{sample}_otsu.pdf', bbox_inches='tight')
        plt.clf()

        p0 = group['P0'][:]
        p1 = group['P1'][:]
        ps = np.floor((p0[:,:,np.newaxis] * red) + (p1[:,:,np.newaxis] * yellow)).astype(np.uint8)

        plt.figure(figsize=(10, 10))
        plt.imshow(ps[::-1])
        plt.xlabel('Voxel intensity')
        plt.ylabel('Distance from implant (Gauss+EDT)')
        #plt.colorbar(shrink=0.8, aspect=20*0.8)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{sample}_probabilities.pdf', bbox_inches='tight')
        plt.clf()

def blood_network(voxel_size):
    scale = 2
    voxel_size = 1.825 * scale
    f = h5py.File(f'{base_dir}/masks/{scale}x/{sample}_sub2.h5', 'r')
    m = f['blood']['mask']
    nz, ny, nx = m.shape
    mm = int(1000 // voxel_size)

    split = nz//4
    ycut = int(np.floor(ny*.45))
    ymid = ycut + ((ny-ycut)//2)
    xmid = nx//2
    print (split, ycut, ymid, xmid, mm)

    vols = [
        ('new_bone', m[:split,ycut:,:].astype(np.uint8)),
        ('old_bone', m[-split:,ycut:,:].astype(np.uint8)),
        ('new_cube', m[:mm,ymid-(mm//2):ymid+(mm//2),xmid-(mm//2):xmid+(mm//2)].astype(np.uint8)),
        ('old_cube', m[-mm:,ymid-(mm//2):ymid+(mm//2),xmid-(mm//2):xmid+(mm//2)].astype(np.uint8))
    ]

    interactive = False

    for name, vol in vols:
        print (name, vol.shape, vol.dtype, np.min(vol), np.max(vol))

        plt = vedo.Plotter() if interactive else vedo.Plotter(offscreen=True)

        v = vedo.Volume(vol)
        bone_cams = {
            2 : dict(
                position=(-2580.06, -1561.14, 31.3219),
                focal_point=(159.441, 302.471, 894.971),
                viewup=(-0.596846, 0.764510, 0.243515),
                distance=3424.00,
                clipping_range=(2359.06, 4772.81),
            ),
            8 : dict(
                position=(-449.814, -464.983, -172.953),
                focal_point=(3.77204, 117.000, 227.564),
                viewup=(-0.807268, 0.586995, 0.0612818),
                distance=839.559,
                clipping_range=(449.673, 1332.37),
            )
        }
        cube_cams = {
            2 : dict(
                position=(463.986, 726.176, 676.861),
                focal_point=(143.584, 141.123, 125.133),
                viewup=(-0.736208, 0.631831, -0.242461),
                distance=865.649,
                clipping_range=(402.720, 1450.47),
            )
        }

        if interactive:
            plt.show(v)
            plt.interactive().close()
        else:
            cam = bone_cams[scale] if name.endswith('bone') else cube_cams[scale]
            plt.show(v, camera=cam)
            plt.screenshot(f'{output_dir}/sample_{name}.png')
            plt.close()

    f.close()

def segmented_slices(scale):
    voxels, voxel_size = load_voxels(sample, scale)
    del voxels
    postfix = ''
    brightness = 0
    implant_boost = 2
    blood_base = .7
    blood_boost = 1
    bone_base = .7
    bone_boost = 1

    with h5py.File(f'{base_dir}/masks/{scale}x/{sample}{postfix}.h5') as f:
        implant = f['implant']['mask']
        nz, ny, nx = implant.shape
        implants = [
            implant[nz//2,:,:],
            implant[:,ny//2,:],
            implant[:,:,nx//2]
        ]

    voxels = np.memmap(f'{base_dir}/binary/voxels/{scale}x/{sample}{postfix}.uint16', dtype=np.uint16, mode='r', shape=(nz, ny, nx))

    names = ['yx', 'zx', 'zy']
    planes = [
        voxels[nz//2,:,:],
        voxels[:,ny//2,:],
        voxels[:,:,nx//2]
    ]
    if sample in crop_map:
        crops = crop_map[sample]
    else:
        crops = default_crop
    crops = [ np.floor((crop // voxel_size)).astype(int) for crop in crops ]

    for name, plane, crop in zip(names, planes, crops):
        ax0, ax1 = name
        plt.figure(figsize=(10, 10))
        plt.imshow(plane[crop[0]:crop[1],crop[2]:crop[3]], cmap=plane_cmap)
        plt.ylabel(f'{ax0}/µm')
        plt.xlabel(f'{ax1}/µm')
        xticks = np.array(plt.xticks()[0][1:-1])
        xticks_labels = np.round((xticks + crop[2]) * voxel_size).astype(int)
        plt.xticks(xticks, xticks_labels)
        yticks = np.array(plt.yticks()[0][1:-1])
        yticks_labels = np.round((yticks + crop[0]) * voxel_size).astype(int)
        plt.yticks(yticks, yticks_labels)
        #plt.colorbar(shrink=0.8, aspect=20*0.8)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{sample}_segmented_{name}_raw.pdf', bbox_inches='tight')
        plt.clf()

    p0 = np.memmap(f'{base_dir}/binary/segmented/gauss+edt/P0/{scale}x/{sample}{postfix}.uint16', dtype=np.uint16, mode='r', shape=(nz, ny, nx))
    p1 = np.memmap(f'{base_dir}/binary/segmented/gauss+edt/P1/{scale}x/{sample}{postfix}.uint16', dtype=np.uint16, mode='r', shape=(nz, ny, nx))

    bloods = [
        p0[nz//2,:,:],
        p0[:,ny//2,:],
        p0[:,:,nx//2]
    ]
    bones = [
        p1[nz//2,:,:],
        p1[:,ny//2,:],
        p1[:,:,nx//2]
    ]
    bloods = [ p / 65536 for p in bloods ]
    bones = [ p / 65536 for p in bones ]

    for name, implant, blood, bone, crop in zip(names, implants, bloods, bones, crops):
        ax0, ax1 = name
        display = np.zeros(implant.shape + (3,), dtype=np.float64)
        display += brightness # Improve overall brightness
        #display += implant[:,:,np.newaxis] * gray * implant_boost
        blo = (blood[:,:,np.newaxis] + blood_base )
        blo[blo < blood_base + .001] = 0
        blo /= blo.max()
        display += blo * red * blood_boost
        bon = (bone[:,:,np.newaxis] + bone_base )
        bon[bon < bone_base + .001] = 0
        bon /= bon.max()
        display += bon * yellow * bone_boost
        display = np.clip(display, 0, 255).astype(np.uint8)
        plt.figure(figsize=(10, 10))
        plt.imshow(display[crop[0]:crop[1],crop[2]:crop[3]])
        plt.ylabel(f'{ax0}/µm')
        plt.xlabel(f'{ax1}/µm')
        xticks = np.array(plt.xticks()[0][1:-1])
        xticks_labels = np.round((xticks + crop[2]) * voxel_size).astype(int)
        plt.xticks(xticks, xticks_labels)
        yticks = np.array(plt.yticks()[0][1:-1])
        yticks_labels = np.round((yticks + crop[0]) * voxel_size).astype(int)
        plt.yticks(yticks, yticks_labels)
        #plt.colorbar(shrink=0.8, aspect=20*0.8)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{sample}_segmented_{name}_colored.pdf', bbox_inches='tight')
        plt.clf()

def load_voxels(sample, scale):
    with h5py.File(f'{base_dir}/hdf5-byte/msb/{sample}.h5', 'r') as f:
        total_shape = np.array(f['voxels'].shape)
        total_shape[0] -= np.sum(f['volume_matching_shifts'][:])
        voxel_size = f['voxels'].attrs['voxelsize']
    global_shape = total_shape // scale

    return np.memmap(f'{base_dir}/binary/voxels/{scale}x/{sample}.uint16', dtype=np.uint16, mode='r', shape=tuple(global_shape)), voxel_size * scale

def global_thresholding(scale, hist_scale, mask_scale):
    with h5py.File(f'{base_dir}/masks/{hist_scale}x/{sample}.h5', 'r') as f:
        mask = f['bone_region']['mask'][:]

    voxels, voxel_size = load_voxels(sample, hist_scale)
    voxels = voxels * mask

    bins, hist = np.histogram(voxels, bins=256, range=(1, voxels.max()))
    # Smooth bins before finding peaks
    bins = scipy.ndimage.gaussian_filter1d(bins, 2)
    peaks = scipy.signal.find_peaks(bins, height=0.01)[0]
    print (peaks)
    # Find the valley between the first two peaks
    valley = np.argmin(bins[peaks[0]:peaks[1]]) + peaks[0]
    print (valley)
    thresh = hist[valley]

    split = np.argwhere(hist > thresh)[0][0]-1
    soft = bins[:split]
    hard = bins[split:]
    soft = (soft - soft.min()) / (soft.max() - soft.min())
    soft += blood_base
    soft[soft < 0 + .001] = 0
    soft /= soft.max()
    soft *= blood_boost
    hard = (hard - hard.min()) / (hard.max() - hard.min())
    hard += bone_base
    hard[hard < 0 + .001] = 0
    hard /= hard.max()
    hard *= bone_boost
    probs = np.zeros(bins.shape + (3,), dtype=np.float64)
    probs[:split] = soft[:,np.newaxis] * red
    probs[split:] = hard[:,np.newaxis] * yellow
    probs = np.clip(probs, 0, 255).astype(np.uint8)

    plt.plot(hist[:-1], bins)
    plt.vlines([hist[peaks[0]], hist[peaks[1]]], 0, np.max(bins), colors='blue')
    plt.vlines([hist[valley]], 0, np.max(bins), colors='red')
    plt.savefig(f'{output_dir}/{sample}_global_hist_1d.pdf', bbox_inches='tight')
    plt.clf()

    if sample == '772_pag':
        thresh = hist[136] # Manually extracted

    voxels, voxel_size = load_voxels(sample, scale)
    nz, ny, nx = voxels.shape
    names = ['yx', 'zx', 'zy']
    planes = [ voxels[nz//2,:,:], voxels[:,ny//2,:], voxels[:,:,nx//2] ]
    with h5py.File(f'{base_dir}/masks/{mask_scale}x/{sample}.h5', 'r') as f:
        mask = f['bone_region']['mask'][:]
    bone_masks = [ mask[nz//mask_scale//2,:,:], mask[:,ny//mask_scale//2,:], mask[:,:,nx//mask_scale//2] ]
    colorbar_scales = [ .8, .75, .75 ]
    if sample in crop_map:
        crops = crop_map[sample]
    else:
        crops = default_crop
    crops = [ np.floor((crop // voxel_size)).astype(int) for crop in crops ]

    for name, plane, bone_mask, crop, colorbar_scale in zip(names, planes, bone_masks, crops, colorbar_scales):
        ax0, ax1 = name

        # Step 1: Find bin_idx for each entry in plane
        # Expand dimensions of plane and hist for broadcasting
        plane_expanded = plane[..., np.newaxis]  # Add an axis for broadcasting
        hist_expanded = hist[np.newaxis, :]  # Add an axis for broadcasting

        # Create a boolean array where True indicates hist > e
        hist_comparison = hist_expanded > plane_expanded

        # Find the first True index along the last axis (hist axis)
        bin_idx = np.argmax(hist_comparison, axis=-1)

        # Step 2: Find the corresponding probability in probs for each bin_idx
        segmented = probs[bin_idx-1]

        # 'probabilities' now contains the corresponding probability for each entry in 'plane'

        # Global thresholding
        #segmented = np.zeros(plane.shape + (3,), dtype=np.uint8)
        #segmented[plane < thresh] = red
        #segmented[plane >= thresh] = yellow
        this_front = np.repeat(np.repeat(bone_mask, mask_scale, axis=0), mask_scale, axis=1)
        segmented = segmented[:this_front.shape[0]] * this_front[:,:,np.newaxis]

        plt.figure(figsize=(10, 10))
        plt.imshow(segmented[crop[0]:crop[1],crop[2]:crop[3]])
        plt.ylabel(f'{ax0}/µm')
        plt.xlabel(f'{ax1}/µm')
        xticks = np.array(plt.xticks()[0][1:-1])
        xticks_labels = np.round((xticks + crop[2]) * voxel_size).astype(int)
        plt.xticks(xticks, xticks_labels)
        yticks = np.array(plt.yticks()[0][1:-1])
        yticks_labels = np.round((yticks + crop[0]) * voxel_size).astype(int)
        plt.yticks(yticks, yticks_labels)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{sample}_global_{name}.pdf', bbox_inches='tight')
        plt.clf()

def otsu_thresholding(scale, mask_scale):
    voxels, voxel_size = load_voxels(sample, scale)

    with h5py.File(f'{base_dir}/masks/{mask_scale}x/{sample}.h5', 'r') as f:
        mask = f['bone_region']['mask'][:]
    bone_region = np.repeat(np.repeat(np.repeat(mask, mask_scale, axis=0), mask_scale, axis=1), mask_scale, axis=2)

    nz, ny, nx = bone_region.shape

    names = ['yx', 'zx', 'zy']
    block_size = 50

    slabs = [
        voxels[nz//2-block_size:nz//2+block_size,:,:],
        voxels[:-1,ny//2-block_size:ny//2+block_size,:].transpose(1,0,2),
        voxels[:-1,:,nx//2-block_size:nx//2+block_size].transpose(2,0,1)
    ]
    mask_slabs = [
        bone_region[nz//2-block_size:nz//2+block_size,:,:],
        bone_region[:,ny//2-block_size:ny//2+block_size,:].transpose(1,0,2),
        bone_region[:,:,nx//2-block_size:nx//2+block_size].transpose(2,0,1)
    ]
    slabs = [ mask_slabs[i] * slabs[i] for i in range(3) ]

    crops = [ np.floor((crop // voxel_size)).astype(int) for crop in default_crop ]

    for name, slab, crop, mask in zip(names, slabs, crops, mask_slabs):
        ax0, ax1 = name
        th3 = np.zeros(slab.shape[1:] + (3,), dtype=np.uint8)
        for y in tqdm(range(crop[0], crop[1])):
            for x in range(crop[2], crop[3]):
                ystart, yend = max(y-block_size, 0), min(y+block_size, slab.shape[1])
                xstart, xend = max(x-block_size, 0), min(x+block_size, slab.shape[2])
                window = slab[:, ystart:yend, xstart:xend]
                ma = np.ma.masked_array(window, window == 0)
                if ma.count() == 0:
                    continue
                bins_otsu, hist_otsu = np.histogram(ma.compressed(), bins=128, range=(1, window.max()))
                thr = threshold_otsu(ma.compressed())
                if np.sum(hist_otsu > thr) == 0:
                    continue
                split = np.argwhere(hist_otsu > thr)[0][0]-1
                soft, hard = bins_otsu[:split], bins_otsu[split:]
                # Normalize and adjust soft and hard segments
                soft = np.clip((soft - soft.min()) / (soft.ptp() + .001), 0, None) * blood_boost + blood_base
                hard = np.clip((hard - hard.min()) / (hard.ptp() + .001), 0, None) * bone_boost + bone_base
                # soft = bins_otsu[:split]
                # hard = bins_otsu[split:]
                # soft = (soft - soft.min()) / (soft.max() - soft.min())
                # soft += blood_base
                # soft[soft < blood_base + .001] = 0
                # soft /= soft.max()
                # soft *= blood_boost
                # hard = (hard - hard.min()) / (hard.max() - hard.min())
                # hard += bone_base
                # hard[hard < bone_base + .001] = 0
                # hard /= hard.max()
                # hard *= bone_boost
                otsu_probs = np.concatenate([soft, hard])
                # otsu_probs = np.zeros_like(bins_otsu, dtype=np.float64)
                # otsu_probs[:split] = soft
                # otsu_probs[split:] = hard
                pixel = slab[block_size, y, x]

                if np.sum(hist_otsu > pixel) == 0:
                    continue
                bin_idx = np.argwhere(hist_otsu > pixel)[0][0]
                p = otsu_probs[bin_idx-1]
                if pixel > thr:
                    th3[y,x] = yellow * p
                else:
                    th3[y,x] = red * p
        th3 = th3 * mask[block_size,:,:,np.newaxis]

        plt.figure(figsize=(10, 10))
        plt.imshow(th3[crop[0]:crop[1],crop[2]:crop[3]], cmap='gray')
        plt.ylabel(f'{ax0}/µm')
        plt.xlabel(f'{ax1}/µm')
        xticks = np.array(plt.xticks()[0][1:-1])
        xticks_labels = np.round((xticks + crop[2]) * voxel_size).astype(int)
        plt.xticks(xticks, xticks_labels)
        yticks = np.array(plt.yticks()[0][1:-1])
        yticks_labels = np.round((yticks + crop[0]) * voxel_size).astype(int)
        plt.yticks(yticks, yticks_labels)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{sample}_global_{name}_otsu.pdf', bbox_inches='tight')
        plt.clf()

def bone_health(scale=1):
    voxels, voxel_size = load_voxels(sample, scale)
    nz, ny, nx = voxels.shape
    bone_prob = np.memmap(f'{base_dir}/binary/segmented/gauss+edt/P1/{scale}x/{sample}.uint16', dtype=np.uint16, mode='r').reshape((nz, ny, nx))
    health_mask = np.load(f'{base_dir}/binary/fields/healthy_bone/{scale}x/{sample}.npy', mmap_mode='r')

    names = ['yx', 'zx', 'zy']
    bone_planes = [ bone_prob[nz//2,:,:], bone_prob[:,ny//2,:], bone_prob[:,:,nx//2] ]
    health_planes = [ health_mask[nz//2,:,:], health_mask[:,ny//2,:], health_mask[:,:,nx//2] ]

    for name, bone_plane, health_plane in zip(names, bone_planes, health_planes):
        display = np.zeros(bone_plane.shape + (3,), dtype=np.uint8)
        display[bone_plane > 0] = yellow
        display[health_plane > 0] = green

        plt.figure(figsize=(10, 10))
        plt.imshow(display)
        plt.ylabel(f'{name[0]}/µm')
        plt.xlabel(f'{name[1]}/µm')
        xticks = np.array(plt.xticks()[0][1:-1])
        xticks_labels = np.round(xticks * voxel_size).astype(int)
        plt.xticks(xticks, xticks_labels)
        yticks = np.array(plt.yticks()[0][1:-1])
        yticks_labels = np.round(yticks * voxel_size).astype(int)
        plt.yticks(yticks, yticks_labels)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{sample}_bone_health_{name}.pdf', bbox_inches='tight')
        plt.clf()

if __name__ == '__main__':
    # Figure 1
    #voxels, voxel_size = load_voxels(sample, 1)
    #full_planes(voxels, voxel_size)

    # Figure 2
    #voxels, voxel_size = load_voxels(sample, 1)
    #roi_planes(voxels, voxel_size)

    # Figure 3
    #voxels, _ = load_voxels(sample, 8)
    #hist_1d(voxels)

    # Figure 4
    #implant_mask(sample, 2)

    # Figure 5 (2dhists) + 6 (fieldhists)
    #hist_2d(sample, '-bone_region')

    # Figure 7 is the "glowing in valleys" plot

    # Figure 8
    #diffusion_slice(2)

    # Figure 9
    #otsu()

    # Figure 10 is the flowchart

    # Figure 11, 12
    #segmented_slices(1)

    # Figure 13
    #blood_network(1)

    # Figure 14
    #global_thresholding(1, 8, 2)
    #otsu_thresholding(1, 2)

    # Figure 15
    bone_health(1)