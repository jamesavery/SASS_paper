# Published data overview

Generated and simulated volumes are made available for external benchmarking and testing.

The sizes with corresponding resolutions are as follows:

| Scale | Resolution [micrometer] | Shape              |
| ----- | ----------------------- | ------------------ |
| 1x    |  1.875                  | (3272, 3456, 3456) |
| 2x    |  3.75                   | (1636, 1728, 1728) |
| 4x    |  7.5                    | (818, 864, 864)    |
| 8x    | 15                      | (409, 432, 432)    |

The input and output volumes are as follows:

| Type    | Content | File format | Data type | Sizes |
| ------- | ------- | ----------- | --------- | ----- |
| Masks | Individual volumes for bone, blood, implant, osteocyt | HDF5 | UINT8 | (1/2/4/8)x |
| Mesh models | Individual volumes for bone, blood, implant, osteocyt | STL | N/A | N/A |
| Ground truth | Single volume | HDF5 | UINT8 | (1/2/4/8)x |
| Simulated tomograms | Single volume | HDF5 + RAW | FLOAT32 | (2/4/8)x |
| Original tomograms | 4-5 subvolumes per tomogram | HDF5 + RAW | FLOAT32 | (1/2/4/8)x |

For all volumes, the first index traverses through the depth of implant, the
second index along the short side of the implant, the third and final index
along the wide side of the implant. The full-size volumes, corresponding to 1x
scaling, have (3272,3456,3456) voxels. Scalings of (2,4,8)x are also provided,
where relevant. Each scaling effectively shortens each dimension to half of its
previous scaling. Scaling 1x thus has 8 times the number of voxels relative to
scaling 2x.

Notice that a full simulation of the 1x scaled tomogram was not possible
in Novi-sim, due to the program crashing. This might either be a software or
hardware limitation.

