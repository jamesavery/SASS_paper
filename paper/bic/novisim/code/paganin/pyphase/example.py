# FROM:
# https://pyphase.readthedocs.io/en/master/examples.html

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyphase
from pyphase import *
import numpy as np

#%% Choose image display to use
displayer = utilities.PyplotImageDisplayer()

# Phase retrieval from a dataset
#%% Load dataset
data = dataset.NanomaxPreprocessed2D('star', version='test')

#%% Align images using default registrator
pyphase.registrator.parameters.NumberOfResolutions = 8
pyphase.registrator.parameters.MaximumNumberOfIterations = 3000

data.align_projection()

#%% Phase retrieval from a dataset using HIO_ER
retriever = phaseretrieval.CTF(data)

# Modify some parameters
retriever.alpha = [1e-3, 1e-8] # Regularisation parameter used for initialisation
retriever.iterations_hio = 2
retriever.iterations_er = 2
retriever.iterations = 2 # Change number of global iterations

# Reconstruct
retriever.reconstruct_projection(dataset=data, projection=0) 

#%% Display reconstruction
displayer.close_all()
displayer.display(data.get_image(projection=0))
displayer.display(data.get_image(projection=0, image_type='attenuation'))

#%% Phase retrieval from images
# Acquisition parameters
energy=13
effective_distance=np.array([0.010054, 0.0155, 0.0178, 0.019, 0.0203])
pixel_size = np.array([0.005924, 0.006043])*1e-6

nyp = nxp = 4096
ny = nx = 2048

#%% Load images
ID = np.zeros((len(effective_distance), nyp, nxp))
for N in range(len(effective_distance)):
    ID[N] = data.get_projection(projection=0, position=N, pad=True)

#%% Phase retrieval from images using HIO_ER
retriever = phaseretrieval.CTF(shape=ID[0].shape, pixel_size=[pixel_size[0], pixel_size[1]], distance=effective_distance, energy=energy, pad=1)

# Modify some parameters
retriever.alpha = [1e-3, 1e-8] # Regularisation parameter used for initialisation
retriever.iterations = 5 # Change number of global iterations

# Reconstruct
phase, attenuation = retriever.reconstruct_image(ID)

#%% Display reconstruction
displayer.close_all()
displayer.display(utilities.resize(phase, [ny, nx]), 'phase')
displayer.display(utilities.resize(attenuation, [ny, nx]), 'attenuation')
