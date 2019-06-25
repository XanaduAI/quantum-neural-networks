# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" This scripts converts Tetris numpy images into a .png figure."""

import numpy as np
import matplotlib.pyplot as plt
import os

##################### set local directories ########
# Model name
model_string = 'tetris'

# Output folder
folder_locator = './outputs/'

# Locations of saved data and output figure
save_string = folder_locator + 'models/' + model_string + '/'


# Loading of images
images_out =np.load(save_string+'images_out.npy')
images_out_big =np.load(save_string+'images_out_big.npy')

num_labels = 7
plot_scale = 1

# Plotting of the final image.
fig_images, axs = plt.subplots(
    nrows=2, ncols=num_labels,figsize=(num_labels*plot_scale,2*plot_scale)
    )

all_images = [images_out,images_out_big]
for i in range(2):
    for lable in range(num_labels):
        ax=axs[i][lable]
        ax.imshow(all_images[i][lable],cmap='gray')
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
plt.tight_layout()
fig_images.savefig(save_string + 'fig_images.png')
