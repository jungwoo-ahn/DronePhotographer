# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

# VENDORED FOR INFERENCE (DronePhotographer AutoPhoto reward): only the low-pass
# ResNet is needed; alexnet/densenet/mobilenet/vgg are not vendored.
from .downsample import *
from .resnet import *
