# MEDIA_CycleMorph

Paper
===============
* CycleMorph: Cycle consistent unsupervised deformable image registration (Boah Kim et al.)

Implementation
===============
A PyTorch implementation of deep-learning-based registration.
We implemented this code based on voxelMorph and original cycleGAN code.
[https://github.com/voxelmorph/voxelmorph]
(*Thanks for voxelMorph.)
[https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix] 
(*Thanks for Jun-Yan Zhu and Taesung Park, and Tongzhou Wang.)

* Requirements
  * OS : Ubuntu
  * Python 3.6
  * PyTorch 1.4.0

Main
===============
* Training: train.py which is handled by scripts/Brain_train.sh
* A code for CycleMorph is in models/cycleMorph_model.py.
