# MEDIA_CycleMorph

Paper
===============
* CycleMorph: Cycle consistent unsupervised deformable image registration (arXiv.org, Boah Kim et al.)
* Unsupervised Deformable Image Registration Using Cycle-Consistent CNN (MICCAI 2019, Boah Kim et al.)

Implementation
===============
A PyTorch implementation of deep-learning-based registration.
We implemented this code based on [voxelMorph](https://github.com/voxelmorph/voxelmorph) and [original cycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) code.
(*Thanks for voxelMorph.)
(*Thanks for Jun-Yan Zhu and Taesung Park, and Tongzhou Wang.)

* Requirements
  * OS : Ubuntu
  * Python 3.6
  * PyTorch 1.4.0

Data
===============
To download the atlas brain and a sample data, visit the [Data](https://drive.google.com/drive/folders/1S7aT_u8YVAcDdR_2Giw2--mGztygH4bd?usp=sharing).
The data should be in folder ./data.

Training
===============
* train.py which is handled by scripts/Brain_train.sh
* You can run the code by running ./scripts/Brain_train.sh
* A code for CycleMorph is in models/cycleMorph_model.py.

Testing
===============
* test.py which is handled by scripts/Brain_test.sh
* You can run the code by running ./scripts/Brain_test.sh
