# CycleMorph
This is an official repository of CycleMorph.

Paper
===============
* CycleMorph: Cycle consistent unsupervised deformable image registration (Medical Image Analysis, Boah Kim et al.)
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

Citations
===============
```
@article{kim2021cyclemorph,
  title={CycleMorph: cycle consistent unsupervised deformable image registration},
  author={Kim, Boah and Kim, Dong Hwan and Park, Seong Ho and Kim, Jieun and Lee, June-Goo and Ye, Jong Chul},
  journal={Medical Image Analysis},
  volume={71},
  pages={102036},
  year={2021},
  publisher={Elsevier}
}

@inproceedings{kim2019unsupervised,
  title={Unsupervised deformable image registration using cycle-consistent cnn},
  author={Kim, Boah and Kim, Jieun and Lee, June-Goo and Kim, Dong Hwan and Park, Seong Ho and Ye, Jong Chul},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={166--174},
  year={2019},
  organization={Springer}
}
```
