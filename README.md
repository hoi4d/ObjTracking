# ObjTracking
Please check out the HOI4D Challenge on the latest project website www.hoi4d.top !
## Overview
This code base provides a benchmark for the HOI4D challenge object tracking task, and we provide the sript to preprocess dataset for Bundletrack, which is a sota categoray-level object pose tracking method.
## Challege
For this challege, you need submmit a pred.npy file(your predicted results) to the leaderboard. The file pred.npy is a ndarray which is the prediction of test_wolabel.h5. You can download the example here: [Link](https://1drv.ms/u/s!ApQF_e_bw-USgjQCKg9hGJIijeqs?e=eGfohd)

## Usage
After you process the data using our sripts, you can easily run [bundletrack](https://github.com/wenbowen123/BundleTrack).
## Citation
```
@InProceedings{Liu_2022_CVPR,
    author    = {Liu, Yunze and Liu, Yun and Jiang, Che and Lyu, Kangbo and Wan, Weikang and Shen, Hao and Liang, Boqiang and Fu, Zhoujie and Wang, He and Yi, Li},
    title     = {HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {21013-21022}
}
```
```
@inproceedings{wen2021bundletrack,
  title={BundleTrack: 6D Pose Tracking for Novel Objects without Instance or Category-Level 3D Models},
  author={Wen, B and Bekris, Kostas E},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems},
  year={2021}
}
```

