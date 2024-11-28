# MotionSNN
This is the PyTorch implementation of paper: Motion-Oriented Hybrid Spiking Neural Networks for Event-based Motion Deblurring (TCSVT 2024). [Paper link](https://ieeexplore.ieee.org/abstract/document/10258440).

## Prerequisites
The Following Setup is tested and it is working:
* Python >= 3.7
* PyTorch >= 1.7.1
* CUDA >= 11.0.2

## Preprocess of GoPro
We convert event data into fixed-length temporal .pt files to reduce training time overhead. The conversion code is in the 'preprocess' folder.
The converted data structure is organized as follows:

datapath/  
├── data/  
│ ├── blurimg/ # All blurred images (.png)  
│ ├── events/ # All event signals (.pt)  
│ └── gt/ # All sharp images (.png)  
└── list/  
│ ├── trainlist.txt # Sample indices for training set  
│ └── testlist.txt # Sample indices for test set  


Download link: Data is uploading (Note: Due to the large size of .pt event files, the uploaded version is in .pbs2 format. )

<!-- Download link: [GoPro Dataset](https://drive.google.com/drive/folders/1rJs8qyTd6EDFYTDSA0N4pv65hxlDFoNf?usp=sharing) (Note: Due to the large size of .pt event files, the uploaded version is in .pbs2 format. ) -->

The pre-trained model on GoPro dataset can be found here [Pre-trained Model](https://drive.google.com/drive/folders/1Lcx-9dQnw5NXOK4gdLw1YLCfBdipHob5?usp=sharing)

## Reference
> Z. Liu, J. Wu, G. Shi, W. Yang, W. Dong and Q. Zhao, "Motion-Oriented Hybrid Spiking Neural Networks for Event-Based Motion Deblurring," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 5, pp. 3742-3754, May 2024, doi: 10.1109/TCSVT.2023.3317976
```bibtex
@ARTICLE{liu2023motionsnn,
  author={Liu, Zhaoxin and Wu, Jinjian and Shi, Guangming and Yang, Wen and Dong, Weisheng and Zhao, Qinghang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Motion-Oriented Hybrid Spiking Neural Networks for Event-Based Motion Deblurring}, 
  year={2024},
  volume={34},
  number={5},
  pages={3742-3754},
  keywords={Feature extraction;Cameras;Image reconstruction;Convolution;Imaging;Image restoration;Image resolution;Hybrid network;motion intensity;spiking transformer;motion deblurring},
  doi={10.1109/TCSVT.2023.3317976}
  }
```