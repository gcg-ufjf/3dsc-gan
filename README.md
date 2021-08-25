# 3DSC-GAN

TensorFlow Implementation for learned compression of 3-D poststack seismic data using Generative Adversarial Networks. The method was developed by Ribeiro et. al. in [Poststack Seismic Data Compression Using a Generative Adversarial Network](http://doi.org/10.1109/LGRS.2021.3103663).

![Comparison of a reconstructed 2-D slice (200 × 650 pixels) from Poseidon3D volume compressed with extremely low target bit rate (0.1 bpv)](images/reconstruction.png)

-------------------------------------------------

## Data
The method was trained and tested using 3-D poststack seismic volumes from the Society of Exploration Geophysicists (SEG) Open Data repository. All the **original volumes** can be downloaded from this [link](http://www.gcg.ufjf.br/files/3dsc-gan/original_data.zip)

## Reconstruction Results
All the **reconstruction results** for the method and baselines can be downloaded from the links bellow:

| target bpv | Kahu3D | Opunake3D| Penobscot3D | Poseidon3D | Waihapa3D |
|------|------|------|------|------|------|
| 0.05 | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Kahu3D_bpv_0.05.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Opunake3D_bpv_0.05.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Penobscot3D_bpv_0.05.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Poseidon3D_bpv_0.05.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Waihapa3D_bpv_0.05.zip) |
| 0.10 | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Kahu3D_bpv_0.10.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Opunake3D_bpv_0.10.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Penobscot3D_bpv_0.10.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Poseidon3D_bpv_0.10.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Waihapa3D_bpv_0.10.zip) |
| 0.25 | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Kahu3D_bpv_0.25.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Opunake3D_bpv_0.25.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Penobscot3D_bpv_0.25.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Poseidon3D_bpv_0.25.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Waihapa3D_bpv_0.25.zip) |
| 0.50 | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Kahu3D_bpv_0.50.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Opunake3D_bpv_0.50.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Penobscot3D_bpv_0.50.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Poseidon3D_bpv_0.50.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Waihapa3D_bpv_0.50.zip) |
| 0.75 | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Kahu3D_bpv_0.75.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Opunake3D_bpv_0.75.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Penobscot3D_bpv_0.75.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Poseidon3D_bpv_0.75.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Waihapa3D_bpv_0.75.zip) |
| 1.00 | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Kahu3D_bpv_1.00.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Opunake3D_bpv_1.00.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Penobscot3D_bpv_1.00.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Poseidon3D_bpv_1.00.zip) | [link](http://www.gcg.ufjf.br/files/3dsc-gan/Waihapa3D_bpv_1.00.zip) |

## Getting Started

### Installation
```bash
# Clone
$ git clone https://github.com/GCG-UFJF/3DSC-GAN.git
$ cd 3DSC-GAN

# Build an image as a container
$ docker build --rm -t 3dsc_gan dockerfiles/

# Run the image as a container
$ docker run -ti --rm --gpus all -v "$(pwd):/workspace" 3dsc_gan bash

# Extract the subvolumes
#  - this step requires the original volumes
#  - verify the path to original_data before running the command
$ python3 data/generate_dataset.py
```

### 3DSC-GAN train/test
```
# Run the image as a container
$ docker run -ti --rm --gpus all -v "$(pwd):/workspace" 3dsc_gan bash

# Train a model
$ cd code/
$ python3 train.py

# Test a trained model
$ cd code/
$ python3 test.py
```


## Citation
If you use this code for your research, please cite the paper.
```
@article{ribeiro2021,
    author={Ribeiro, Kevyn Swhants dos Santos and Schiavon, Ana Paula and Navarro, João Paulo and Vieira, Marcelo Bernardes and Villela, Saulo Moraes and Silva, Pedro Mário Cruz e},
    journal={IEEE Geoscience and Remote Sensing Letters},
    title={Poststack Seismic Data Compression Using a Generative Adversarial Network},
    year={2021},
    volume={},
    number={},
    pages={1-5},
    doi={10.1109/LGRS.2021.3103663}
}
```