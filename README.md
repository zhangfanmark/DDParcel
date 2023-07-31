# DDSurfer
DDSurfer: deep learning anatomical brain parcellation from diffusion MRI. 


# Dependencies
* [3D Slicer](https://www.slicer.org)
* [PNL conversion](https://github.com/pnlbwh/conversion)

# Installation

    conda create --name DDSurfer python=3.9
    conda activate DDSurfers
    pip install torchvision torch
    pip install cudatoolkit
    pip install h5py nibabel numpy 
    pip install pillow scikit-image scipy

# Pretrained model

Download [weights.zip](https://github.com/zhangfanmark/DDSurfer/releases), and uncompress to the code root folder.

# Example

Downlaod [100HCP-population-mean-T2.nii.gz] (https://zenodo.org/record/2648292/files/100HCP-population-mean-T2.nii.gz?download=1), and place it under the code root folder. 
Download [testdata.zip](https://github.com/zhangfanmark/DDSurfer/releases), and uncompress to the code root folder.

    bash process.sh

