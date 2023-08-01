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

Downlaod [100HCP-population-mean-T2-1mm.nii.gz](https://github.com/zhangfanmark/DDSurfer/releases), and place it under the code root folder. 

Download [testdata.zip](https://github.com/zhangfanmark/DDSurfer/releases), and uncompress to the code root folder.

    bash process.sh

**NOTE**: Depending on how the brain mask is created, ``--flip`` option in the above shell script ([Lines 89-92](https://github.com/zhangfanmark/DDSurfer/blob/0b9e6fb4c3ff0d348857e8dfdb92ae6a54f55e42/process.sh#L89C1-L92C108)) may need to be adjusted to make the mask and the DTI maps in the same space after loading the nifti files as numpy arrays. For most cases, ``--flip 0`` is used. For the HCP data with the provided brain mask, ``--flip 1`` is needed. To check this, visualize the "XXX-dti-FractionalAnisotropy-Reg-NormMasked.nii.gz" file in the output folder using Slicer or other software; if there is any orientation issue, change the setting of ``--flip``. 
