# IMPORTS
import argparse
import time
import sys
import glob
import logging
import os
import copy
import nibabel as nib
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from scipy.ndimage.filters import median_filter, gaussian_filter
from skimage.measure import regionprops
from skimage.measure import label
from collections import OrderedDict

from data_loader.load_neuroimaging_data import OrigDataThickSlices_Fused_Input
from data_loader.load_neuroimaging_data import map_label2aparc_aseg
from data_loader.load_neuroimaging_data import map_prediction_sagittal2full
from data_loader.load_neuroimaging_data import get_largest_cc
from data_loader.load_neuroimaging_data import load_and_conform_image

from data_loader.augmentation import ToTensorTest
from models.networks import FastSurferCNN_Fuse_Unet_v3_extended

HELPTEXT = """
DDSurfer: deep learning anatomical brain parcellation from diffusion MRI. \n

Authors: \n
Fan Zhang \n
Kang Ik Kevin Cho \n
Johanna Seitz-Holland \n
Lipeng Ning \n
Jon Haitz Legarreta \n
Yogesh Rathi \n
Carl-Fredrik Westin \n
Lauren J. O\'Donnell \n
Ofer Pasternak

Date: 2023-07

Acknowledgement: This code is build upon FasterSurfer (deep-mi.org/research/fastsurfer)
"""

def options_parse():
    """
    Command line option parser
    """
    parser = argparse.ArgumentParser(description=HELPTEXT)

    # 1. Options for the MRI volumes (name of in and output, order of interpolation if not conformed)
    parser.add_argument('--in_dir', dest='iname', help='input directory, generated by preprocessing.py', default='./testdata/HCP-100337-b1000/')
    parser.add_argument('--out_dir', dest='oname', help='output directory', default='./testdata/HCP-100337-b1000/')
    parser.add_argument('--weights_dir', dest='weights', help="path to pre-trained weights", default='./weights/')

    # 2. Options for model parameters setup (only change if model training was changed)
    parser.add_argument('--num_filters', type=int, default=64, help='Filter dimensions for DenseNet (all layers same). Default=64')
    parser.add_argument('--num_classes_ax_cor', type=int, default=79, help='Number of classes to predict in axial and coronal net, including background. Default=79')
    parser.add_argument('--num_classes_sag', type=int, default=51, help='Number of classes to predict in sagittal net, including background. Default=51')
    parser.add_argument('--num_channels', type=int, default=7, help='Number of input channels. Default=7 (thick slices)')
    parser.add_argument('--kernel_height', type=int, default=5, help='Height of Kernel (Default 5)')
    parser.add_argument('--kernel_width', type=int, default=5, help='Width of Kernel (Default 5)')
    parser.add_argument('--stride', type=int, default=1, help="Stride during convolution (Default 1)")
    parser.add_argument('--stride_pool', type=int, default=2, help="Stride during pooling (Default 2)")
    parser.add_argument('--pool', type=int, default=2, help='Size of pooling filter (Default 2)')

    # 3. Clean up and GPU/CPU options (disable cuda, change batchsize)
    parser.add_argument('--clean', dest='cleanup', help="Flag to clean up segmentation", action='store_true')
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for inference. Default: 8")
    parser.add_argument('--simple_run', action='store_true', default=True, help='Simplified run: only analyse one given image specified by --in_name (output: --out_name). Need to specify absolute path to both --in_name and --out_name if this option is chosen.')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')

    sel_option = parser.parse_args()

    if sel_option.iname[-1] == '/':
        sel_option.iname = sel_option.iname[:-1]
    subjectID = os.path.basename(sel_option.iname)

    sel_option.iname = sorted(glob.glob(os.path.join(sel_option.iname, '*-dti-*-NormMasked.nii.gz')))
    os.makedirs(sel_option.oname, exist_ok=True)
    sel_option.oname = os.path.join(sel_option.oname, subjectID + '-DDSurfer-wmparc-Reg.mgz')

    sel_option.network_sagittal_path = os.path.join(sel_option.weights, 'Fused-Unet-v3-noMax-sagittal.pkl')
    sel_option.network_coronal_path = os.path.join(sel_option.weights, 'Fused-Unet-v3-noMax-coronal.pkl')
    sel_option.network_axial_path = os.path.join(sel_option.weights, 'Fused-Unet-v3-noMax-axial.pkl')

    sel_option.backbone_sagittal_path = sorted(glob.glob(os.path.join(sel_option.weights, 'backbones', '*sagittal.pkl')))
    sel_option.backbone_coronal_path = sorted(glob.glob(os.path.join(sel_option.weights, 'backbones', '*coronal.pkl')))
    sel_option.backbone_axial_path = sorted(glob.glob(os.path.join(sel_option.weights, 'backbones', '*axial.pkl')))

    sel_option.num_classes_ax_cor = 82
    sel_option.num_classes_sag = 54
    sel_option.num_channels = 7
    sel_option.num_modality = len(sel_option.iname)
    sel_option.batch_size = 4

    return sel_option

def run_network(img_filename, orig_data, prediction_probability, plane, ckpts, params_model, model, logger):

    # Set up DataLoader
    test_dataset = OrigDataThickSlices_Fused_Input(img_filename, orig_data, plane=plane, transforms=transforms.Compose([ToTensorTest()]))
    test_data_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=params_model["batch_size"])

    # Set up state dict for model
    logger.info("  -Loading {} Net from {}".format(plane, ckpts))

    backbones = copy.deepcopy(model.FastSurferCNNs) # Somthing related to parallel: may need to be "model.module.FastSurferCNNs"

    model_state = torch.load(ckpts, map_location=params_model["device"])
    new_state_dict = OrderedDict()
    for k, v in model_state["model_state_dict"].items():
        if k[:7] == "module." and not params_model["model_parallel"]:
            new_state_dict[k[7:]] = v
        elif k[:7] != "module." and params_model["model_parallel"]:
            new_state_dict["module." + k] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    print('Load backbones!')
    model.FastSurferCNNs = backbones
    model.eval()

    logger.info("  -{} model loaded.".format(plane))
    with torch.no_grad():

        start_index = 0
        for batch_idx, sample_batch in enumerate(test_data_loader):

            images_batch = Variable(sample_batch["image"])

            if params_model["use_cuda"]:
                images_batch = images_batch.cuda()

            temp, temp_list = model(images_batch)

            for t_idx, temp in enumerate(temp_list):
                start_index_tmp = start_index
                if plane == "Axial":
                    temp = temp.permute(3, 0, 2, 1)
                    prediction_probability[t_idx, :, start_index_tmp:start_index_tmp + temp.shape[1], :, :] += torch.mul(temp.cpu(), 0.4)
                    start_index_tmp += temp.shape[1]

                elif plane == "Coronal":
                    temp = temp.permute(2, 3, 0, 1)
                    prediction_probability[t_idx, :, :, start_index_tmp:start_index_tmp + temp.shape[2], :] += torch.mul(temp.cpu(), 0.4)
                    start_index_tmp += temp.shape[2]

                else:
                    temp = map_prediction_sagittal2full(temp).permute(0, 3, 2, 1)
                    prediction_probability[t_idx, start_index_tmp:start_index_tmp + temp.shape[0], :, :, :] += torch.mul(temp.cpu(), 0.2)
                    start_index_tmp += temp.shape[0]

            start_index = start_index_tmp

    return prediction_probability


def fastsurfercnn(img_filename, save_as, logger, args):

    start_total = time.time()

    # Set up model for axial and coronal networks
    params_network = {'num_channels': args.num_channels, 'num_filters': args.num_filters, "num_modality": args.num_modality, "backbone_model": args.backbone_axial_path,
                      'kernel_h': args.kernel_height, 'kernel_w': args.kernel_width,
                      'stride_conv': args.stride, 'pool': args.pool,
                      'stride_pool': args.stride_pool, 'num_classes': args.num_classes_ax_cor,
                      'kernel_c': 1, 'kernel_d': 1}

    # Put it onto the GPU or CPU
    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("On GPU!") if use_cuda else print("On CPU ...")
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda and torch.cuda.device_count() > 1:
        model_parallel = False # TODO: Issue with using parallel with new torch verion.
    else:
        model_parallel = False

    params_model = {'device': device, "use_cuda": use_cuda, "batch_size": args.batch_size, "model_parallel": model_parallel}
    logger.info("Cuda available: {}, # Available GPUS: {}, Cuda user disabled (--no_cuda flag): {}, --> Using device: {}".format(torch.cuda.is_available(), torch.cuda.device_count(),  args.no_cuda, device))

    # Set up tensor to hold probabilities
    pred_prob = torch.zeros((args.num_modality+2, 256, 256, 256, args.num_classes_ax_cor), dtype=torch.float)

    orig_data = []
    for idx in range(len(img_filename)):
        header_info, affine_info, orig_data_ = load_and_conform_image(img_filename[idx], interpol=1, logger=logger)
        orig_data.append(orig_data_)
        logger.info("Reading volume {}".format(img_filename[idx]))
        del orig_data_

    # Axial
    params_network["num_classes"] = args.num_classes_ax_cor
    params_network["num_channels"] = args.num_channels
    params_network["backbone_model"] = args.backbone_axial_path
    params_network["use_cuda"] = use_cuda
    model = FastSurferCNN_Fuse_Unet_v3_extended(params_network)
    if model_parallel:
        model = nn.DataParallel(model)
    model.to(device)
    start = time.time()
    pred_prob = run_network(img_filename, orig_data, pred_prob, "Axial", args.network_axial_path, params_model, model, logger)
    logger.info("Axial View Tested in {:0.4f} seconds".format(time.time() - start))

    # Coronal
    params_network["num_classes"] = args.num_classes_ax_cor
    params_network["num_channels"] = args.num_channels
    params_network["backbone_model"] = args.backbone_coronal_path
    model = FastSurferCNN_Fuse_Unet_v3_extended(params_network)
    if model_parallel:
        model = nn.DataParallel(model)
    model.to(device)
    start = time.time()
    pred_prob = run_network(img_filename,  orig_data, pred_prob, "Coronal", args.network_coronal_path, params_model, model, logger)
    logger.info("Coronal View Tested in {:0.4f} seconds".format(time.time() - start))

    # Sagittal
    params_network["num_classes"] = args.num_classes_sag
    params_network["num_channels"] = args.num_channels
    params_network["backbone_model"] = args.backbone_sagittal_path
    model = FastSurferCNN_Fuse_Unet_v3_extended(params_network)
    if model_parallel:
        model = nn.DataParallel(model)
    model.to(device)
    start = time.time()
    pred_prob = run_network(img_filename, orig_data, pred_prob, "Sagittal", args.network_sagittal_path, params_model, model, logger)
    logger.info("Sagittal View Tested in {:0.4f} seconds".format(time.time() - start))

    # Postprocessing
    # Get predictions and map to freesurfer label space
    pred_prob = pred_prob[0]
    _, pred_prob = torch.max(pred_prob, 3)
    pred_prob = pred_prob.numpy()
    pred_prob = map_label2aparc_aseg(pred_prob)

    # Post processing - Splitting classes
    # Quick Fix for 2026 vs 1026; 2029 vs. 1029; 2025 vs. 1025
    rh_wm = get_largest_cc(pred_prob == 41)
    lh_wm = get_largest_cc(pred_prob == 2)
    rh_wm = regionprops(label(rh_wm, background=0))
    lh_wm = regionprops(label(lh_wm, background=0))
    centroid_rh = np.asarray(rh_wm[0].centroid)
    centroid_lh = np.asarray(lh_wm[0].centroid)

    # labels needs to be separated
    labels_list = np.array([1001, 1003, 1006, 1007, 1008, 1009, 1011,
                            1015, 1018, 1019, 1020, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035])

    for label_current in labels_list:
        label_img = label(pred_prob == label_current, connectivity=3, background=0)
        for region in regionprops(label_img):
             if region.label != 0:  # To avoid background
                if np.linalg.norm(np.asarray(region.centroid) - centroid_rh) < np.linalg.norm(np.asarray(region.centroid) - centroid_lh):
                    mask = label_img == region.label
                    pred_prob[mask] = label_current + 1000

    # Quick Fixes for overlapping classes
    aseg_lh = gaussian_filter(1000 * np.asarray(pred_prob == 2, dtype=np.float32), sigma=3)
    aseg_rh = gaussian_filter(1000 * np.asarray(pred_prob == 41, dtype=np.float32), sigma=3)
    lh_rh_split = np.argmax(np.concatenate((np.expand_dims(aseg_lh, axis=3), np.expand_dims(aseg_rh, axis=3)), axis=3), axis=3)

    # Problematic classes: 1026, 1011, 1029, 1019
    for prob_class_lh in [1011, 1019, 1026, 1029, 1032]:
        prob_class_rh = prob_class_lh + 1000

        mask_lh = ((pred_prob == prob_class_lh) | (pred_prob == prob_class_rh)) & (lh_rh_split == 0)
        mask_rh = ((pred_prob == prob_class_lh) | (pred_prob == prob_class_rh)) & (lh_rh_split == 1)

        pred_prob[mask_lh] = prob_class_lh
        pred_prob[mask_rh] = prob_class_rh

    # Clean-Up
    if args.cleanup is True:

        labels = [2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                  15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                  46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                  77, 1026, 2026]

        start = time.time()
        pred_prob_medfilt = median_filter(pred_prob, size=(3, 3, 3))
        mask = np.zeros_like(pred_prob)
        tolerance = 25

        for current_label in labels:
            current_class = (pred_prob == current_label)
            label_image = label(current_class, connectivity=3)

            for region in regionprops(label_image):

                if region.area <= tolerance:
                    mask_label = (label_image == region.label)
                    mask[mask_label] = 1

        pred_prob[mask == 1] = pred_prob_medfilt[mask == 1]
        logger.info("Segmentation Cleaned up in {:0.4f} seconds.".format(time.time() - start))

    pred_prob[orig_data[0] == -4] = 0

    # Saving image
    header_info.set_data_dtype(np.int16)
    mapped_aseg_img = nib.MGHImage(pred_prob, affine_info, header_info)
    mapped_aseg_img.to_filename(save_as)

    logger.info("Saving Segmentation to {}".format(save_as))
    logger.info("Total processing time: {:0.4f} seconds.".format(time.time() - start_total))


if __name__ == "__main__":

    # Command Line options and error checking done here
    options = options_parse()

    # Set up the logger
    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))

    fastsurfercnn(options.iname, options.oname, logger, options)


