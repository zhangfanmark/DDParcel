#DWIToDTIEstimation=/Applications/Slicer5.2.2.app/Contents/Extensions-31382/SlicerDMRI/lib/Slicer-5.2/cli-modules/DWIToDTIEstimation
#DiffusionTensorScalarMeasurements=/Applications/Slicer5.2.2.app/Contents/Extensions-31382/SlicerDMRI/lib/Slicer-5.2/cli-modules/DiffusionTensorScalarMeasurements
#BRAINSFit=/Applications/Slicer5.2.2.app/Contents/lib/Slicer-5.2/cli-modules/BRAINSFit
#ResampleScalarVectorDWIVolume=/Applications/Slicer5.2.2.app/Contents/lib/Slicer-5.2/cli-modules/ResampleScalarVectorDWIVolume

Slicer_cli=${Slicer_root}/lib/Slicer-5.2/cli-modules
Dmri_cli=${Slicer_root}/Extensions-31382/SlicerDMRI/lib/Slicer-5.2/cli-modules

DWIToDTIEstimation="${Dmri_cli}/DWIToDTIEstimation"
DiffusionTensorScalarMeasurements="${Dmri_cli}/DiffusionTensorScalarMeasurements"
BRAINSFit="${Slicer_cli}/BRAINSFit"
ResampleScalarVectorDWIVolume="${Slicer_cli}/ResampleScalarVectorDWIVolume"

atlas_T2=./100HCP-population-mean-T2-1mm.nii.gz

subID=HCP-100337-b1000

inputdir=testdata
outputdir=$inputdir/$subID/
mkdir -p $outputdir

# input data
dwi=$inputdir/$subID.nii.gz
bval=$inputdir/$subID.bval
bvec=$inputdir/$subID.bvec
mask=$inputdir/$subID-mask.nii.gz

nrrd_dwi=$inputdir/$subID.nhdr
nrrd_mask=$inputdir/$subID-mask.nhdr
if [ ! -f  $nrrd_mask ]; then
	$1 nhdr_write.py --nifti $dwi --bval $bval --bvec $bvec --nhdr $nrrd_dwi
	$1 nhdr_write.py --nifti $mask --nhdr $nrrd_mask
fi

# DTI parameter computation
nrrd_dti=$outputdir/$subID-dti.nhdr
nrrd_b0=$outputdir/$subID-b0.nhdr
if [ ! -f $nrrd_b0 ]; then
    $1 $DWIToDTIEstimation --enumeration LS $nrrd_dwi $nrrd_dti $nrrd_b0 #-m $nrrd_mask
fi

nrrd_fa=$outputdir/$subID-dti-FractionalAnisotropy.nhdr
nrrd_trace=$outputdir/$subID-dti-Trace.nhdr
nrrd_minEig=$outputdir/$subID-dti-MinEigenvalue.nhdr
nrrd_midEig=$outputdir/$subID-dti-MidEigenvalue.nhdr

nii_fa=$outputdir/$subID-dti-FractionalAnisotropy.nii.gz
nii_trace=$outputdir/$subID-dti-Trace.nii.gz
nii_minEig=$outputdir/$subID-dti-MinEigenvalue.nii.gz
nii_midEig=$outputdir/$subID-dti-MidEigenvalue.nii.gz
 
if [ ! -f $nii_midEig ]; then
    $1 $DiffusionTensorScalarMeasurements --enumeration FractionalAnisotropy $nrrd_dti $nrrd_fa
    $1 $DiffusionTensorScalarMeasurements --enumeration Trace $nrrd_dti $nrrd_trace
    $1 $DiffusionTensorScalarMeasurements --enumeration MinEigenvalue $nrrd_dti $nrrd_minEig
    $1 $DiffusionTensorScalarMeasurements --enumeration MidEigenvalue $nrrd_dti $nrrd_midEig

    $1 nifti_write.py -i $nrrd_fa     -p ${nrrd_fa//.nhdr/}
    $1 nifti_write.py -i $nrrd_trace  -p ${nrrd_trace//.nhdr/}
    $1 nifti_write.py -i $nrrd_minEig -p ${nrrd_minEig//.nhdr/}
    $1 nifti_write.py -i $nrrd_midEig -p ${nrrd_midEig//.nhdr/}
fi

# register data to MNI
tfm=$outputdir/$subID-b0ToAtlasT2.tfm 
tfminv=$outputdir/$subID-b0ToAtlasT2_Inverse.h5
if [ ! -f $tfm ]; then
	$1 $BRAINSFit --fixedVolume $atlas_T2 --movingVolume $nrrd_b0 --linearTransform $outputdir/$subID-b0ToAtlasT2.tfm --useRigid --useAffine
fi

nii_fa_reg=$outputdir/$subID-dti-FractionalAnisotropy-Reg.nii.gz
nii_trace_reg=$outputdir/$subID-dti-Trace-Reg.nii.gz
nii_minEig_reg=$outputdir/$subID-dti-MinEigenvalue-Reg.nii.gz
nii_midEig_reg=$outputdir/$subID-dti-MidEigenvalue-Reg.nii.gz
nii_mask_reg=$outputdir/$subID-mask-Reg.nii.gz
if [ ! -f $nii_mask_reg ]; then
	$1 $ResampleScalarVectorDWIVolume -i linear ${nii_fa}     --Reference ${atlas_T2}     --transformationFile $tfm $nii_fa_reg
	$1 $ResampleScalarVectorDWIVolume -i linear ${nii_trace}  --Reference ${atlas_T2}  --transformationFile $tfm $nii_trace_reg
	$1 $ResampleScalarVectorDWIVolume -i linear ${nii_minEig} --Reference ${atlas_T2} --transformationFile $tfm $nii_minEig_reg
	$1 $ResampleScalarVectorDWIVolume -i linear ${nii_midEig} --Reference ${atlas_T2} --transformationFile $tfm $nii_midEig_reg
	$1 $ResampleScalarVectorDWIVolume -i nn     ${mask}       --Reference ${atlas_T2}       --transformationFile $tfm $nii_mask_reg
fi


# normalizing 
nii_fa_reg_norm=$outputdir/$subID-dti-FractionalAnisotropy-Reg-NormMasked.nii.gz
nii_trace_reg_norm=$outputdir/$subID-dti-Trace-Reg-NormMasked.nii.gz
nii_minEig_reg_norm=$outputdir/$subID-dti-MinEigenvalue-Reg-NormMasked.nii.gz
nii_midEig_reg_norm=$outputdir/$subID-dti-MidEigenvalue-Reg-NormMasked.nii.gz

if [ ! -f $nii_midEig_reg_norm ]; then
	$1 python normalize.py --input $nii_fa_reg --mask $nii_mask_reg --output $nii_fa_reg_norm --flip 1
	$1 python normalize.py --input $nii_trace_reg --mask $nii_mask_reg --output $nii_trace_reg_norm --flip 1
	$1 python normalize.py --input $nii_minEig_reg --mask $nii_mask_reg --output $nii_minEig_reg_norm --flip 1
	$1 python normalize.py --input $nii_midEig_reg --mask $nii_mask_reg --output $nii_midEig_reg_norm --flip 1
fi

# DDSurfer
mgz_wmparc_reg=$outputdir/$subID-DDSurfer-wmparc-Reg.mgz
if [ ! -f $mgz_wmparc_reg ]; then
	$1 python DDSurfer_Pred.py --in_dir $outputdir --out_dir $outputdir --weights_dir ./weights/
fi

nii_wmparc=$outputdir/$subID-DDSurfer-wmparc.nii.gz
if [ ! -f $nii_wmparc ]; then
 $1 $ResampleScalarVectorDWIVolume --Reference $mask --transformationFile $tfminv --interpolation nn $mgz_wmparc_reg $nii_wmparc
fi








