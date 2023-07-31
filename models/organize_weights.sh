parameters=(dti-FractionalAnisotropy-reg-NormMasked-T1w dti-Trace-reg-NormMasked-T1w dti-MidEigenvalue-reg-NormMasked-T1w dti-MinEigenvalue-reg-NormMasked-T1w)
views=(axial coronal sagittal)
mkdir -p backbones
for para in ${parameters[@]}
do

	for view in ${views[@]}
	do

		pkl=/Users/fan/Downloads/model_slice_mixed_HCP/model_slice_mixed_HCP/$para/log/$view/ckpts/Epoch_30_training_state.pkl

		$1 cp $pkl backbones/${para//-T1w/}-$view.pkl

	done
done


views=(axial coronal sagittal)
mkdir -p backbones

for view in ${views[@]}
do

	pkl=/Users/fan/Downloads/model_slice_mixed_HCP/model_slice_mixed_HCP/Fused-Unet-v3-noMax-T1w/log/$view/ckpts/Epoch_20_training_state.pkl

	$1 cp $pkl Fused-Unet-v3-noMax-$view.pkl

done
