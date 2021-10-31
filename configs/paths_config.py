dataset_paths = {
	#  Face Datasets (FFHQ - train, CelebA-HQ - test)
	'ffhq': '',
	'ffhq_val': '',

	#  Cars Dataset (Stanford cars)
	'cars_train': '',
	'cars_val': '',
}

model_paths = {
	'stylegan_ffhq': './pretrained/stylegan2-ffhq-config-f.pt',
	'ir_se50': './pretrained/model_ir_se50.pth',
	'shape_predictor': './pretrained/shape_predictor_68_face_landmarks.dat',
	'moco': './pretrained/moco_v2_800ep_pretrain.pt'
}
