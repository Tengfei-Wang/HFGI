from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['ffhq_val'],
		'test_target_root': dataset_paths['ffhq_val'],
	},
	'cars_encode': {
		'transforms': transforms_config.CarsEncodeTransforms,
		'train_source_root': dataset_paths['cars_train'],
		'train_target_root': dataset_paths['cars_train'],
		'test_source_root': dataset_paths['cars_val'],
		'test_target_root': dataset_paths['cars_val'],
	}
}
