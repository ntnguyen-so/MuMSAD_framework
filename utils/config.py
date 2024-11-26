########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: utils
# @file : config
#
########################################################################

from models.model.convnet import ConvNet
from models.model.inception_time import InceptionModel
from models.model.resnet import ResNetBaseline
from models.model.sit import SignalTransformer


# Important paths
OBSEA_data_path = "data/OBSEA/data/"
OBSEA_metrics_path = "data/OBSEA/metrics/"
OBSEA_scores_path = "data/OBSEA/scores/"

save_done_training = 'results/done_training/'	# when a model is done training a csv with training info is saved here
path_save_results = 'results/raw_predictions'	# when evaluating a model, the predictions will be saved here

# Detector
detector_names = [
    'AutoEncoder (AE)', 
      'CBLOF',
      'COPOD', 
      'DenoisingAutoEncoder (DAE)',
      'HBOS',
      'LOF',
      'PCC', 
      'RobustPCA',
      'Random Black Forest (RR)', 
      'Torsk',
]


# Dict of model names to Constructors
deep_models = {
	'convnet':ConvNet,
	'inception_time':InceptionModel,
	'inception':InceptionModel,
	'resnet':ResNetBaseline,
	'sit':SignalTransformer,
}

num_dimensions = 3

