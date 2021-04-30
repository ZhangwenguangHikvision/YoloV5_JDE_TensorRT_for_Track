import _init_paths
import torch
import struct

from core.mot.general import non_max_suppression_and_inds, non_max_suppression_jde, non_max_suppression, scale_coords
from core.mot.torch_utils import intersect_dicts
from models.mot.cstrack import Model

from mot_online import matching
from mot_online.kalman_filter import KalmanFilter
from mot_online.log import logger
from mot_online.utils import *

from mot_online.basetrack import BaseTrack, TrackState
import datasets as track_datasets
import tracker_utils  as track_utils

# Initialize
opt_device = torch.device('cpu')
# Load model
ckpt = torch.load('runs/exp61_mot_test/weights/best_mot_test.pt', map_location=opt_device) # load to FP32
model = Model('experiments/model_set/CSTrack3_0.yaml', ch=3, nc=1).to(opt_device)
exclude = ['anchor'] if 'experiments/model_set/CSTrack3_0.yaml' else []  # exclude keys
state_dict = ckpt['model']
state_dict = intersect_dicts(state_dict.state_dict(), model.state_dict(), exclude=exclude)  # intersect
model.load_state_dict(state_dict, strict=False)  # load
model.to(opt_device).eval()

f = open('../jde.wts', 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')
