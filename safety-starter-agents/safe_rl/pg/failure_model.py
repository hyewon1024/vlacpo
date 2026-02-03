from collections import deque
import numpy as np
import torch
import yaml
from failure_prob.model import get_model
from failure_prob.model.base import BaseModel
from failure_prob.conf import process_cfg

from collections import deque
import numpy as np
import torch
from omegaconf import OmegaConf

from failure_prob.model import get_model
from failure_prob.model.base import BaseModel
from failure_prob.conf import process_cfg


class FailureCostModel:
    def __init__(self, cfg_path, ckpt_path, input_dim, feature_fn, window=20, device="cpu"):
        raw_cfg = OmegaConf.load(cfg_path)
        self.cfg = process_cfg(raw_cfg)
        
        self.model: BaseModel = get_model(self.cfg, input_dim)
        
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict) 
        
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.feature_fn = feature_fn
        self.window = window
        self.buf = deque(maxlen=window)

    def reset(self):
        self.buf.clear()

    @torch.no_grad()
    def __call__(self, o):
        feat = self.feature_fn(o) 
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat).float()
        else:
            feat = feat.float().detach().cpu()

        self.buf.append(feat)

        features = torch.stack(list(self.buf), dim=0).unsqueeze(0).to(self.device)

        batch = {"features": features}

        score = self.model(batch)  
        cost = score[:, -1].squeeze().item() 
        return float(cost)