# -*- coding: utf-8 -*-
"""Abstract base model"""

from abc import ABC, abstractmethod

import numpy as np
import torch

from utils.config import Config


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, cfg):
        self.config = Config.from_json(cfg)

        # fix random seeds for reproducibility
        np.random.seed(self.config.train.seed)
        torch.manual_seed(self.config.train.seed)
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.manual_seed(self.config.train.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    # @abstractmethod
    # def evaluate(self):
    #     pass
