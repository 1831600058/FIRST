import os
import shutil
import numpy as np
from pathlib import Path
from utils.util import makedirs


class ConfigArgs(object):
    def __init__(self, config):
        names = self.__dict__
        for k, v in config.items():
            names[k] = v
        # self.ser_list = np.arange(config['ser_list'][0], config['ser_list'][1] + 1, 2)

        # output dirs
        self.outpt_path = self.output_dir + self.project + '/' + self.workspace + '/'
        # model dir
        self.model_path = self.outpt_path + 'checkpoints/'
        # log path
        self.log_path = self.outpt_path
        # est dir
        self.validation_path = self.outpt_path + 'est_validations/'
        self.prediction_path = self.outpt_path + 'est_predictions/'
        makedirs([self.model_path, self.validation_path, self.prediction_path])
