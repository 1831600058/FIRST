import shutil
import random

import numpy as np
import os
import re
import logging


def init_log(file_path):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    fl = logging.FileHandler(file_path, 'a')
    fl.setLevel(logging.INFO)
    log.addHandler(fl)
    fmt = logging.Formatter('%(asctime)s [%(levelname)-8s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    cl = logging.StreamHandler()
    cl.setFormatter(fmt)
    log.addHandler(cl)


def normalize_wav(sig):
    scale = np.max(np.abs(sig)) + 1e-7
    sig = sig / scale

    return sig, scale


def snr(s, s_p):
    r""" calculate signal-to-noise ratio (SNR)

        Parameters
        ----------
        s: clean speech
        s_p: processed speech
    """
    return 10.0 * np.log10(np.sum(s ** 2) / (np.sum((s_p - s) ** 2) + 1.0e-8))


def read_list(list_file):
    f = open(list_file, "r")
    lines = f.readlines()
    list_sig = []
    for x in lines:
        list_sig.append(x[:-1])
    f.close()
    return list_sig


def gen_list(wav_dir, append):
    l = []
    lst = os.listdir(wav_dir)
    lst.sort()
    for f in lst:
        if re.search(append, f):
            l.append(f)
    return l


def write_log(file, name, train, validate):
    message = ''
    for m, val in enumerate(train):
        message += ' --TRerror%i=%.3f ' % (m, val.data.numpy())
    for m, val in enumerate(validate):
        message += ' --CVerror%i=%.3f ' % (m, val.data.numpy())
    file.write(name + ' ')
    file.write(message)
    file.write('/n')


def makedirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def saveConfig(yaml, yaml_name, src, dst):
    f_params = open(dst + '/' + yaml_name, 'w')
    for k, v in yaml.items():
        f_params.write('{}:\t{}\n'.format(k, v))
    shutil.copy(os.path.join(src, 'train.py'), os.path.join(dst, 'train.py'))
    shutil.copy(os.path.join(src, 'test.py'), os.path.join(dst, 'test.py'))


def simu_nonlinear(a, input, para):
    type = a
    # para = random.randint(2, 5)
    if type < 4:
        input = input / max(abs(input))
        if type == 1:
            a = 1 / para * 5
            output = a * input / np.sqrt(a * a + input * input)
        elif type == 2:
            a = para / 10
            output = 1 - np.exp(-input * a)
        elif type == 3:
            a = np.log(para - 1) + 0.1
            output = a * 2 * input + a * input ** 2 + input ** 3
        output = output / max(abs(output)) * 0.5
    if type == 4:
        farSig_max = random.uniform(0.5, 0.9) * np.amax(abs(input))  # 0.5 ~ 0.9
        farSigNL = np.clip(input, -farSig_max, farSig_max)
        bn = farSigNL * 1.5 - 0.3 * (farSigNL ** 2)
        a1 = bn > 0
        a2 = bn <= 0
        ab = bn * (4 * a1) + bn * (0.5 * a2)
        output = 4 * (2 / (1 + np.exp(-1 * ab)) - 1)
    if type == 5:
        farSig_max = random.uniform(0.5, 0.9) * np.amax(abs(input))  # 0.5 ~ 0.9
        x_soft_n = (farSig_max * input) / (np.sqrt(farSig_max ** 2 + input ** 2 + 1e-7) + 1e-7)
        x_hard_n = np.clip(input, -farSig_max, farSig_max)
        bn = 1.5 * x_soft_n - 0.3 * (x_soft_n ** 2)
        a1 = bn > 0
        a2 = bn <= 0
        ab = bn * (4 * a1) + bn * (0.5 * a2)
        output = 4 * (2 / (1 + np.exp(-1 * ab)) - 1)

    return output
