#    Author:  a101269
#    Date  :  2020/3/6
import random
import os
import numpy as np
import re
import logging
import torch
import tensorflow as tf
from pathlib import Path
import utils.colorer

logger = logging.getLogger()


def init_logger(log_file=None):
    '''
    logging
    Example:
        >>> from utils.utils import init_logger
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    # log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # DEBUG，INFO，WARNING，ERROR，CRITICAL
    console = logging.StreamHandler()
    console.setFormatter(log_format)
    logger.handlers = [console]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def seed_everything(seed=123):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def has_ens_num(str0):
    return bool(re.search('[a-zA-Z0-9]', str0))


def device(use_cuda):
    """
    setup GPU device if available, move model into configured device
    # 如果n_gpu_use为数字，则使用range生成list
    # 如果输入的是一个list，则默认使用list[0]作为controller
    Example:
        use_gpu = '' : cpu
        use_gpu = '0': cuda:0
        use_gpu = '0,1' : cuda:0 and cuda:1
     """

    if not use_cuda:
        device_type = 'cpu'
    else:
        device_type = f"cuda"
    device = torch.device(device_type)
    return device
