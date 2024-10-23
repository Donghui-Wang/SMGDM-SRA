import cv2
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from BAR import conf_mgt
from BAR.utils import yamlread
from core.wandb_logger import WandbLogger
#from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
#import wandb
import random
from pathlib import Path
from calflops import calculate_flops

import yaml
from  BAR.test import  main




import BAR.test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/shadowgeneration.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    parser.add_argument('--conf_path', type=str, required=False, default="config/test_inet256_ev2li.yml")
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    config_file_path = os.path.join('config', 'test_inet256_ev2li.yml')


    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
  #  tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    use_bar = opt['use_bar']
    Status = False


    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')


    # ######### Set Seeds ###########
    # random.seed(1234)
    # np.random.seed(1234)
    # torch.manual_seed(1234)
    # torch.cuda.manual_seed_all(1234)

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    # 使用 os.path.abspath 获取完整的绝对路径
    BAR_path = os.path.abspath(result_path)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=Status)
        visuals = diffusion.get_current_visuals(need_LR=False)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        mask_img = Metrics.tensor2img(visuals['mask'])
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]), '{}/{}_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # grid img
            shadow_img = Metrics.tensor2img(visuals['SR'])  # uint8
            Metrics.save_img(
                shadow_img, '{}/{}_{}_shadow.png'.format(result_path, current_step, idx))
            # Metrics.save_img(
            #     Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}.png'.format(result_path, current_step, idx))


