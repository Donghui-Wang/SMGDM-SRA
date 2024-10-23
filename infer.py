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
    parser.add_argument('-c', '--config', type=str, default='config/shadow.json',
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

    if(val_set.data_len == 540):
        Status = True
    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

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
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            without_BAR_path = BAR_path + '/without_BAR'
            Path(without_BAR_path).mkdir(parents=True, exist_ok=True)
            Metrics.save_img(
                sr_img, '{}/{}_{}_sr.png'.format(without_BAR_path, current_step, idx))
            # Metrics.save_img(
            #     Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}.png'.format(result_path, current_step, idx))

        Metrics.save_img(
            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
        # Metrics.save_img(
        #     fake_img, '{}/{}_{}.png'.format(result_path, current_step, idx))
        Metrics.save_img1(
            mask_img, '{}/{}_{}_mask.png'.format(result_path, current_step, idx))
        # Metrics.save_img(
        #     fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

        # if wandb_logger and opt['log_infer']:
        #     wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img)

    if use_bar:
        # 定义源目录和目标目录
        source_dir = result_path
        mask_boundary_dir = BAR_path+'/boundary_mask'
        complete_path = BAR_path+'/complete_path'

        # 创建目标目录，如果不存在则创建
        Path(mask_boundary_dir).mkdir(parents=True, exist_ok=True)
        Path(complete_path).mkdir(parents=True, exist_ok=True)

        # 遍历源目录中的所有以_mask.png结尾的文件
        for filename in os.listdir(source_dir):
            if filename.endswith('_mask.png'):
                # 读取掩膜图像
                image_path = os.path.join(source_dir, filename)
                mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if mask is None:
                    print(f"无法读取图像: {image_path}")
                    continue

                # 查找掩膜中的轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 创建一个空白掩膜用于绘制边界
                boundary_mask = np.zeros_like(mask)

                # 在空白掩膜上绘制轮廓（边界），增加厚度以使边界半径更大
                cv2.drawContours(boundary_mask, contours, -1, (255), thickness=3)

                # 反相操作，将黑色变为白色，白色变为黑色
                inverted_boundary_mask = cv2.bitwise_not(boundary_mask)

                # 保存边界掩膜
                new_filename = filename.replace('_mask', '')
                boundary_mask_path = os.path.join(mask_boundary_dir, new_filename)
                cv2.imwrite(boundary_mask_path, inverted_boundary_mask)
                print(f"边界掩膜已保存到: {boundary_mask_path}")
        # 读取 YAML 文件
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)

        # 动态调整配置项的值
        # config['data']['eval']['lama_inet256_ev2li_n100_test']['paths']['srs'] = complete_path
        config['data']['eval']['lama_inet256_ev2li_n100_test']['gt_path'] = without_BAR_path
        config['data']['eval']['lama_inet256_ev2li_n100_test']['mask_path'] = mask_boundary_dir

        # 将修改后的配置写回到 YAML 文件
        with open(config_file_path, 'w') as file:
            yaml.dump(config, file)

        print('YAML 文件已成功更新并保存')
        main(conf_arg)
        if wandb_logger and opt['log_infer']:
            wandb_logger.log_eval_table(commit=True)
