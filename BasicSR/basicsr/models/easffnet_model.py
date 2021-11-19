import torch
from collections import OrderedDict
from tqdm import tqdm
import os
import wandb
import numpy as np
import cv2
import lpips
from copy import deepcopy
import logging

from .esrgan_model import ESRGANModel

from basicsr.metrics import calculate_metric
from basicsr.metrics.lpips import calculate_lpips
from basicsr.utils import imwrite, tensor2img
from basicsr.archs import build_network
from basicsr.utils.registry import MODEL_REGISTRY


logger = logging.getLogger('basicsr')

@MODEL_REGISTRY.register()
class EASFFNetModel(ESRGANModel):
    """EASFFNet model for artifact reduction with exemplar reference_image"""

    def feed_data(self, data):
        self.compressed = data['compressed'].to(self.device)
        self.reference = data['reference'].to(self.device)
        self.landmark = data['landmark'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    '''
    def load_network(self, net, load_path, strict=True, param_key='params'):
        net = self.get_bare_model(net)
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]

        with torch.no_grad():
            for name in ["compressed", "reference"]:
                # print("Here!")
                # print(net.state_dict()[f"{name}_conv_first.weight"].shape, net.state_dict()[f"{name}_conv_first.weight"].dtype, net.state_dict()[f"{name}_conv_first.weight"][0])
                net.state_dict()[f"{name}_conv_first.weight"].copy_(load_net["conv_first.weight"])
                # print(net.state_dict()[f"{name}_conv_first.weight"].shape, net.state_dict()[f"{name}_conv_first.weight"].dtype, net.state_dict()[f"{name}_conv_first.weight"][0])
                net.state_dict()[f"{name}_conv_first.bias"].copy_(load_net["conv_first.bias"])
                for i in range(net.num_block):
                    for j in range(1, 4):
                        for k in range(1, 6):
                            net.state_dict()[f"{name}_body.{i}.rdb{j}.conv{k}.weight"].copy_(load_net[f"body.{i}.rdb{j}.conv{k}.weight"])
                            net.state_dict()[f"{name}_body.{i}.rdb{j}.conv{k}.bias"].copy_(load_net[f"body.{i}.rdb{j}.conv{k}.bias"])
                net.state_dict()[f"{name}_conv_body.weight"].copy_(load_net["conv_body.weight"])
                net.state_dict()[f"{name}_conv_body.bias"].copy_(load_net["conv_body.bias"])
        for other_name, param in net.named_parameters():
            # print(other_name, ':', param.requires_grad)
            param.requires_grad = True
    '''

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.compressed, self.reference, self.landmark)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pixel'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_perceptual'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            if self.cri_gan:
                fake_g_pred = self.net_d(self.output)
                l_g_gan = self.cri_gan(fake_g_pred, target_is_real=False, is_disc=False)

                l_g_total += l_g_gan
                loss_dict['l_g_gan'] = l_g_gan

                loss_dict['l_g_total'] = l_g_total

                l_g_total.backward()
                self.optimizer_g.step()

                # optimize net_d
                for p in self.net_d.parameters():
                    p.requires_grad = True

                self.optimizer_d.zero_grad()

                # gan loss

                fake_d_pred = self.net_d(self.output).detach()
                real_d_pred = self.net_d(self.gt)
                l_d_real = self.cri_gan(real_d_pred, target_is_real=True, is_disc=True)
                l_d_fake = self.cri_gan(fake_d_pred, target_is_real=False, is_disc=True)
                l_d = l_d_real + l_d_fake
                l_d.backward()
                self.optimizer_d.step()

                loss_dict['l_d_real'] = l_d_real
                loss_dict['l_d_fake'] = l_d_fake
                loss_dict['l_d_total'] = l_d
            else:
                l_g_total.backward()
                self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.compressed, self.reference, self.landmark)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.compressed, self.reference, self.landmark)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='reference_image')

        if with_metrics:
            for name, opt_ in self.opt['val']['metrics'].items():
                if name == 'lpips':
                    lpips_net = lpips.LPIPS(net='alex').cuda()

        for idx, val_data in enumerate(dataloader):
            img_name = os.path.splitext(os.path.basename(val_data['compressed_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            result_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.compressed
            del self.reference
            del self.landmark
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = os.path.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = os.path.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = os.path.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(result_img, save_img_path)
            if int(img_name.split("_")[0]) % 100 == 0:
                compressed_img = cv2.cvtColor(tensor2img([visuals['compressed']]), cv2.COLOR_BGR2RGB)
                reference_img = cv2.cvtColor(tensor2img([visuals['reference']]), cv2.COLOR_BGR2RGB)
                landmark_img = np.expand_dims(tensor2img([visuals['landmark']]), 2)
                landmark_img = cv2.merge((landmark_img, landmark_img, landmark_img))
                stacked_img = np.hstack((compressed_img, reference_img, landmark_img, cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)))
                wandb.log({img_name: wandb.Image(stacked_img)})

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=result_img, img2=gt_img)
                    if name == 'lpips':
                        self.metric_results[name] += calculate_lpips(result_img, gt_img, lpips_net)
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                if metric == 'lpips':
                    del lpips_net

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['compressed'] = self.compressed.detach().cpu()
        out_dict['reference'] = self.reference.detach().cpu()
        out_dict['landmark'] = self.landmark.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
