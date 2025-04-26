import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import HRModel
from .utils import Progbar, create_dir
from .metrics import PSNR
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import kpn.utils as kpn_utils
import torchvision
import lpips


class HAFN():
    def __init__(self, config):
        self.config = config

        self.debug = False
        self.inpaint_model = HRModel(config).to(config.DEVICE)

        self.transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)


        self.psnr = PSNR(255.0).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
            print('test dataset:'.format(len(self.test_dataset)))
        else:
            self.train_dataset = Dataset(config, config.TRAIN_INPUT_FLIST, config.TRAIN_MASK_FLIST, config.TRAIN_GT_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_INPUT_FLIST, config.VAL_MASK_FLIST, config.VAL_GT_FLIST, augment=False, training=False)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

            print('train dataset:{}'.format(len(self.train_dataset)))
            print('eval dataset:{}'.format(len(self.val_dataset)))

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True


    def load(self):
        self.inpaint_model.load(self.config.MODEL_LOAD)

    def save(self):
        self.inpaint_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=0,
            drop_last=True,
            shuffle=True
            #shuffle=False
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        max_psnr = 0
        while(keep_training):
            epoch += 1
            #if(epoch % 5)
            print('\n\nTraining epoch: %d' % epoch)

            for items in train_loader:
                self.inpaint_model.train()

                inputs, gts, unetgts= self.cuda(*items)
                
                
                #**************************************************************************************************************1
                unetout, unetloss, ulogs=self.inpaint_model.Unetprocess(inputs, unetgts)
                self.inpaint_model.unetbackward(unetloss)
                unetout1 = unetout.detach()
                # Define the RGB to grayscale conversion weights
                weights = torch.tensor([0.299, 0.587, 0.114], device=self.config.DEVICE)
                # Expand weights to match the input dimensions and apply them
                unetout1 = torch.sum(unetout1 * weights.view(1, 3, 1, 1), dim=1, keepdim=True)
          
                
                #print("Mask of channels (method 1):", num_channels_2)
                outputs,x1, gen_loss, dis_loss, logs = self.inpaint_model.process(inputs, gts, unetout1)
                
                
                # backward
                self.inpaint_model.backward(gen_loss, dis_loss)
                iteration = self.inpaint_model.iteration

                

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs+ulogs


                # sample
                if iteration % self.config.TRAIN_SAMPLE_INTERVAL == 0:
                    img_list2 = [inputs,  outputs, gts]
                    name_list2 = ['in', 'pre_1', 'gt']
                    kpn_utils.save_sample_png(sample_folder=self.config.TRAIN_SAMPLE_SAVE,
                                              sample_name='ite_{}_{}'.format(self.inpaint_model.iteration,
                                                                             0), img_list=img_list2,
                                              name_list=name_list2, pixel_max_cnt=255, height=-1,
                                              width=-1)


                # save model at checkpoints
                #if iteration % self.config.SAVE_INTERVAL == 0:
                #    self.save()

                # evaluate model at checkpoints
                if iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    cur_psnr = self.eval()
                    self.inpaint_model.iteration = iteration

                    #if iteration>self.config.SAVE_INTERVAL/2 and cur_psnr > max_psnr:
                    #    max_psnr = cur_psnr
                    #    self.save()
                    #    print('---increase-iteration:{}'.format(iteration))
                
                if iteration % self.config.LOG_INTERVAL == 0:
                    print(logs)
                
                if iteration >= max_iteration:
                    keep_training = False
                    break

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            drop_last=True,
            shuffle=False
            
        )

        model = self.config.MODEL

        self.inpaint_model.eval()

        psnr_all = []
        ssim_all = []
        l1_list = []
        lpips_list = []
        
        unet_list = []

        iteration = self.inpaint_model.iteration
        with torch.no_grad():
            for items in val_loader:
                inputs,  gts, unetgts = self.cuda(*items)

                
                
                #**************************************************************************************************************1
                unetout, unetloss, ulogs = self.inpaint_model.Unetprocess(inputs, unetgts)
                
                # Define the RGB to grayscale conversion weights
                weights = torch.tensor([0.299, 0.587, 0.114], device=self.config.DEVICE)
                # Expand weights to match the input dimensions and apply them
                unetout = torch.sum(unetout * weights.view(1, 3, 1, 1), dim=1, keepdim=True)
                outputs,x1, gen_loss, dis_loss, logs = self.inpaint_model.process(inputs, gts,unetout)
                
                
                psnr, ssim = self.metric(gts, outputs)
                #psnr, ssim = self.metric(merge_gts, unetout)
                psnr_all.append(psnr)
                ssim_all.append(ssim)

                l1_loss = torch.nn.functional.l1_loss(outputs, gts, reduction='mean').item()
                l1_list.append(l1_loss)
                
                #unet_loss = torch.nn.functional.l1_loss(unetout, unetgts, reduction='mean').item()
                #unet_list.append(unet_loss)
                unet_loss = torch.nn.functional.mse_loss(unetout, unetgts, reduction='mean').item()
                unet_list.append(unet_loss)
                
                
                pl = 1.0
                lpips_list.append(pl)
                # if torch.cuda.is_available():
                #     pl = loss_fn_vgg(transf(outputs_merged[0].cpu()).cuda(), transf(images[0].cpu()).cuda()).item()
                #     lpips_list.append(pl)
                # else:
                #     pl = loss_fn_vgg(transf(outputs_merged[0].cpu()), transf(images[0].cpu())).item()
                #     lpips_list.append(pl)


                # sample
                #if iteration>80000 and iteration % self.config.EVAL_SAMPLE_INTERVAL == 0:
                #if iteration==15000 or iteration ==60000 or iteration == 120000 or iteration >290000:
                if iteration==1520 or iteration ==7600 or iteration == 15200 or iteration ==22800 or iteration ==30400 or iteration ==38000 or iteration ==45600 or iteration ==60800 or iteration >68400:
                    img_list2 = [inputs, outputs,x1, gts, unetout, unetgts]
                    name_list2 = ['in',  'outputs', 'x1','gt', 'unetout', 'unetgts']
                    kpn_utils.save_sample_png(sample_folder=self.config.EVAL_SAMPLE_SAVE,
                                          sample_name='ite_{}_{}'.format(iteration, len(psnr_all)), img_list=img_list2,
                                          name_list=name_list2, pixel_max_cnt=255, height=-1,
                                          width=-1)
                    print('psnr:{}/{}  ssim:{}/{} unet:{}/{}  l1:{}/{}  lpips{}/{}  {}/{}'.format(psnr, np.average(psnr_all),
                                                                                   ssim, np.average(ssim_all),
                                                                                   unet_loss,np.average(unet_list),
                                                                                   l1_loss, np.average(l1_list),
                                                                                   pl, np.average(lpips_list),
                                                                                   len(psnr_all), len(self.val_dataset)))
                # sample unet
                #if iteration>self.config.SAVE_INTERVAL*0.6 and len(psnr_all) % self.config.EVAL_SAMPLE_INTERVAL == 0:
                #    img_list2 = [inputs, gts, unetout, unetgts]
                #    name_list2 = ['in', 'gt', 'unetout', 'unetgts']
                #    kpn_utils.save_sample_png(sample_folder=self.config.EVAL_SAMPLE_SAVE,
                #                          sample_name='ite_{}_{}'.format(iteration, len(psnr_all)), img_list=img_list2,
                #                          name_list=name_list2, pixel_max_cnt=255, height=-1,
                #                          width=-1)
                #    print('psnr:{}/{}  ssim:{}/{} unet:{}/{}  lpips{}/{}  {}/{}'.format(psnr, np.average(psnr_all),
                #                                                                   ssim, np.average(ssim_all),
                #                                                                   unet_loss,np.average(unet_list),
                #                                                                   pl, np.average(lpips_list),
                #                                                                   len(psnr_all), len(self.val_dataset)))
                if len(psnr_all) >= 2002:
                    break

            print('iteration:{} ave_psnr:{}  ave_ssim:{} ave_l1:{}  ave_lpips:{}   ave_unetloss:{}'.format(
                iteration,
                np.average(psnr_all),
                np.average(ssim_all),
                np.average(l1_list),
                np.average(lpips_list),
                np.average(unet_list)
            ))
            
            return np.average(psnr_all)

    def test(self):
        self.inpaint_model.eval()

        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        psnr_list = []
        ssim_list = []
        l1_list = []
        lpips_list = []

        index = 0
        with torch.no_grad():
            for items in test_loader:
                inputs, masks, gts = self.cuda(*items)
                index += 1

                outputs = self.inpaint_model(inputs, masks)
                outputs_merged = (outputs * masks) + (inputs * (1 - masks))

                psnr, ssim = self.metric(gts, outputs_merged)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                if torch.cuda.is_available():
                    pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()).cuda(), self.transf(gts[0].cpu()).cuda()).item()
                    lpips_list.append(pl)
                else:
                    pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()), self.transf(gts[0].cpu())).item()
                    lpips_list.append(pl)

                l1_loss = torch.nn.functional.l1_loss(outputs_merged, gts, reduction='mean').item()
                l1_list.append(l1_loss)

                print("psnr:{}/{}  ssim:{}/{} l1:{}/{}  lpips:{}/{}  {}".format(psnr, np.average(psnr_list),
                                                                                ssim, np.average(ssim_list),
                                                                                l1_loss, np.average(l1_list),
                                                                                pl, np.average(lpips_list),
                                                                                len(ssim_list)))


                if len(ssim_list) % 1 == 0:
                    #images_masked = inputs * (1 - masks)
                    img_list = [inputs, gts, outputs, outputs_merged]
                    name_list = ['in', 'gt', 'pre1', 'pre2']

                    kpn_utils.save_sample_png(sample_folder=self.config.TEST_SAMPLE_SAVE, sample_name='{}_'.format(len(ssim_list)),
                                              img_list=img_list,
                                              name_list=name_list, pixel_max_cnt=255, height=-1, width=-1)



            print('psnr_ave:{} ssim_ave:{} l1_ave:{} lpips:{}'.format(np.average(psnr_list),
                                                                                 np.average(ssim_list),
                                                                                 np.average(l1_list),
                                                                                 np.average(lpips_list)))


    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def metric(self, gt, pre):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        psnr = min(100, compare_psnr(gt, pre))
        ssim = compare_ssim(gt, pre, multichannel=True, data_range=255)

        return psnr, ssim