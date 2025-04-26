import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss
#os.environ['NCCL_P2P_DISABLE'] = '1'
#os.environ['NCCL_IB_DISABLE'] = '1'
from .UNet import UNet

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.model_save = config.PATH
        self.model_save = os.path.join(self.model_save, 'DR_Model')

    def load(self, type):
        self.gen_weights_path =self.model_save + '/' + type + '_gen.pth'
        self.dis_weights_path =self.model_save + '/' + type + '_dis.pth'

        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self):

        if len(self.config.GPU) > 1:
            generate_param = self.generator.module.state_dict()
            dis_param = self.discriminator.module.state_dict()
            print('save...multiple GPU')
        else:
            generate_param = self.generator.state_dict()
            dis_param = self.discriminator.state_dict()
            print('save...single GPU')

        torch.save({
            'iteration': self.iteration,
            'generator': generate_param
        }, os.path.join(self.model_save, '{}_{}_gen.pth'.format(self.iteration, self.name)))

        torch.save({
            'discriminator': dis_param
        }, os.path.join(self.model_save, '{}_{}_dis.pth'.format(self.iteration, self.name)))

        print('\nsaving %s...\n' % self.name)


class HRModel(BaseModel):
    def __init__(self, config):
        super(HRModel, self).__init__('HRModel', config)

        generator = InpaintGenerator(config=config)
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        #**************************************************************************************************************2
        UNet1 = UNet(input_channels=3, output_channels=3)
        self.add_module('UNet1', UNet1)
        self.UNet_loss = nn.MSELoss()
        self.optimizerUNet = torch.optim.Adam(params = UNet1.parameters(),
                                         lr=float(config.LR), betas=(0.5, 0.999))
        
        
        
        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )


        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )
    
    
    #**************************************************************************************************************3
    def update_lr(self, epoch):
        new_lr = max(1e-5, self.gen_optimizer.param_groups[0]['lr'] * 0.8 ** (epoch / 5))
        self.gen_optimizer.param_groups[0]['lr'] = new_lr
        self.optimizerUNet.param_groups[0]['lr'] = new_lr
    
    def Unetprocess(self, images, gts):
        #self.iteration += 1
        self.optimizerUNet.zero_grad()
        inputs = images
        unetout = self.UNet1(inputs)
        
        unet_loss = self.UNet_loss(unetout, gts)
        logs = [
            ("unet_loss", unet_loss.item())
        ]
        return unetout, unet_loss, logs
    
    
    
    def process(self, images, gts, unetout):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs,x1 = self(images, unetout)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = gts
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss


        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, gts) * self.config.L1_LOSS_WEIGHT 
        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, gts)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs , gts)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss


        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs,x1, gen_loss, dis_loss, logs

    def forward(self, images, unetout):
        #images_masked = images * (1 - masks)
        
        inputs = torch.cat((images, unetout), dim=1)
        #print("Number of channels (method 1):", num_channels_2)
        outputs,x1 = self.generator(inputs)
        return outputs,x1

    def backward(self, gen_loss=None, dis_loss=None):
        gen_loss.backward()
        self.gen_optimizer.step()

        dis_loss.backward()
        self.dis_optimizer.step()
#**************************************************************************************************************4    
    def unetbackward(self, unetloss=None):
        unetloss.backward()
        self.optimizerUNet.step()
        
        
        
