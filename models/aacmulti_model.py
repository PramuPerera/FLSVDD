import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np

class AACMULTIModel(BaseModel):
    def name(self):
        return 'AACMultiModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
	self.opt = opt
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['En', 'De', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['En', 'De', 'D']
        # load/define networks
        self.netEn = networks.define_G(opt.input_nc, opt.output_nc,opt.ngf,  'multien', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netDe = networks.define_G(opt.input_nc, opt.output_nc ,opt.ngf, 'multide', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        use_sigmoid = opt.no_lsgan
        self.netD = networks.define_D(512, opt.ndf, 'fc',
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_En = torch.optim.Adam(self.netEn.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_De = torch.optim.Adam(self.netDe.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=0.01*opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_En)
	    self.optimizers.append(self.optimizer_De)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input.to(self.device)
        self.real_B = input.to(self.device)

    def forward(self):
	[code1 , code2] = self.netEn(self.real_A)
	temp = self.netDe(code1)
	MSE = torch.unsqueeze(torch.mean(torch.mean(torch.mean(torch.pow((code1-self.real_A),2),1),1),1),dim=0)
	code1= torch.tensor(torch.cat((code1,  torch.transpose(MSE,0,1).repeat(1,self.code.shape[1])),1))
        pred_code1 = torch.mean(self.netD(code1.view(code1.size(0), -1)),1).detach()
	temp = self.netDe(code2)
	MSE = torch.unsqueeze(torch.mean(torch.mean(torch.mean(torch.pow((code2-self.real_A),2),1),1),1),dim=0)
	code2= torch.tensor(torch.cat((code2,  torch.transpose(MSE,0,1).repeat(1,self.code.shape[1])),1))
        pred_code2 = torch.mean(self.netD(code2.view(code2.size(0), -1)),1).detach()
	lbl1 = pred_code1>=pred_code2
	lbl2 = pred_code2>pred_code1
	t1 = (lbl1).nonzero() # these are class 1 entries
	t1 = t1[:,0]
	class1=	torch.index_select(code1, 0, t1)
	t2 = (lbl2).nonzero() # these are class 1 entries
	t2 = t2[:,0]
	class2=	torch.index_select(code2, 0, t2)
	self.code= torch.cat((class1,class2),0)
        self.fake_B = self.netDe(code1,code2,t1,t2)
 	self.code = self.code.view(self.code.size(0), -1)
	self.real_B = torch.index_select(self.real_B, 0, torch.cat((t1,t2)))
	self.real_A = torch.index_select(self.real_A, 0, torch.cat((t1,t2)))
	MSE = torch.unsqueeze(torch.mean(torch.mean(torch.mean(torch.pow((self.fake_B-self.real_A),2),1),1),1),dim=0)
	self.code= torch.tensor(torch.cat((self.code,  torch.transpose(MSE,0,1).repeat(1,self.code.shape[1])),1))
	#self.code = t


    def test(self):
        with torch.no_grad():

		[code1 , code2] = self.netEn(self.real_A)
        	pred_code1 = torch.mean(self.netD(code1.view(code1.size(0), -1)),1)
        	pred_code2 = torch.mean(self.netD(code2.view(code2.size(0), -1)),1)
		lbl1 = pred_code1>=pred_code2
		lbl2 = pred_code2>pred_code1
		t1 = (lbl1).nonzero() # these are class 1 entries
		t1 = t1[:,0]
		class1=	torch.index_select(code1, 0, t1)
		t2 = (lbl2).nonzero() # these are class 1 entries
		t2 = t2[:,0]
		class2=	torch.index_select(code2, 0, t2)
		self.code= torch.cat((class1,class2),0)
        	self.fake_B =  self.netDe(code1,code2,t1,t2)
 		self.code = self.code.view(self.code.size(0), -1)
		MSE = torch.unsqueeze(torch.mean(torch.mean(torch.mean(torch.pow((self.fake_B-self.real_A),2),1),1),1),dim=0)
		self.code= torch.tensor(torch.cat((self.code,  torch.transpose(MSE,0,1).repeat(1,self.code.shape[1])),1))
	return [self.fake_B, self.code]

    def backward_D(self):
        fake_AB = self.code
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
	btchsz = self.code.shape[0]
	N = self.code.shape[1]
	#m = torch.distributions.cauchy.Cauchy(torch.zeros(N), torch.eye(N)) 
	m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(N), 2* torch.eye(N))
	self.real_code = m.sample((btchsz,1));
        pred_real = self.netD((self.real_code))
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = self.code
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_En.zero_grad()
	self.optimizer_De.zero_grad()
        self.backward_G()
        self.optimizer_En.step()
        self.optimizer_De.step()
