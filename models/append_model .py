import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np

class AACModel(BaseModel):
    def name(self):
        return 'AACModel'

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
            self.model_names = ['En', 'De']
        # load/define networks
        self.netEn = networks.define_G(opt.input_nc, opt.output_nc,opt.ngf,  'en', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netDe = networks.define_G(opt.input_nc, opt.output_nc ,opt.ngf, 'de', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
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
	self.code = self.netEn(self.real_A)
        self.fake_B = self.netDe(self.code)
 	self.code = self.code.view(self.code.size(0), -1)
	#MSE = torch.unsqueeze(torch.mean(torch.mean(torch.mean(torch.pow((self.fake_B-self.real_A),2),1),1),1),dim=0)
	#t = self.code.view(self.code.size(0), -1).clone().detach()
	#t = t.cpu().detach().numpy()
	#self.code= torch.tensor(np.concatenate((t, MSE.cpu().detach().numpy().transpose().repeat(t.shape[1],1)),1)).cuda()
	#self.code = t


    def test(self):
        with torch.no_grad():
		self.code = self.netEn(self.real_A)
		self.fake_B = self.netDe(self.code)
 		self.code = self.code.view(self.code.size(0), -1)
        	
		#MSE = torch.unsqueeze(torch.mean(torch.mean(torch.mean(torch.pow((self.fake_B-self.real_A),2),1),1),1),dim=0)
		#t = self.code.view(self.code.size(0), -1).clone().detach()
		#t = t.cpu().detach().numpy()
		#self.code= torch.tensor(np.concatenate((t, MSE.cpu().detach().numpy().transpose().repeat(t.shape[1],1)),1)).cuda()
		#self.code = t
	return [self.fake_B, self.code]

    def backward_D(self):
        fake_AB = self.code
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
	btchsz = self.code.shape[0]
	N = self.code.shape[1]
	#m = torch.distributions.cauchy.Cauchy(torch.zeros(N), torch.eye(N)) 
	m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(N), 2* torch.eye(N))


	'''while True:
		m = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([0.5]));
		A = m.sample((btchsz*10,N));
		for k in A:
			print(np.shape(k))
			print(torch.norm(k))	
			print(SD)
		B =[k for k in A if torch.norm(k)<1 ]
		print(len(B))
		if len(B)> btchsz:
			self.real_code = B[0:btchsz]
			self.real_code = torch.tensor(np.array(self.real_code))
			print(self.real_code.shape)
			print(self.code.shape)
			break
	

	m = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([0.5]));
	A = m.sample((btchsz,N));
	A = A/torch.norm(A, p=2, dim=1, keepdim =True)
	l = m.sample((btchsz,1))
	self.real_code = torch.squeeze(A*l.expand_as(A))
	'''

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
