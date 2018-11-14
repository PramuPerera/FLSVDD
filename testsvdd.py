import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from data_loader import get_loader
import numpy as np
from sklearn.metrics import roc_curve, auc
import scipy.io as io

def run(opt):
    output={}
    #opt.cname = 0
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    cname = opt.cname
    opt.display_id = -1   # no visdom display
    datatrain, data_loader , dataset_size = get_loader(opt,classname = opt.cname)
    data_iter = iter(data_loader) 
    model = create_model(opt)
    model.setup(opt)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    if opt.eval:
        model.eval()
    cnt = 0
    for data, lbl in datatrain:
        cnt +=1
	model.set_input(data)
        out, code = model.test()
	if cnt ==1:
		f = code.view(code.size(0), -1).cpu().numpy().tolist()
	else:
		f += code.view(code.size(0), -1).cpu().numpy().tolist()
    output['train']=f
    cnt = 0	
    for data, lbl in data_loader:
        cnt +=1
        model.set_input(data)
        out, code = model.test()
	if cnt ==1:
		tlbl = (lbl==cname).numpy().tolist()
		g  = code.view(code.size(0), -1).cpu().numpy().tolist()
		tloss =  np.mean(((out.cpu()-data).numpy())**2,(1,2,3)).tolist()
	else:
		g += code.view(code.size(0), -1).cpu().numpy().tolist()
		tlbl += (lbl==cname).numpy().tolist()
		tloss +=  np.mean(((out.cpu()-data).numpy())**2,(1,2,3)).tolist()
	
        #visuals = model.get_current_visuals()
        #img_path = model.get_image_paths()
        #save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # save the website
    #webpage.save()
    output['test']=g
    output['lbl']=tlbl
    io.savemat('feat.mat',output)
    fpr, tpr, _ = roc_curve(tlbl, tloss, 0)
    roc_auc1 = auc(fpr, tpr)
    return(roc_auc1)

if __name__ == '__main__':
    opt = TestOptions().parse()
    roc = run(opt)

