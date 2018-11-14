import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from data_loader import get_loader
import numpy as np
from sklearn.metrics import roc_curve, auc
if __name__ == '__main__':
    cname = 8
    opt = TestOptions().parse()
    opt.model = 'ocgan'
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    _, data_loader , dataset_size = get_loader(opt,classname = cname)
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
    for data, lbl in data_loader:
        cnt +=1
        model.set_input(data)
        out, code, order = model.test()
	tlbl = torch.index_select(tlbl, 0, order)
	if cnt ==1:
		tlbl = (lbl==cname).numpy().tolist()	
		tloss =  np.mean(((out.cpu()-data).numpy())**2,(1,2,3)).tolist()
	else:
		
		tlbl += (lbl==cname).numpy().tolist()
		tloss +=  np.mean(((out.cpu()-data).numpy())**2,(1,2,3)).tolist()

        #visuals = model.get_current_visuals()
        #img_path = model.get_image_paths()
        #save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # save the website
    #webpage.save()
    fpr, tpr, _ = roc_curve(tlbl, tloss, 0)
    roc_auc1 = auc(fpr, tpr)
    print(roc_auc1)
