import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import Sampler

class OCSampler(Sampler):

    def __init__(self, dataset, cname):
	self.mask = [i for i,e  in enumerate(dataset) if e[1].numpy()==cname]
        #self.mask = [i for i,e  in enumerate(dataset) if e[1]==cname]

    def __iter__(self):
        return (iter(self.mask))

    def __len__(self):
        return len(self.mask)



def get_loader(config , classname = 1):
    transform = transforms.Compose([
                    transforms.Scale(config.fineSize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    mnist = datasets.MNIST(root='../data/mnist', download=True, transform=transform)
    mnist_val = datasets.MNIST(root='../data/mnist', train=False, download=True, transform=transform)
    MNISTsampler = OCSampler(mnist, classname)
    SVHNsampler = OCSampler(mnist, classname)
    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size, sampler=MNISTsampler,
                                               
                                               num_workers=config.num_threads)
    mnist_val_loader = torch.utils.data.DataLoader(dataset=mnist_val,
                                               batch_size=100,
                                               
                                               num_workers=config.num_threads)
    return  mnist_loader, mnist_val_loader, len(MNISTsampler)
