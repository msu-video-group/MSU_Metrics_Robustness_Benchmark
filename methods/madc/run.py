import torch

from torch.autograd import Variable
from evaluate import test_main 


def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu'):
    init_image = compress_image.clone()
    compress_image = Variable(compress_image, requires_grad=True).to(device)
    init_image = Variable(init_image, requires_grad=False).to(device)
    lr = 0.001
    eps = 10 / 255
    iters = 8
    for i in range(iters):
        score = model(ref_image.to(device), compress_image.to(device)) if ref_image is not None else model(compress_image.to(device))
        score = score.mean()
        loss = 1 - score / metric_range
        loss.backward() 
        g2 = compress_image.grad.clone()
        compress_image.grad.zero_()
    
        if (i < 1):
          pg = g2.clone()
        else:
          loss = ((compress_image - init_image)**2).mean()**0.5
          loss.backward()
          g1 = compress_image.grad.clone()
          compress_image.grad.zero_()
          pg = g2 - (g1*g2).sum() / (g1*g1).sum() * g1
    
    
        pg = torch.sign(pg)
        compress_image.data -=  lr * pg
        compress_image.grad.zero_()
    
        cur_score = ((compress_image - init_image)**2).mean()**0.5
        while cur_score > eps:
          cur_score.backward() 
          g2 = torch.sign(compress_image.grad)
          compress_image.data -= 0.0005 * g2
          compress_image.grad.zero_()
          compress_image.data.clamp_(0., 1)
          cur_score = ((compress_image - init_image)**2).mean()**0.5
        
        compress_image.data.clamp_(0., 1.)
        compress_image.grad.zero_()
    res_image = (compress_image).data.clamp_(min=0, max=1)

    return res_image


if __name__ == "__main__":
    test_main(attack)

