import torch

from torch.autograd import Variable
#from evaluate import test_main 

         

def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu'):
    eps = 10 / 255
    iters = 10
    alpha = 1/255
    compress_image = Variable(compress_image.clone().to(device), requires_grad=True)
    
    p = torch.zeros_like(compress_image).to(device)
    p = Variable(p, requires_grad=True)
    for i in range(iters):
        res = compress_image + p
        res.data.clamp_(0., 1.)
        score = model(ref_image.to(device), res.to(device)) if ref_image is not None else model(res.to(device))
        loss = 1 - score / metric_range
        loss.backward() 
        g = p.grad
        g = torch.sign(g)
        p.data -= alpha * g
        p.data.clamp_(-eps, +eps)
        p.grad.zero_()
    res_image = compress_image + p

    res_image = (res_image).data.clamp_(min=0, max=1)
    return res_image


# if __name__ == "__main__":
#     test_main(attack)
