import torch

from torch.autograd import Variable
# from evaluate import test_main 

def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu'):
    eps = 10 / 255
    compress_image = Variable(compress_image.clone().to(device), requires_grad=True)
    score = model(ref_image.to(device), compress_image.to(device)) if ref_image is not None else model(compress_image.to(device))
    loss = 1 - score / metric_range
    loss.backward() 
    g = compress_image.grad
    g = torch.sign(g)
    compress_image.data -= eps * g
    compress_image.data.clamp_(0., 1.)
    compress_image.grad.zero_()

    res_image = (compress_image).data.clamp_(min=0, max=1)
    return res_image
            
# if __name__ == "__main__":
#     test_main(attack)

