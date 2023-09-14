import torch

from torch.autograd import Variable
#from evaluate import test_main 


def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu'):
    eps = 10 / 255
    iters = 10
    nu = 1.0
    compress_image = Variable(compress_image.clone().to(device), requires_grad=True)
    
    g_prev = 0
    lr = eps / iters
    for i in range(iters):
        score = model(ref_image.to(device), compress_image.to(device)) if ref_image is not None else model(compress_image.to(device))
        score = score.mean()
        loss = 1 - score / metric_range
        loss.backward() 
        g = compress_image.grad + g_prev * nu
        g_prev = g
        g = torch.sign(g)
        compress_image.data -= lr * g
        compress_image.data.clamp_(0., 1.)
        compress_image.grad.zero_()

    res_image = (compress_image).data.clamp_(min=0, max=1)
    return res_image

# if __name__ == "__main__":
#     test_main(attack)

