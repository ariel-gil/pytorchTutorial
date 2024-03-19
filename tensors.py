## tensor is multidimensional Array. A matrix of vecrors

import torch 
x = torch.empty(1)
y=torch.rand(1)
print(x)


z= torch.add(x,y)

s=torch.mul(x,y)

y=torch.add_(x)

y = x.view(-1,8) ## resize -1 dimension, by 8 

if torch.cuda.is_available():
    device = torch.device("cuda")
    x=torch.ones(5, device=device)
    y=torch.ones(5)
    y=y.to(device) ## move to device (GPU) 
    z=x+y
    # z.numpy() ## would return an error, numpy is only on GPU.  z.to("cpu")
    