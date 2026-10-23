"""
Acording to Pytorch, tensors are data structures that are similar to arrays and matrices.


"""
import torch
import numpy as np

# directly from data

data =[[1,2],[3,4]]

x_data = torch.tensor(data= data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
print(f"Ones Tensors: \n {x_ones} \n ")

x_rand = torch.rand_like(x_data, dtype = torch.float)
print(f"Random Tensors: \n {x_rand} \n")

