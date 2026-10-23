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


print("Testing shape")
shape = (2,4) 

rand_tensors = torch.rand(shape)
ones_tensors = torch.ones(shape)
zeros_tensors = torch.zeros(shape)

print(f"Random Tensors: \n {rand_tensors}")
print(f"Random Tensors: \n {ones_tensors}")
print(f"Zeros Tensor: \n {zeros_tensors} \n")


tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Data Type of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}\n")

if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

tensor = torch.ones(4,4)

print(f"First row: {tensor[:,0]}")
print(f"First column: {tensor[:, 0]}") 
print(f"Last column: {tensor[..., -1]}") 
tensor[:,1] = 0
print(tensor) 

t1 = torch.cat([tensor, tensor, tensor], dim = 1)
print(t1)


y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor,tensor.T, out= y3)

print(y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out =z3)

print(z3)

print("Finding the sum of Tensor variable")
print(tensor)
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"{tensor} \n") 
tensor.add_(5) # the '_' operations that store the reuslt into the operand
print(tensor)

# tensor to numpy array

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)

print(f"T: {t}")
print(f"N: {n}")

n = np.ones(5)
t = torch.from_numpy(n)



print(f"Numpy: {n} to Torch: {t}")