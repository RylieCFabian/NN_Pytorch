# was lazy with the imports
from pytorch_test import *

"""
This is where I loaded the Models in after training and made it guess certain clothing Items.

Honestly I'm trying to make it show the images it's self but I still don't know how to do that. But I made the Neural Network in like 20 min. 

PyTorch is peak!
"""

model = NueralNetwork().to(device)
model.load_state_dict(torch.load("model.pth" , weights_only = True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x , y = test_data[0][0] , test_data[0][1]

with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted , actual = classes[pred[0].argmax(0)] , classes[y] 
    print(f'Predicted: "{predicted}" , Actual: "{actual}"')

