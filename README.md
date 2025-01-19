# Neural network theorum
create a neural network using pytorch

input->network-> output

network is composed of multiple layer transform , the transform type usually is Linear transform or convolution transform: input->o1->o2->o3->output

linear: y=ax+b

convolution: x* kernal=y

so we want to find best each layer a,b or kernal can that y closed to y actual 
loss_function=y-y_actual
loss_function=abs(y-y_actual)


# Neural network tranining

we use following iterative to find best model parameter let output closed to y actual
![image](https://github.com/user-attachments/assets/7043663f-c5ff-4898-b086-7e8e484f3c94)


corresponding python code:
```python
import torch
import torch.nn as nn
import torch.optim as optim

a=torch.tensor([1.0,2.0,3.0,4.0])


class simplenn(nn.Module):
  def __init__(self):
      # self.transform1=nn.Conv2d(3,32,kernel_size=8,stride=4)
      super(simplenn,self).__init__()
      # self.transform2=nn.Conv2d(32,64,kernel_size=8,stride=4)
      self.transform3=nn.Sequential(nn.Linear(4,1),nn.ReLU())
  def forward(self,x):
        # x=self.transform1(x)
        # x=self.transform2(x)
        # x=x.view(x.size(0),-1)
        x=self.transform3(x)
        return x
  


network=simplenn()
b=network.forward(a)
print(b)

loss_function=nn.MSELoss()
optimizer=optim.SGD(network.parameters(),lr=0.01)
''' 
[1,2,3,4]*a+b->>>ouput, we want to find best a,b so that ,output can closed to 0

'''
for name, param in network.named_parameters():
    print(f"Parameter name: {name}")
    print(f"Values:\n{param.data}")

episode=100
y_actual=3
for i in range(episode):
    y_pred=network.forward(a)
    loss=loss_function(y_pred,y_actual)
    loss.backward()
    for name,param in network.named_parameters():
      print(f"gradient of {name}:{param.grad}")
    optimizer.step()
```
