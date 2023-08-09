from tinygrad.tensor import Tensor

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * x
y.backward()

print(x.grad)  # This will show the gradient of y w.r.t. x
