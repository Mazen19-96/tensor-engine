from typing import List, NamedTuple, Callable , Optional , Union
import numpy as np

class Dependency(NamedTuple):
    tesnor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)



class Tensor:
    def __init__(self,data:Arrayable, 
                 requires_grad: bool = False, 
                 depends_on = None) -> None:
        self.data = ensure_array(data)
        self.requires_grad =  requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad:  Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self.gard = Tensor(np.zeros_like(self.data))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, gard: 'Tensor') -> None:
        assert self.requires_grad, "non_requires_grad tensor"
         
        self.grad += grad.data

        for Dependency in self.depends_on:
            backward_grad = Dependency.grad_fn(grad.data)
            Dependency.tensor.backward(tensor(backward_grad))


    def sum(self) -> 'Tensor':  
        return tesnor_sum(self) 


def tesnor_sum(t: Tensor) -> 'Tesnor':

    data = t.data.sum()
    requires_grad = t.requires_grad
    
    if requires_grad:
        def grad_fn(grad: np.ndarray) ->np.ndarray:
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t,grad_fn)]
    else :
        depends_on = []

    return Tensor(data,requires_grad,depends_on)