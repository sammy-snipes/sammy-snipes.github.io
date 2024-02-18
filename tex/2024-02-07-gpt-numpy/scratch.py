class Parameter:
    def __init__(self) -> None:
        pass


def function(x : Parameter) -> Parameter:
    out = Parameter(.....)
    def _backward():
        x.grad += .....
    out._backward = _backward
    return out 
