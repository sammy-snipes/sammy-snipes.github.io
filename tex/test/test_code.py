from einops import repeat

test = f"{self.main} -> {nigga}"

def _einsum(ptrn, x, w):
    out = Thing(np.einsum(ptrn, x.data, w.data), _children=[x, w])
    def _backward():
        x_ptrn, w_ptrn, z_ptrn = *ptrn.split('->')[0].split(','), ptrn.split('->')[-1]
        z_ptrn = z_ptrn if z_ptrn else 'zy'
        
        w_grad_ptrn = ''.join([c for c in w_ptrn if c in set(x_ptrn + z_ptrn)])
        x_grad_ptrn = ''.join([c for c in x_ptrn if c in set(w_ptrn + z_ptrn)])
        
        x_grad = np.einsum(f'{z_ptrn},{w_ptrn}->{x_grad_ptrn}', out.grad, w.data)
        w_grad = np.einsum(f'{z_ptrn},{x_ptrn}->{w_grad_ptrn}', out.grad, x.data)

        w_shape = dict(zip(w_ptrn, w.data.shape))
        x_shape = dict(zip(x_ptrn, x.data.shape))

        w_broadcast_string = f"{' '.join(w_grad_ptrn)} -> {' '.join(w_shape.keys())}"
        w_grad = repeat(w_grad, w_broadcast_string, **w_shape)

        x_broadcast_string = f"{' '.join(x_grad_ptrn)} -> {' '.join(x_shape.keys())}"
        x_grad = repeat(x_grad, x_broadcast_string, **x_shape)

        x.grad += x_grad
        w.grad += w_grad

    out._backward = _backward
    return out

def _sigmoid(x):
    out = Thing(1 / (1 + np.exp(-x.data)), _children=[x])
    def _backward():
        x.grad += out.data * (1 - out.data) * out.grad
    out._backward = _backward
    return out