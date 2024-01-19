---
title:  "Einsum Gradient"
mathjax: true
layout: post
categories: media
---

<script>window.MathJax = { 
    tex: { 
        tags: "ams", 
        scale: 200,
    }, 
}; </script> 
 <script async='async' id='MathJax-script' src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js' type='text/javascript'></script>  
<!-- l. 10 --><p class='noindent'>Last time I talked about how to find the jacobian of einsum. You can
use that in backpropogation but you’ll run out of RAM very quickly.
Today I want to explain how you can backpropogate through einsum
operations without needing the full jacobian. Here is the setup: we have neural
network \(N : \mathbb {R}^{np} \rightarrow \mathbb {R}\), and somewhere in that network is an einsum operation that
we want to backpropogate across. For simplicity we'll make it matrix
multiplication
</p><!-- l. 22 --><p class='indent'>   \begin {equation}  X_{ij}W_{jk} = Z_{jk}  \end {equation}<a id='x1-2r1'></a>
</p><!-- l. 24 --><p class='indent'>   We have the upstream gradient \(G\), which is the same shape as \(Z\), and we want to
find the gradients w.r.t. \(X, W\), and then generalize the process to any einsum
operation. If we calculate the gradient w.r.t. to a single element of \(W\) we see
that
</p><!-- l. 33 --><p class='indent'>   \begin {equation}  \bigl (\nabla N_{W}\bigr )_{jk} = \sum _i G_{ik}X_{ij}  \end {equation}<a id='x1-3r2'></a>
</p><!-- l. 35 --><p class='indent'>   This is the sum of the product of the \(k\)’th column of \(G\) and the \(j\)’th column of \(X\).
Notice that (2) can be written as an einsum
</p><!-- l. 40 --><p class='indent'>   \begin {equation}  X_{ij}G_{ik} = \bigl (\nabla N_{W} \bigr )_{jk}  \end {equation}<a id='x1-4r3'></a>
</p><!-- l. 42 --><p class='indent'>   This is similar to (1). We left the subscripts alone, and put the upstream
gradient \(G\) in place of \(W\). Lets do the same thing to find \(\nabla N_X\).
</p><!-- l. 47 --><p class='indent'>   \begin {equation}  \bigl (\nabla N_{X}\bigr )_{ij} = \sum _k G_{ik}W_{jk}  \end {equation}<a id='x1-5r4'></a>
</p><!-- l. 51 --><p class='indent'>   \begin {equation}  G_{ik}W_{jk} = \bigl (\nabla N_{X} \bigr )_{ij}  \end {equation}<a id='x1-6r5'></a>
</p><!-- l. 53 --><p class='indent'>   For \(\nabla N_X\) we take the sum of the product of the \(i\)’th row of \(G\) with the \(j\)’th row of \(W\).
Comparing (1), (2), (5) we can see the pattern. We leave the subscripts alone and
put \(G\) in place of whatever we are differentiating with respect to. This
rule almost works, but we can easily think of a situation where it fails.
Consider
</p><!-- l. 61 --><p class='indent'>   \begin {equation}  X_{ab}W_{cd} = Z_{ab}  \end {equation}<a id='x1-7r6'></a>
</p><!-- l. 63 --><p class='indent'>   our rule would give
</p><!-- l. 67 --><p class='indent'>   \begin {equation}  X_{ab}G_{ab} \xrightarrow []{??} \bigl (\nabla N_W\bigr )_{cd}  \end {equation}<a id='x1-8r7'></a>
</p><!-- l. 69 --><p class='indent'>   which doesnt work. There is no way to yield a \(cd\) matrix from the operands on
the LHS. For another example that wouldnt work Consider
</p><!-- l. 74 --><p class='indent'>   \begin {equation}  X_{bij}W_{jk} = Z_{ik}  \end {equation}<a id='x1-9r8'></a>
</p><!-- l. 76 --><p class='indent'>   Here our rule says \(\nabla N_X\) can be found with \(G_{ik}W_{jk} \xrightarrow []{??} \bigl (\nabla N_X\bigr )_{bij}\) but again this operation doesnt
make sense since the \(b\) dimension is missing from \(G,W\). We notice that this
is a <span class='cmti-10x-x-109'>shape </span>issue and not a <span class='cmti-10x-x-109'>value </span>issue. Looking at (8) we can see the
gradient will be the same across all batches, meaning the gradient we
want is some \(ij\) matrix repeated across the \(b\) dimension. To get it we just
need to reshape our the output from our rule . Let \(A\) be a placeholder
matrix to indicate the output of the einsum. The correct gradient for (8)
is

</p><!-- l. 87 --><p class='indent'>   \begin {equation}  1_{b} \otimes \bigl (G_{ik}W_{jk}\rightarrow A_{ij}\bigr ) = \bigl (\nabla N_X\bigr )_{bij}  \end {equation}<a id='x1-10r9'></a>
</p><!-- l. 89 --><p class='indent'>   We can revisit (6), (7) and get the gradient with the right shape
via
</p><!-- l. 93 --><p class='indent'>   \begin {equation}  1_{cd} \otimes \bigl (X_{ab}G_{ab} \rightarrow a\bigr ) = \bigl (\nabla N_W\bigr )_{cd}  \end {equation}<a id='x1-11r10'></a>
</p><!-- l. 96 --><p class='indent'>   At this point we are done. Backpropogating over an einsum operation can be
done in two steps:
     </p><ol class='enumerate1'>
<li class='enumerate' id='x1-13x1'>Take the original operation and replace the target variable with the
     upstream gradient
     </li>
<li class='enumerate' id='x1-15x2'>Reshape/broadcast the result</li></ol>



## Putting it in code

Were going to write a little autograd engine. We need a class for holding data, and then einsum and sigmoid functions. 
#### The class

```python
from collections import deque
import numpy as np

class Thing:
    def __init__(self, data, _children=[]) -> None:
        self.data = data if data.shape else data.reshape(1,1)
        self._children = _children
        self._backward = lambda: None  
        self.grad = np.zeros_like(self.data)

    def backward(self):
        self.grad, visited, queue = np.ones((1, 1)), set(), deque([self])
        while queue: v = queue.popleft(); [visited.add(v), v._backward(), queue.extend(n for n in v._children if n not in visited)][0]
```
This just holds data/grad and backwards stuff. Im reshaping scalars
to `(1, 1)` because its easier to handle. 
Calling `.backward()` does breadth first and calls `._backward()` on all children. Pretty jank one line BFS.

#### Sigmoid

```python
def _sigmoid(x):
    out = Thing(1 / (1 + np.exp(-x.data)), _children=[x])
    def _backward():
        x.grad += out.data * (1 - out.data) * out.grad
    out._backward = _backward
    return out
```
Pretty standard. The `out.grad` just needs to be multiplied by the simgoid derivative. 

#### Einsum

This one is a little more involved. As described above we are doing the original
einsum but replacing the target variable with the upstream gradient, but we need to be careful here. From (8), a pattern like `bij,jk->ik` will throw an error
when we try to backpropogate into $$X$$ with `ik,jk->bij`. In this example the $$X$$ grad 
string pattern,
the `->...` part, needs to be only the subscripts of $$X$$ contained in the subscripts
of $$W$$ and upstream gradient. Its hard to explain this in words but you'll probably get it
if you read the code. Were going to use `einops.repeat` for the reshape/broadcasting

```python
from einops import repeat

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
```
1. We start by making the `_grad_ptrn` for `x, w`. These have to come from
the set of subscripts in the other two operands for it to 
make sense. 
2. After that we calculate the gradient
3. After that we reshape the gradient by broadcasting it along
the missing dimensions. For an example like (8) `repeat` would get passed
 `'i j -> b i j', b = x.shape[0]`

#### Testing

Lets spin up a random NN and see how it goes. I dont want to code a loss
function so were just going to sum stuff at the end. 

```python
import torch
from torch import einsum
torch.manual_seed(10)
np.random.seed(10)


x = torch.randn(2, 3, requires_grad = True)
thing = Thing(x.detach().numpy())

shapes = [(3, 4), (4, 5), (1, 2, 5), (5, 3)]
ptrns = ['ij,jk->ik', 'ij,jk->ik', 'ij,cij->ij', 'ab,bd->']

torch_weights = [torch.randn(s, requires_grad = True) for s in shapes]
thing_weights = [Thing(w.detach().numpy()) for w in torch_weights]

for (ptrn, w, thing_w) in zip(ptrns, torch_weights, thing_weights):
    x = torch.sigmoid(einsum(ptrn, x, w))
    thing = _sigmoid(_einsum(ptrn, thing, thing_w))

x.backward()
thing.backward()

for w, thing_w in zip(torch_weights, thing_weights):
    print(np.allclose(w.grad.detach().numpy(), thing_w.grad))
# prints all true
```
alright thats about it. We can now backpropogate over an arbitrary
einsum without relying on the jacobian. This is a lot faster. You
can build an autograd engine around this and the backward pass reduces 
to an einsum and potential reshape at each layer. 














