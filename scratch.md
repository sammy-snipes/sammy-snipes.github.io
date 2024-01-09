<!-- The way I ended up doing this was by viewing `einsum('', x, w)` as a function
$$E:\mathbb{R}^{\text{x.shape}} \rightarrow \mathbb{R}^{(\text{x.shape}) \times (\text{w.shape})} $$
$$E(X)  = Z$$

Usually einsum only maps into a small subspace of that codomain, but looking at it this way is helpful for building the jacobian. $E$ will have jacobian of shape $(\text{x.shape})\times(\text{w.shape})\times(\text{x.shape})$.   -->










<!-- We are viewing the output $Z \in \mathbb{R}^{(\text{x.shape})\times(\text{w.shape})}$. Since no dimensions are summed in this view, $Z_{ijkl}$ is always independent of $x_{mn}$ for $(mn) \neq (ij)$. So right off the bat we know all the non-zero elements of the jacobian are living on the diagonal `jacobian[i, j, :, :, i, j]`

When we multiply axes together in the original einsum e.g. `ij,jk`, this means that the output is only defined on the slice `z[:, i, j, :]` where $i = j$. 

To summarize, we are interested in the slice of the jacobian where the following conditions are met:
1. `jacobian[i, j, :, :, m, n]`, $(ij) = (mn)$, the slice where the output is defined
2. `jacobian[something]`, the slice where along the multiplied axes 

We can select this intersection with 

if you look at the input string, and the string we use to select the the diagonal of the jacobian you can start to see the pattern

On the left of `->` its going to be (input operand 1) + (input operand 2) + (input operand 1). On the right side of `->` its going to be the part of input operand 1 unique from operand 2, plus operand 2. We can code it like this

```python
def jacobian_diagonal_(ptrn):
    op1, op2 = ptrn.split('->')[0].split(',')
    start = (op1 + op2) + op1
    end = "".join([c for c in op1 if c not in op2]) + op2
    return f"{start}->{end}"
```
Using a set would be quicker, but these need to be in order. With this step 2 is done, our jacobian has the right values and it just needs to be reshaped.  -->









 <!-- Weve been viewing the jacobian in a higher dimension and in this step we collapse it into the correct dimension. Whatever axes were multiplied together in the original einsum need to be collapsed in the jacobian. For example in matrix multiplication

 ```python
 x = torch.randn(3, 4)
 w = torch.randn(4, 5)
 z = einsum('ij,jk->ik', x, w)
 ```
This clearly a function 

$$E:\mathbb{R}^{3 \times 4}\rightarrow \mathbb{R}^{3 \times 5}$$

But weve been looking at it in 

$$E:\mathbb{R}^{3 \times 4} \rightarrow \mathbb{R}^{(3 \times 4) \times (4 \times 5)}$$

So to get the *true* jacobian, not our expanded view of it, we need to collapse it along dimensions 1, 2

```python
jacobian= einsum('ijjkab->ikab', jacobian)
```
The string used here doesnt really matter as long as it sums the right directions. You could do `wizard->ward`, `abcdef->adef`....

The general pattern is whatever axes are multiplied together in the original einsum need to be summed over. We can do it like this

```python
def jacobian_sum_dims(ptrn):
    op1, op2 = ptrn.split('->')[0].split(',')
    start = op1 + op2
    end = "".join([c for c in start if start.count(c) == 1])
    end += "".join(list(set(string.ascii_lowercase))[:len(op1)])
    return f"{start}->{end}"
```

#### Step 4

Now we need to account for any reshaping/summing that happened in the original einsum. For example if we had something like `ij,jk->ki`, matrix multiplication and then transpose, our current jacobian would be wrong. Similarly if we did matrix multiplication and sum `ij,jk->`.
# Not sure
### Kronecker Product

```python
x, w = torch.randn(1, 2, 3), torch.randn(4, 5, 6)
z = einsum('abd,def->abcdef', x, w)
```
Lets initialize our jacobian

```python
jacobian = torch.zeros(*x.shape, *w.shape, *x.shape)
```
Now we look at $Z$ to figure out what partials go where. An element of $Z$ is given by

$$Z_{ijklmn} = x_{ijk}w_{lmn}$$

or in block form

$$Z_{ijk} = x_{ijk}W$$

From this we can see clearly that

$$
\frac{\partial Z_{ijk}}{\partial x_{lmn}} = 
\begin{cases}
W \ \ \ if \ \ \ (ijk) = (lmn) \\ 
0 \ \ \ else
\end{cases}
$$
Meaning the slice of the jacobian `jacobian[i, j, k, :, :, :, l, m, n]` should be equal to $W$ where $(ijk)=(lmn)$ and zero everywhere else. We can achieve this using.......einsum

```python
einsum('ijkabcijk->ijkabc', jacobian)[:] = w 
```
In certain situation einsum returns a view and we can use that to broadcast to the *diagonal* we want. 

### Matrix multiplication

```python
x, w = torch.randn(2, 3), torch.randn(3, 4)
z = einsum('ij,jk->ik', x, w)
```
Obviously $Z$ lives in $\mathbb{R}^{2 \times 4}$, but were going to look at it in $\mathbb{R}^{(2 \times 3) \times (3 \times 4)}$. An element of $Z$ is given by

$$Z_{ijkl} = 
\begin{cases}
x_{ij}w_{kl} \ \ \ if \ \ \ j = k \\
0 \ \ \ \ \ \ \ \ \ \  else
\end{cases}
$$

differentiating that is simple enough

$$\frac{\partial Z_{ijkl}}{\partial x_{mn}} = 
\begin{cases}
w_{kl} \ \ \ if \ \ \ j = k \ \ and \  \ (ij) = (lm)\\
0  \ \ \ \ \  else
\end{cases}
$$

cool lets make the jacobian. 

```python
jacobian = torch.zeros(*x.shape, *w.shape, *x.shape)
einsum('ijjkij->ijk', jacobian)[:] = w
```
The values are right but this needs to be collapsed since we are viewing it in a higher dimension

```python
jacobian = einsum('ijklmn->ijlm', jacobian)
```
Thats it. -->