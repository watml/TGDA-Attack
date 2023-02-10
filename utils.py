import math
import random

import numpy as np

import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def norm(tensors):
    return math.sqrt(sum([torch.sum(tensor ** 2).item() for tensor in tensors]))


def concat(tensors_one, tensors_two):
    return list(tensors_one) + list(tensors_two)


def dot(tensors_one, tensors_two):
    ret = tensors_one[0].new_zeros((1, ), requires_grad=True)

    for t1, t2 in zip(tensors_one, tensors_two):
        ret = ret + torch.sum(t1 * t2)

    return ret


def images2vectors(images):
    return images.view(images.size(0), 784)


def vectors2images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


@torch.no_grad()
def confidence(discriminator, data, generator=None):
    if generator is not None:
        data = generator(data)

    if discriminator.__class__.__name__ == "Discriminator":
        return discriminator(data).mean()
    else:
        return discriminator(data).sigmoid().mean()


@torch.no_grad()
def conjugate_gradient(_hvp, b, maxiter=None, tol=1e-30, lam=0.0, use_cache=0):
    """
    Minimize 0.5 x^T H^T H x - b^T H x, where H is symmetric
    Args:
        _hvp (function): hessian vector product, only takes a sequence of tensors as input
        b (sequence of tensors): b
        maxiter (int): number of iterations
        lam (float): regularization constant to avoid singularity of hessian. lam can be positive, zero or negative
    (Q = H^T H)
    """
    def hvp(inputs):
        with torch.enable_grad():
            outputs = _hvp(inputs)

        outputs = [xx + lam * yy for xx, yy in zip(outputs, inputs)]

        return outputs

    with torch.enable_grad():
        Hb = hvp(b)

    # zero initialization
    xxs = [hb.new_zeros(hb.size()) for hb in Hb]
    ggs = [- hb.clone().detach() for hb in Hb]
    dds = [- hb.clone().detach() for hb in Hb]

    i = 0

    while True:
        i += 1

        with torch.enable_grad():
            Qdds = hvp(hvp(dds))

        # print(dot(ggs, ggs))
        # print(norm(ggs))

        # if dot(ggs, ggs) < tol:
        if norm(ggs) < tol:
            break

        # one step steepest descent
        alpha = - dot(dds, ggs) / dot(dds, Qdds)
        xxs = [xx + alpha * dd for xx, dd in zip(xxs, dds)]

        # update gradient
        ggs = [gg + alpha * Qdd for gg, Qdd in zip(ggs, Qdds)]

        # compute the next conjugate direction
        beta = dot(ggs, Qdds) / dot(dds, Qdds)
        dds = [gg - beta * dd for gg, dd in zip(ggs, dds)]

        if maxiter is not None and i >= maxiter:
            break

    return xxs


def test_conjugate_gradient():
    """Solving A x = grads"""
    print('testing conjugate gradient:')

    def hvp(lst_tensors):
        A = torch.tensor([[2, 1], [1, 3]], dtype=torch.float, device=device)
        return [A.mm(tensor) for tensor in lst_tensors]

    grads = [torch.tensor([[3], [4]], dtype=torch.float, device=device)]
    ret = conjugate_gradient(hvp, grads, maxiter=1)
    print(ret)

    ret = conjugate_gradient(hvp, grads, maxiter=2)
    print(ret)

    ret = conjugate_gradient(hvp, grads, maxiter=2, lam=0.01)
    print(ret)

    # expect ret = [[1], [1]] after two iterations


if __name__ == "__main__":
    device = "cuda:0"
    test_conjugate_gradient()
