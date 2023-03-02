from torch.autograd import Function


class GradReversal(Function):
    @staticmethod
    def forward(ctx, x, l):
        ctx.l = l
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return - grad_output * ctx.l, None


def grad_reverse(inputs, l):
    return GradReversal.apply(inputs, l)
