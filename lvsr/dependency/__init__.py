from lvsr.dependency.debug import debugTheanoVar
from blocks.algorithms import StepRule
from theano import tensor

class RemoveNans(StepRule):
    def __init__(self, scaler=1):
        self.scaler = scaler

    def compute_step(self, parameter, previous_step):
        step_sum = tensor.sum(previous_step)
        not_finite = (tensor.isnan(step_sum) +
                      tensor.isinf(step_sum))
        step = tensor.switch(
            not_finite > 0, (1 - self.scaler) * parameter, 0)
        return step, []

def get_var_path(theano_var):
    path = []
    brick = theano_var.tag.annotations[0]
    while len(brick.parents) > 0:
        path += [brick.name]
        brick = brick.parents[0]
    path += [brick.name]
    path.reverse()
    return '{}.{}'.format('/'.join(path), theano_var.name)