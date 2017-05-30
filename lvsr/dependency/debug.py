import theano

def debugTheanoVar(name, var, print_str=False):
    attrs = ["shape"]
    if print_str:
        attrs += ["__str__"]
    return theano.printing.Print(name, attrs=tuple(attrs))(var)