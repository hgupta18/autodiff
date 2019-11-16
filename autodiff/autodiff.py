class AutoDiff():

    def __init__(self, val, der = 1):
        self.val = val
        self.der = der

    ## binary operators
    def __add__(self, other):
        try:
            new_val = self.val + other.val
            new_der = self.der + other.der
        except AttributeError:
            new_val = self.val + other
            new_der = self.der
        return AutoDiff(new_val, new_der)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        try: # assumes two AutoDiff objects
            new_val = self.val - other.val
            new_der = self.der - other.der
        except AttributeError: # assumes other is scalar
            new_val = self.val - other
            new_der = self.der
        return AutoDiff(new_val, new_der)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        try:
            new_val = self.val*other.val
            new_der = self.der*other.val + other.der*self.val
        except AttributeError:
            new_val = self.val*other
            new_der = self.der*other
        return AutoDiff(new_val, new_der)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other): # form self /other
        try: # assumes self & other are autodiff objects
            new_val = self.val / other.val
            new_der = (self.der * other.val - other.der * self.val) / (other.val ** 2)
        except AttributeError: # assumes self is autodiff object and other is scalar
            new_val = self.val / other
            new_der = self.der / other
        return AutoDiff(new_val, new_der)


    def __rtruediv__(self, other): # form other /self
        try: # assumes self & other are autodiff objects
            new_val = other.val / self.val
            new_der = (other.der * self.val - self.der * other.val) / (self.val ** 2)
        except AttributeError: # assumes self is autodiff object and other is scalar
            new_val = other / self.val
            new_der = other * ( - self.der / (self.val ** 2))
        return AutoDiff(new_val, new_der)

    def __pow__(self, other):
        try:
            new_val = self.val*other.val
            new_der = self.der*other.val + other.der*self.val
        except AttributeError:
            new_val = self.val*other
            new_der = self.der*other
        return AutoDiff(new_val, new_der)

    ## unary operators
    def __neg__(self):
        new_val = -self.val
        new_der = -self.der
        return AutoDiff(new_val, new_der)

    ## elementary functions
    def sin(x):
        pass

    def cos(x):
        pass

    def tan(x):
        pass

    def ln(x):
        pass

    def log10(x):
        pass

    def log2(x):
        pass

    def log(x, base=10):
        pass
    
    def exp(x):
        pass

    def sqrt(x):
        pass



if __name__ == "__main__":
    # defining parameters
    a = 2.0
    x = AutoDiff(a)
    # b = 4.0
    # y = AutoDiff(b)
    alpha = 10.0
    beta = 5.0

    # demo 1
    f = 3*x/(2*x+4)
    print(f.val, f.der)

    # demo 2
    f = beta / (alpha * x)
    print(f.val, f.der)

    # demo 3
    f = beta - alpha * x
    print(f.val, f.der)

    # demo 4
    f = beta + x * alpha
    print(f.val, f.der)



