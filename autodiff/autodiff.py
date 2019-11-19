import numpy as np 

class AutoDiff():
    """
    Implementation of Forward Auto Differentiation using the Chain Rule

    Attributes:

    val: value of custom function evaluated at x
    der: derivative of custom function evaluated at x

    """

    def __init__(self, val, der = 1): 
        """ 
        Initializes AutoDiff object w/ inputs val and der. Der set to 1 initially.

        """
        self.val = val
        self.der = der

    """binary operators""" 
    def __add__(self, other): 
        """
        inputs: self: AD object, other: AD object or scalar 
        returns AD object of addition of two inputs in the form self + other

        """
        try: # assumes two AutoDiff objects
            new_val = self.val + other.val
            new_der = self.der + other.der
        except AttributeError: # assumes other is scalar
            new_val = self.val + other
            new_der = self.der
        return AutoDiff(new_val, new_der)

    def __str__(self):
        """ 
        returns string value of the function
        
        """
        return "AutoDiff({},{})".format(self.val, self.der)

    def __repr__(self):
        """ 
        returns string value of the function
        
        """
        return "AutoDiff({},{})".format(self.val, self.der)

    def __radd__(self, other):
        """
        inputs: self: AD object, other: AD object or scalar 
        returns AD object of addition of two inputs in the form other + self

        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        inputs: self: AD object, other: AD object or scalar 
        returns AD object of subtraction of two inputs in the form self - other

        """
        try: # assumes two AutoDiff objects
            new_val = self.val - other.val
            new_der = self.der - other.der
        except AttributeError: # assumes other is scalar
            new_val = self.val - other
            new_der = self.der
        return AutoDiff(new_val, new_der)

    def __rsub__(self, other):
        """
        inputs: self: AD object, other: AD object or scalar 
        returns AD object of subtraction of two inputs in the form other - self

        """
        return - self.__sub__(other)

    def __mul__(self, other):
        """
        inputs: self: AD object, other: AD object or scalar 
        returns AD object of multiplication of two inputs in the form self * other

        """
        try: # assumes two AutoDiff objects
            new_val = self.val * other.val
            new_der = self.der * other.val + other.der * self.val
        except AttributeError: # assumes other is scalar
            new_val = self.val * other
            new_der = self.der * other
        return AutoDiff(new_val, new_der)

    def __rmul__(self, other):
        """
        inputs: self: AD object, other: AD object or scalar 
        returns AD object of multiplication of two inputs in the form other * self

        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        inputs: self: AD object, other: AD object or scalar 
        returns AD object of true division of two inputs in the form self / other

        """
        try: # assumes self & other are autodiff objects
            new_val = self.val / other.val
            new_der = (self.der * other.val - other.der * self.val) / (other.val ** 2)
        except AttributeError: # assumes self is autodiff object and other is scalar
            new_val = self.val / other
            new_der = self.der / other
        return AutoDiff(new_val, new_der)


    def __rtruediv__(self, other):
        """
        inputs: self: AD object, other: AD object or scalar 
        returns AD object of true division of two inputs in the form other / self

        """
        try: # assumes self & other are autodiff objects
            new_val = other.val / self.val
            new_der = (other.der * self.val - self.der * other.val) / (self.val ** 2)
        except AttributeError: # assumes self is autodiff object and other is scalar
            new_val = other / self.val
            new_der = other * ( - self.der / (self.val ** 2))
        return AutoDiff(new_val, new_der)

    def __pow__(self, other):
        """
        inputs: self: AD object, other: AD object or scalar 
        returns AD object of power function of the form self ** other

        """
        try: # assumes two AutoDiff objects
            new_val = self.val ** other.val
            new_der = other.val * (self.val ** (other.val - 1)) * self.der
        except AttributeError: # assumes other is scalar
            new_val = self.val ** other
            new_der = other * (self.val ** (other - 1)) * self.der
        return AutoDiff(new_val, new_der)

    """ unary operators """
    def __neg__(self): 
        """
        inputs: self: AD object
        returns AD object of negative function of the form - self 

        """
        new_val = - self.val
        new_der = - self.der
        return AutoDiff(new_val, new_der)

    """ elementary functions """
    def sin(self):
        """
        inputs: self: AD object
        returns AD object of sine function of the form sine(self)

        """
        new_val = np.sin(self.val)
        new_der = self.der * np.cos(self.val)
        return AutoDiff(new_val, new_der)

    def cos(self):
        """
        inputs: self: AD object
        returns AD object of cosine function of the form cosine(self)

        """
        new_val = np.cos(self.val)
        new_der = self.der * ( - np.sin(self.val))
        return AutoDiff(new_val, new_der)

    def tan(self):
        """
        inputs: self: AD object
        returns AD object of tangent function of the form tangent(self)

        """
        new_val = np.tan(self.val)
        new_der = self.der / ( np.cos(self.val) ** 2 )
        return AutoDiff(new_val, new_der)

    def ln(self):
        """
        inputs: self: AD object
        returns AD object of natural log (ln) function of the form ln(self)

        """
        new_val = np.log(self.val)
        new_der = self.der * (1 / self.val)
        return AutoDiff(new_val, new_der)

    def log(self, base):
        """
        inputs: self: AD object, base: scalar
        returns AD object of log function with base = base of the form log(self, base)

        """
        new_val = np.log(self.val) / np.log(base)
        new_der = self.der * (1 / (self.val * np.log(base)))
        return AutoDiff(new_val, new_der)
    
    def exp(self):
        """
        inputs: self: AD object
        returns AD object of exponential function of the form exp(self)

        """
        new_val = np.exp(self.val)
        new_der = self.der * np.exp(self.val)
        return AutoDiff(new_val, new_der)

    def sqrt(self):
        """
        inputs: self: AD object
        returns AD object of sqrt function of the form sqrt(self)

        """ 
        new_val = self.val ** (1/2)
        new_der = self.der * ((1/2) * (self.val ** (- 1/2)))
        return AutoDiff(new_val, new_der)