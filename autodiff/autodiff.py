import numpy as np 

class AutoDiff():
    """
    Implementation of Forward Auto Differentiation using the Chain Rule

    Attributes:

    val: value of custom function evaluated at x
    der: derivative of custom function evaluated at x

    """

    def __init__(self, values, der = [1]): 
        """ 
        Initializes AutoDiff object w/ inputs val and der. Der set to 1 initially.

        """
        # assert isinstance(val, int) or isinstance(val, float), "Please provide value as int or float"
        # assert isinstance(der, int) or isinstance(der, float), "Please provide derivative as int or float"
        self.val = np.array(values) # set to np array
        self.der = np.array(der) # set to np array
        

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


    """ Comparison operators """
    def __eq__(self, other):
        '''
        inputs: self: AD object, other: AD object or scalar
        returns boolean, True if value of self is equal to other else False
        '''
        try: # assumes two autodiff objects
            return self.val == other.val
        except AttributeError: # assumes other is scalar
            return self.val == other


    def __ne__(self, other):
        '''
        inputs: self: AD object, other: AD object or scalar
        returns boolean, True if value of self is not equal to other else False
        '''
        try: # assumes two autodiff objects
            return self.val != other.val
        except AttributeError: # assumes other is scalar
            return self.val != other


    def __gt__(self, other):
        '''
        inputs: self: AD object, other: AD object or scalar
        returns boolean, True if value of self is greater than or equal to other else False
        '''
        try: # assumes two autodiff objects
            return self.val > other.val
        except AttributeError: # assumes other is scalar
            return self.val > other

    def __ge__(self, other):
        '''
        inputs: self: AD object, other: AD object or scalar
        returns boolean, True if value of self is greater than other else False
        '''
        try: # assumes two autodiff objects
            return self.val >= other.val
        except AttributeError: # assumes other is scalar
            return self.val >= other

    def __lt__(self, other):
        '''
        inputs: self: AD object, other: AD object or scalar
        returns boolean, True if value of self is less than or equal to other else False
        '''
        try: # assumes two autodiff objects
            return self.val < other.val
        except AttributeError: # assumes other is scalar
            return self.val < other

    def __le__(self, other):
        '''
        inputs: self: AD object, other: AD object or scalar
        returns boolean, True if value of self is less than other else False
        '''
        try: # assumes two autodiff objects
            return self.val <= other.val
        except AttributeError: # assumes other is scalar
            return self.val <= other


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

    """ inverse trig functions """
    def arcsine(self):
        """
        inputs: self: AD object
        returns AD object of arcsine function of the form arcsine(self)

        """
        new_val = np.arcsin(self.val)
        new_der = self.der / ((1 - self.val ** 2) ** (1/2))
        return AutoDiff(new_val, new_der)

    def arccosine(self):
        """
        inputs: self: AD object
        returns AD object of arccosine function of the form arccosine(self)

        """
        new_val = np.arccos(self.val)
        new_der = - self.der / ((1 - self.val ** 2) ** (1/2))
        return AutoDiff(new_val, new_der)

    def arctangent(self):
        """
        inputs: self: AD object
        returns AD object of arctangent function of the form arctangent(self)

        """
        new_val = np.arctan(self.val)
        new_der = self.der / (1 + self.val ** 2)
        return AutoDiff(new_val, new_der)

    """ hyperbolic functions """
    def sinh(self):
        """
        inputs: self: AD object
        returns AD object of sinh function of the form sinh(self)

        """
        new_val = np.sinh(self.val)
        new_der = self.der * np.cosh(self.val)
        return AutoDiff(new_val, new_der)

    def cosh(self):
        """
        inputs: self: AD object
        returns AD object of cosh function of the form cosh(self)

        """
        new_val = np.cosh(self.val)
        new_der = self.der * np.sinh(self.val)
        return AutoDiff(new_val, new_der)

    def tanh(self):
        """
        inputs: self: AD object
        returns AD object of tanh function of the form tanh(self)

        """
        new_val = np.tanh(self.val)
        new_der = self.der * 1 / (np.cosh(self.val) ** 2)
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

    def expm(self, base):
        """
        inputs: self: AD object
        returns AD object of exponential function of the form exp(self)

        """
        new_val = base ** self.val
        new_der = self.der * ( base ** self.val ) * np.log(base)
        return AutoDiff(new_val, new_der)

    def logistic(self):
        """
        inputs: self: AD object
        returns AD object of logistic function of the form logistic(self)

        """
        new_val = 1 / ( 1 + np.exp(- self.val))
        new_der = np.exp(self.val) / ((1 + np.exp(self.val)) ** 2)
        return AutoDiff(new_val, new_der)


    def sqrt(self):
        """
        inputs: self: AD object
        returns AD object of sqrt function of the form sqrt(self)

        """ 
        new_val = self.val ** (1/2)
        new_der = self.der * ((1/2) * (self.val ** (- 1/2)))
        return AutoDiff(new_val, new_der)

# if __name__ == "__main__":
    #Demo
    # AD1 = AutoDiff([[2,3],[4,5]], [2,6])
    # AD2 = 2 ** AD1
    # print(AD1)
    # print(AD2)


    # AD1 = AutoDiff(1)
    # AD2 = AutoDiff(3)
    # AD3 = AD1 + AD2
    # AD4 = AD1 + 4
    # AD5 = AutoDiff([AD1, AD2])

    # print(AD1)
    # print(AD2)
    # print(AD3)
    # print(AD4)
    # # print(AD5)

    # x = AutoDiff([1], [1, 0, 0])
    # y = AutoDiff([2], [0, 1, 0])
    # z = AutoDiff([3], [0, 0, 1])
    # print('x: ',format(x))
    # print('y: ',format(y))
    # print('z: ',format(z))

    # f = x + y + z
    # print(f)

    # g = 2 * x + x * y
    # print(x)
    # print(y)
    # print(g)
    # print(g.val)
    # print(g.der)

    # z = AutoDiff(3, [1, 0]) + AutoDiff(4, [0, 1])
    # print(z)

    # z = AutoDiff(3, [1, 0]) - AutoDiff(4, [0, 1])
    # print(z)

    # x = AutoDiff(3.0, [1, 0])
    # y = AutoDiff(2, [0, 1])
    # z = x * y
    # print(z)

    # x = AutoDiff(3.0, [1, 0, 0])
    # y = AutoDiff(1.0, [0, 1, 0])
    # w = AutoDiff(2.0, [0, 0, 1])
    # z = x + y ** 2 + x * w
    # print(z)

    # x = AutoDiff(3.0, [1, 0, 0])
    # y = AutoDiff(1.0, [0, 1, 0])
    # w = AutoDiff(2.0, [0, 0, 1])
    # z = (x + y ** 2 + x * w)/2
    # print(z)
    # print(z.val)
    # print(z.der)

    # x = AutoDiff(3.0)
    # a = AutoDiff([ 1., x, x, 4.])
    # z = 3 / a
    # print(z.val) #not working

    # x = AutoDiff(3.0, [1, 0, 0])
    # y = AutoDiff(1.0, [0, 1, 0])
    # z = AutoDiff(2.0, [0, 0, 1])
    # f = 2 * x + y - 1 + z ** 2
    # z = f.__pow__(2)
    # print(z)
    # print(z.val)
    # print(z.der)



    #x = AutoDiff(1, [1 , 0]) 
    #y = AutoDiff(2, [0 , 1])
    #f1 = 2 * x + 3 * y
    #f2 = AutoDiff.exp(x) + y ** 3
    #f3 = AutoDiff.cos(x)  + 3 * AutoDiff.sin(y)
    #print(f1, f2, f3)

    # x = AutoDiff(1) 
    # y = AutoDiff(2, [0 , 1])
    # f1 = AutoDiff.tanh(x)
    # print(f1)
