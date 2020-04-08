# 
"""
@author: xxw
@time: 2020/3/28 11:48
@desc: 
"""

class Mullayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forword(self,x,y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self,dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy


apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = Mullayer()
mul_tax_layer = Mullayer()

apple_price = mul_apple_layer.forword(apple,apple_num)
price = mul_tax_layer.forword(apple_price,tax)

print(price)


dprices = 1
dapple_price,dtax = mul_tax_layer.backward(dprices)
dapple,dapple_num = mul_apple_layer.backward(dapple_price)

price(dapple,dapple_num,dtax)