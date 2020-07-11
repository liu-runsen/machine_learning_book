'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''



import sympy
import math
x = sympy.symbols("x")
print(sympy.integrate(x**2,(x,1,2)))
# 常量
r = sympy.symbols('r',positive=True)
# 圆的面积
circle_area = 2 * sympy.integrate(math.sqrt(r**2-x**2),(x,-r,r))
print(circle_area)
# 球的体积
print(sympy.integrate(circle_area,(x,-r,r)))


from scipy import integrate
f = lambda y, x: x*y**2
val1, err1 = integrate.dblquad(f,  # 函数
                     0,  # x下界0
                     2,  # x上界2
                     lambda x: 0,  # y下界0
                     lambda x: 1)  # y上界1
print('二重积分结果：', val1)


from scipy import integrate
# 三重积分
val2, err2 = integrate.tplquad(lambda z, y, x: x*y*z,  # 函数
                     0,  # x下界0
                     3,  # x上界3
                     lambda x: 0,  # y下界0
                     lambda x: 2,  # y上界2
                     lambda x, y: 0,  # z下界0
                     lambda x, y: 1)  # z上界1
print('三重积分结果：', val2)



