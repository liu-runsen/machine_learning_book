'''
@Author： Runsen
@微信公众号： 润森笔记
@博客： https://blog.csdn.net/weixin_44510615
@Date： 2020/7/5
'''


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.figure(dpi=100)

#K=1
plt.plot(np.linspace(0,15,100),stats.chi2.pdf(np.linspace(0,15,100),df=1))
plt.fill_between(np.linspace(0,15,100),stats.chi2.pdf(np.linspace(0,15,100),df=1),alpha=0.15)

#K=3
plt.plot(np.linspace(0,15,100),stats.chi2.pdf(np.linspace(0,15,100),df=3))
plt.fill_between(np.linspace(0,15,100),stats.chi2.pdf(np.linspace(0,15,100),df=3),alpha=0.15)

#K=6
plt.plot(np.linspace(0,15,100),stats.chi2.pdf(np.linspace(0,15,100),df=6))
plt.fill_between(np.linspace(0,15,100),stats.chi2.pdf(np.linspace(0,15,100),df=6),alpha=0.15)

#图例
plt.text(x=0.5,y=0.7,s="$ k=1$",rotation=-65,alpha=.75,weight="bold",color="#008fd5")
plt.text(x=1.5,y=.35,s="$ k=3$",alpha=.75,weight="bold",color="#fc4f30")
plt.text(x=5,y=.2,s="$ k=6$",alpha=.75,weight="bold",color="#e5ae38")

#坐标轴
plt.tick_params(axis="both",which="major",labelsize=18)
plt.axhline(y=0,color="black",linewidth=1.3,alpha=.7)
plt.show()


from scipy.stats import chi2_contingency
table = [[28,172],[60,140]]
chi2,pval,dof,expected = chi2_contingency(table)
print("卡方检验chi2:",chi2)
print("显著性值:",pval)
print("理论数列联表如下\n",expected)
if pval < 0.05:
   print("拒绝原假设，实验组和对照组存在差异")
else:
   print("支持原假设")
