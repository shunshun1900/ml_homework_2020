import numpy as np

y = np.array([[1,2,3],[2,2,2]])
x = np.array([2,3])
z = np.zeros((2,3))
r = np.random.rand(2)
r1=np.random.randn(3)
r2=np.random.randint(10)
print(r,r1,r2)
k = range(3)
for i in k:
    print(i)

z[0]=1
print(z)
#print(np.dot(y.T,x))

#print(np.math.factorial(x))

