from ekf import EKF
import matplotlib.pyplot as plt
from math import sin
from math import pi
import numpy as np

px = 0.0
py = 0.0
qw = 1.0
qz = 0.0
v = 1.0
w = 0.0

xs = []
ys = []
xHs = []
yHs = []
times = []

ekfE = EKF()

data = np.genfromtxt('ekf_comp.csv', delimiter=',')

for ii in range(600):
    if ii > 0:
        dt = data[ii,0]-data[ii-1,0]
    else:
        dt = 1./30.

    px = data[ii,1]
    py = data[ii,2]

    qw = data[ii,4]
    qz = data[ii,3]

    qN = qw**2+qz**2
    qw /= qN
    qz /= qN

    print("time: %.4f)"%data[ii,0])
    if data[ii,0]<8.5 or data[ii,0]>10.2:
        ekfE.learn(px,py,qw,qz,dt)
        print("learn")
    else:
        ekfE.predict(dt)
        print("predict")
    print("px "+str(round(px,4))+" py "+str(round(py,4))+" qw "+str(round(qw,4))+" qz "+str(round(qz,4))+" v "+str(round(v,4))+" w "+str(round(w,4)))
    print("pxH "+str(round(ekfE.xH[0].item(),4))+" py "+str(round(ekfE.xH[1].item(),4))+" qw "+str(round(ekfE.xH[2].item(),4))+" qz "+str(round(ekfE.xH[3].item(),4))+" v "+str(round(ekfE.xH[4].item(),4))+" w "+str(round(ekfE.xH[5].item(),4))+"\n")
    xs.append(px)
    ys.append(py)
    xHs.append(ekfE.xH[0].item())
    yHs.append(ekfE.xH[1].item())
    times.append(data[ii,0])

plt.figure()
plt.plot(times,xs)
plt.plot(times,xHs)
plt.plot(times,ys)
plt.plot(times,yHs)
plt.legend(["x","xH","y","yH"])
plt.show()
