from ekf import EKF
import matplotlib.pyplot as plt
from math import sin
from math import pi

px = 0.0
py = 0.0
qw = 1.0
qz = 0.0
v = 1.0
w = 0.0
dt = 0.1

xs = []
ys = []
xHs = []
yHs = []
times = []

ekfE = EKF()
for ii in range(100):
    # position
    # pD = q*v*q^-1, v is x axis
    w = sin(2.0*pi*(1.0/5.0)*(ii*dt))
    
    pxD = (qw**2-qz**2)*v
    pyD = 2.0*qw*qz*v
    
    # orientation
    # qD = 0.5*B*w, w is z axis
    qwD = -0.5*qz*w
    qzD = 0.5*qw*w

    px += dt*pxD
    py += dt*pyD

    qw += dt*qwD
    qz += dt*qzD

    qN = qw**2+qz**2
    qw /= qN
    qz /= qN

    print("time "+str(round(ii*dt,4)))
    if ii*dt<1.0 or ii*dt>6.0:
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
    times.append(ii*dt)

plt.figure()
plt.plot(times,xs, "-", color="blue")
plt.plot(times,xHs, ".", color="blue")
plt.plot(times,ys, "-", color="red")
plt.plot(times,yHs, ".", color="red")
plt.legend(["x","xH","y","yH"])
plt.show()