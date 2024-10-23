import numpy as np

# class for 3DOF ekf for nonholonomic in body frame

class EKF():

    # constructor to initialize a Dynamics object

    def __init__(self):

        """

        Initialize the EKF \n

        State:

        -------

        \t px: x position \n

        \t py: y position \n

        \t qw: real component of orientation \n

        \t qz: z componenet of oreintation \n

        \t v: x linear velocity in body frame \n

        \t w: z angular velocity in body frame \n

        Inputs:

        -------

        

        Returns:

        -------

        """
        
        # state px, py, qw, qz, v, w

        self.xH = np.zeros(6)

        self.xH[2] = 1.0 # make it identity for rotation

        

        # covariance

        self.P = np.diag([10**-4,10**-4,10**-6,10**-6,10**-2,10**-2])

        # process variance

        self.Q = np.diag([10**-2,10**-2])


        # measurement variance

        self.R = np.diag([10**-4,10**-4,10**-6,10**-6])

        # measurement jacobian

        self.H = np.zeros((4,6))

        self.H[0:4,0:4] = np.eye(4)
        
        self.initialized = False


    def reset(self,px=0.0,py=0.0,qw=1.0,qz=0.0):

        """

        Reset the state and covariance \n

        Inputs:

        -------

        \t px: x position \n

        \t py: y position \n

        \t qw: real rotation element \n

        \t qz: z rotation element \n

        

        Returns:

        -------

        """

        self.xH = np.zeros(6)

        self.xH[0] = px

        self.xH[1] = py

        self.xH[2] = qw

        self.xH[3] = qz

        self.P = 0.1*np.eye(6)



    def predict(self,dt):

        """

        Predict the state and covariance forward in time \n

        Inputs:

        -------

        \t dt: time step \n

        

        Returns:

        -------

        """
        if not self.initialized:
            self.xH = np.array([-1, -1, 1, 0,0])
            return


        # predict the state, normalize the orientation before using it

        pxH = self.xH[0]

        pyH = self.xH[1]

        qwH = self.xH[2]

        qzH = self.xH[3]

        qN = qwH**2 + qzH**2

        qwH /= qN

        qzH /= qN

        vH  = self.xH[4]

        wH  = self.xH[5]

        

        # position

        # pD = q*v*q^-1, v is x axis

        pxHD = (qwH**2-qzH**2)*vH

        pyHD = 2.0*qwH*qzH*vH

        

        # orientation

        # qD = 0.5*B*w, w is z axis

        qwHD = -0.5*qzH*wH

        qzHD = 0.5*qwH*wH

        # velocity

        # veclocity is piecewise constant

        vHD = 0.0

        wHD = 0.0


        # predict the state

        xHD = np.array([pxHD,pyHD,qwHD,qzHD,vHD,wHD])

        self.xH += dt*xHD



        # next predict covariance

        # x(k+1) = x(k) + dt*xHD

        pxDqw = 2.0*qwH*vH*dt

        pxDqz = -2.0*qzH*vH*dt

        pxDv  = (qwH**2-qzH**2)*dt

        pyDqw = 2.0*qzH*vH*dt

        pyDqz = 2.0*qwH*vH*dt

        pyDv  = 2.0*qwH*qzH*dt

        qwDqz = -0.5*wH*dt

        qwDw  = -0.5*qzH*dt

        qzDqw = 0.5*wH*dt

        qzDw  = 0.5*qwH*dt

        # px, py, qw, qz, v, w

        F = np.eye(6) + np.array([[0,0,pxDqw,pxDqz,pxDv,   0],

                                  [0,0,pyDqw,pyDqz,pyDv,   0],

                                  [0,0,    0,qwDqz,   0,qwDw],

                                  [0,0,qzDqw,    0,   0,qzDw],

                                  [0,0,    0,    0,   0,   0],

                                  [0,0,    0,    0,   0,   0]])



        # position

        # pD = q*v*q^-1, v is x axis

        pxHD = (qwH**2-qzH**2)*vH

        pyHD = 2.0*qwH*qzH*vH

        

        # orientation

        # qD = 0.5*B*w, w is z axis

        qwHD = -0.5*qzH*wH

        qzHD = 0.5*qwH*wH



        # velocity

        # veclocity is piecewise constant

        vHD = 0.0

        wHD = 0.0



        # predict the state

        xHD = np.array([pxHD,pyHD,qwHD,qzHD,vHD,wHD])

        self.xH += dt*xHD



        # px, py, qw, qz, v, w

        L = np.array([[pxDv,0],

                      [pyDv,0],

                      [0,qwDw],

                      [0,qzDw],

                      [dt,0],

                      [0,dt]])


        

        # predict the covariance

        self.P = (F@self.P@F.T + L@self.Q@L.T)


    def learn(self,px=0.0,py=0.0,qw=1.0,qz=0.0,dt=1.0/30.0):

        """

        Predict and update the state and covariance using measurement \n

        Inputs:

        -------

        \t px: x position \n

        \t py: y position \n

        \t qw: real rotation element \n

        \t qz: z rotation element \n


        Returns:

        -------

        """
        if not self.initialized:
            self.reset(px, py, qw, qz)
            self.initialized = True


        # first call predict to update the state and covariance

        self.predict(dt)


        # now perform measurement update

        # get the innovation covariance

        S = self.H@self.P@self.H.T + self.R


        # calculate the gain

        K = self.P@self.H.T@np.linalg.inv(S)


        # update the state and covariance

        qN = qw**2 + qz**2

        if qN > 0.001:

            qw /= qN

            qz /= qN

        else:

            print("THERES SOMETHING HAPPENING HERE BUT WHAT IT IS AINT EXACTLY CLEAR")

        self.xH += K@(np.array([px,py,qw,qz])-self.xH[0:4])

        self.P = (np.eye(6)-K@self.H)@self.P
