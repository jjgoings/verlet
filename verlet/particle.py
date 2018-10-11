import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.core import sympify
from verlet.integrators import NoseHoover

class Particle(NoseHoover):
    """ class to contain information on position, momentum, and forces for a 
        single particle in 1- or 2-dimensions.
    """
    def __init__(self,r,p,mass=1.0,temp=298.15,dt=0.1,gm=10.0,numsteps=1000):
        self.r      = np.asarray(r)  
        self.p      = np.asarray(p)  
        assert np.shape(self.r) == np.shape(self.p)
        self.dim    = np.shape(self.r)[0]
        self.m      = mass
        self.z      = np.zeros(self.dim)

        kB = 0.0019872041   # Boltzmann in kcal/(mol * K)
        self.kT = kB*temp
        self.dt = dt
        self.gm = gm
        self.numsteps = numsteps
        self.R = [] # position time series
        self.P = [] # momentum time series
        self.Z = [] # thermostat time series
        self.updatePotential()

    def updatePotential(self,Ux='0',Uy='0'):
        # Define potential; if 1D y and Uy are ignored later on
        x,y = symbols('x y')
        Ux = sympify(Ux)
        Uy = sympify(Uy)
        self.Fx = lambdify(x,-diff(Ux,x)) # x force function
        self.Fy = lambdify(y,-diff(Uy,y)) # y force function
        self.Ux = lambdify(x,Ux) # x potential function
        self.Uy = lambdify(y,Uy) # y potential function

    def force(self,r):
        """ return force at given position """
        if self.dim == 1:
            return np.array([self.Fx(r[0])])
        elif self.dim == 2:
            return np.array([self.Fx(r[0]),self.Fy(r[1])])
        else:
            sys.exit('Error in force dimension: 1D or 2D only!')

    def potential(self,r):
        """ return potential at given position """
        if self.dim == 1:
            return np.array([self.Ux(r[0])])
        elif self.dim == 2:
            return np.array([self.Ux(r[0]),self.Uy(r[1])])
        else:
            sys.exit('Error in potential dimension: 1D or 2D only!')

    def run(self):
        for step in range(self.numsteps):
            self.R.append(self.r)
            self.P.append(self.p)
            self.Z.append(self.z)
            self.integrate()

    @property
    def xs(self):
        return np.asarray(self.R)[:,0]

    @property
    def ys(self):
        return np.asarray(self.R)[:,1]
       
    @property
    def pys(self):
        return np.asarray(self.P)[:,0]

    @property
    def pys(self):
        return np.asarray(self.P)[:,1]

if __name__ == '__main__':        
    simulation = Particle(r=[1.0,1.0],p=[0.0,0.0])
    simulation.updatePotential(Ux='2*x**2',Uy='0.1*y**2')
    simulation.run()
    plt.plot(simulation.xs)
    plt.plot(simulation.ys)
    plt.show()
   
