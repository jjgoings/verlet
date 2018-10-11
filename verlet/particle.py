import numpy as np
from sympy import *
from sympy.core import sympify
from verlet.integrators import Integrator

class Particle(Integrator):
    """ class to contain information on position, momentum, and forces for a 
        single particle in 1- or 2-dimensions.
    """
    def __init__(self,r,p,mass=1.0,temp=298.15,dt=0.1,sig=5.0,numsteps=1000,
                 integrator='brownian'):
        self.r      = np.asarray(r)  
        self.p      = np.asarray(p)  
        assert np.shape(self.r) == np.shape(self.p)
        self.dim    = np.shape(self.r)[0]
        self.m      = mass
        self.z      = 0.0 #np.zeros(self.dim)
        self.integrator = integrator

        kB = 0.0019872041   # Boltzmann in kcal/(mol * K)
        self.kT = kB*temp
        self.dt = dt
        self.sig = sig 
        self.numsteps = numsteps
        self.R = [] # position time series
        self.P = [] # momentum time series
        self.Z = [] # thermostat time series
        self.updatePotential()

    def updatePotential(self,U='0'):
        # Define potential and forces
        if self.dim == 1:
            x = symbols('x')
            Uxy = sympify(U)
            self.Fx = lambdify(x,-diff(Uxy,x),modules='numpy') # x force function
            self.Uxy = lambdify(x,Uxy,modules='numpy') # x potential function
        elif self.dim == 2:
            x,y = symbols('x y')
            Uxy = sympify(U)
            self.Fx = lambdify((x,y),-diff(Uxy,x),modules='numpy')
            self.Fy = lambdify((x,y),-diff(Uxy,y),modules='numpy')
            self.Uxy = lambdify((x,y),Uxy) # x,y potential function
        #Ux = sympify(Ux)
        #Uy = sympify(Uy)
        #self.Fx = lambdify(x,-diff(Ux,x)) # x force function
        #self.Fy = lambdify(y,-diff(Uy,y)) # y force function
        #self.Ux = lambdify(x,Ux) # x potential function
        #self.Uy = lambdify(y,Uy) # y potential function

    def force(self,r):
        """ return force at given position """
        if self.dim == 1:
            return np.array([self.Fx(r[0])])
        elif self.dim == 2:
            return np.array([self.Fx(r[0],r[1]),self.Fy(r[0],r[1])])
        else:
            sys.exit('Error in force dimension: 1D or 2D only!')

    def run(self):
        try:
            from tqdm import tqdm
            time = tqdm(range(self.numsteps))
        except ImportError:
            time = range(self.numsteps)

        for step in time:
            self.R.append(self.r)
            self.P.append(self.p)
            self.Z.append(self.z)
            if self.integrator.lower() == 'brownian':
                self.brownian()
            elif self.integrator.lower() == 'nosehoover':
                self.nosehoover()
       
    @property
    def xs(self):
        return np.asarray(self.R)[:,0]

    @property
    def ys(self):
        return np.asarray(self.R)[:,1]
       
    @property
    def pxs(self):
        return np.asarray(self.P)[:,0]

    @property
    def pys(self):
        return np.asarray(self.P)[:,1]

if __name__ == '__main__':        
    simulation = Particle(r=[1.0,1.0],p=[0.0,0.0])
    simulation.updatePotential(U='2*x**2 + 0.1*y**2')
    simulation.run()
   
