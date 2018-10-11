import numpy as np
from sympy import *
from sympy.core import sympify

class NoseHoover(object):
    """
       Class to set up and run dynamics of constant temperature single particle 
       according to the Nose-Hoover-Langevin thermostat in one or two dimensions

       r:    position (nm)
       p:    momenta (amu * nm / ps)
       z:    thermostat coordinate (kcal/amu)^(1/2); init as zero 
       mass: mass (amu = g/mol)
       temp: temperature (K)
       dt:   timestep (ps)
       gm:   friction (1/ps)
    """
    def integrate(self):
        dt = self.dt
        kT = self.kT
        gm = self.gm 
        N  = self.dim
        r  = self.r 
        p  = self.p 
        z  = self.z 
        F  = self.force 
        m  = self.m
        h = np.random.randn()

        # integration adapted from Juan M. Bello-Rivas 
        # (https://scicomp.stackexchange.com/users/15/juan-m-bello-rivas), 
        # How to integrate numerically Nosé Hoover equation?, 
        # URL (version: 2016-10-14): https://scicomp.stackexchange.com/q/25202

        kinetic = lambda p: p*p/(2*m)  # kinetic energy (kcal/mol)

        # begin integration 
        z = np.exp(-0.5*gm*dt)*z + np.sqrt(kT/(m*(1. - np.exp(-gm*dt))))*h

        p = p + 0.5*dt*F(r)

        p = p*np.exp(-0.25*z*dt)
        z = z + 0.5*dt*(2*kinetic(p) - N*kT)/m
        p = p*np.exp(-0.25*z*dt)

        r = r + dt*p/m

        p = p*np.exp(-0.25*z*dt)
        z = z + 0.5*dt*(2*kinetic(p) - N*kT)/m
        p = p*np.exp(-0.25*z*dt)

        p = p + 0.5*dt*F(r)

        z = np.exp(-0.5*gm*dt)*z + np.sqrt(kT/(m*(1. - np.exp(-gm*dt))))*h
    
        # update state 
        self.r = r 
        self.p = p 
        self.z = z

