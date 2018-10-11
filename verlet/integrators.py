import numpy as np

class Integrator(object):
    """
       Class to set up and run dynamics of constant temperature single particle 

       r:    position (nm)
       p:    momenta (amu * nm / ps)
       z:    thermostat coordinate (kcal/amu)^(1/2); init as zero 
       mass: mass (amu = g/mol)
       temp: temperature (K)
       dt:   timestep (ps)
       sig:  friction (1/ps)
    """
    def nosehoover(self):
        dt  = self.dt
        kT  = self.kT
        sig = self.sig 
        N   = self.dim
        r   = self.r 
        p   = self.p 
        z   = self.z 
        F   = self.force 
        m   = self.m
        h   = np.random.randn()
        mu = 0.5 

        p = p + 0.5*dt*F(r)
        r = r + 0.5*dt*p
        p = np.exp(-0.5*dt*z)*p

        denom = 1 + (dt*sig**2)/(4*mu)
        z = (1 - (dt*sig**2)/(4*mu))*z + (dt/mu)*(p*p/m - N*kT) + sig*np.sqrt(dt)*h
        z = z/denom

        p = np.exp(-0.5*dt*z)*p
        r = r + 0.5*dt*p
        p = p + 0.5*dt*F(r)

        # update state 
        self.r = r 
        self.p = p 
        self.z = z

    def brownian(self):
        # From stochastic integrator in LAMMPS (last pg of PDF below)
        #https://www2.ph.ed.ac.uk/~dmarendu/MVP/MVP03.pdf
        dt  = self.dt
        kT  = self.kT
        sig = self.sig
        N   = self.dim
        r   = self.r
        p   = self.p
        F   = self.force
        m   = self.m
        h   = np.random.randn()

        p = p - 0.5*dt*(-F(r) + sig*p/m) + \
             np.sqrt(dt*kT*sig/m)*h
        r = r + dt*p/m
        p = p - 0.5*dt*(-F(r)/m + sig*p/m) + \
             np.sqrt(dt*kT*sig/m)*h

        # update state
        self.r = r
        self.p = p

