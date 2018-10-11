from verlet.particle import Particle

# set up 2D particle, with initital position r=[x,y] and momentum p = [P_x, P_y]
simulation = Particle(r=[1.0,-1.0],p=[0.0,0.0],numsteps=5)

# Add analytic potential for particle dynamics
simulation.updatePotential(U='x**2 + y**2')

simulation.run()

print('Particle x-position time series:\n',simulation.xs)
print('Particle y-position time series:\n',simulation.ys)



