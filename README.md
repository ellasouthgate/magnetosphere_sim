# Developer Documentation for the Trajectory Simulation.py
## Aims of the Simulation 
Initially I had hoped to create the simulation such that we could see a stream of solar wind particles, and show where they create the aurora at the intersection of the Earth and Sun's dipoles. With both the time and coding experience this was not feasible. Doing so would have also caused the problem with scale as the Larmor radii are in a range of 1-5m and would not be visible next to the Earth's radius . 

Due to these two aspects the simulation focuses on the Larmor radius of a singular particle with changing field strengths, pitch angles, and initial energy of the electron in the near-Earth field of 200km. 

This simulation neglects the effects of the Sun's dipole, Collisions of particles within the Solar wind, and Relativistic effects. 

## Constants 
k = mu0/4pi, this is the magnetic field constant that turns the dipole moment into the field (1x10^-7 Tm/A)
q = Elementary charge (1.602176634 x10^-19 C)
m_e = electron mass (9.1093837015 x 10^ -31 kg)
earth_radius = mean radius of the Earth (6.371x10^6 m)
dipole_moment = magnitude of the Earths magnetic dipole moment (8 x10^22 Am^2)
tilt = the angle between the z axis and the dipole axis (23.4 degrees)

