# Developer Documentation for the Trajectory Simulation.py
## Aims of the Simulation 
Initially I had hoped to create the simulation such that we could see a stream of solar wind particles, and show where they create the aurora at the intersection of the Earth and Sun's dipoles. With both the time and coding experience this was not feasible. Doing so would have also caused the problem with scale as the Larmor radii are in a range of 1-5m and would not be visible next to the Earth's radius . 

Due to these two aspects the simulation focuses on the Larmor radius of a singular particle with changing field strengths, pitch angles, and initial energy of the electron in the near-Earth field of 200km. 

This simulation neglects the effects of the Sun's dipole, Collisions of particles within the Solar wind, and Relativistic effects. 

## Constants 
### Physical Constants
k = mu0/4pi, this is the magnetic field constant that turns the dipole moment into the field (1x10^-7 Tm/A)

q = Elementary charge (1.602176634 x10^-19 C)

m_e = electron mass (9.1093837015 x 10^ -31 kg)

earth_radius = mean radius of the Earth (6.371x10^6 m)

dipole_moment = magnitude of the Earths magnetic dipole moment (8 x10^22 Am^2)

tilt = the angle between the z axis and the dipole axis (23.4 degrees)

### Numerical Settings 
dt = Time step of the integrator (1x10^-9s =1ns)

steps_per_frame = Euler steps per screen refresh ( 50 steps)

sample_every = every nth step is then stored and plotted ( this had to be done as we did not have enough compute power to store and plot every step) 

trail_samples = maximum number of stored data on the trail length 

### Display Settings 
panel_width = the width of the control panel 310 pixels

axis_ticks = (-8,-4,0,4,8)

path_colour, arrow_colour, marker_colour can be changed using different hex ocde values to change the display colors. 

### Changing Variables
energy_kev: (options 1,3,5 keV)

pitch_deg: (options 30,45,75 degrees)

storm_mode changes the field strength from the initial quiet condition (average field strength approx 58 micro Tesla. This alters the B_z by -0.6x10^-6

## Field Model
The initial Field model starts with the z-axis pointing through the pole. This is then change so that the dipole moment vector is tilted using 'tilt' in the x-z plane

m = ( -M sin(tilt), 0, -M cos(tilt) )      M = 8.0e22 A m^2

dipole_field uses the following dipole formula 

B(r) = k * ( 3 r (m . r) - m |r|^2 ) / |r|^5

get_magnetic_field then calls on this value for the varying conditions. 

## Mechanisms 




