# Developer Documentation for the Trajectory Simulation
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
### Initial Velocity (calculate_velocity_matrix)
For each energy and pitch angle the code forms the initial velocity and stores it in a nested dictionary v_p(energy)(pitch)

1.  The field direction at the starting point is found. The sign is chosen so that the electron's parallel velocity points in the direction of travel (down towards the atmosphere).

2.  A vector perpendicular to the field is made with a cross product against the x-axis.
  
3.  The speed comes from the non-relativistic kinetic energy:

speed = sqrt(2 E q / m), with E converted from keV to joules.

4.  The velocity is split by the pitch angle alpha, measured from the field:

v = speed * cos(alpha) * b_hat  +  speed * sin(alpha) * perpendicular

### Integrating the motion (euler_step)
The force acting on the electron takes into account the Earths magnetic field only. 

a = (charge / mass) * ( v x B )          charge = -q 

Each step splits the velocity into parrallel and perpendicular components to B then updates the perpendicular velocity v_perp_new = v_perp + a * dt. 

The new perpendicular velocity is then scaled back to the original length and the position is moved acording to the new velocity which is a combination of the parallel and new perpendicular velocity. 

The perpendicular velocity is rescaled as initial versions of the code showed the Larmor radius increasing as time went on at an increasing rate. As the magnetic force does no work the speed remains constant. By doing this some angular error remains which is not taken into account in this version of the simulation. 


### Guiding center
get_guiding_center is the center of the circle the electron is orbiting at a particular point. 

R_gc = r + ( m / (charge * |B|) ) * ( v x b_hat )

The guiding center trail is drawn as the electron continues its path. 

### Larmor radius
r_L = m * v_perp / ( q * |B| )          unit: kg (m/s) / (C T) = m

### Current and induced field
While I ran into some issues incorporating the current induced by the particle into the code I still chose to calculate the current and induced field to show within the data window alonside the plot of the decreasing larmor radius as it adds context. 

The following provides the acceleration and equivalant current and the field that would be produced. 

I = q^2 |B| / (2 pi m)                  unit: A  (charge times gyrofrequency)

B_induced = (4 pi k) * I / (2 r_L)      unit: T  (center of a circular loop)

### View update_3d_view
update_3d_view subtracts the lastest guiding center position from the stored points so that the electron remains visible on the screen within the plot. 

## Program Structure 
__init__ Sets defaults (5 keV, 75 degrees, quiet field), builds the interface, resets and starts the loop

_build_interface configures the window grid and calls the two builders below

_build_3d_view creates the 3d axes, line objects, marker, title text and legend

_build_control_panel creates the buttons, reset/play/pause the telemetry box and the plot of the larmor radius. _segmented, _make_button, _section contribute to this. 

_style_selected, _update_button_highlights highlight the active input choices 

_set_storm, _set_energy, _set_pitch store the new choice and call _reset_path

_play_sim, _pause_sim switches the anim_running on and off allowing you to pause the simulation 

_animate_loop is the main loop which uses _update_telemetry, _update_radius_plot to update the telemetry and the larmor plot, and _apply_view, _update_field_arrow to set the aix and redraw the field arrow. 


## Limitations and Issues 
1.  No mirror force. The parallel velocity is held constant within each step and the perpendicular speed is held constant by the rescale. In a real converging field the magnetic moment is conserved, so v_perp would grow and v_parallel would shrink, eventually reflecting the electron. Here the Larmor radius falls only because |B| rises.
  
2.  By using non-relativistic speeds creates slight errors.
  
3.  When rescaling perpendicular velocity the phase error is not addresed.






