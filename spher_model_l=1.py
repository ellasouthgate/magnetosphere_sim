import numpy as np
import matplotlib.pyplot as plt

'''
Model of magnetic field using spherical harmonics, expanded to l=1 (equivalent to dipole model)
Uses B = -∇Ψ(r,theta, phi) = -(r^ ∂_r + (1/r)theta^ ∂_theta + (1/rsin(theta))phi^ ∂_phi)Ψ

Ψ(r,theta,phi) is the scalar magnetic potential (units A):

Ψ(r,theta,phi) = Re(∑_(l=1)^N ((Re/r)^(l+1)(∑_(m=0)^l(P_l^m (cos(theta))(g_l^m cos(m*phi) +h_l^m sin(m*phi))))))

where P_l^m (cos(theta)) are the associated Schmidt quasi-normalised Legendre polynomials and g_l^m and h_l^m are
Gauss coefficients (values taken from https://igpppublic.ucsd.edu/~cathy/Classes/SIO229/geomag_2009_3.pdf)
'''

'''Using R_E = 631 but can correct later'''
re=631

def spherical_mag(r,t,p):
    '''
    Takes r, theta, phi and returns mag field at corresponding point in r^, theta^,
    and phi^ directions (each Bi multiplies e^_i)
    '''
    Br = -2*((re/r)**3)*(29615*np.cos(t) - 1728*np.sin(t)*np.cos(p) + 5186*np.sin(p)*np.sin(t))
    Bt = ((re/r)**3)*(-29615*np.sin(t) - 1728*np.cos(t)*np.cos(p) + 5186*np.cos(t)*np.sin(p))
    Bp = ((re/r)**3)*(1728*np.sin(p) + 5186**np.cos(p))
    return Br, Bt, Bp

'''Setting ranges for r, theta, and phi, avoiding division by zero in above function'''
r_range=np.linspace(0.1, 631.0, 8)
t_range=np.linspace(0.1, np.pi-0.1, 10)
p_range=np.linspace(0, 2*np.pi, 10, endpoint=False)

R, T, P = np.meshgrid(r_range, t_range, p_range)

'''Function evaluated at R, T, P'''
BR, BT, BP = spherical_mag(R, T, P)
    
'''Transformations of coordinates to cartesian'''
X=R*np.sin(T)*np.cos(P)
Y=R*np.sin(T)*np.sin(P)
Z=R*np.cos(T)

'''Transformations of vector components to cartesian'''
U=BR*np.sin(T)*np.cos(P) + BT*np.cos(T)*np.cos(P) - BP*np.sin(P)
V=BR*np.sin(T)*np.sin(P) + BT*np.cos(T)*np.sin(P) + BP*np.cos(P)
W=BR*np.cos(T) - BT*np.sin(T)


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
quiver = ax.quiver(X, Y, Z, U, V, W, length=50, normalize=True, cmap='viridis') 


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('l=1 spherical harmonics model')

max_range= np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() /2.0
mid_x = (X.max()+X.min())*0.5
mid_y = (Y.max()+Y.min())*0.5
mid_z = (Z.max()+Z.min())*0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()


