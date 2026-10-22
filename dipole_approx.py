import numpy as np
import matplotlib.pyplot as plt

ax = plt.figure().add_subplot(projection='3d')


'''R_E 6317km, k=mu_0/4pi~e-7, scale not properly adjusted for earth but will correct later'''

k=10**-7

'''Southern hemisphere'''
x_range = np.linspace(-631.0, 631.0, 8)
y_range = np.linspace(-631.0, 631.0, 8)
z_range = np.linspace(-631.0, 00.0, 8)
x, y, z = np.meshgrid(x_range, y_range, z_range)


'''Northern hemisphere'''
a_range = np.linspace(-631.0, 631.0, 8)
b_range = np.linspace(-631.0, 631.0, 8)
c_range = np.linspace(0.0, 631.0, 8)
a, b, c = np.meshgrid(a_range, b_range, c_range)


'''Using B_i=k*(3*r_i*m -m_i*|r|**2)/|r|**5, component form of dipole magnetic field'''
Bx1=k*(3*x*(z*10))/(np.sqrt(x**2 +y**2 +z**2))**5
By1=k*(3*y*(z*10))/(np.sqrt(x**2 +y**2 +z**2))**5
Bz1=k*(3*z*(z*10)-10*(np.sqrt(x**2 +y**2 +z**2))**2)/(np.sqrt(x**2 +y**2 +z**2))**5

Bx2=k*(3*a*(c*10))/(np.sqrt(a**2 +b**2 +c**2))**5
By2=k*(3*b*(c*10))/(np.sqrt(a**2 +b**2 +c**2))**5
Bz2=k*(3*c*(c*10)-10*(np.sqrt(a**2 +b**2 +c**2))**2)/(np.sqrt(a**2 +b**2 +c**2))**5


ax.quiver(x, y, z, Bx1, By1, Bz1, length=50, normalize=True, color='b')
ax.quiver(a, b, c, Bx2, By2, Bz2, length=50, normalize=True, color='r')
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Dipole approx. of Earth\'s magnetic field')
plt.grid()
plt.show()
