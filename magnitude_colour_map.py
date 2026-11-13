import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

'''
Colour map of local magnetic field magnitude at Earth's surface
Uses B = -∇Ψ(r,theta, phi) = -(r^ ∂_r + (1/r)theta^ ∂_theta + (1/rsin(theta))phi^ ∂_phi)Ψ(r,theta, phi)

Ψ(r,theta,phi) is the scalar magnetic potential (units A):

Ψ(r,theta,phi) = Re(∑_(l=1)^N ((Re/r)^(l+1)(∑_(m=0)^l(P_l^m (cos(theta))(g_l^m cos(m*phi) +h_l^m sin(m*phi))))))

where P_l^m (cos(theta)) are the associated Schmidt quasi-normalised Legendre polynomials and g_l^m and h_l^m are
Gauss coefficients (values taken from https://igpppublic.ucsd.edu/~cathy/Classes/SIO229/geomag_2009_3.pdf) for
the IGRF-2000 model

Normalisation is carried out in line with the IGRF-2000 convention to agree with these coefficients,
so P_l^m (cos(theta)) is given by:

P_l^m (cos(theta)) = sqrt((e(l-m)!/(l+m)!)) * sin^m 0 * (d/d(cos0))^m (P_l (cos(theta))

where e=1 for m=0, e=2 for m>0
the Cordon-Shortley phase term (-1)^m is not included as this is incorporated into the coefficients

Works for expansion up to N=4
'''

'''Radius of earth R_E = 6.371e6 m '''
R_E=6371000


'''List of Gauss coefficients g'''
g1_0=-29615
g1_1=-1728
g2_0=-2267
g2_1=3072
g2_2=1672
g3_0=1341
g3_1=-2290
g3_2=1253
g3_3=715
g4_0=935
g4_1=787
g4_2=251
g4_3=-405
g4_4=110

g_list=[[g1_0, g1_1], [g2_0, g2_1, g2_2],[g3_0, g3_1, g3_2, g3_3],[g4_0, g4_1, g4_2, g4_3, g4_4]]

'''List of Gauss coefficients h'''
h1_0=0
h1_1=5186
h2_0=0
h2_1=-2478
h2_2=-458
h3_0=0
h3_1=-227
h3_2=296
h3_3=-492
h4_0=0
h4_1=272
h4_2=-232
h4_3=119
h4_4=-304

h_list=[[h1_0, h1_1], [h2_0, h2_1, h2_2],[h3_0, h3_1, h3_2, h3_3],[h4_0, h4_1, h4_2, h4_3, h4_4]]



def spherical_mag(t,p):
    '''
    Takes r, theta, phi and returns magnitude of mag. field at corresponding point. Bi determined
    in each of r^, theta^, and phi^ directions (ie. each Bi multiplies e^_i)
    '''
    
    P1_0=np.cos(t)
    P1_1=np.sin(t)
    
    P2_0=(3*(np.cos(t))**2 -1)/2
    P2_1=(3*np.sin(t)*np.cos(t))/np.sqrt(3)
    P2_2=(3*(np.sin(t))**2)/np.sqrt(12)
    
    P3_0=(5*(np.cos(t))**3 -3*np.cos(t))/2
    P3_1=(15*(np.cos(t)**2)*(np.sin(t)) -3*(np.sin(t)))/(2*np.sqrt(6))
    P3_2=(np.sqrt(15)*np.cos(t)*(np.sin(t))**2)/2
    P3_3=(np.sqrt(10)*(np.sin(t))**3)/4
    
    P4_0=(35*(np.cos(t))**4 - 30*(np.cos(t))**2 +3)/8
    P4_1=(-15*np.cos(t)*np.sin(t) +35*np.sin(t)*(np.cos(t)**3))/(2*np.sqrt(10))
    P4_2=(np.sqrt(5)*(35*(np.sin(t)**2)*(np.cos(t))**2 -5*(np.sin(t)**2)))/20
    P4_3=(105*(np.cos(t)*(np.sin(t)))**3)/np.sqrt(2520)
    P4_4=(105*(np.sin(t))**4)/np.sqrt(20160)
    

    P_list=[[P1_0, P1_1], [P2_0, P2_1, P2_2], [P3_0, P3_1, P3_2, P3_3], [P4_0, P4_1, P4_2, P4_3, P4_4]]

    '''Derivatives with respect to theta of Schmidt quasi-normalised associated Legendre polynomials'''
    dP1_0=-np.sin(t)
    dP1_1=np.cos(t)
    
    dP2_0=-3*np.cos(t)*np.sin(t)
    dP2_1=(3*((np.cos(t))**2 - np.sin(t)**2))/np.sqrt(3)
    dP2_2=(6*np.sin(t)*np.cos(t))/np.sqrt(12)
    
    dP3_0=(-15*(np.sin(t))*(np.cos(t))**2 +3*np.sin(t))/2
    dP3_1=(15*(-2*np.cos(t)*(np.sin(t))**2 +(np.cos(t))**3) -3*np.cos(t))/(2*np.sqrt(6))
    dP3_2=(np.sqrt(15)*(2*np.sin(t)*(np.cos(t))**2 -(np.sin(t))**3))/2
    dP3_3=(3*np.sqrt(10)*np.cos(t)*(np.sin(t))**2)/4
    
    dP4_0=(-35*np.sin(t)*(np.cos(t))**3 +15*np.sin(t)*np.cos(t))/2
    dP4_1=(-15*(np.cos(t)**2 - (np.sin(t)**2)) +35*(np.cos(t))**4 -3*(np.sin(t)**2)*(np.cos(t)**2))/(2*np.sqrt(10))
    dP4_2=(np.sqrt(5)*(70*np.sin(t)*(np.cos(t))**3 -70*np.cos(t)*(np.sin(t)**3) -10*np.sin(t)*np.cos(t)))/20
    dP4_3=(105*(3*((np.cos(t)**2)*np.sin(t))**2 -(np.sin(t))**4))/np.sqrt(2520)
    dP4_4=(420*(np.sin(t))**3)*(np.cos(t))/np.sqrt(20160)

    dP_list=[[dP1_0,dP1_1],[dP2_0, dP2_1, dP2_2],[dP3_0, dP3_1, dP3_2, dP3_3],[dP4_0, dP4_1, dP4_2, dP4_3, dP4_4]]

    '''Presets for function'''
    l_sum1=0
    m_sum1=0
    l_sum2=0
    m_sum2=0
    l_sum3=0
    m_sum3=0

    '''Input N'''
    N=2


    for i in range(1,N+1,1):
        for j in range(0,i+1,1):
            m_sum1 = m_sum1 + (P_list[i-1][j])*((g_list[i-1][j])*np.cos(j*p) + (h_list[i-1][j])*np.sin(j*p))
        l_sum1 = l_sum1 - (i+1)*m_sum1*((R_E/R_E)**(i+2))
        m_sum1=0

    Br = -l_sum1


    for i in range(1,N+1,1):
        for j in range(0,i+1,1):
            m_sum2 = m_sum2 + (dP_list[i-1][j])*((g_list[i-1][j])*np.cos(j*p) + (h_list[i-1][j])*np.sin(j*p))
        l_sum2 = l_sum2 + m_sum2*((R_E/R_E)**(i+2))
        m_sum2=0 
    
    Bt = -l_sum2


    for i in range(1,N+1,1):
        for j in range(0,i+1,1):
            m_sum3 = m_sum3 + (P_list[i-1][j])*(-j*(g_list[i-1][j])*np.sin(j*p) + j*(h_list[i-1][j])*np.cos(j*p))
        l_sum3 = l_sum3 + m_sum3*((R_E/R_E)**(i+2))
        m_sum3=0
    
    Bp = -l_sum3/(np.sin(t))

    mag = np.sqrt(Br**2 + Bt**2 + Bp**2)
    
    return mag





'''Setting ranges for r, theta, and phi, avoiding division by zero in above function'''
t_range=np.linspace(0.1, np.pi-0.1, 100)
p_range=np.linspace(0, 2*np.pi, 100)

T, P = np.meshgrid(t_range, p_range)

'''Transformations of coordinates R, T, P to cartesian'''
X=R_E*np.sin(T)*np.cos(P)
Y=R_E*np.sin(T)*np.sin(P)
Z=R_E*np.cos(T)

F = spherical_mag(T,P)
print(F)

norm = Normalize(vmin=F.min(), vmax=F.max())
F_norm = norm(F)

colors = cm.coolwarm(F_norm)

fig = plt.figure()
ax = fig.add_subplot( 1, 1, 1, projection='3d')

surface = ax.plot_surface(
X,Y,Z,
facecolors=colors,
rstride=1,
cstride=1,
linewidth=0,
antialiased=True,
shade=False
)


mappable = cm.ScalarMappable(norm=norm, cmap='coolwarm')
mappable.set_array(F)

fig.colorbar(
mappable,
ax=ax,
shrink=0.65,
label='Local field strength (nT)'
)


ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Magnitude of Earth\'s magnetic field at surface')

plt.tight_layout()
plt.show()

