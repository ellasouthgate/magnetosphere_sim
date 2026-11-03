import numpy as np
import matplotlib.pyplot as plt

'''
Model of magnetic field using spherical harmonics
Uses B = -∇Ψ(r,theta, phi) = -(r^ ∂_r + (1/r)theta^ ∂_theta + (1/rsin(theta))phi^ ∂_phi)Ψ(r,theta, phi)

Ψ(r,theta,phi) is the scalar magnetic potential (units A):

Ψ(r,theta,phi) = Re(∑_(l=1)^N ((Re/r)^(l+1)(∑_(m=0)^l(P_l^m (cos(theta))(g_l^m cos(m*phi) +h_l^m sin(m*phi))))))

where P_l^m (cos(theta)) are the associated Schmidt quasi-normalised Legendre polynomials and g_l^m and h_l^m are
Gauss coefficients (values taken from https://igpppublic.ucsd.edu/~cathy/Classes/SIO229/geomag_2009_3.pdf)
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

    
def spherical_mag(r,t,p):
    '''
    Takes r, theta, phi and returns mag field at corresponding point in r^, theta^,
    and phi^ directions (each Bi multiplies e^_i)
    '''

    '''List of Schmidt quasi-normalised associated Legendre polynomials
    (from https://en.wikipedia.org/wiki/Associated_Legendre_polynomials)
    '''
    P1_0=np.cos(t)
    P1_1=-np.sin(t)
    P2_0=(3*(np.cos(t))**2 -1)/2
    P2_1=-3*np.sin(t)*np.cos(t)
    P2_2=3*(np.sin(t))**2
    P3_0=(5*(np.cos(t))**3 -3*np.cos(t))/2
    P3_1=3*(1-5*(np.cos(t))**2)*(np.sin(t))/2
    P3_2=15*np.cos(t)*(np.sin(t))**2
    P3_3=-15*(np.sin(t))**3
    P4_0=(35*(np.cos(t))**4 - 30*(np.cos(t))**2 +3)/8
    P4_1=(-5*(7*(np.cos(t))**3 -3*np.cos(t))*np.sin(t))/2
    P4_2=(15*(7*(np.cos(t))**2 -1)*(np.sin(t))**2)/2
    P4_3=-105*(np.cos(t)*(np.sin(t)))**3
    P4_4=105*(np.sin(t))**4

    

    P_list=[[P1_0, P1_1], [P2_0, P2_1, P2_2], [P3_0, P3_1, P3_2, P3_3], [P4_0, P4_1, P4_2, P4_3, P4_4]]

    '''Derivatives of Schmidt quasi-normalised associated Legendre polynomials'''
    dP1_0=-np.sin(t)
    dP1_1=-np.cos(t)
    dP2_0=-3*np.cos(t)*np.sin(t)
    dP2_1=-3*np.cos(2*t)
    dP2_2=6*np.sin(t)*np.cos(t)
    dP3_0=(-15*(np.sin(t))*(np.cos(t))**2 +3*np.sin(t))/2
    dP3_1=(3*((1-5*(np.cos(t))**2)*np.cos(t) -10*np.sin(t)*np.cos(t)))/2
    dP3_2=15*(2*np.sin(t)*(np.cos(t))**2 -(np.sin(t))**3)
    dP3_3=-45*np.cos(t)*(np.sin(t))**2
    dP4_0=(-140*np.sin(t)*(np.cos(t))**3 +60*np.sin(t)*np.cos(t))/8
    dP4_1=(-5*(7*(np.cos(t))**4 -3*(np.cos(t))**2 -21*(np.cos(t)*np.sin(t))**2 +3*(np.sin(t))**2))/2
    dP4_2=(15*(14*np.sin(t)*(np.cos(t))**3 -2*np.sin(t)*np.cos(t) -14*np.cos(t)*(np.sin(t))**3))/2
    dP4_3=-105*(3*((np.cos(t))*np.sin(t))**2 -(np.sin(t))**4)
    dP4_4=(420*(np.sin(t))**3)*(np.cos(t))

    dP_list=[[dP1_0,dP1_1],[dP2_0, dP2_1, dP2_2],[dP3_0, dP3_1, dP3_2, dP3_3],[dP4_0, dP4_1, dP4_2, dP4_3, dP4_4]]

    '''Presets for function'''
    l_sum1=0
    m_sum1=0
    l_sum2=0
    m_sum2=0
    l_sum3=0
    m_sum3=0

    '''Input N'''
    N=4

    for i in range(1,N+1,1):
        for j in range(0,i+1,1):
            m_sum1 = m_sum1 + (P_list[i-1][j])*((g_list[i-1][j])*np.cos(j*p) + (h_list[i-1][j])*np.sin(j*p))
        l_sum1 = l_sum1 - (i+1)*m_sum1*((R_E/r)**(i+2))
        m_sum1=0

    Br = -l_sum1


    for i in range(1,N+1,1):
        for j in range(0,i+1,1):
            m_sum2 = m_sum2 + (dP_list[i-1][j])*((g_list[i-1][j])*np.cos(j*p) + (h_list[i-1][j])*np.sin(j*p))
        l_sum2 = l_sum2 + m_sum2*((R_E/r)**(i+2))
        m_sum2=0 
    
    Bt = -l_sum2


    for i in range(1,N+1,1):
        for j in range(0,i+1,1):
            m_sum3 = m_sum3 + (P_list[i-1][j])*(-j*(g_list[i-1][j])*np.sin(j*p) + j*(h_list[i-1][j])*np.cos(j*p))
        l_sum3 = l_sum3 + m_sum3*((R_E/r)**(i+2))
        m_sum3=0
    
    Bp = -l_sum3/(np.sin(t))
    
    return Br, Bt, Bp

'''Setting ranges for r, theta, and phi, avoiding division by zero in above function'''
r_range=np.linspace(0.1, 1000000.0, 12)
t_range=np.linspace(0.1, np.pi-0.1, 15)
p_range=np.linspace(0, 2*np.pi, 15, endpoint=False)

R, T, P = np.meshgrid(r_range, t_range, p_range)

'''Magneic field function evaluated at R, T, P'''
BR, BT, BP = spherical_mag(R, T, P)

'''Transformations of coordinates R, T, P to cartesian'''
X=R*np.sin(T)*np.cos(P)
Y=R*np.sin(T)*np.sin(P)
Z=R*np.cos(T)

'''Transformations of magnetic field vector components to cartesian coordinates'''
U=BR*np.sin(T)*np.cos(P) + BT*np.cos(T)*np.cos(P) - BP*np.sin(P)
V=BR*np.sin(T)*np.sin(P) + BT*np.cos(T)*np.sin(P) + BP*np.cos(P)
W=BR*np.cos(T) - BT*np.sin(T)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
quiver = ax.quiver(X, Y, Z, U, V, W, length=50000, normalize=True, cmap='viridis') 


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('spherical harmonics model')

max_range= np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() /2.0
mid_x = (X.max()+X.min())*0.5
mid_y = (Y.max()+Y.min())*0.5
mid_z = (Z.max()+Z.min())*0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()


