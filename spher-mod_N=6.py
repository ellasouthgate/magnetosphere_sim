import numpy as np
import matplotlib.pyplot as plt

'''
Model of magnetic field using spherical harmonics
Uses B = -∇Ψ(r,theta, phi) = -(r^ ∂_r + (1/r)theta^ ∂_theta + (1/rsin(theta))phi^ ∂_phi)Ψ(r,theta, phi)

Ψ(r,theta,phi) is the scalar magnetic potential (units A):

Ψ(r,theta,phi) = Re(∑_(l=1)^N ((Re/r)^(l+1)(∑_(m=0)^l(P_l^m (cos(theta))(g_l^m cos(m*phi) +h_l^m sin(m*phi))))))

where P_l^m (cos(theta)) are the associated Schmidt quasi-normalised Legendre polynomials and g_l^m and h_l^m are
Gauss coefficients (values taken from https://igpppublic.ucsd.edu/~cathy/Classes/SIO229/geomag_2009_3.pdf)
Works for expansion up to N=6
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
g5_0=-217
g5_1=351
g5_2=222
g5_3=-131
g5_4=-169
g5_5=-12
g6_0=72
g6_1=68
g6_2=74
g6_3=-161
g6_4=-5
g6_5=17
g6_6=-91

g_list=[[g1_0, g1_1], [g2_0, g2_1, g2_2],[g3_0, g3_1, g3_2, g3_3],[g4_0, g4_1, g4_2, g4_3, g4_4], [g5_0, g5_1, g5_2, g5_3, g5_4, g5_5], [g6_0, g6_1, g6_2, g6_3, g6_4, g6_5, g6_6]]

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
h5_0=0
h5_1=44
h5_2=172
h5_3=-134
h5_4=-40
h5_5=107
h6_0=0
h6_1=-17
h6_2=64
h6_3=65
h6_4=-61
h6_5=1
h6_6=44

h_list=[[h1_0, h1_1], [h2_0, h2_1, h2_2],[h3_0, h3_1, h3_2, h3_3],[h4_0, h4_1, h4_2, h4_3, h4_4],[h5_0, h5_1, h5_2, h5_3, h5_4, h5_5], [h6_0, h6_1, h6_2, h6_3, h6_4, h6_5, h6_6]]

    
def spherical_mag(r,t,p):
    '''
    Takes r, theta, phi and returns mag field at corresponding point in r^, theta^,
    and phi^ directions (each Bi multiplies e^_i)
    '''

    '''List of Schmidt quasi-normalised associated Legendre polynomials
    Up to P5_0 from https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html ,
    the rest calculated using generating function:

    Pn_m(x)=(1/n!2^n)(1-x^2)^(m/2) (d/dx)^(m+n) (x^2 -1)^n

    (https://academic.oup.com/gji/article/160/2/487/659348)
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
    P5_0=((np.cos(t))*(63*(np.cos(t))**4 -70*(np.cos(t))**2 +15))/8
    P5_1=(np.sin(t)*(315*(np.cos(t))**4 -210*(np.cos(t))**2 +15))/8
    P5_2=((np.sin(t))**2)*(315*(np.cos(t))**3 -105*np.cos(t))/2
    P5_3=((np.sin(t))**3)*(945*(np.cos(t))**2 -105)/2
    P5_4=945*(np.cos(t))*(np.sin(t))**4
    P5_5=945*(np.sin(t))**5
    P6_0=(231*(np.cos(t))**6 -315*(np.cos(t))**4 +105*(np.cos(t))**2 -5)/16
    P6_1=(np.sin(t))*(693*(np.cos(t))**5 -630*(np.cos(t))**3 +210*np.cos(t))/8
    P6_2=((np.sin(t))**2)*(3465*(np.cos(t))**4 -1890*(np.cos(t))**2 +105)/8
    P6_3=((np.sin(t))**3)*(3465*(np.cos(t))**3 -945*np.cos(t))/2
    P6_4=((np.sin(t))**4)*(10395*(np.cos(t))**2 -945)/2
    P6_5=((np.sin(t))**5)*(10395*np.cos(t))
    P6_6=10395*(np.sin(t))**6

    

    P_list=[[P1_0, P1_1], [P2_0, P2_1, P2_2], [P3_0, P3_1, P3_2, P3_3], [P4_0, P4_1, P4_2, P4_3, P4_4], [P5_0, P5_1, P5_2, P5_3, P5_4, P5_5], [P6_0, P6_1, P6_2, P6_3, P6_4, P6_5, P6_6]]

    '''Derivatives with respect to theta of Schmidt quasi-normalised associated Legendre polynomials'''
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
    dP5_0=(-(315*(np.cos(t))**4 -210*(np.cos(t))**2 +15)*np.sin(t))/8
    dP5_1=(-15*np.cos(t))*((84*(np.cos(t))**2 -28)*(np.sin(t))**2 -21*(np.cos(t))**4 +14*(np.cos(t))**2 -1)/8
    dP5_2=(-np.sin(t))*((945*(np.cos(t))**2 -105)*(np.sin(t))**2 +210*(np.cos(t))**2 -630*(np.cos(t))**4)/2
    dP5_3=(-315*np.cos(t)*(np.sin(t))**2)*(6*(np.sin(t))**2 -9*(np.cos(t))**2 +1)/2
    dP5_4=3780*((np.cos(t))**2)*(np.sin(t))**3 -945*(np.sin(t))**5
    dP5_5=4725*np.cos(t)*(np.sin(t))**4
    dP6_0=(-np.sin(t))*(693*(np.cos(t))**5 -630*(np.cos(t))**3 +105*np.cos(t))/8
    dP6_1=-((3465*(np.cos(t))**4 -1890*(np.cos(t))**2 +210)*(np.sin(t))**2 -693*(np.cos(t))**6 +630*(np.cos(t))**4 -210*(np.cos(t))**2)/8
    dP6_2=-105*np.cos(t)*np.sin(t)*((66*(np.cos(t))**2 -18)*(np.sin(t))**2 -33*(np.cos(t))**4 +18*(np.cos(t))**2 -1)/4
    dP6_3=(-(np.sin(t))**2)*((10395*(np.cos(t))**2 -945)*(np.sin(t))**2 +2835*(np.cos(t))**2 -10395*(np.cos(T))**4)/2
    dP6_4=(-945*np.cos(t)*(np.sin(t))**3)*(11*(np.sin(t))**2 -22*(np.cos(t))**2 +2)
    dP6_5=51975*((np.cos(t))**2)*(np.sin(t))**4 -10395*(np.sin(t))**6
    dP6_6=62370*np.cos(t)*(np.sin(t))**5

    dP_list=[[dP1_0,dP1_1],[dP2_0, dP2_1, dP2_2],[dP3_0, dP3_1, dP3_2, dP3_3],[dP4_0, dP4_1, dP4_2, dP4_3, dP4_4],[dP5_0, dP5_1, dP5_2, dP5_3, dP5_4, dP5_4], [dP6_0, dP6_1, dP6_2, dP6_3, dP6_4, dP6_5, dP6_6]]

    '''Presets for function'''
    l_sum1=0
    m_sum1=0
    l_sum2=0
    m_sum2=0
    l_sum3=0
    m_sum3=0

    '''Input N'''
    N=6

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
ax.set_title('Spherical harmonics model of geomagnetosphere')

max_range= np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() /2.0
mid_x = (X.max()+X.min())*0.5
mid_y = (Y.max()+Y.min())*0.5
mid_z = (Z.max()+Z.min())*0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()


