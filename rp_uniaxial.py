# -*- coding: utf-8 -*-
"""

Calculate polariton dispersion of isotropic-uniaxial-isotropic layer with arbitrarily-oriented optic axis
J. Lekner, Optical Properties of a uniaxial layer, Pure Appl. Opt. 3 (1994) 821-837

@author: Francesco L. Ruta
Basov Infrared Spectroscopy Laboratory
Columbia University, Department of Physics

Original upload 7/11/2019
"""

import numpy as np
import csv
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.scimath import sqrt as sqrt
import matplotlib.pyplot as plt


# ------------------//------------------------- User Input Begins Here -----------------------//-------------------- #


# CSV files with dielectric function along the ordinary and extraordinary axes of the uniaxial layer, respectively.
# Organized such that the first column lists the frequencies in cm^-1, the second column lists the real part of the
# relative dielectric function, and the third column lists the imaginary part of the relative dielectric function
uniaxial_ordinary = 'pco_ab.csv'
uniaxial_extraordinary = 'pco_c.csv'
d = 100  # nm, thickness of uniaxial layer

# CSV file with dielectric function of isotropic substrate, organized same as specified above. Bottom isotropic layer
# Upper isotropic layer is vacuum/air, with dielectric constant = 1
substrate = 'si.csv'

# orientation of optic axis
theta = 0  # polar angle from z axis pointing normal to the surface, towards air
phi = 0    # azimuthal angle from x axis in plane of incidence (along p-polarization)

# frequency axis - specify the limits of the plot
wstart = 1000  # cm^-1
wstop = 7000  # cm^-1
wstep = 5     # step size

# momentum axis - specify the limits of the plot
qstart = 1  # 10^5 cm^-1
qstop = 10  # 10^5 cm^-1
qstep = 0.1  # step size

# color axis - specify the limits of the plot
cmax = 0.1
cmin = 0
cstep = 100   # number of steps

# ------------------//-------------------------- User Input Ends Here ------------------------//-------------------- #


# ------------ Conversions ------------- #

c = 3 * 10 ** 8  # m/s, speed of light
d = d * 10 ** (-9)  # m, thickness of uniaxial layer
theta = np.pi-theta  # converting to proper coordinate system

omega = np.arange(wstart, wstop, wstep)  # cm^-1
ks = omega * 0.02998 * 10 ** 12 / c  # m^-1

qs = np.arange(qstart, qstop, qstep)  # 10^5 cm^1
Ks = qs * 10 ** 5 * 100  # m^-1

# Convert polar and azimuthal angles to direction cosines
alpha = -np.sin(theta) * np.cos(phi)
gamma = np.cos(theta)
beta = -np.sin(theta) * np.sin(phi)

# ------------ Substrate --------------- #

with open(substrate, newline='') as csvfile:
    s = list(csv.reader(csvfile))

w_s = np.asfarray(np.transpose(s)[0])
e1_s = np.asfarray(np.transpose(s)[1])
e2_s = np.asfarray(np.transpose(s)[2])

f_e1_si = interpolate.interp1d(w_s, e1_s, kind='cubic')
f_e2_si = interpolate.interp1d(w_s, e2_s, kind='cubic')
es = f_e1_si(omega) + 1j * f_e2_si(omega)

# ------------ Air/Vacuum -------------- #

ea = np.ones(len(es))  # air/vacuum just ones

# ---------- Uniaxial layer ------------ #

# ordinary axis
with open(uniaxial_ordinary, newline='') as csvfile:
    pco_ab = list(csv.reader(csvfile))

w_pco_ab = np.asfarray(np.transpose(pco_ab)[0])   # frequencies (cm^-1)
e1_pco_ab = np.asfarray(np.transpose(pco_ab)[1])  # real part
e2_pco_ab = np.asfarray(np.transpose(pco_ab)[2])  # imaginary part

f_e1_pco_ab = interpolate.interp1d(w_pco_ab, e1_pco_ab, kind='cubic')
f_e2_pco_ab = interpolate.interp1d(w_pco_ab, e2_pco_ab, kind='cubic')
eo = f_e1_pco_ab(omega) + 1j * f_e2_pco_ab(omega)

# extraordinary axis
with open(uniaxial_extraordinary, newline='') as csvfile:
    pco_c = list(csv.reader(csvfile))

w_pco_c = np.asfarray(np.transpose(pco_c)[0])    # frequencies (cm^-1)
e1_pco_c = np.asfarray(np.transpose(pco_c)[1])   # real part
e2_pco_c = np.asfarray(np.transpose(pco_c)[2])   # imaginary part

f_e1_pco_c = interpolate.interp1d(w_pco_c, e1_pco_c, kind='cubic')
f_e2_pco_c = interpolate.interp1d(w_pco_c, e2_pco_c, kind='cubic')
ee = f_e1_pco_c(omega) + 1j * f_e2_pco_c(omega)

# ----------- Calculation ------------- #


# Layer matrix (L=MPM^-1)
def lmat(k, K, m, alp, bet, gam):

    ko = sqrt(eo[m])*k
    qo = sqrt(ko**2-K**2)

    ed = ee[m]-eo[m]
    egam = eo[m]+gam**2*ed
    qbar = sqrt(eo[m]*(ee[m]*egam*k**2-K**2*(ee[m]-bet**2*ed))/egam**2)
    qep = qbar - alp*gam*K*ed/egam
    qem = -qbar - alp*gam*K*ed/egam

    p = np.asmatrix([[np.exp(1j*qo*d), 0, 0, 0],
                    [0, np.exp(-1j*qo*d), 0, 0],
                    [0, 0, np.exp(1j*qep*d), 0],
                    [0, 0, 0, np.exp(1j*qem*d)]])

    m = np.asmatrix([[-bet*qo, bet*qo, alp*qo**2-gam*qep*K, alp*qo**2-gam*qem*K],
                    [alp*qo-gam*K, -alp*qo-gam*K, bet*ko**2, bet*ko**2],
                    [-bet*ko**2, -bet*ko**2, (alp*qep-gam*K)*ko**2, (alp*qem-gam*K)*ko**2],
                    [(alp*qo-gam*K)*qo, (alp*qo+gam*K)*qo, bet*qep*ko**2, bet*qem*ko**2]])

    return np.matmul(m, np.matmul(p, np.linalg.inv(m)))


# Reflection coefficient of the structure
def rp(k, K, n, alp, bet, gam):

    costh1 = sqrt(1-(K/(sqrt(ea[i])*k))**2)
    costh2 = sqrt(1-(K/(sqrt(es[i])*k))**2)

    l_ = lmat(k, K, n, alp, bet, gam)
    k1 = sqrt(ea[n]) * k
    k2 = sqrt(es[n]) * k
    q1 = sqrt(ea[n] * k ** 2 - K ** 2)
    q2 = sqrt(es[n] * k ** 2 - K ** 2)

    p1 = k2*(costh1*l_[0, 0]+k1*l_[0, 2])-costh2*(costh1*l_[2, 0]+k1*l_[2, 2])
    p2 = q2*(costh1*l_[1, 0]+k1*l_[1, 2])-costh1*l_[3, 0]-k1*l_[3, 2]
    a1 = k2*(l_[0, 1]-q1*l_[0, 3])-costh2*(l_[2, 1]-q1*l_[2, 3])
    a2 = q2*(l_[1, 1]-q1*l_[1, 3])-l_[3, 1]+q1*l_[3, 3]
    b1 = k2*(costh1*l_[0, 0]-k1*l_[0, 2])-costh2*(costh1*l_[2, 0]-k1*l_[2, 2])
    b2 = q2*(costh1*l_[1, 0]-k1*l_[1, 2])-costh1*l_[3, 0]+k1*l_[3, 2]

    rpp = -(p1*a2-a1*p2)/(a1*b2-b1*a2)
    rps = -(b1*p2-p1*b2)/(a1*b2-b1*a2)

    return rpp+rps


# beta = 0 simplification (plane of incidence)
# def rp_beta0(k, K, i, gam):
#
#     ed = ee[i]-eo[i]
#     eg = eo[i] + gam**2*ed
#     qG = sqrt(eg*k**2-K**2)
#     q1 = sqrt(ea[i]*k**2-K**2)
#     q2 = sqrt(es[i] * k ** 2 - K ** 2)
#
#     Q1 = q1/ea[i]
#     Q2 = q2/es[i]
#     Q = qG/(sqrt(eo[i])*sqrt(ee[i]))
#     p1 = (Q-Q1)/(Q+Q1)
#     p2 = (Q2-Q)/(Q2+Q)
#     qbar = sqrt(eo[i])*sqrt(ee[i])*qG/eG
#     Z = np.exp(2j*qbar*d)
#
#     return -(p1+p2*Z)/(1+p1*p2*Z)


rp_arr = np.ones((len(omega), len(qs)), dtype=complex)

for k_it in ks:
    w = k_it * c / (0.02998 * 10 ** 12)
    i = int((w - wstart) / wstep)

    for K_it in Ks:
        q = K_it / (10 ** 5 * 100)
        j = int((q - qstart) / qstep)

#        rp_arr[i, j] = rp_beta0(k_it, K_it, i, gamma) #beta = 0 simplification
        rp_arr[i, j] = rp(k_it, K_it, i, alpha, beta, gamma)

# remove scars (some instability somewhere that I should fix)
# print(rp_arr[27, :]) # for example
omega = omega[np.logical_not(rp_arr[:, 0] == 1. + 0.j)]
rp_arr = rp_arr[np.logical_not(rp_arr[:, 0] == 1. + 0.j)]

# get poles
imrp_arr = np.imag(rp_arr)

# --------------- Plot -------------- #

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(121)
ax.set_aspect('auto')
X, Y = np.meshgrid(qs, omega)
mesh = plt.contourf(X, Y, imrp_arr, np.linspace(cmin, cmax, cstep))
plt.xlabel(r'$q\ (10^{5}\ cm^{-1})$')
plt.ylabel(r'$\omega\ (cm^{-1})$')
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label(r'$Im(r_p)$', rotation=270)
cbar.ax.set_yticklabels([])

# idx = int((wQCL - wstart) / wstep)
# print(qs[np.argmax(imrps[idx])])

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot([0, -alpha], [0, beta], zs=[0, 0], color='r', linewidth=1.0, marker='o')
ax2.plot([0, -alpha], [0, 0], zs=[0, -gamma], linewidth=1.0, marker='o')
ax2.plot([0, 0], [0, beta], zs=[0, -gamma], color='g', linewidth=1.0, marker='o')
ax2.plot([0, -alpha], [0, beta], zs=[0, -gamma], color='k', linewidth=7.0, marker='o', markersize=14.0)
ax2.set_xlim([-1, 1])
ax2.set_ylim([-1, 1])
ax2.set_zlim([-1, 1])
ax2.title.set_text('Optic Axis')

plt.savefig('./animation/figure.png')
plt.show(block=True)