# -*- coding: utf-8 -*-
"""

Calculate polariton dispersion of isotropic-biaxial-isotropic layer with arbitrarily-oriented crystal axes
I. Abdulhalim, Journal of Optics A: Pure and Applied Optics 1 (1999) 646
I. Abdulhalim, Optics Communications 157 (1998) 265

@author: Francesco L. Ruta
Basov Infrared Spectroscopy Laboratory
Columbia University, Department of Physics

Original upload 7/16/2019
Modified 7/24/2019
"""

import numpy as np
import csv
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.scimath import sqrt as sqrt
import matplotlib.pyplot as plt


# ------------------//------------------------- User Input Begins Here -----------------------//-------------------- #


# CSV files with dielectric function along the crystallographic axes of the biaxial layer, respectively.
# Organized such that the first column lists the frequencies in cm^-1, the second column lists the real part of the
# relative dielectric function, and the third column lists the imaginary part of the relative dielectric function
a_axis_df = 'pco_ab.csv'
b_axis_df = 'pco_ab.csv'
c_axis_df = 'pco_c.csv'
H = 200  # nm, thickness of biaxial layer

# CSV file with dielectric function of isotropic substrate, organized same as specified above. Bottom isotropic layer
# Upper isotropic layer is vacuum/air, with dielectric constant = 1
substrate = 'si.csv'

# orientation of crystallographic frame abc relative to xyz frame (xz plane of incidence, xy plane is interface)
psi = 0     # yaw angle (azimuthal angle for c axis)
theta = 0      # pitch angle (polar angle for c axis)
phi = 0        # roll angle

# frequency axis - specify the limits of the plot
wstart = 1000  # cm^-1
wstop = 7000  # cm^-1
wstep = 10     # step size

# momentum axis - specify the limits of the plot
qstart = 0  # 10^5 cm^-1
qstop = 10  # 10^5 cm^-1
qstep = 0.05  # step size

# color axis - specify the limits of the plot
cmax = 0.1
cmin = 0
cstep = 100   # number of steps

# ------------------//-------------------------- User Input Ends Here ------------------------//-------------------- #


# ------------ Conversions ------------- #

C = 3 * 10 ** 8  # m/s, speed of light
H = H * 10 ** (-9)  # m, thickness of anisotropic layer

omega = np.arange(wstart, wstop, wstep)  # cm^-1
ks = omega * 0.02998 * 10 ** 12 / C  # m^-1
kstart = wstart * 0.02998 * 10 ** 12 / C  # m^-1
kstep = wstep * 0.02998 * 10 ** 12 / C  # m^-1

qs = np.arange(qstart, qstop, qstep)  # 10^5 cm^1
Ks = qs * 10 ** 5 * 100  # m^-1
Kstart = qstart * 10 ** 5 * 100  # m^-1
Kstep = qstep * 10 ** 5 * 100  # m^-1


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

# ---------- Anisotropic layer ------------ #

# a axis
with open(a_axis_df, newline='') as csvfile:
    pco_a = list(csv.reader(csvfile))

w_pco_a = np.asfarray(np.transpose(pco_a)[0])   # frequencies (cm^-1)
e1_pco_a = np.asfarray(np.transpose(pco_a)[1])  # real part
e2_pco_a = np.asfarray(np.transpose(pco_a)[2])  # imaginary part

f_e1_pco_a = interpolate.interp1d(w_pco_a, e1_pco_a, kind='cubic')
f_e2_pco_a = interpolate.interp1d(w_pco_a, e2_pco_a, kind='cubic')
eps_a = f_e1_pco_a(omega) + 1j * f_e2_pco_a(omega)

# eps_inf_a = 4.0
# w_lo_a = 972
# w_to_a = 820
# gam_a = 4.0
# eps_a = eps_inf_a*(1 + (w_lo_a**2 - w_to_a**2)/(w_to_a**2 - omega**2 - 1j*omega*gam_a))

# b  axis
with open(b_axis_df, newline='') as csvfile:
    pco_b = list(csv.reader(csvfile))

w_pco_b = np.asfarray(np.transpose(pco_b)[0])   # frequencies (cm^-1)
e1_pco_b = np.asfarray(np.transpose(pco_b)[1])  # real part
e2_pco_b = np.asfarray(np.transpose(pco_b)[2])  # imaginary part

f_e1_pco_b = interpolate.interp1d(w_pco_b, e1_pco_b, kind='cubic')
f_e2_pco_b = interpolate.interp1d(w_pco_b, e2_pco_b, kind='cubic')
eps_b = f_e1_pco_b(omega) + 1j * f_e2_pco_b(omega)

# eps_inf_b = 2.4
# w_lo_b = 1004
# w_to_b = 958
# gam_b = 2.0
# eps_b = eps_inf_b*(1 + (w_lo_b**2 - w_to_b**2)/(w_to_b**2 - omega**2 - 1j*omega*gam_b))

# c axis
with open(c_axis_df, newline='') as csvfile:
    pco_c = list(csv.reader(csvfile))

w_pco_c = np.asfarray(np.transpose(pco_c)[0])    # frequencies (cm^-1)
e1_pco_c = np.asfarray(np.transpose(pco_c)[1])   # real part
e2_pco_c = np.asfarray(np.transpose(pco_c)[2])   # imaginary part

f_e1_pco_c = interpolate.interp1d(w_pco_c, e1_pco_c, kind='cubic')
f_e2_pco_c = interpolate.interp1d(w_pco_c, e2_pco_c, kind='cubic')
eps_c = f_e1_pco_c(omega) + 1j * f_e2_pco_c(omega)

# eps_inf_c = 5.2
# w_lo_c = 851
# w_to_c = 545
# gam_c = 4.0
# eps_b = eps_inf_c*(1 + (w_lo_c**2 - w_to_c**2)/(w_to_c**2 - omega**2 - 1j*omega*gam_c))


# ----------- Calculation ------------- #


# rotate dielectric tensor to xyz frame with xz being plane of incidence (called in d_mat)
def rot_eps(m, th, ph, ps, opt):

    a1 = np.cos(ps)*np.cos(ph) - np.cos(th)*np.sin(ph)*np.sin(ps)
    a2 = -np.sin(ps)*np.cos(ph) - np.cos(th)*np.sin(ph)*np.cos(ps)
    a3 = np.sin(th)*np.sin(ph)
    b1 = np.cos(ps)*np.sin(ph) + np.cos(th)*np.cos(ph)*np.sin(ps)
    b2 = -np.sin(ps)*np.sin(ph) + np.cos(th)*np.cos(ph)*np.cos(ps)
    b3 = -np.sin(th)*np.cos(ph)
    c1 = np.sin(th)*np.sin(ps)
    c2 = np.sin(th)*np.cos(ps)
    c3 = np.cos(th)

    # rotation matrix is orthonormal
    rot = np.asmatrix([[a1, a2, a3],
                       [b1, b2, b3],
                       [c1, c2, c3]])

    if opt == [0, 0, 0]:
        # diagonal dielectric tensor at specific frequency
        eps_mat = np.asmatrix([[eps_a[m], 0, 0],
                              [0, eps_b[m], 0],
                              [0, 0, eps_c[m]]])

        return np.matmul(np.matmul(rot, eps_mat), np.linalg.inv(rot))

    else:

        return np.matmul(rot, np.transpose(opt))


# Delta matrix (Maxwell's equations matrix form) (called in p_mat)
def d_mat(k, K, m, alp, bet, gam):

    vx = K/k  # normalized x-direction propagation constant

    eps = rot_eps(m, alp, bet, gam, [0, 0, 0])

    exx = eps[0, 0]
    exy = eps[0, 1]
    exz = eps[0, 2]
    eyx = eps[1, 0]
    eyy = eps[1, 1]
    eyz = eps[1, 2]
    ezx = eps[2, 0]
    ezy = eps[2, 1]
    ezz = eps[2, 2]

    dmat = np.zeros((4, 4), dtype=np.complex128)

    dmat[0, 0] = -vx*ezx/ezz
    dmat[0, 1] = 1.-vx**2./ezz
    dmat[0, 2] = -vx*ezy/ezz
    dmat[1, 0] = exx - exz*ezx/ezz
    dmat[1, 1] = -vx*exz/ezz
    dmat[1, 2] = exy - exz*ezy/ezz
    dmat[2, 3] = 1.
    dmat[3, 0] = eyx - eyz*ezx/ezz
    dmat[3, 1] = -vx*eyz/ezz
    dmat[3, 2] = eyy - vx**2. - eyz*ezy/ezz

    return dmat


# Propagation matrix is exp(D) = V*exp(Lambda)*V^-1 (eigendecomposition)
def p_mat(k, K, m, alp, bet, gam):

    d = d_mat(k, K, m, alp, bet, gam)

    # eigenvalues and eigenvectors numerically - fast
    vz, eigv = np.linalg.eig(d)
    vzmat = np.asmatrix([[np.exp(1.j*k*H*vz[0]), 0, 0, 0],
                         [0, np.exp(1.j*k*H*vz[1]), 0, 0],
                         [0, 0, np.exp(1.j*k*H*vz[2]), 0],
                         [0, 0, 0, np.exp(1.j*k*H*vz[3])]])

    p = eigv*vzmat*np.linalg.inv(eigv)

    return p


def rp(k, K, n, alp, bet, gam):

    ni = sqrt(ea[n])
    nt = sqrt(es[n])

    cos_gi = sqrt(1. - (K / (ni * k)) ** 2.)
    cos_gt = sqrt(1. - (K / (nt * k)) ** 2.)

    p = p_mat(k, K, n, alp, bet, gam)

    # checked once, these seem correct
    a1 = ni*(nt*p[0, 1] - cos_gt*p[1, 1]) + cos_gi*(nt*p[0, 0] - cos_gt*p[1, 0])
    a2 = ni*(nt*p[0, 1] - cos_gt*p[1, 1]) - cos_gi*(nt*p[0, 0] - cos_gt*p[1, 0])
    a4 = (nt*p[0, 2] - cos_gt*p[1, 2]) - ni*cos_gi*(nt*p[0, 3] - cos_gt*p[1, 3])
    a5 = ni*(nt*cos_gt*p[2, 1] - p[3, 1]) + cos_gi*(nt*cos_gt*p[2, 0] - p[3, 0])
    a6 = ni*(nt*cos_gt*p[2, 1] - p[3, 1]) - cos_gi*(nt*cos_gt*p[2, 0] - p[3, 0])
    a8 = (nt*cos_gt*p[2, 2] - p[3, 2]) - ni*cos_gi*(nt*cos_gt*p[2, 3] - p[3, 3])

    return (a1*a8-a4*a5)/(a4*a6-a2*a8)


rp_arr = np.ones((len(omega), len(qs)), dtype=complex)

for k_it in ks:
    i = int((k_it - kstart) / kstep)

    for K_it in Ks:
        j = int((K_it - Kstart) / Kstep)

        rp_arr[i, j] = rp(k_it, K_it, i, theta, phi, psi)


# remove horizontal scars
omega = omega[np.logical_not(rp_arr[:, 0] == 1. + 0.j)]
rp_arr = rp_arr[np.logical_not(rp_arr[:, 0] == 1. + 0.j)]

# remove vertical scars
qs = qs[np.logical_not(rp_arr[127, :] == 1. + 0.j)]
rp_arr = rp_arr[:, np.logical_not(rp_arr[127, :] == 1. + 0.j)]

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

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot([0,  1.5], [0, 0], zs=[0, 0], color='k', linewidth=1.0, marker='o')
ax2.plot([0, 0], [0, 1.5], zs=[0, 0], color='k', linewidth=1.0, marker='o')
ax2.plot([0, 0], [0, 0], zs=[0, 1.5], color='k', linewidth=1.0, marker='o')
xrot = rot_eps(0, theta, phi, psi, [1, 0, 0])
yrot = rot_eps(0, theta, phi, psi, [0, 1, 0])
zrot = rot_eps(0, theta, phi, psi, [0, 0, 1])
ax2.plot([0, xrot[0, 0]], [0, xrot[0, 1]], zs=[0, xrot[0, 2]], color='b', linewidth=7.0, marker='o', markersize=14.0)
ax2.plot([0, yrot[0, 0]], [0, yrot[0, 1]], zs=[0, yrot[0, 2]], color='g', linewidth=7.0, marker='o', markersize=14.0)
ax2.plot([0, zrot[0, 0]], [0, zrot[0, 1]], zs=[0, zrot[0, 2]], color='r', linewidth=7.0, marker='o', markersize=14.0)
ax2.set_xlim([-2, 2])
ax2.set_ylim([-2, 2])
ax2.set_zlim([-2, 2])
ax2.title.set_text('Crystal Axes')

plt.show(block=True)
