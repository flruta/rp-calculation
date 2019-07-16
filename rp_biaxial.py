# -*- coding: utf-8 -*-
"""

Calculate polariton dispersion of isotropic-biaxial-isotropic layer with arbitrarily-oriented crystal axes
I. Abdulhalim, Journal of Optics A: Pure and Applied Optics 1 (1999) 646
I. Abdulhalim, Optics Communications 157 (1998) 265
W. Premerlani, "Computing Euler Angles from Direction Cosines" (2010)


@author: Francesco L. Ruta
Basov Infrared Spectroscopy Laboratory
Columbia University, Department of Physics

Original upload 7/16/2019
"""

import numpy as np
import csv
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.scimath import sqrt as sqrt
import matplotlib.pyplot as plt
import sys


# ------------------//------------------------- User Input Begins Here -----------------------//-------------------- #


# CSV files with dielectric function along the crystallographic axes of the biaxial layer, respectively.
# Organized such that the first column lists the frequencies in cm^-1, the second column lists the real part of the
# relative dielectric function, and the third column lists the imaginary part of the relative dielectric function
a_axis_df = 'pco_ab.csv'
b_axis_df = 'pco_ab.csv'
c_axis_df = 'pco_c.csv'
H = 100  # nm, thickness of biaxial layer

# CSV file with dielectric function of isotropic substrate, organized same as specified above. Bottom isotropic layer
# Upper isotropic layer is vacuum/air, with dielectric constant = 1
substrate = 'si.csv'

# orientation of crystallographic frame abc relative to xyz frame (xz plane of incidence, xy plane is interface)
psi = np.pi/12        # yaw angle (azimuthal angle for c axis)
theta = np.pi/2       # pitch angle (polar angle for c axis)
phi = 0         # roll angle

# frequency axis - specify the limits of the plot
wstart = 1000  # cm^-1
wstop = 7000  # cm^-1
wstep = 10     # step size

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

C = 3 * 10 ** 8  # m/s, speed of light
H = H * 10 ** (-9)  # m, thickness of anisotropic layer

omega = np.arange(wstart, wstop, wstep)  # cm^-1
ks = omega * 0.02998 * 10 ** 12 / C  # m^-1

qs = np.arange(qstart, qstop, qstep)  # 10^5 cm^1
Ks = qs * 10 ** 5 * 100  # m^-1


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

# b  axis
with open(b_axis_df, newline='') as csvfile:
    pco_b = list(csv.reader(csvfile))

w_pco_b = np.asfarray(np.transpose(pco_b)[0])   # frequencies (cm^-1)
e1_pco_b = np.asfarray(np.transpose(pco_b)[1])  # real part
e2_pco_b = np.asfarray(np.transpose(pco_b)[2])  # imaginary part

f_e1_pco_b = interpolate.interp1d(w_pco_b, e1_pco_b, kind='cubic')
f_e2_pco_b = interpolate.interp1d(w_pco_b, e2_pco_b, kind='cubic')
eps_b = f_e1_pco_b(omega) + 1j * f_e2_pco_b(omega)

# c axis
with open(c_axis_df, newline='') as csvfile:
    pco_c = list(csv.reader(csvfile))

w_pco_c = np.asfarray(np.transpose(pco_c)[0])    # frequencies (cm^-1)
e1_pco_c = np.asfarray(np.transpose(pco_c)[1])   # real part
e2_pco_c = np.asfarray(np.transpose(pco_c)[2])   # imaginary part

f_e1_pco_c = interpolate.interp1d(w_pco_c, e1_pco_c, kind='cubic')
f_e2_pco_c = interpolate.interp1d(w_pco_c, e2_pco_c, kind='cubic')
eps_c = f_e1_pco_c(omega) + 1j * f_e2_pco_c(omega)


# ----------- Calculation ------------- #


# rotate dielectric tensor to xyz frame with xz being plane of incidence (called in d_mat)
def rot_eps(m, th, ph, ps, opt):

    # alp, bet, gam are the three euler angles of abc reference frame
    # convert to direction cosines
    a1 = np.cos(th)*np.cos(ps)
    a2 = np.sin(ph)*np.sin(th)*np.cos(ps) - np.cos(ph)*np.sin(ps)
    a3 = np.cos(ph)*np.sin(th)*np.cos(ps) + np.sin(ph)*np.sin(ps)
    b1 = np.cos(th)*np.sin(ps)
    b2 = np.sin(ph)*np.sin(th)*np.sin(ps) + np.cos(ph)*np.cos(ps)
    b3 = np.cos(ph)*np.sin(th)*np.sin(ps) - np.sin(ph)*np.cos(ps)
    c1 = -np.sin(th)
    c2 = np.sin(ph)*np.cos(th)
    c3 = np.cos(ph)*np.cos(th)

    # rotation matrix is orthonormal
    rot = np.asmatrix([[a1, a2, a3],
                       [b1, b2, b3],
                       [c1, c2, c3]])

    if opt == [0, 0, 0]:
        # diagonal dielectric tensor at specific frequency
        eps_mat = np.asmatrix([[eps_a[m], 0, 0],
                              [0, eps_b[m], 0],
                              [0, 0, eps_c[m]]])

        return np.matmul(np.matmul(rot, eps_mat), np.transpose(rot))

    else:

        return np.matmul(rot, np.transpose(opt))


# Delta matrix (Maxwell's equations matrix form) (called in p_mat)
def d_mat(k, K, m, alp, bet, gam):

    vx = K/k  # normalized x-direction propagation constant
    vy = 0  # xz is plane of incidence (no propagation in y)

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
    dmat[0, 1] = 1-vx**2/ezz
    dmat[0, 2] = -vx*ezy/ezz
    dmat[0, 3] = -vx*vy/ezz
    dmat[1, 0] = exx - vy**2 - exz*ezx/ezz
    dmat[1, 1] = -vx*exz/ezz
    dmat[1, 2] = exy + vx*vy - exz*ezy/ezz
    dmat[1, 3] = -vy*exz/ezz
    dmat[2, 0] = -vy*ezx/ezz
    dmat[2, 1] = dmat[0, 3]
    dmat[2, 2] = -vy*ezy/ezz
    dmat[2, 3] = 1-vy**2/ezz
    dmat[3, 0] = eyx + vx*vy - eyz*ezx/ezz
    dmat[3, 1] = -vx*eyz/ezz
    dmat[3, 2] = eyy - vx**2 - eyz*ezy/ezz
    dmat[3, 3] = -vy*eyz/ezz

    return dmat


# Propagation matrix is exp(D) = V*exp(Lambda)*V^-1 (eigendecomposition)
def p_mat(k, K, m, alp, bet, gam):

    d = d_mat(k, K, m, alp, bet, gam)

    # eigenvalues and eigenvectors numerically - fast
    vz, eigv = np.linalg.eig(d)
    vzmat = np.asmatrix([[np.exp(1j*k*H*vz[0]), 0, 0, 0],
                         [0, np.exp(1j*k*H*vz[1]), 0, 0],
                         [0, 0, np.exp(1j*k*H*vz[2]), 0],
                         [0, 0, 0, np.exp(1j*k*H*vz[3])]])

    p = eigv*vzmat*np.linalg.inv(eigv)

    return p


# # Analytic propagation matrix using Lagrange-Sylvester interpolation polynomial
# # solution is not consistent with numerics (misprint in paper, likely for dd variable)
# def p_mat_analytic(k, K, m, alp, bet, gam):
#
#     d = d_mat(k, K, m, alp, bet, gam)
#
#     # with vy = 0, we can make this simplification for a, b, and c:
#     a = -(d[0, 0] + d[1, 1])
#     b = d[1, 1]*d[0, 0] - d[0, 1]*d[1, 0] - d[3, 2]
#     c = d[3, 2]*(d[0, 0] + d[1, 1]) - d[0, 2]*d[3, 0] - d[3, 1]*d[1, 2]
#     dd = (d[0, 0]*(d[1, 2]*d[3, 1] - d[1, 1]*d[3, 2]) + d[0, 1]*(d[3, 2]*d[1, 0] - d[3, 0]*d[1, 2])
#           + d[0, 2]*(d[1, 1]*d[3, 0] - d[1, 0]*d[3, 1]))
#
#     # calculate eigenvalues
#     vz = np.zeros(4, dtype=np.complex128)
#
#     f = -4 * (b ** 2 - 3 * a * c + 12 * dd) ** 3 + (
#                 2 * b ** 3 - 9 * a * b * c + 27 * c ** 2 + 27 * a ** 2 * dd - 72 * b * dd) ** 2
#     g = a ** 2 / 4 - 2 * b / 3 + 2 ** (1 / 3) * (b ** 2 - 3 * a * c + 12 * dd)
#     h = 2 * b ** 3 - 9 * a * b * c + 27 * c ** 2 + 27 * a ** 2 * dd - 72 * b * dd + sqrt(f)
#     cbrth = h**(np.longdouble(1/3.))
#     w = sqrt(g / (3 * cbrth) + cbrth*(1 / 32) ** (1 / 3))  # made a correction here
#     u = a ** 2 / 2 - 4 * b / 3 - 2 ** (1 / 3) * (b ** 2 - 3 * a * c + 12 * dd) / (3 * cbrth) - cbrth*(1 / 32) ** (
#                 1 / 3)  # made a correction here
#     q = (-a ** 3 + 4 * a * b - 8 * c) / (4 * w)
#
#     vz[0] = -a / 4 - w / 2 + sqrt(u - q) / 2
#     vz[1] = -a / 4 - w / 2 - sqrt(u - q) / 2
#     vz[2] = -a / 4 + w / 2 + sqrt(u - q) / 2
#     vz[3] = -a / 4 + w / 2 - sqrt(u - q) / 2
#
#     f1 = f_vals(k, 1, vz)
#     f2 = f_vals(k, 2, vz)
#     f3 = f_vals(k, 3, vz)
#     f4 = f_vals(k, 4, vz)
#
#     # now construct the propagation matrix
#     p = np.zeros((4, 4), dtype=np.complex128)
#
#     # p[0, 0] = (f1*(d[0, 0]**3 + d[0, 1]*d[1, 0]*(2*d[0, 0] + d[1, 1]) + d[0, 2]*d[3, 0])
#     #             - f2 - f3*(d[0, 0]**2 + d[0, 1]*d[1, 0]) + f4*d[0, 0])
#     p[0, 0] = (f1*(d[0, 0]**3 + 3*d[0, 1]*d[1, 0]*d[0, 0] + d[0, 2]*d[1, 2])
#                 - f2 -f3*(d[0, 0]**2 + d[0, 1]*d[1, 0]) + f4*d[0, 0])
#
#     p[0, 1] = (f1*(d[0, 1]*(d[0, 1]*d[1, 0] + d[1, 1]**2) + d[0, 2]*d[3, 1]
#                 - a*d[0, 0]*d[0, 1]) + f3*a*d[0, 1] + f4*d[0, 1])
#
#     # # made a correction here
#     p[0, 2] = -f1*(d[0, 2]*(a*d[0, 0] + b) + a*d[0, 1]*d[1, 2]) - f3*(d[0, 0]*d[0, 2] + d[0, 1]*d[1, 2]) + f4*d[0, 2]
#
#     p[0, 3] = f1*(d[0, 0]*d[0, 2] + d[0, 1]*d[1, 2]) - f3*d[0, 2]
#
#     p[1, 0] = (f1*(d[1, 0]*(d[0, 0]**2 + d[0, 1]*d[1, 0]) - a*d[1, 0]*d[1, 1] + d[1, 2]*d[3, 0])
#                 + f3*a*d[1, 0] + f4*d[1, 0])
#
#     # p[1, 1] = (f1*(d[1, 1]**3 + d[0, 1]*d[1, 0]*(2*d[1, 1] + d[0, 0]) + d[1, 2]*d[3, 1])
#     #           - f2 - f3*(d[1, 1]**2 + d[0, 1]*d[1, 0]) + f4*d[1, 1])
#     p[1, 1] = p[0, 0]   # symmetric dielectric tensor simplification
#
#     p[1, 2] = -f1*(d[1, 2]*(a*d[1, 1] + b) + a*d[0, 2]*d[1, 0]) - f3*(d[1, 0]*d[0, 2] + d[1, 1]*d[1, 2]) + f4*d[1, 2]
#
#     p[1, 3] = f1*(d[0, 2]*d[1, 0] + d[1, 1]*d[1, 2]) - f3*d[1, 2]
#
#     # p[2, 0] = f1*(d[0, 0]*d[3, 0] + d[1, 0]*d[3, 1]) - f3*d[3, 0]
#     p[2, 0] = p[1, 3]   # symmetric dielectric tensor simplification
#
#     # p[2, 1] = f1*(d[0, 1]*d[3, 0] + d[1, 1]*d[3, 1]) - f3*d[3, 1]
#     p[2, 1] = p[0, 3]   # symmetric dielectric tensor simplification
#
#     p[2, 2] = -f1*(a*d[3, 2] + c) - f2 - f3*d[3, 2]
#
#     p[2, 3] = f1*d[3, 2] + f4
#
#     # # p[3, 0] = (-f1*(d[3, 0]*(a*d[0, 0] + b) + a*d[1, 0]*d[3, 1])
#     #              - f3*(d[0, 0]*d[3, 0] + d[1, 0]*d[3, 1]) + f4*d[3, 0])
#     p[3, 0] = p[1, 2]   # symmetric dielectric tensor simplification
#
#     # # p[3, 1] = (-f1*(d[3, 1]*(a*d[1, 1] + b) + a*d[0, 1]*d[3, 0])
#     #              - f3*(d[0, 1]*d[3, 0] + d[1, 1]*d[3, 1]) + f4*d[3, 1])
#     p[3, 1] = p[0, 2]   # symmetric dielectric tensor simplification
#
#     p[3, 2] = (f1*(d[3, 0]*(d[0, 0]*d[0, 2] + d[0, 1]*d[1, 2]) + d[3, 1]*(d[0, 2]*d[1, 0] + d[1, 1]*d[1, 2])
#                 + d[3, 2]**2) + f3*(a*d[3, 2] + c) + f4*d[3, 2])
#
#     p[3, 3] = p[2, 2]
#
#     return p
#
#
# # f parameters from Lagrange-Sylvester (called in p_mat for analytic solution)
# def f_vals(k, num, vz):
#
#     # calculate f values
#     if num == 1:
#
#         f1 = 0
#         for I in [0, 1, 2, 3]:
#
#             f1mult = 1
#             for J in [0, 1, 2, 3]:
#
#                 if J != I:
#                     f1mult = f1mult/(vz[I] - vz[J])
#
#             f1 = f1 + np.exp(1j * k * H * vz[I]) * f1mult
#
#         return f1
#
#     elif num == 2:
#
#         f2 = 0
#         for I in [0, 1, 2, 3]:
#
#             f2mult = 1
#             for J in [0, 1, 2, 3]:
#
#                 if J != I:
#                     f2mult = f2mult*vz[J]/(vz[I] - vz[J])
#
#             f2 = f2 + np.exp(1j * k * H * vz[I]) * f2mult
#
#         return f2
#
#     elif num == 3:
#
#         f3 = 0
#         for I in [0, 1, 2, 3]:
#
#             f3mult = 1
#             f3add = 0
#             for J in [0, 1, 2, 3]:
#
#                 if J != I:
#                     f3add = f3add + vz[J]
#                     f3mult = f3mult/(vz[I] - vz[J])
#
#             f3 = f3 + np.exp(1j * k * H * vz[I]) * f3mult * f3add
#
#         return f3
#
#     # not sure about this guy:
#     elif num == 4:
#
#         f4 = 0
#         for I in [0, 1, 2, 3]:
#
#             f4mult = 1
#             f4add = 0
#             for J in [0, 1, 2, 3]:
#
#                 if J != I:
#                     f4mult = f4mult / (vz[I] - vz[J])
#
#                     for M in [0, 1, 2, 3]:
#
#                         if M != I and M != J:
#
#                             f4add = f4add + vz[J]*vz[M]
#
#             f4 = f4 + np.exp(1j * k * H * vz[I]) * f4mult * f4add
#
#         return f4
#
#     else:
#         sys.exit("ERROR: Invalid number for f value")
# Reflection coefficient of the structure


def rp(k, K, n, alp, bet, gam):

    ni = sqrt(ea[n])
    nt = sqrt(es[n])

    cos_gi = sqrt(1 - (K / (ni * k)) ** 2)
    cos_gt = sqrt(1 - (K / (nt * k)) ** 2)

    p = p_mat(k, K, n, alp, bet, gam)

    # checked once, these seem correct
    a1 = ni*(nt*p[0, 1] - cos_gt*p[1, 1]) + cos_gi*(nt*p[0, 0] - cos_gt*p[1, 0])
    a2 = ni*(nt*p[0, 1] - cos_gt*p[1, 1]) - cos_gi*(nt*p[0, 0] - cos_gt*p[1, 0])
    # a3 = (nt*p[0, 2] - cos_gt*p[1, 2]) + ni*cos_gi*(nt*p[0, 3] - cos_gt*p[1, 3])
    a4 = (nt*p[0, 2] - cos_gt*p[1, 2]) - ni*cos_gi*(nt*p[0, 3] - cos_gt*p[1, 3])
    a5 = ni*(nt*cos_gt*p[2, 1] - p[3, 1]) + cos_gi*(nt*cos_gt*p[2, 0] - p[3, 0])
    a6 = ni*(nt*cos_gt*p[2, 1] - p[3, 1]) - cos_gi*(nt*cos_gt*p[2, 0] - p[3, 0])
    # a7 = (nt*cos_gt*p[2, 2] - p[3, 2]) + ni*cos_gi*(nt*cos_gt*p[2, 3] - p[3, 3])
    a8 = (nt*cos_gt*p[2, 2] - p[3, 2]) - ni*cos_gi*(nt*cos_gt*p[2, 3] - p[3, 3])

    den = a4*a6-a2*a8
    rpp = (a1*a8-a4*a5)/den
    # rps = (a3*a8-a4*a7)/den

    return rpp


rp_arr = np.ones((len(omega), len(qs)), dtype=complex)

for k_it in ks:
    w_it = k_it * C / (0.02998 * 10 ** 12)
    i = int((w_it - wstart) / wstep)

    for K_it in Ks:
        q_it = K_it / (10 ** 5 * 100)
        j = int((q_it - qstart) / qstep)

        rp_arr[i, j] = rp(k_it, K_it, i, theta, phi, psi)


# remove horizontal scars
omega = omega[np.logical_not(rp_arr[:, 0] == 1. + 0.j)]
rp_arr = rp_arr[np.logical_not(rp_arr[:, 0] == 1. + 0.j)]

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
