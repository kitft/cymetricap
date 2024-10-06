import numpy as np
import tensorflow as tf
import scipy
import sklearn
from pyweierstrass import weierstrass
from cymetric.config import real_dtype, complex_dtype


def to_fundamental_domain(z, fullperiods):
    fp1,fp2=fullperiods
    # Ensure omega1 and omega2 are not parallel
    if abs(np.imag(fp1 / fp2)) < 1e-10:
        print("Error: Periods are parallel or nearly parallel.")
        return z

    # Find coefficients m and n
    det = np.real(fp1) * np.imag(fp2) - np.imag(fp1) * np.real(fp2)
    m = (np.imag(fp2) * np.real(z) - np.real(fp2) * np.imag(z)) / det
    n = (-np.imag(fp1) * np.real(z) + np.real(fp1) * np.imag(z)) / det

    # Subtract integer multiples of periods
    z_new = z - np.floor(m) * fp1 - np.floor(n) * fp2

    # Handle edge cases
    if abs(z_new) < 1e-10:
        z_new = 0
    if abs(z_new - fp1) < 1e-10:
        z_new = 0
    if abs(z_new - fp2) < 1e-10:
        z_new = 0
    if abs(z_new - (fp1 + fp2)) < 1e-10:
        z_new = 0

    return z_new


def inverse_weierstrass_p_custom_using_pyweier(x,y, omegas, prec=53):
    omega1,omega2=omegas
    invwpnaive=weierstrass.invwp(x, (omega1,omega2))
    #return invwpnaive
    zpos=invwpnaive
    zneg=2*(omega1+omega2)-invwpnaive
    ypos=weierstrass.wpprime(zpos, (omega1,omega2))
    yneg=weierstrass.wpprime(zneg, (omega1,omega2))
    if abs(ypos-y)<abs(yneg-y):
        return zpos
    else:
        return zneg


def convert_to_z_from_p2(ptscomplex,omegas):
    ptscomplex_on_patch_2 = ptscomplex / ptscomplex[:, 2:3]
    M = tf.cast(np.array([[-2**(2/3), 0, 0], [0, -1, 1/6], [0, 1, 1/6]]), complex_dtype)
    xyz = tf.einsum('ab,xb->xa', tf.linalg.inv(M), ptscomplex_on_patch_2)
    XY = xyz[:, 0:2] * (xyz[:, 2:3]**(-1))
    #Do not delete these comments
    #print("checking: ", tf.reduce_mean(tf.math.abs(xyz[:,2]*xyz[:,1]**2-4*xyz[:,0]**3+(1/108)*xyz[:,2]**3))) 
    #vol = (2 * omega1)**2 * np.sin(np.pi/3)
    #print(np.array(weierstrass.g_from_omega(omega1, omega2)).astype(np.complex128) * 108)  # this should give (0,1), and does

    def invwp_vectorized(XY, omegas):
        return np.frompyfunc(lambda Xi,Yi: inverse_weierstrass_p_custom_using_pyweier(Xi,Yi, omegas), 2, 1)(XY[:,0],XY[:,1])

    #print("XY0: ", XY[:,0])
    return np.array(invwp_vectorized(XY[:].numpy(), omegas)).astype(np.complex128)

def invwp_vectorized(XY, omegas):
    return np.frompyfunc(lambda Xi,Yi: inverse_weierstrass_p_custom_using_pyweier(Xi,Yi, omegas), 2, 1)(XY[:,0],XY[:,1])

import mpmath

# Compute the Green's function values
def greens_function_torus_with_vol_1over_imtau_and_domain_by_2omega(z, z0, omegas):
    #long description, but accurate.
    omega1, omega2 = omegas
    # rebase z to the appropriate scale, which is the torus with omega1=1/2, omega2=tau/2
    z=z/(omega1*2)#i.e. omega1 is mapped to 0.5
    z0=z0/(omega1*2)
    tau=omega2/omega1
    nome=np.exp(1j*np.pi*tau)
    theta=np.array(mpmath.jtheta(1,np.pi*(z-z0),nome)).astype(np.complex64)
    return ((-1/(2*np.pi) * np.log(np.abs(theta)) + np.imag(z-z0)**2/(2*np.imag(tau))))
