import os
import numpy as np
#import scipy
#from scipy import sparse

'''

############################
# helper function delete_if_present
############################

def delete_if_present(file):
    if os.path.exists(file):
        os.remove(file)

############################
# frequency noise power spectrum
############################

def f_min(white,oof):
    if oof == 'Y':
        return 1.15*10**(-5)
    else:
        return 0

def f_knee(white,oof):
    if oof == 'Y':
        return 0.1
    else:
        return 0

@np.vectorize
def spectrum(f,white,oof,NET,f_knee):
    if white == 'Y':
        return NET**2
        if oof == 'Y':
            return NET**2*(1+f_knee/abs(f))
    else:
        if oof == 'Y':
            return NET**2*f_knee/abs(f)
        else:
            return 0
'''

############################
# I,Q,U,V coefficients given Mueller matrix, alpha and beta
############################

def coeff_I(m, a, b):
    return (m[0,0]+m[1,0]*np.cos(b)+m[2,0]*np.sin(b))/2
    # reduces to m[0,0]*1/2 for diagonal mueller mat.

def coeff_Q(m, a, b):
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    return (m[0,1]*ca-m[0,2]*sa+(m[1,1]*ca-m[1,2]*sa)*cb+(m[2,1]*ca-m[2,2]*sa)*sb)/2
    # reduces to (m[1,1]*ca*cb-m[2,2]*sa*sb)/2 for diagonal mueller mat

def coeff_U(m, a, b):
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    return (m[0,1]*sa+m[0,2]*ca+(m[1,1]*sa+m[1,2]*ca)*cb+(m[2,1]*sa+m[2,2]*ca)*sb)/2
    # reduces to (m[1,1]*sa*cb+m[2,2]*ca*sb)/2 for diagonal mueller mat

'''
def coeff_V(m, a, b):
    return (m[0,3]+m[1,3]*np.cos(b)-m[2,3]*np.sin(b))/2

def coeff_vec(m, a, b):
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    c_I = (m[0,0]+m[1,0]*cb-m[2,0]*sb)/2
    c_Q = (m[0,1]*ca-m[0,2]*sa+(m[1,1]*ca-m[1,2]*sa)*cb-(m[2,1]*ca-m[2,2]*sa)*sb)/2
    c_U = (m[0,1]*sa+m[0,2]*ca+(m[1,1]*sa+m[1,2]*ca)*cb-(m[2,1]*sa+m[2,2]*ca)*sb)/2
    return np.stack((c_I,c_Q,c_U), axis=-1)

def coeff_mat(m, a, b_vec):
    nobs = len(a)
    ndet = len(b_vec[0])
    mat = np.zeros((nobs,ndet,3))
    for d in range(ndet):
        b = b_vec[:,d]
        mat[:,d] = np.transpose(np.array([coeff_I(m, a, b), coeff_Q(m, a, b), coeff_U(m, a, b)]))
    return mat

############################
# maps_func takes M and turns it in a map (multiplying by the exponential and summing)
############################

def maps_func(M, t_array, f_array):
    nobs = len(t_array)
    nfre = len(f_array)
    ndet = len(M[0])//nobs
    npix = len(M[:,0])

    exp_mat = np.exp(-1j*2*np.pi*np.outer(t_array, f_array))

    def M_map(d):
        M_tilde = M[:,d*nobs:(d+1)*nobs].dot(exp_mat) #scalar product between M and exp_mat
        return M_tilde.transpose()

def coeff_U(m, a, b):
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    return (m[0,1]*sa+m[0,2]*ca+(m[1,1]*sa+m[1,2]*ca)*cb-(m[2,1]*sa+m[2,2]*ca)*sb)/2

def coeff_V(m, a, b):
    return (m[0,3]+m[1,3]*np.cos(b)-m[2,3]*np.sin(b))/2

def coeff_vec(m, a, b):
    sa, ca, sb, cb = np.sin(a), np.cos(a), np.sin(b), np.cos(b)
    c_I = (m[0,0]+m[1,0]*cb-m[2,0]*sb)/2
    c_Q = (m[0,1]*ca-m[0,2]*sa+(m[1,1]*ca-m[1,2]*sa)*cb-(m[2,1]*ca-m[2,2]*sa)*sb)/2
    c_U = (m[0,1]*sa+m[0,2]*ca+(m[1,1]*sa+m[1,2]*ca)*cb-(m[2,1]*sa+m[2,2]*ca)*sb)/2
    return np.stack((c_I,c_Q,c_U), axis=-1)

def coeff_mat(m, a, b_vec):
    nobs = len(a)
    ndet = len(b_vec[0])
    mat = np.zeros((nobs,ndet,3))
    for d in range(ndet):
        b = b_vec[:,d]
        mat[:,d] = np.transpose(np.array([coeff_I(m, a, b), coeff_Q(m, a, b), coeff_U(m, a, b)]))
    return mat

############################
# maps_func takes M and turns it in a map (multiplying by the exponential and summing)
############################

def maps_func(M, t_array, f_array):
    nobs = len(t_array)
    nfre = len(f_array)
    ndet = len(M[0])//nobs
    npix = len(M[:,0])

    exp_mat = np.exp(-1j*2*np.pi*np.outer(t_array, f_array))

    def M_map(d):
        M_tilde = M[:,d*nobs:(d+1)*nobs].dot(exp_mat) #scalar product between M and exp_mat
        return M_tilde.transpose()

    for d in np.arange(ndet):
        maps[d:(d+1)] = M_map(d) #add M_vec to every row for different values of d and f

    return maps

def maps_func_tensor(M, t_array, f_array):
    nobs = len(t_array)
    nfre = len(f_array)
    ndet = len(M[0,0,:])
    npix = len(M[:,0,0])

    exp_mat = np.exp(-1j*2*np.pi*np.outer(t_array, f_array))

    def M_map(d):
        M_tilde = M[:,:,d].dot(exp_mat) #scalar product between M and exp_mat
        return M_tilde.transpose()

    maps = np.empty((ndet*nfre,npix),dtype=complex)

    for d in np.arange(ndet):
        maps[d*nfre:(d+1)*nfre] = M_map(d) #add M_vec to every row for different values of d and f

    return maps

############################
# func_A and func_AT apply A and A^T to a vector, returning a vector, b should be a ndet*nobs matrix!
############################

def func_A(b, pix, ndet, m, alpha, beta):
    nobs = len(pix)
    idx = np.arange(nobs)
    x = np.empty(nobs*ndet)
    for d in np.arange(ndet):
        x[d*nobs + idx] = b[pix]*coeff_I(m, alpha, beta[d]) + b[pix+1]*coeff_Q(m, alpha, beta[d]) + b[pix+2]*coeff_U(m, alpha, beta[d])
    return x
    
def func_AT(b, pix, nsto, m, alpha, beta):
    npix = len(set(list(pix)))
    pix_idx = np.arange(npix)
    def where_idx(pix,p):
        return np.where[pix==p]
    x = np.zeros(nsto*npix)
    for p in pix_idx:
        idx_where = idx[where_idx(pix,p)]
        for d in np.array(4):
            x[nsto*p_idx+0] += b[d*nobs+idx_where]*coeff_I(m, alpha, beta[d,idx_where])
            x[nsto*p_idx+1] += b[d*nobs+idx_where]*coeff_Q(m, alpha, beta[d,idx_where])
            x[nsto*p_idx+2] += b[d*nobs+idx_where]*coeff_U(m, alpha, beta[d,idx_where])
    return x

############################
# evaluation of frequency power spectrum (trapeziodal rule)
############################

def power(func,f_array):
    f_boundaries = np.concatenate(([f_array[0]], 0.5*(f_array[:-1]+f_array[1:]), [f_array[-1]]))
    f_val = func(f_boundaries)
    return (f_boundaries[1:]-f_boundaries[:-1])*0.5*(f_val[:-1]+f_val[1:])

############################
# components of the noise matrix
############################

def noise_comp(t1,t2,func,f_array,fsamp):
    f_pos = f_array[f_array>=0]
    where_pos = np.where(f_array>=0)
    f_neg = f_array[f_array<0]
    where_neg = np.where(f_array<0)

    f_boundaries = np.concatenate(([f_neg[0]], 0.5*(f_neg[:-1]+f_neg[1:]), [f_neg[-1]]))
    f_val = np.multiply(func(f_boundaries),np.exp(1j*2*np.pi*(t1-t2)*f_boundaries))
    integral_neg = np.sum((f_boundaries[1:]-f_boundaries[:-1])*0.5*(f_val[:-1]+f_val[1:]))

    f_boundaries = np.concatenate(([f_pos[0]], 0.5*(f_pos[:-1]+f_pos[1:]), [f_pos[-1]]))
    f_val = np.multiply(func(f_boundaries),np.exp(1j*2*np.pi*(t1-t2)*f_boundaries))
    integral_pos = np.sum((f_boundaries[1:]-f_boundaries[:-1])*0.5*(f_val[:-1]+f_val[1:]))

    return fsamp*(integral_neg + integral_pos)
    
'''

