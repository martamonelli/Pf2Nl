import numpy as np
from numpy.linalg import inv
from math import pi

import scipy.sparse
from scipy.sparse import linalg
from scipy.sparse import coo_matrix, csr_matrix, block_diag
from scipy.sparse.linalg import LinearOperator

import healpy as hp
import matplotlib.pyplot as plt

import _params as param
import _utils as utils

import sys

import time

from matplotlib.legend_handler import HandlerBase

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], 
                           color=orig_handle[0])
        l2 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                           color=lighten_color(orig_handle[0],0.6),linestyle='dashed')
        return [l2, l1]
        
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

plt.rcParams.update({
    "font.size":10.95,
    "text.usetex":True,
    "font.family":"serif", 
    "font.serif":"cm"
    })

##########################################################
# mock focal plane and scanning strategy specifics
##########################################################

xi_dets = param.xi_dets
ndet = len(xi_dets)
dets = np.arange(0,ndet)

f_samp = param.f_samp
delta_t = 1/f_samp

omega = param.omega

nside = param.nside
npix = 12*nside**2
nobs = npix

pix = np.arange(nobs)%npix

##########################################################
# mock noise spectrum
##########################################################

ff = np.fft.fftfreq(nobs, d=delta_t)

def spectrum_white(f):
    return param.NET**2*(1+f-f)

def spectrum_oof(f):
    return param.NET**2*((f**2+param.f_knee**2)/(f**2+param.f_min**2))**param.alpha
    
#plt.loglog(ff[ff > 0],spectrum_white(ff[ff > 0]),linestyle='dashed',color='darkorange')
#plt.loglog(ff[ff > 0],spectrum_oof(ff[ff > 0]),color='orangered')
#plt.xlabel(r'$f$')
#plt.ylabel(r'$P(f)$')
#plt.savefig('Pf.pdf')
#plt.clf()

##########################################################
# stuff
##########################################################

exp_mat = np.exp(-1j*2*pi*np.einsum('f,t->tf',ff,np.arange(nobs)*delta_t))

# best shape for alpha, beta: (nobs, ndet)
alpha = 2*(0*xi_dets.reshape((1,-1))+(omega*np.arange(nobs)*delta_t).reshape((-1,1)))
beta =  2*(  xi_dets.reshape((1,-1))-(omega*np.arange(nobs)*delta_t).reshape((-1,1)))

nsto = 3

##########################################################
# mueller matrix elements
##########################################################

mueller_mat = np.identity(3)
mueller_mat[2,2] *= -1		#to switch the HWP on

print('now running with mueller matrix:')
print(mueller_mat)
  
##########################################################
# time-dependent quantities (pix, psi...)
##########################################################

coeff_IQU = np.empty((nobs,ndet,nsto))
coeff_IQU[:,:,0] = utils.coeff_I(mueller_mat,alpha,beta)
coeff_IQU[:,:,1] = utils.coeff_Q(mueller_mat,alpha,beta)
coeff_IQU[:,:,2] = utils.coeff_U(mueller_mat,alpha,beta)
del alpha, beta

##########################################################
# playing around
##########################################################

lmax = 2*nside

print('entering loop')
Cls_white = np.zeros((6,lmax))
Cls_oof = np.zeros((6,lmax))
    
ell = np.arange(len(Cls_white[0]))[2:]

#'''

for d in np.arange(len(xi_dets)):
    start = time.time()

    Iij, Qij, Uij = np.zeros((3,nobs))
    
    for p in np.arange(npix):
        A = coeff_IQU[np.where(pix==p)]		# A 	 has dimensions [nobs, ndet, nsto]
        AT = A.transpose()        			# AT 	 has dimensions [nsto, ndet, nobs]
        ATA = np.einsum('sdi,idS->sS', AT, A)		# ATA 	 has dimensions [nsto, nsto]
        block = inv(ATA)				# block  has dimensions [nsto, nsto]
        AT_new = AT[:,d,:]				# AT_new has dimensions [nsto, nobs]
        
        temp = np.einsum('sS,Si->s', block, AT_new)	# temp	 has dimensions [nsto]
        
        Iij[np.where(pix==p)] = temp[0]
        Qij[np.where(pix==p)] = temp[1]
        Uij[np.where(pix==p)] = temp[2]
        
    print(np.sqrt(Qij**2 + Uij**2))
      
    I_mat = csr_matrix((Iij, (pix, np.arange(nobs))), shape = (npix, nobs)) 
    Q_mat = csr_matrix((Qij, (pix, np.arange(nobs))), shape = (npix, nobs)) 
    U_mat = csr_matrix((Uij, (pix, np.arange(nobs))), shape = (npix, nobs)) 
     
    I_map = I_mat.dot(exp_mat)
    Q_map = Q_mat.dot(exp_mat)
    U_map = U_mat.dot(exp_mat)
    
    for i in np.arange(len(ff)):
        M_map = np.array([I_map[:,i],Q_map[:,i],U_map[:,i]])
        Cls = hp.sphtfunc.anafast(np.real(M_map),lmax=lmax-1) + hp.sphtfunc.anafast(np.imag(M_map),lmax=lmax-1)
        Cls_white += spectrum_white(ff[i])*Cls
        Cls_oof += spectrum_oof(ff[i])*Cls
    mins = (time.time() - start)/60
    
    print('detector '+str(d)+' done in '+str(mins)+' minutes')
    
print('exited loop')

#'''

plt.figure(figsize=(6,4))
plt.loglog(ell,Cls_oof[0,2:],color='dimgrey')
plt.loglog(ell,Cls_white[0,2:],linestyle='dashed',color=lighten_color('dimgrey',0.6))
plt.loglog(ell,Cls_oof[1,2:],color='darkorange')
plt.loglog(ell,Cls_white[1,2:],linestyle='dashed',color=lighten_color('darkorange',0.6))
plt.loglog(ell,Cls_oof[2,2:],color='orangered')
plt.loglog(ell,Cls_white[2,2:],linestyle='dashed',color=lighten_color('orangered',0.6))
plt.ylim(1e-11, 1e-10)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\widehat{N}_\ell^{XY}$ [$\mu$K$^2$]')
plt.legend([("dimgrey","--"), ("darkorange",":"), ("orangered",":")], [r'$TT$', r'$EE$', r'$BB$'],
           handler_map={tuple: AnyObjectHandler()})
plt.yticks(ticks=[1e-11,2e-11,3e-11,4e-11,5e-11,6e-11,7e-11,8e-11,9e-11,1e-10], labels=[r'$10^{-11}$','','','','','','','','',r'$10^{-10}$'])
plt.savefig('Nl.pdf')
plt.clf()
