import scipy
import scipy.sparse.linalg as linalg
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.numpy.fft import fft, ifft, fftfreq
from jax import jit

from jax.config import config
config.update("jax_enable_x64", True)

def construct_integrator(version, full, nu, L, nx, nt):

    # wave number mesh (multiply back by nx because fftfreq takes it out - we raise k to powers so it affects us and has to be accounted for)
    k = (2 * jnp.pi / L) * fftfreq(nx) * nx
    dealias = jnp.zeros_like(k)
    dealias = jnp.abs(k) > jnp.max(k) * 2/3

    # Fourier Transform of the linear and nonlinear operators
    FL = (k ** 2) - nu * (k ** 4)
    FN = - (1 / 2) * (1j) * k

    if version == 'np' and full:
        def propagator(u0, T):
            dt = T/nt

            # Crank-Nicholson algebraic relations between uhat[n+1] and uhat[n]
            CN1 = (1 + (dt / 2) * FL)
            CN2 = 1 / (1 - (dt / 2) * FL)

            # solution mesh in real and Fourier space
            u      = np.empty((nt, nx), dtype=np.float64)
            u_hat  = np.empty((nt, nx), dtype=np.complex128)
            u_hat2 = np.empty((nt, nx), dtype=np.complex128)

            # set initial condition in real and Fourier space
            u[0]      = u0
            u_hat[0]  = (1 / nx) * np.fft.fft(u0)
            u_hat2[0] = (1 / nx) * np.fft.fft(u0**2)

            # first timestep (no advanced restarting, just Euler)
            u_hat[1]  = CN2 * ( CN1 * u_hat[0] + FN * u_hat2[0] * dt)

            # auxiliaries of first timestep
            u[1]      = nx * np.real(np.fft.ifft(u_hat[1]))
            u_hat2[1] = (1 / nx) * np.fft.fft(u[1]**2)

            # compute solution through time via finite difference method in Fourier space
            for j in range(1,nt-1): 
                # Cranck-Nicholson + Adams-Bashforth
                u_hat[j+1]  = CN2 * ( CN1 * u_hat[j] + ( 1.5 * FN * u_hat2[j] - 0.5 * FN * u_hat2[j-1] ) * dt)
                u_hat[j+1, dealias] = 0
                
                # go back to real space
                u[j+1]      = nx * np.real(np.fft.ifft(u_hat[j+1]))
                u_hat2[j+1] = (1 / nx) * np.fft.fft(u[j+1]**2)
            
            return u
        
    elif version == 'np' and not full:
        def propagator(u0, T):
            dt = T/nt

            # Crank-Nicholson algebraic relations between uhat[n+1] and uhat[n]
            CN1 = (1 + (dt / 2) * FL)
            CN2 = 1 / (1 - (dt / 2) * FL)

            # set initial condition in real and Fourier space
            uL      = np.float64(u0)
            u_hatL  = np.complex128((1 / nx) * np.fft.fft(u0))
            u_hat2L = np.complex128((1 / nx) * np.fft.fft(u0**2))

            # first timestep (no advanced restarting, just Euler)
            u_hat  = CN2 * ( CN1 * u_hatL + FN * u_hat2L * dt)
            u      = nx * np.real(np.fft.ifft(u_hat))
            u_hat2 = (1 / nx) * np.fft.fft(u**2)

            # compute solution through time via finite difference method in Fourier space
            for j in range(1,nt-1): 
                # Cranck-Nicholson + Adams-Bashforth
                u_hatN  = CN2 * ( CN1 * u_hat + ( 1.5 * FN * u_hat2 - 0.5 * FN * u_hat2L ) * dt)
                u_hatN[dealias] = 0
                
                # go back to real space
                uN      = nx * np.real(np.fft.ifft(u_hatN))
                u_hat2N = (1/nx) * np.fft.fft(uN**2)

                # new states
                u, u_hat, u_hat2, u_hat2L = uN, u_hatN, u_hat2N, u_hat2
            
            return u

    elif version == 'jax' and full:
        @jit
        def propagator(u0, T):
            dt = T/nt

            # Crank-Nicholson algebraic relations between uhat[n+1] and uhat[n]
            CN1 = (1 + (dt / 2) * FL)
            CN2 = 1 / (1 - (dt / 2) * FL)

            # solution mesh in real and Fourier space
            u      = jnp.empty((nt, nx), dtype=jnp.float64)
            u_hat  = jnp.empty((nt, nx), dtype=jnp.complex128)
            u_hat2 = jnp.empty((nt, nx), dtype=jnp.complex128)

            # set initial condition in real and Fourier space
            u      = u     .at[0].set( u0 )
            u_hat  = u_hat .at[0].set( (1 / nx) * fft(u[0])    )
            u_hat2 = u_hat2.at[0].set( (1 / nx) * fft(u[0]**2) )

            # first timestep (no advanced restarting, just Euler)
            u_hat  = u_hat .at[1].set( CN2 * ( CN1 * u_hat[0] + FN * u_hat2[0] * dt) )
            u      = u     .at[1].set( nx * jnp.real(ifft(u_hat[1])) )
            u_hat2 = u_hat2.at[1].set( (1 / nx) * fft(u[1]**2)       )

            # compute solution through time via finite difference method in Fourier space
            def finite_step(j, state):
                u, u_hat, u_hat2 = state

                # Cranck-Nicholson + Adams-Bashforth
                u_hat  = u_hat.at[j+1].set( CN2 * ( CN1 * u_hat[j] + ( 1.5 * FN * u_hat2[j] - 0.5 * FN * u_hat2[j-1] ) * dt) )
                u_hat  = u_hat.at[j+1, dealias].set(0)
                
                # go back to real space
                u      = u     .at[j+1].set( nx * jnp.real(ifft(u_hat[j+1])) )
                u_hat2 = u_hat2.at[j+1].set( (1 / nx) * fft(u[j+1]**2)       )

                return (u, u_hat, u_hat2)

            # compute solution through time via finite difference method in Fourier space
            u = jax.lax.fori_loop(1,nt-1,finite_step,(u, u_hat, u_hat2))[0]
            return jnp.real(u)
    
    elif version == jax and not full:
        @jit
        def propagator(u0, T):
            dt = T/nt
            
            # Crank-Nicholson algebraic relations between uhat[n+1] and uhat[n]
            CN1 = (1 + (dt / 2) * FL)
            CN2 = 1 / (1 - (dt / 2) * FL)

            # set initial condition in real and Fourier space
            u      = jnp.float64(u0)
            u_hat  = jnp.complex128((1 / nx) * fft(u[0]))
            u_hat2 = jnp.complex128((1 / nx) * fft(u[0]**2))

            # first timestep (no advanced restarting, just Euler)
            u_hatL  = CN2 * ( CN1 * u_hat + FN * u_hat2 * dt)
            uL      = nx * jnp.real(ifft(u_hatL))
            u_hat2L = (1/nx) * fft(uL**2)

            # compute solution through time via finite difference method in Fourier space
            def finite_step(j, state):
                u, u_hat, u_hat2, u_hat2L = state

                # Cranck-Nicholson + Adams-Bashforth
                u_hatN =  CN2 * ( CN1 * u_hat + ( 1.5 * FN * u_hat2 - 0.5 * FN * u_hat2L ) * dt)
                u_hatN = u_hat2N.at[dealias].set[0]
                
                # go back to real space
                uN      = nx * jnp.real(ifft(u_hatN))
                u_hat2N = (1/nx) * fft(uN**2)

                return (uN, u_hatN, u_hat2N, u_hat2)

            # compute solution through time via finite difference method in Fourier space
            u = jax.lax.fori_loop(1,nt-1,finite_step,(u, u_hat, u_hat2, u_hat2L))[0]
            return jnp.real(u)
  
    return propagator

def PO_autocorrelation(u,nt,RAM_limited=True,exclusion=0.1):
    if RAM_limited:
        bigdiff = np.empty((nt,nt))
        mindiff = np.inf
        minind  = (0,0)
        x       = np.arange(0,nt,1)
        for i in range(nt):
            diff = np.linalg.norm(u[:] - u[i], axis=-1)
            bigdiff[i] = diff 
            diff[abs(x-i)<exclusion*nt] += np.inf
            if np.min(diff) < mindiff:
                mindiff = np.min(diff)
                minind  = (i,np.argmin(diff))
        
        return min(minind), max(minind), bigdiff
    
    else:
        diff = np.linalg.norm(u[:,np.newaxis] - u[np.newaxis,:],axis=-1)
        bigdiff = np.copy(diff)
        x, y = np.mgrid[0:1:(1j*nt),0:1:(1j*nt)]
        diff[abs(x-y)<0.1] += np.inf
        a = np.argmin(diff) % nt
        b = np.argmin(diff) // nt
        return min(a,b), max(a,b), bigdiff
