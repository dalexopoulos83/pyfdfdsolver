#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:14:29 2022

@author: thkam, dimitrisalexopoulos
"""

import numpy as np
import matplotlib.pyplot as plt
from fdfd_2D_solver import yee_grid


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.serif": ["Tahoma"],
})

case = 3

if case == 1:
    #Case 1
    # This parameters are for step index fiber
    n1 = 1.45
    n2 = 0
    n3 = 0
    n_rect = 1.00
    Lx = 15.0
    Ly = 15.0
    outer_radius = 3
    midle_radius = 0
    inner_radius = 0
    di = 0
    target_neff = 1.438604
    l = 1.50
    ntarget = n1
    NN = np.arange(100, 120, 10)
    
elif case == 2:
    #Case 2
    # This are for Microstractured Optical Fiber
    n1 = 1.45
    n2 = 1.42
    n3 = 1.0
    n_rect = 1.42
    Lx = 20.0
    Ly = 20.0
    outer_radius = 2
    midle_radius = 0
    inner_radius = 2
    di = 5
    target_neff = 1.4353607
    l = 1.50
    ntarget = n1
    NN = np.arange(100, 120, 10)
    
elif case == 3:
    #Case 3
    # Cylindrical hybrid plasmonic waveguide - Daru Chen
    n1 = 3.455 # This is the outer cyrcle
    n2 = 1.445 # This is the midle cyrcle
    n3 = 0.1453+11.3587j # This is the inner cyrcle
    n_rect = 1.455
    Lx = 1.5
    Ly = 1.5
    outer_radius = 0.350 # This is the outer cyrcle
    midle_radius = 0.150 # This is the midle cyrcle
    inner_radius = 0.100 # This is the inner cyrcle
    di = 0
    target_neff = 2.3
    l = 1.55
    ntarget = 2.3 
    NN = np.arange(100, 120, 10)


nmodes = 2
ploting_mode = 0
C0 = 3e8
M0 = 1.257e-06


fiber_dict = [
    { 'type' : 'rectangle',
      'x1' : -np.inf,
      'x2' : +np.inf,
      'y1' : -np.inf,
      'y2' : +np.inf,
      'e_value_inside' : n_rect ** 2.0,
    },
    { 'type' : 'disk',
      'x0' : 0.0,
      'y0' : 0.0,
      'radius' : outer_radius,
      'e_value_inside' : n1 ** 2.0,
  },
  { 'type' : 'midle_disk',
    'x0' : 0.0,
    'y0' : 0.0,
    'midle_radius' : midle_radius,
    'e_value_inside' : n2 ** 2.0,
  },
  { 'type' : 'inner_disk',
    'theta' : [15, 75, 135, 195, 255, 315],
    'di' : di,  # This is Λ
    'inner_radius' : inner_radius,
    'e_value_inside' : n3 ** 2.0,
  }
]
 

neff_no_avg = np.zeros([NN.size, nmodes], dtype = complex)
neff_tensor = np.zeros([NN.size, nmodes], dtype = complex)
neff_tensor_pml = np.zeros([NN.size, nmodes], dtype = complex)
neff_straight = np.zeros([NN.size, nmodes], dtype = complex)
neff_inverse = np.zeros([NN.size, nmodes], dtype = complex)


for i, N in enumerate(NN):
    Nx = N
    Ny = N
    Dx = Lx / Nx
    Dy = Ly / Ny
    xmin = -Nx / 2 * Dx
    ymin = -Ny / 2 * Dy 
    omega = 2 * np.pi * C0 / l
    
    
# -----------------------------------------------------------------------------  
    Yt = yee_grid(Nx, Ny, Dx, Dy, fiber_dict,
                  xmin = xmin, ymin = ymin, omega = omega, averaging = 'tensor',
                  nmodes = nmodes, ntarget = ntarget)
    Yt.solve()
    neff_tensor[i, :] = Yt.neff_q

    print('N = ', N, 'neff tensor = ', neff_tensor[i,:])
    
# -----------------------------------------------------------------------------
    Yn = yee_grid(Nx, Ny, Dx, Dy, fiber_dict,
                  xmin = xmin, ymin = ymin, omega = omega, averaging = 'none',
                  nmodes = nmodes, ntarget = ntarget)

    Yn.solve()
    neff_no_avg[i,:] = Yn.neff_q
    
    print('N = ', N, 'neff no avg = ', neff_no_avg[i,:])
    
# -----------------------------------------------------------------------------
    Ys = yee_grid(Nx, Ny, Dx, Dy, fiber_dict,
                xmin = xmin, ymin = ymin, omega = omega, averaging = 'straight',
                nmodes = nmodes, ntarget = ntarget)
    Ys.solve()
    neff_straight[i, :] = Ys.neff_q
    print('N = ', N, 'neff straight = ', neff_straight[i,:])
   
# -----------------------------------------------------------------------------
    Yi = yee_grid(Nx, Ny, Dx, Dy, fiber_dict,
                 xmin = xmin, ymin = ymin, omega = omega, averaging = 'inverse',
                 nmodes = nmodes, ntarget = ntarget)
    Yi.solve()
    neff_inverse[i, :] = Yi.neff_q
    print('N = ', N, 'neff inverse = ', neff_inverse[i,:])

# -----------------------------------------------------------------------------

plt.figure()
plt.plot(NN[:i],np.real(neff_tensor[:i,0]), 'r-*', label = 'tensor averaging')
plt.plot(NN[:i],np.real(neff_straight[:i,0]), 'm--', label = '$<\epsilon>^{-1}$')
plt.plot(NN[:i],np.real(neff_inverse[:i,0]), 'b--', label = '$<\epsilon^{-1}>$')
plt.plot(NN[:i],np.real(neff_no_avg[:i,0]), 'c', label = 'no averaging')

plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
plt.tight_layout()




