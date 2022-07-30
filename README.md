# pyfdfdsolver

This is a 2D Finite Difference Frequency Domain mode solver with dielectric tensor smoothing for photonic waveguides. 

We have implement three different flavors of averaging: 
- Tensor Averaging 
- Straight Averaging
- Inverse Averaging 
- no averaging

The entry point of the application is fdfd_2D.py file, there you can select one of the three implemented structures  you want to run:
1. Step Index Fiber
2. Microstructured Optical Fiber
3. Cylindrical Hybrid Plasmonic Waveguide

If you want to select a specific averaging shema, when you call the yee_grid you have to select one of the following: 
- tensor
- straight
- inverse
- none

Bellow we have some more information for the structure parameters that you can tune according to your needs


    The simulated structures, as the code is implemented, can be cylindrical and for all three of them we can 
    set the radius and the refractive index of the material named outer_radius, middle_radius, inner_radius. 
    For the inner_radius case only we can set the 'di' and theta' parameters and have multiple with a distance 
    di from core center to this cylinder center and the peripheral cylinder arranged on the angles we have set 
    on 'thete'
    
    n1 = 1.45 -> refractive index of outer_radius (cylindrical structure)
    n2 = 1.42 -> refractive index of midle_radius (cylindrical structure)
    n3 = 1.0  -> refractive index of inner_radius (cylindrical structure)
    n_rect = 1.42 -> refractive index surrounding the structure 
    Lx = 20.0 -> computational window x axis size
    Ly = 20.0 -> computational window y axis size
    outer_radius = 2
    middle_radius = 0
    inner_radius = 2
    di = 5 -> for the case of mof this is Λ, the period of air holes in this example. For fibers this must be set to 0
    l = 1.50 -> operational wavelength in μm
    ntarget = n1 -> target n effective for eigenproblem solution 
    NN = np.arange(100, 420, 10) -> grid resolution per axis, it starts from 100 to 420 with step 10
    file_name = './results/ms_fiber/ms_fiber' -> file name to save the results
