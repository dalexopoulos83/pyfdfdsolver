# pyfdfdsolver

This is a 2D Finite Difference Frequency Domain mode solver for photonic waveguides. It calculates and plots the effective refractive index for Step Index Fiber (SI), Microstructured Optical Fiber (MOF) and Cylindrical Hybrid Plasmonic Waveguide (CHPW). Also we solve and compare different dielectric averaging schema, 1. Tensor Averaging, 2. Straight Averaging, 3. Inverse Averaging and 4. with no averaging.
The entry point of the application is fdfd_2D.py, there you can select for which structure the application want to run:

1: Step Index Fiber
2: Microstructured Optical Fiber
3: Cylindrical Hybrid Plasmonic Waveguide

Then you have to select the structure parameters:

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

If you want to select a specific averaging shema, when you call the yee_grid you have to select one of the following: 'tensor', 'straight', 'inverse' and 'none'.
