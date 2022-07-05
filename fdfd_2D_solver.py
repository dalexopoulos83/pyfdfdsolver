import sys
import traceback
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack, eye
from scipy.sparse.linalg import eigs

DPHI_ORTH_VECTORS = np.pi / 25
E0 = 8.854e-12
M0 = 1.257e-06
C0 = 3e8



def calc_dist_e(calldict, x, y):
    xr = x.reshape(-1)
    yr = y.reshape(-1)
    e = np.ones(xr.size, dtype=complex)

    for d in calldict:
        if d['type'] == 'disk':
            r = d['radius']
            x0 = d['x0']
            y0 = d['y0']
            v_in = d['e_value_inside']
            ii = np.where((xr - x0) ** 2.0 + (yr - y0) ** 2.0 <= r ** 2.0)
            e[ii] = v_in

        elif d['type'] == 'rectangle':
            x1 = d['x1']
            y1 = d['y1']
            x2 = d['x2']
            y2 = d['y2']
            v_in = d['e_value_inside']
            ii = np.where((x1 <= xr) & (xr < x2) & (y1 <= yr) & (yr < y2))
            e[ii] = v_in

        elif d['type'] == 'midle_disk':
            if d['midle_radius'] > 0:
                rm = d['midle_radius']
                xm = d['x0']
                ym = d['y0']
                v_in = d['e_value_inside']
                ii = np.where((xr - xm) ** 2.0 + (yr - ym) ** 2.0 <= rm ** 2.0)
                e[ii] = v_in

        elif d['type'] == 'inner_disk':
            if d['inner_radius'] > 0:
                ri = d['inner_radius']
                di = d['di']
                v_in = d['e_value_inside']
                theta = d['theta']
                if di > 0:
                    for i in range(len(theta)):
                        rads = np.radians(theta[i])
                        c, s = np.cos(rads), np.sin(rads)
                        xp = (c * di).reshape(-1)
                        yp = (s * di).reshape(-1)
                        ii = np.where((xr - xp) ** 2.0 + (yr - yp) ** 2.0 <= ri ** 2.0)
                        e[ii] = v_in
                else:
                    xp = 0
                    yp = 0
                    ii = np.where((xr - xp) ** 2.0 + (yr - yp) ** 2.0 <= ri ** 2.0)
                    e[ii] = v_in

    return e.reshape(x.shape)


def vectorize(M):
    return M.reshape(-1)


class yee_grid:

    def devectorize(self, v):
        return v.reshape(self.Nx, self.Ny)

    def __init__(self, Nx, Ny, Dx, Dy, calldicts, xmin=0.0, ymin=0.0,
                 voxel_xsize=100, voxel_ysize=100,
                 dPML=5, order=2, R0 = 1e-17, sigma_max=1.0, omega=1.0,
                 averaging='tensor', nmodes=1, ntarget=None):

        self.Nx = Nx
        self.Ny = Ny
        self.Dx = Dx
        self.Dy = Dy
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmin + Nx * Dx - Dx / 2
        self.ymax = ymin + Ny * Dy - Dy / 2
        self.dPML = dPML * Dx
        self.order = order
        self.voxel_xsize = voxel_xsize
        self.voxel_ysize = voxel_ysize
        self.calldicts = calldicts
        self.omega = omega
        self.sigma_max = sigma_max
        self.averaging = averaging
        self.nmodes = nmodes
        self.ntarget = ntarget

        self.ie = np.arange(0, 2 * self.Nx)
        self.je = np.arange(0, 2 * self.Ny)
        self.im = np.arange(0, 2 * self.Nx)
        self.jm = np.arange(0, 2 * self.Ny)

        self.xe = self.x(self.ie)
        self.ye = self.y(self.je)
        self.xm = self.x(self.im)
        self.ym = self.y(self.jm)

        self.yye, self.xxe = np.meshgrid(self.ye, self.xe)
        self.yym, self.xxm = np.meshgrid(self.ym, self.xm)
        self.calc_e()
        self.calc_orth_vectors()

        self.calc_eavg()
        self.calc_tensor()
        self.calc_VU()
        self.calc_matrices()
        
    def ijgrid(self, di=0.0, dj=0.0):
        i = np.arange(di, 2 * self.Nx, 2).astype(int)
        j = np.arange(dj, 2 * self.Ny, 2).astype(int)
        [jj, ii] = np.meshgrid(j, i)
        return [ii, jj]

    def x(self, i):
        return self.xmin + i * self.Dx * 0.5

    def y(self, j):
        return self.ymin + j * self.Dy * 0.5

    def i(self, x):
        p = np.round(2 * (x - self.xmin) / self.Dx)
        return p.astype(int)

    def j(self, y):
        q = np.round(2 * (y - self.ymin) / self.Dy)
        return q.astype(int)

    def xEz(self, i):
        return self.Dx * i

    def yEz(self, j):
        return self.Dy * j

    def xEy(self, i):
        return self.Dx * i

    def yEy(self, j):
        return self.Dy * j + self.Dy / 2.0

    def xEx(self, i):
        return self.Dx * i + self.Dx / 2.0

    def yEx(self, j):
        return self.Dy * j

    def xHz(self, i):
        return self.Dx * i + self.Dx / 2.0

    def yHz(self, j):
        return self.Dy * j + self.Dy / 2.0

    def xHy(self, i):
        return self.Dx * i + self.Dx / 2.0

    def yHy(self, j):
        return self.Dy * j

    def xHx(self, i):
        return self.Dx * i

    def yHx(self, j):
        return self.Dy * j + self.Dy / 2.0

    def calc_e(self, dx=0.0, dy=0.0):
        self.e = calc_dist_e(self.calldicts, self.xxe - dx, self.yye - dy)

    def calc_exy(self, x, y):
        return calc_dist_e(self.calldicts, x, y)
    
    def calc_exy_real(self, x, y):
        e_real = np.real(calc_dist_e(self.calldicts, x, y))
        return e_real
    
    def calc_exy_imag(self, x, y):
        e_imag = np.imag(calc_dist_e(self.calldicts, x, y))
        return e_imag

    def calc_coarse_avg(self):
        if not hasattr(self, 'e'):
            self.calc_e()

        dx = self.Dx / 4.0
        dy = self.Dy / 4.0

        displacements = [(+dx, +dy),
                         (-dx, +dy),
                         (+dx, -dy),
                         (-dx, -dx)]

        self.e_cavg = np.zeros(self.xxe.shape, dtype=complex)
        for dx, dy in displacements:
            self.e_cavg += 0.25 * calc_dist_e(self.calldicts, self.xxe - dx, self.yye - dy)
            
    def calc_boundaries(self):
        if not hasattr(self, 'e_cavg'):
            self.calc_coarse_avg()

        self.ib, self.jb = np.where(self.e != self.e_cavg)
        self.xb = self.x(self.ib)
        self.yb = self.y(self.jb)


    def calc_orth_vectors(self):
        r0 = np.min([self.Dx, self.Dy]) * 0.5

        if not hasattr(self, 'xb'):
            self.calc_boundaries()

        self.nx = 0.5 * np.sqrt(2) * np.ones(self.xxe.shape)
        self.ny = 0.5 * np.sqrt(2) * np.ones(self.xxe.shape)

        for i, ib in enumerate(self.ib):
            x0 = self.xb[i]
            y0 = self.yb[i]
            jb = self.jb[i]

            theta = np.arange(0, 2 * np.pi, DPHI_ORTH_VECTORS)

            xint = x0 + r0 * np.cos(theta)
            yint = y0 + r0 * np.sin(theta)

            integrand_x = self.calc_exy(xint, yint) * (xint - x0)
            integrand_y = self.calc_exy(xint, yint) * (yint - y0)

            nx = np.sum(integrand_x)
            ny = np.sum(integrand_y)

            self.nx[ib, jb] = np.real(nx / np.lib.scimath.sqrt(nx ** 2.0 + ny ** 2.0))
            self.ny[ib, jb] = np.real(ny / np.lib.scimath.sqrt(nx ** 2.0 + ny ** 2.0))

    def voxel_xy(self, x, y):
        xmin = x - self.Dx / 2
        xmax = x + self.Dx / 2
        ymin = y - self.Dy / 2
        ymax = y + self.Dy / 2

        xv = np.linspace(xmin, xmax, self.voxel_xsize)
        yv = np.linspace(ymin, ymax, self.voxel_ysize)
        [yy, xx] = np.meshgrid(yv, xv)
        return xx, yy

    def calc_eavg(self):
        global epmlx
        if not hasattr(self, 'ib'):
            self.calc_boundaries()

        self.eavg_col = np.zeros(self.ib.shape, dtype=complex)
        self.eiavg_col = np.zeros(self.ib.shape, dtype=complex)
        self.eiavg = 1 / np.copy(self.e)
        self.eavg = np.copy(self.e)

        if self.averaging != 'none':
            for i, ib in enumerate(self.ib):
                jb = self.jb[i]
                x0 = self.x(ib)
                y0 = self.y(jb)
                xv, yv = self.voxel_xy(x0, y0)
                exy_real = self.calc_exy_real(xv, yv)
                exy_imag = self.calc_exy_imag(x0, y0)
                self.eavg_col[i] = complex(np.mean(exy_real), exy_imag)
                self.eiavg_col[i] = 1 / complex(1 / np.mean(1 / exy_real), exy_imag)
                self.eavg[ib, jb] = self.eavg_col[i]
                self.eiavg[ib, jb] = self.eiavg_col[i]

    def calc_tensor(self):

        if not hasattr(self, 'eavg'):
            self.calc_eavg()

        if self.averaging == 'tensor':
            self.fyy = 1 / self.eavg[0::2, 1::2] + \
                       self.ny[0::2, 1::2] * self.ny[0::2, 1::2] * (self.eiavg[0::2, 1::2] - 1 / self.eavg[0::2, 1::2])

            self.fyx = self.nx[0::2, 1::2] * self.ny[0::2, 1::2] * \
                       (self.eiavg[0::2, 1::2] - 1 / self.eavg[0::2, 1::2])

            self.fxx = 1 / self.eavg[1::2, 0::2] + \
                       self.nx[1::2, 0::2] * self.nx[1::2, 0::2] * (self.eiavg[1::2, 0::2] - 1 / self.eavg[1::2, 0::2])

            self.fxy = self.nx[1::2, 0::2] * self.ny[1::2, 0::2] * \
                       (self.eiavg[1::2, 0::2] - 1 / self.eavg[1::2, 0::2])

            self.fzz = (1 / self.eavg[0::2, 0::2])

        elif self.averaging == 'none':

            self.fyy = 1 / self.e[0::2, 1::2]
            self.fxx = 1 / self.e[1::2, 0::2]

            self.fxy = np.zeros(self.eiavg.shape)
            self.fyx = np.copy(self.fxy)
            self.fzz = 1 / self.e[0::2, 0::2]
            
        elif self.averaging == 'inverse':
            self.fyy = self.eiavg[0::2, 1::2]    
            self.fxx = self.eiavg[1::2, 0::2]
                       
            self.fxy = np.zeros(self.eiavg.shape)            
            self.fyx = np.zeros(self.eiavg.shape)            
                        
            self.fzz = self.eiavg[0::2, 0::2]
        
        elif self.averaging == 'straight':
     
            self.fyy = 1 / self.eavg[0::2, 1::2]    
            self.fxx = 1 / self.eavg[1::2, 0::2]
                       
            self.fxy = np.zeros(self.eavg.shape)            
            self.fyx = np.zeros(self.eavg.shape)            
            
            self.fzz = 1 / self.eavg[0::2, 0::2]

    def s_diags(self, dql, vl):

        p = np.array([], dtype=int)
        q = np.array([], dtype=int)
        d = np.array([], dtype=int)

        for dq, v in zip(dql, vl):
            p1, q1, _, _ = self.pq(dq)
            v = np.array(v)
            if v.size == 1:
                v = np.ones(p1.size) * v

            p = np.concatenate((p, p1))
            q = np.concatenate((q, q1))
            d = np.concatenate((d, v))

        return csr_matrix((d, (p, q)))

    def pq_slow(self, dq):
        N = self.Nx * self.Ny
        pss = []
        qss = []
        iss = []
        jss = []

        for p in np.arange(0, N, dtype=int):
            q = p - dq
            if q >= 0 and q < N:
                pss.append(p)
                qss.append(q)
                i = int(np.floor(p / self.Ny))
                j = int(p - i * self.Ny)
                iss.append(i)
                jss.append(j)

        return np.array(pss), np.array(qss), np.array(iss), np.array(jss)

    def pq(self, dq):

        N = self.Nx * self.Ny

        if dq >= 0:
            q = np.arange(0, N - dq)
            p = dq + q
        else:
            q = np.arange(-dq, N)
            p = q + dq

        i = np.floor(p / self.Ny)
        j = p - i * self.Ny
        return p.astype(int), q.astype(int), i.astype(int), j.astype(int)

    def s_diags_2D(self, dql, fl):
        ps = np.array([], dtype=int)
        qs = np.array([], dtype=int)
        ds = np.array([], dtype=int)

        for dq, f in zip(dql, fl):
            p, q, _, _ = self.pq(dq)
            f = np.array([f])
            if f.size == 1:
                fv = f[0] * np.ones(p.size)
            else:
                fv = vectorize(f)[0:p.size]

            ds = np.concatenate((ds, fv))
            ps = np.concatenate((ps, p))
            qs = np.concatenate((qs, q))

        return csr_matrix((ds, (ps, qs)))

    def calc_VU(self):
        self.Uy = self.s_diags([0, -1], [-1 / self.Dy, +1 / self.Dy])
        self.Ux = self.s_diags([0, -self.Ny], [-1 / self.Dx, +1 / self.Dx])
        self.Vy = self.s_diags([0, 1], [1 / self.Dy, -1 / self.Dy])
        self.Vx = self.s_diags([0, self.Ny], [1 / self.Dx, -1 / self.Dx])

    def calc_sF(self):
        self.iFxx = self.s_diags_2D([0], [1 / self.fxx])
        self.iFyy = self.s_diags_2D([0], [1 / self.fyy])
        self.Fxx = self.s_diags_2D([0], [self.fxx])
        self.Fyy = self.s_diags_2D([0], [self.fyy])
        self.Fzz = self.s_diags_2D([0], [self.fzz])
        self.Sxy = self.s_diags_2D([0, -self.Ny, +1, -self.Ny + 1],
                                   [0.25, 0.25, 0.25, 0.25])
        self.Syx = self.s_diags_2D([0, +self.Ny, -1, +self.Ny - 1],
                                   [0.25, 0.25, 0.25, 0.25])
        self.fxyd = self.s_diags_2D([0], [self.fxy])
        self.fyxd = self.s_diags_2D([0], [self.fyx])
        self.Fxy = self.fxyd * self.Sxy
        self.Fyx = self.fyxd * self.Syx
        
        self.ifxyd = self.s_diags_2D([0], [np.linalg.pinv(self.fxy) ])
        self.ifyxd = self.s_diags_2D([0], [np.linalg.pinv(self.fyx) ])
        self.iFxy = self.ifxyd * self.Sxy
        self.iFyx = self.ifyxd * self.Syx


    def calc_sQB(self):
        I = eye(self.Nx * self.Ny)
        w = self.omega   
        self.Qxx = w**2.0 / C0**2.0 * I + self.Uy * self.Fzz * self.Vy + self.Fyy * self.Vx * self.Ux \
                    - self.Fyx * self.Vy * self.Ux

        self.Qyy = w**2.0 / C0**2.0 * I + self.Ux * self.Fzz * self.Vx + self.Fxx * self.Vy * self.Uy \
                    - self.Fxy * self.Vx * self.Uy

        self.Qxy = - self.Uy * self.Fzz * self.Vx + self.Fyy * self.Vx * self.Uy \
                    - self.Fyx * self.Vy * self.Uy

        self.Qyx = - self.Ux * self.Fzz * self.Vy + self.Fxx * self.Vy * self.Ux \
                    - self.Fxy * self.Vx * self.Ux

        self.Q = vstack((hstack( (self.Qxx, self.Qxy), format = 'csr'),
                        hstack( (self.Qyx, self.Qyy), format = 'csr')),
                        format = 'csr' )

        self.Bqxx = self.Fyy
        self.Bqyy = self.Fxx
        self.Bqxy = -self.Fyx
        self.Bqyx = -self.Fxy

        self.Bq = vstack((hstack((self.Bqxx, self.Bqxy), format='csr'),
                         hstack((self.Bqyx, self.Bqyy), format='csr')),
                         format='csr')

    def calc_matrices(self):
        self.calc_sF()
        self.calc_sQB()

    def solve(self):
        try:
            beta0 = self.omega / C0 * self.ntarget
            self.targ = beta0 ** 2.0
            self.beta0 = beta0
            self.wq, self.vq = eigs(self.Q, self.nmodes, M=self.Bq, sigma=self.targ)
            self.k0 = self.omega / C0
            self.neff_q = np.emath.sqrt(self.wq / self.k0 ** 2.0)
        except:
            # printing stack trace
            traceback.print_exception(*sys.exc_info())

            

            
            
            
               