# ******************************************************
#     Program: stencil2d-cupy
#      Author: Stefano Ubbiali, Oliver Fuhrer
#       Email: subbiali@phys.ethz.ch, ofuhrer@ethz.ch
#        Date: 04.06.2020
# Description: CuPy implementation of 4th-order diffusion
# ******************************************************
import click
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

try:
    import cupy as xp
    print('Running on GPU with cupy.')
except ImportError:
    xp = np
    print('Running on CPU with numpy.')


def get_asnumpy(z):
    try:
        return z.get()
    except AttributeError:
        return z


def laplacian(in_field, lap_field, num_halo, extend=0):
    """ Compute the Laplacian using 2nd-order centered differences.

    Parameters
    ----------
    in_field : array-like
        Input field (nz x ny x nx with halo in x- and y-direction).
    lap_field : array-like
        Result (must be same size as ``in_field``).
    num_halo : int
        Number of halo points.
    extend : `int`, optional
        Extend computation into halo-zone by this number of points.
    """
    ib = num_halo - extend
    ie = -num_halo + extend
    jb = num_halo - extend
    je = -num_halo + extend

    lap_field[:, jb:je, ib:ie] = (
        -4.0 * in_field[:, jb:je, ib:ie]
        + in_field[:, jb:je, ib - 1 : ie - 1]
        + in_field[:, jb:je, ib + 1 : ie + 1 if ie != -1 else None]
        + in_field[:, jb - 1 : je - 1, ib:ie]
        + in_field[:, jb + 1 : je + 1 if je != -1 else None, ib:ie]
    )


def halo_update(field, num_halo):
    """ Update the halo-zone using an up/down and left/right strategy.

    Parameters
    ----------
    field : array-like
        Input/output field (nz x ny x nx with halo in x- and y-direction).
    num_halo : int
        Number of halo points.
    
    Note
    ----
        Corners are updated in the left/right phase of the halo-update.
    """
    # bottom edge (without corners)
    field[:, :num_halo, num_halo:-num_halo] = field[:, -2*num_halo:-num_halo, num_halo:-num_halo]

    # top edge (without corners)
    field[:, -num_halo:, num_halo:-num_halo] = field[:, num_halo:2*num_halo, num_halo:-num_halo]

    # left edge (including corners)
    field[:, :, :num_halo] = field[:, :, -2*num_halo:-num_halo]

    # right edge (including corners)
    field[:, :, -num_halo:] = field[:, :, num_halo:2*num_halo]


def apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=1):
    """ Integrate 4th-order diffusion equation by a certain number of iterations.

    Parameters
    ----------
    in_field : array-like
        Input field (nz x ny x nx with halo in x- and y-direction).
    lap_field : array-like
        Result (must be same size as ``in_field``).
    alpha : float
        Diffusion coefficient (dimensionless).
    num_iter : `int`, optional
        Number of iterations to execute.
    """
    tmp_field = xp.empty_like(in_field)

    for n in range(num_iter):
        halo_update(in_field, num_halo)

        laplacian(in_field, tmp_field, num_halo=num_halo, extend=1)
        laplacian(tmp_field, out_field, num_halo=num_halo, extend=0)

        out_field[:, num_halo:-num_halo, num_halo:-num_halo] = (
            in_field[:, num_halo:-num_halo, num_halo:-num_halo]
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]
        )

        if n < num_iter - 1:
            in_field, out_field = out_field, in_field
        else:
            halo_update(out_field, num_halo)


@click.command()
@click.option(
    '--nx', type=int, required=True, help='Number of gridpoints in x-direction'
)
@click.option(
    '--ny', type=int, required=True, help='Number of gridpoints in y-direction'
)
@click.option(
    '--nz', type=int, required=True, help='Number of gridpoints in z-direction'
)
@click.option('--num_iter', type=int, required=True, help='Number of iterations')
@click.option(
    '--num_halo',
    type=int,
    default=2,
    help='Number of halo-pointers in x- and y-direction',
)
@click.option(
    '--plot_result', type=bool, default=False, help='Make a plot of the result?'
)
def main(nx, ny, nz, num_iter, num_halo=3, plot_result=False):
    """Driver for apply_diffusion that sets up fields and does timings"""

    assert 0 < nx <= 1024 * 1024, 'You have to specify a reasonable value for nx'
    assert 0 < ny <= 1024 * 1024, 'You have to specify a reasonable value for ny'
    assert 0 < nz <= 1024, 'You have to specify a reasonable value for nz'
    assert 0 < num_iter <= 1024 * 1024, 'You have to specify a reasonable value for num_iter'
    assert 2 <= num_halo <= 256, 'Your have to specify a reasonable number of halo points'

    alpha = 1.0 / 32.0

    xsize = nx + 2 * num_halo
    ysize = ny + 2 * num_halo
    zsize = nz

    in_field = xp.zeros((zsize, ysize, xsize))

    imin = int(0.25 * zsize + 0.5)
    imax = int(0.75 * zsize + 0.5)
    jmin = int(0.25 * ysize + 0.5)
    jmax = int(0.75 * ysize + 0.5)
    kmin = int(0.25 * xsize + 0.5)
    kmax = int(0.75 * xsize + 0.5)

    in_field[imin:imax+1, jmin:jmax+1, kmin:kmax+1] = 1

    out_field = xp.copy(in_field)

    np.save('in_field', get_asnumpy(in_field))

    if plot_result:
        plt.ioff()
        plt.imshow(get_asnumpy(in_field[in_field.shape[0] // 2, :, :]), origin='lower')
        plt.colorbar()
        plt.savefig('in_field.png')
        plt.close()

    # warmup caches
    apply_diffusion(in_field, out_field, alpha, num_halo)

    # time the actual work
    tic = time.time()
    apply_diffusion(in_field, out_field, alpha, num_halo, num_iter=num_iter)
    toc = time.time()

    print(f'Elapsed time for work = {toc - tic}s.')

    np.save('out_field', get_asnumpy(out_field))

    if plot_result:
        plt.ioff()
        plt.imshow(get_asnumpy(out_field[out_field.shape[0] // 2, :, :]), origin='lower')
        plt.colorbar()
        plt.savefig('out_field.png')
        plt.close()


if __name__ == '__main__':
    main()
