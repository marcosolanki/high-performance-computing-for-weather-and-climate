# ******************************************************
#     Program: stencil2d-cupy
#      Author: Stefano Ubbiali, Oliver Fuhrer
#       Email: subbiali@phys.ethz.ch, ofuhrer@ethz.ch
#        Date: 04.06.2020
# Description: CuPy implementation of 4th-order diffusion
# ******************************************************
import time
import click
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp


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
    # Bottom edge (without corners):
    field[:, :num_halo, num_halo:-num_halo] = field[:, -2*num_halo:-num_halo, num_halo:-num_halo]

    # Top edge (without corners):
    field[:, -num_halo:, num_halo:-num_halo] = field[:, num_halo:2*num_halo, num_halo:-num_halo]

    # Left edge (including corners):
    field[:, :, :num_halo] = field[:, :, -2*num_halo:-num_halo]

    # Right edge (including corners):
    field[:, :, -num_halo:] = field[:, :, num_halo:2*num_halo]


def apply_diffusion(field, alpha, num_halo, num_iter=1):
    """ Integrate 4th-order diffusion equation by a certain number of iterations.

    Parameters
    ----------
    field : array-like
        Input/output field (nz x ny x nx with halo in x- and y-direction).
    alpha : float
        Diffusion coefficient (dimensionless).
    num_iter : `int`, optional
        Number of iterations to execute.
    """
    lap = cp.empty_like(field)

    for n in range(num_iter):
        # Halo update:
        halo_update(field, num_halo)

        # Run the stencil:
        imin = num_halo - 1
        imax = -num_halo + 1
        jmin = num_halo - 1
        jmax = -num_halo + 1

        lap[:, jmin:jmax, imin:imax] = (
             -4 * field[:, jmin:jmax, imin:imax]
                + field[:, jmin:jmax, imin-1:imax-1]
                + field[:, jmin:jmax, imin+1:imax+1 if imax != -1 else None]
                + field[:, jmin-1:jmax-1, imin:imax]
                + field[:, jmin+1:jmax+1 if jmax != -1 else None, imin:imax])

        imin = num_halo
        imax = -num_halo
        jmin = num_halo
        jmax = -num_halo

        field[:, jmin:jmax, imin:imax] -= alpha * (
             -4 * lap[:, jmin:jmax, imin:imax]
                + lap[:, jmin:jmax, imin-1:imax-1]
                + lap[:, jmin:jmax, imin+1:imax+1]
                + lap[:, jmin-1:jmax-1, imin:imax]
                + lap[:, jmin+1:jmax+1, imin:imax])

    halo_update(field, num_halo)


@click.command()
@click.option('--nx', type=int, required=True, help='Number of gridpoints in x-direction')
@click.option('--ny', type=int, required=True, help='Number of gridpoints in y-direction')
@click.option('--nz', type=int, required=True, help='Number of gridpoints in z-direction')
@click.option('--num_iter', type=int, required=True, help='Number of iterations')
@click.option('--num_halo', type=int, default=2, help='Number of halo-pointers in x- and y-direction')
@click.option('--plot_result', type=bool, default=False, help='Make a plot of the result?')

def main(nx, ny, nz, num_iter, num_halo=2, plot_result=False):
    """Driver for apply_diffusion that sets up fields and does timings"""

    assert 0 < nx <= 1024 * 1024, 'You have to specify a reasonable value for nx'
    assert 0 < ny <= 1024 * 1024, 'You have to specify a reasonable value for ny'
    assert 0 < nz <= 1024, 'You have to specify a reasonable value for nz'
    assert 0 < num_iter <= 1024 * 1024, 'You have to specify a reasonable value for num_iter'
    assert 2 <= num_halo <= 256, 'You have to specify a reasonable number of halo points'

    alpha = 1 / 32

    # Allocate input field:
    xsize = nx + 2 * num_halo
    ysize = ny + 2 * num_halo
    zsize = nz

    field_np = np.zeros((zsize, ysize, xsize))

    # Prepare input field:
    imin = int(0.25 * xsize + 0.5)
    imax = int(0.75 * xsize + 0.5)
    jmin = int(0.25 * ysize + 0.5)
    jmax = int(0.75 * ysize + 0.5)

    field_np[:, jmin:jmax+1, imin:imax+1] = 1

    # Write input field to file:
    np.save('in_field', field_np)

    if plot_result:
        # Plot initial field:
        plt.ioff()
        plt.imshow(field_np[field_np.shape[0] // 2, :, :], origin='lower')
        plt.colorbar()
        plt.savefig('in_field.png')
        plt.close()

    # Timed region:
    tic = time.time()

    field = cp.array(field_np)

    apply_diffusion(field, alpha, num_halo, num_iter=num_iter)

    field_np = field.get()

    toc = time.time()
    print(f'Elapsed time for work = {toc - tic}s.')

    # Save output field:
    np.save('out_field', field_np)

    if plot_result:
        # Plot output field:
        plt.ioff()
        plt.imshow(field_np[field_np.shape[0] // 2, :, :], origin='lower')
        plt.colorbar()
        plt.savefig('out_field.png')
        plt.close()


if __name__ == '__main__':
    main()
