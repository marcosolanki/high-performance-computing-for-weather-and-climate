# ******************************************************
#     Program: stencil2d-gt4py
#      Author: Stefano Ubbiali
#       Email: subbiali@phys.ethz.ch
#        Date: 04.06.2020
# Description: GT4Py implementation of 4th-order diffusion
# ******************************************************
import time
import click
import numpy as np
import matplotlib.pyplot as plt
import gt4py as gt

try:
    # Modern GT4Py (as is available on PyPI):
    import gt4py.cartesian.gtscript as gtscript
    legacy_api = False
except ImportError:
    # Ancient GT4Py (as is installed on Piz Daint):
    import gt4py.gtscript as gtscript
    legacy_api = True


@gtscript.function
def laplacian(in_field):
    lap_field = (
        -4 * in_field[0, 0, 0]
        + in_field[-1, 0, 0]
        + in_field[1, 0, 0]
        + in_field[0, -1, 0]
        + in_field[0, 1, 0])
    return lap_field


def diffusion_defs(in_field: gtscript.Field[float], out_field: gtscript.Field[float], *, alpha: float):

    from __externals__ import laplacian
    from __gtscript__ import PARALLEL, computation, interval

    with computation(PARALLEL), interval(...):
        lap = laplacian(in_field)
        laplap = laplacian(lap)
        out_field = in_field - alpha * laplap


def update_halo(field, num_halo):
    # Bottom edge (without corners):
    field[num_halo:-num_halo, :num_halo, :] = field[num_halo:-num_halo, -2*num_halo:-num_halo, :]

    # Top edge (without corners):
    field[num_halo:-num_halo, -num_halo:, :] = field[num_halo:-num_halo, num_halo:2*num_halo, :]

    # Left edge (including corners):
    field[:num_halo, :, :] = field[-2*num_halo:-num_halo, :, :]

    # Right edge (including corners):
    field[-num_halo:, :, :] = field[num_halo:2*num_halo, :, :]


def apply_diffusion(diffusion_stencil, in_field, out_field, alpha, num_halo, num_iter=1):
    
    # Origin and extent of the computational domain:
    origin = (num_halo, num_halo, 0)
    domain = (in_field.shape[0] - 2 * num_halo, in_field.shape[1] - 2 * num_halo, in_field.shape[2])

    for n in range(num_iter // 2):
        # Halo update:
        update_halo(in_field, num_halo)

        # Run the stencil:
        diffusion_stencil(in_field=in_field, out_field=out_field, alpha=alpha, origin=origin, domain=domain)

        # Halo update:
        update_halo(out_field, num_halo)

        # Run the stencil:
        diffusion_stencil(in_field=out_field, out_field=in_field, alpha=alpha, origin=origin, domain=domain)

    # Halo update:
    update_halo(in_field, num_halo)

    if num_iter % 2 == 1:
        # Run the stencil:
        diffusion_stencil(in_field=in_field, out_field=out_field, alpha=alpha, origin=origin, domain=domain)

        # Halo update:
        update_halo(out_field, num_halo)

        # Right edge update (for some reason necessary to make results match with CuPy):
        out_field[-num_halo:, :, :] = out_field[num_halo:2*num_halo, :, :]

    else:
        # Right edge update (for some reason necessary to make results match with CuPy):
        in_field[-num_halo:, :, :] = in_field[num_halo:2*num_halo, :, :]


@click.command()
@click.option('--nx', type=int, required=True, help='Number of gridpoints in x-direction')
@click.option('--ny', type=int, required=True, help='Number of gridpoints in y-direction')
@click.option('--nz', type=int, required=True, help='Number of gridpoints in z-direction')
@click.option('--num_iter', type=int, required=True, help='Number of iterations')
@click.option('--num_halo', type=int, default=2, help='Number of halo-points in x- and y-direction')
@click.option('--backend', type=str, required=False, default='numpy', help='GT4Py backend')
@click.option('--plot_result', type=bool, default=False, help='Make a plot of the result?')

def main(nx, ny, nz, num_iter, num_halo=2, backend='numpy', plot_result=False):
    """Driver for apply_diffusion that sets up fields and does timings."""

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

    in_field_np = np.zeros((xsize, ysize, zsize), dtype=float)

    # Prepare input field:
    imin = int(0.25 * xsize + 0.5)
    imax = int(0.75 * xsize + 0.5)
    jmin = int(0.25 * ysize + 0.5)
    jmax = int(0.75 * ysize + 0.5)

    in_field_np[imin:imax+1, jmin:jmax+1, :] = 1
    
    # Write input field to file:
    np.save('in_field', np.swapaxes(in_field_np, 0, 2))

    if plot_result:
        # Plot initial field:
        plt.ioff()
        plt.imshow(in_field_np[:, :, 0], origin='lower')
        plt.colorbar()
        plt.savefig('in_field.png')
        plt.close()
    
    # Compile diffusion stencil:
    diffusion_stencil = gtscript.stencil(
        definition=diffusion_defs,
        backend=backend,
        externals={'laplacian': laplacian},
        rebuild=False)
    
    # Timed region:
    tic = time.time()
    
    if legacy_api:
        in_field = gt.storage.from_array(backend=backend, default_origin=(num_halo, num_halo, 0), data=in_field_np)
        out_field = gt.storage.empty(backend=backend, default_origin=(num_halo, num_halo, 0),
                                     shape=(nx+2*num_halo, ny+2*num_halo, nz), dtype=float)
    else:
        in_field = gt.storage.from_array(backend=backend, data=in_field_np)
        out_field = gt.storage.empty(backend=backend, shape=(nx+2*num_halo, ny+2*num_halo, nz), dtype=float)

    apply_diffusion(diffusion_stencil, in_field, out_field, alpha, num_halo, num_iter=num_iter)
    
    if legacy_api:
        out_field_np = np.asarray(in_field if num_iter % 2 == 0 else out_field)
    else:
        out_field_np = (in_field if num_iter % 2 == 0 else out_field).get()
    
    toc = time.time()
    print(f'Elapsed time for work = {toc - tic}s.')

    # Save output field:
    np.save('out_field', np.swapaxes(out_field_np, 0, 2))

    if plot_result:
        # Plot output field:
        plt.ioff()
        plt.imshow(out_field[:, :, 0], origin='lower')
        plt.colorbar()
        plt.savefig('out_field.png')
        plt.close()


if __name__ == '__main__':
    main()
