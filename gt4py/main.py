# ******************************************************
#     Program: stencil2d-gt4py
#      Author: Stefano Ubbiali
#       Email: subbiali@phys.ethz.ch
#        Date: 04.06.2020
# Description: GT4Py implementation of 4th-order diffusion
# ******************************************************
import click
import gt4py as gt
import gt4py.cartesian.gtscript as gtscript
import matplotlib.pyplot as plt
import numpy as np
import time


@gtscript.function
def laplacian(in_field):
    lap_field = (
        -4.0 * in_field[0, 0, 0]
        + in_field[0, 0, -1]
        + in_field[0, 0, 1]
        + in_field[0, -1, 0]
        + in_field[0, 1, 0]
    )
    return lap_field


def diffusion_defs(
    in_field: gtscript.Field[float],
    out_field: gtscript.Field[float],
    *,
    a1: float,
    a2: float,
    a8: float,
    a20: float,
):
    from __gtscript__ import PARALLEL, computation, interval

    with computation(PARALLEL), interval(...):
        out_field = (
              a1 * in_field[0, -2, 0]
            + a2 * in_field[0, -1, -1]
            + a8 * in_field[0, -1, 0]
            + a2 * in_field[0, -1, 1]
            + a1 * in_field[0, 0, -2]
            + a8 * in_field[0, 0, -1]
            + a20 * in_field[0, 0, 0]
            + a8 * in_field[0, 0, 1]
            + a1 * in_field[0, 0, 2]
            + a2 * in_field[0, 1, -1]
            + a8 * in_field[0, 1, 0]
            + a2 * in_field[0, 1, 1]
            + a1 * in_field[0, 2, 0]
        )


def update_halo(field, num_halo):

    # bottom edge (without corners)
    field[:, :num_halo, num_halo:-num_halo] = field[:, -2 * num_halo : -num_halo, num_halo:-num_halo]

    # top edge (without corners)
    field[:, -num_halo:, num_halo:-num_halo] = field[:, num_halo : 2 * num_halo, num_halo:-num_halo]

    # left edge (including corners)
    field[:, :, :num_halo] = field[:, :, -2 * num_halo : -num_halo]

    # right edge (including corners)
    field[:, :, -num_halo:] = field[:, :, num_halo : 2 * num_halo]


def apply_diffusion(
    diffusion_stencil, in_field, out_field, alpha, num_halo, num_iter=1
):
    # origin and extent of the computational domain
    origin = (0, num_halo, num_halo)
    domain = (
        in_field.shape[0],
        in_field.shape[1] - 2 * num_halo,
        in_field.shape[2] - 2 * num_halo,
    )

    for n in range(num_iter):
        # halo update
        update_halo(in_field, num_halo)

        # run the stencil
        diffusion_stencil(
            in_field=in_field,
            out_field=out_field,
            a1=-alpha,
            a2=-2 * alpha,
            a8=8 * alpha,
            a20=1 - 20 * alpha,
            origin=origin,
            domain=domain,
        )

        if n < num_iter - 1:
            # swap input and output fields
            in_field, out_field = out_field, in_field
        else:
            # halo update
            update_halo(out_field, num_halo)


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
    help='Number of halo-points in x- and y-direction',
)
@click.option(
    '--backend', type=str, required=False, default='numpy', help='GT4Py backend.'
)
@click.option(
    '--plot_result', type=bool, default=False, help='Make a plot of the result?'
)
def main(nx, ny, nz, num_iter, num_halo=2, backend='numpy', plot_result=False):
    """Driver for apply_diffusion that sets up fields and does timings."""

    assert 0 < nx <= 1024 * 1024, 'You have to specify a reasonable value for nx'
    assert 0 < ny <= 1024 * 1024, 'You have to specify a reasonable value for ny'
    assert 0 < nz <= 1024, 'You have to specify a reasonable value for nz'
    assert 0 < num_iter <= 1024 * 1024, 'You have to specify a reasonable value for num_iter'
    assert 2 <= num_halo <= 256, 'You have to specify a reasonable number of halo points'

    alpha = 1.0 / 32.0

    # default origin
    dorigin = (0, num_halo, num_halo)

    # allocate input field
    xsize = nx + 2 * num_halo
    ysize = ny + 2 * num_halo
    zsize = nz

    in_field_np = np.zeros((zsize, ysize, xsize), dtype=float)

    # prepare input field
    imin = int(0.25 * zsize + 0.5)
    imax = int(0.75 * zsize + 0.5)
    jmin = int(0.25 * ysize + 0.5)
    jmax = int(0.75 * ysize + 0.5)
    kmin = int(0.25 * xsize + 0.5)
    kmax = int(0.75 * xsize + 0.5)

    in_field_np[imin:imax+1, jmin:jmax+1, kmin:kmax+1] = 1

    # write input field to file
    # swap first and last axes for compatibility with day1/stencil2d.py
    np.save('in_field', in_field_np)

    if plot_result:
        # plot initial field
        plt.ioff()
        plt.imshow(in_field_np[nz//2, :, :], origin='lower')
        plt.colorbar()
        plt.savefig('in_field.png')
        plt.close()

    # compile diffusion stencil
    kwargs = {'verbose': True} if backend in ('gtx86', 'gtmc', 'gtcuda') else {}
    diffusion_stencil = gtscript.stencil(
        definition=diffusion_defs,
        backend=backend,
        externals={'laplacian': laplacian},
        rebuild=False,
        **kwargs,
    )

    # apply_diffusion(diffusion_stencil, in_field, out_field, alpha, num_halo)

    # time the actual work
    tic = time.time()
    out_field = gt.storage.zeros(
        backend=backend, shape=(nz, ny + 2 * num_halo, nx + 2 * num_halo), dtype=float
    )
    in_field = gt.storage.from_array(data=in_field_np, backend=backend)
    apply_diffusion(
        diffusion_stencil, in_field, out_field, alpha, num_halo, num_iter=num_iter
    )
    out_field_np = out_field.get()
    toc = time.time()
    print(f'Elapsed time for work = {toc - tic}s.')

    # save output field
    # swap first and last axes for compatibility with day1/stencil2d.py
    #np.save('out_field', np.swapaxes(out_field_np, 0, 2))
    np.save('out_field', out_field_np)

    if plot_result:
        # plot the output field
        plt.ioff()
        plt.imshow(out_field_np[nz//2, :, :], origin='lower')
        plt.colorbar()
        plt.savefig('out_field.png')
        plt.close()


if __name__ == '__main__':
    main()
