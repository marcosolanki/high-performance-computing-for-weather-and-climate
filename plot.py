import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_field_from_file(filename):
    nx, ny, nz = pd.read_csv(filename, nrows=1, dtype=int, header=None).to_numpy().reshape(-1)
    data = pd.read_csv(filename, skiprows=1, dtype=np.float64, header=None).to_numpy().reshape(-1)
    return np.reshape(data, (nz, ny, nx))


def validate_results():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    in_field = read_field_from_file('in_field.csv')
    im1 = axs[0].imshow(in_field[in_field.shape[0] // 2, :, :], origin='lower', vmin=-0.1, vmax=1.1)
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title('Initial condition')
    out_field = read_field_from_file('out_field.csv')
    im2 = axs[1].imshow(out_field[out_field.shape[0] // 2, :, :], origin='lower', vmin=-0.1, vmax=1.1)
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title('Final result')
    plt.show()


if __name__ == '__main__':
    validate_results()
