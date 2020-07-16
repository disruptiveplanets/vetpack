"""Centroid inspection."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


def plot_pipeline_mask(ax, tpf):
    for i, j in np.ndindex(tpf.pipeline_mask.shape):
        if tpf.pipeline_mask[i, j]:
            ax.add_patch(
                patches.Rectangle(
                    (j+tpf.column, i+tpf.row), 1, 1,
                    color='pink', fill=True, alpha=0.5
                )
            )

    return ax


def median_image(tpf, ax=None):
    # Target position in the TPF
    radec = np.vstack([tpf.ra, tpf.dec]).T
    coords = tpf.wcs.all_world2pix(radec, 0)
    tx = coords[0][0]+tpf.column
    ty = coords[0][0]+tpf.row

    # Limits of the plot
    xlim = (tpf.column, tpf.column+tpf.shape[1])
    ylim = (tpf.row, tpf.row+tpf.shape[2])
    extent = (
        tpf.column, tpf.column+tpf.shape[1], tpf.row, tpf.row+tpf.shape[2]
    )

    # Calculate the median difference image
    med_image = np.nanmedian(tpf.flux, axis=0)

    # Plot it
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(
        med_image, extent=extent, interpolation='nearest', origin='lower'
    )
    ax.plot(tx, ty, 'ro', alpha=0.5)
    plot_pipeline_mask(ax, tpf)
    ax.set_xlabel('Pixel Column')
    ax.set_ylabel('Pixel Row')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.colorbar(im, label='Flux (e$^{-1} s^{-1}$)')

    return ax
