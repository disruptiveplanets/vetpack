"""Centroid inspection."""
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import transitleastsquares as tls
from astropy.time import Time
from matplotlib import patches
from scipy.signal import find_peaks


def median_image(tpf, ax=None):
    """
    Plot the median image from a target pixel file.

    Parameters
    ----------
    tpf : `~lightkurve.TargetPixelFile`
        A target pixel file object.

    ax : `~matplotlib.axes.Axes` or None, optional.
        The `~matplotlib.axes.Axes` object to be drawn on.
        If None, creates a new ``Figure`` and ``Axes``.

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        An ``Axes`` object with the median image plotted.
    """
    # Target position in the TPF
    ra, dec = _pmcorrected_coordinates(tpf)
    radec = np.vstack([ra, dec]).T
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
    ax = _plot_pipeline_mask(ax, tpf)
    ax.set_xlabel('Pixel Column')
    ax.set_ylabel('Pixel Row')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.colorbar(im, label='Flux (e$^{-1} s^{-1}$)')

    return ax


def difference_image(tpf, t0, period, duration, ax=None):
    """
    Plot the median in/out transit difference image from a target pixel file.

    Parameters
    ----------
    tpf : `~lightkurve.TargetPixelFile`
        A target pixel file object.

    t0 : float
        The central transit time.

    period : float
        The orbital period.

    duration : float
        The duration of the transit.

    ax : `~matplotlib.axes.Axes` or None, optional.
        The `~matplotlib.axes.Axes` object to be drawn on.
        If None, creates a new ``Figure`` and ``Axes``.

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        An ``Axes`` object with the median difference image plotted.
    """
    # Target position in the TPF
    ra, dec = _pmcorrected_coordinates(tpf)
    radec = np.vstack([ra, dec]).T
    coords = tpf.wcs.all_world2pix(radec, 0)
    tx = coords[0][0]+tpf.column
    ty = coords[0][0]+tpf.row

    # Limits of the plot
    xlim = (tpf.column, tpf.column+tpf.shape[1])
    ylim = (tpf.row, tpf.row+tpf.shape[2])
    extent = (
        tpf.column, tpf.column+tpf.shape[1], tpf.row, tpf.row+tpf.shape[2]
    )

    # Identify transits and flanking windows
    in_transit = tls.transit_mask(
        tpf.time, period, 0.9*duration, t0
    )
    before_transit = tls.transit_mask(
        tpf.time, period, 0.9*duration, t0-duration
    )
    after_transit = tls.transit_mask(
        tpf.time, period, 0.9*duration, t0+duration
    )
    oot = before_transit+after_transit
    idx_transits, _ = find_peaks(in_transit)

    # Calculate per-transit difference images
    idx_transits, _ = find_peaks(in_transit)
    diff_images = []
    for i, idx_transit in enumerate(idx_transits):
        tt = tpf.time[idx_transit]
        orbit = np.abs(tpf.time - tt) < 0.5*period
        tpf_in = tpf[in_transit & orbit]
        tpf_out = tpf[oot & orbit]
        med_in = np.nanmedian(tpf_in.flux, axis=0)
        med_out = np.nanmedian(tpf_out.flux, axis=0)
        diff_image = med_in-med_out
        diff_images.append(diff_image)

    # Calculate the median difference image
    med_diff = np.nanmedian(diff_images, axis=0)

    # Plot it
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(
        med_diff, extent=extent, interpolation='nearest', origin='lower'
    )
    ax.plot(tx, ty, 'ro', alpha=0.5)
    ax = _plot_pipeline_mask(ax, tpf)
    ax.set_xlabel('Pixel Column')
    ax.set_ylabel('Pixel Row')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.colorbar(im, label='Flux (e$^{-1} s^{-1}$)')

    return ax


def _plot_pipeline_mask(ax, tpf):
    """
    Plot the pipeline aperture mask.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~matplotlib.axes.Axes` object to be drawn on.

    tpf : `~lightkurve.TargetPixelFile`
        The relevant target pixel file object.

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        An ``Axes`` object with the pipeline mask overplotted.
    """
    for i, j in np.ndindex(tpf.pipeline_mask.shape):
        if tpf.pipeline_mask[i, j]:
            ax.add_patch(
                patches.Rectangle(
                    (j+tpf.column, i+tpf.row), 1, 1,
                    color='pink', fill=True, alpha=0.5
                )
            )

    return ax


def _pmcorrected_coordinates(tpf):
    """
    Get the proper-motion-corrected coordinates of the target
    from a target pixel file header.

    Parameters
    ----------
    tpf : `~lightkurve.TargetPixelFile`
        A target pixel file object.

    Returns
    -------
    ra : float
        The proper-motion-corrected right ascension in degrees.

    dec : float
        The proper-motion-corrected declination in degrees.
    """
    # Calculate time since J2000
    h = tpf.get_header()
    t_start = Time(h['DATE-OBS'])
    t_end = Time(h['DATE-END'])
    t_obs = t_start + (t_end-t_start)/2
    dt = ((t_obs.jd - Time('J2000').jd)*u.day).to(u.year)

    # Apply proper motion correction
    pmra = ((h['PMRA']*u.mas/u.yr)*dt).to(u.deg).value
    pmdec = ((h['PMDEC']*u.mas/u.yr)*dt).to(u.deg).value
    ra = tpf.ra + pmra
    dec = tpf.dec + pmdec

    return ra, dec
