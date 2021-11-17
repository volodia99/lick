#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LightSource
from scipy.interpolate import griddata
from skimage.util import random_noise
from skimage import exposure

from licplot import lic_internal

def interpol(
    xx,
    yy,
    v1,
    v2,
    field,
    *,
    method="linear",
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    size_interpolated=None,
):
    if xmin is None:
        xmin = xx.min()
    if xmax is None:
        xmax = xx.max()
    if ymin is None:
        ymin = yy.min()
    if ymax is None:
        ymax = yy.max()

    if size_interpolated is None:
        size_interpolated = 800

    # evenly spaced grid (same spacing in x and y directions)
    nyi = size_interpolated
    nxi = int((xmax-xmin)/(ymax-ymin)*nyi)
    if nxi < nyi:
        nxi = size_interpolated
        nyi = int((ymax-ymin)/(xmax-xmin)*nxi)

    x = np.linspace(xmin, xmax, nxi)
    y = np.linspace(ymin, ymax, nyi)

    xi, yi = np.meshgrid(x, y)

    # then, interpolate your data onto this grid:

    px = xx.flatten()
    py = yy.flatten()
    pv1 = v1.flatten()
    pv2 = v2.flatten()
    pfield = field.flatten()

    gv1 = griddata((px, py), pv1, (xi, yi), method=method)
    gv2 = griddata((px, py), pv2, (xi, yi), method=method)
    gfield = griddata((px, py), pfield, (xi, yi), method="nearest")

    return (x, y, gv1, gv2, gfield)

def lick(v1, v2, *, niter_lic=5, kernel_length=101, lightsource=True):
    v1 = v1.astype(np.float32)
    v2 = v2.astype(np.float32)
    texture = random_noise(
            np.zeros((v1.shape[0], v1.shape[1])),
            mode="gaussian",
            mean=0.5,
            var=0.001,
            seed=0,
    ).astype(np.float32)
    kernel = np.sin(np.arange(kernel_length) * np.pi / kernel_length).astype(np.float32)
    #kernel = np.ones(70).astype(np.float32)
    image = lic_internal.line_integral_convolution(v1, v2, texture, kernel)
    for _ in range(niter_lic - 1):
        image = lic_internal.line_integral_convolution(
            v1, v2, image.astype(np.float32), kernel
        )

    image = exposure.equalize_hist(image)
    image /= image.max()

    if lightsource:
        # Illuminate the scene from the northwest
        ls = LightSource(azdeg=0, altdeg=45)
        image = ls.hillshade(image, vert_exag=5)

    return image

# xmin = 0.3
# xmax=2.5
# ymin=-0.8
# ymax=0.8
# refinement = 3
# niter_lic = 10
# kernel_length=70
# log = False
# cmap = cb.cbmap("cb.extreme_rainbow")
# vmin = -6.4
# vmax = 0
# density = 3
# fig, ax = plt.subplots()
# lightsource=True, 
# streamlines=False, 
# alpha_transparency=True,

def lick_box(
    x, 
    y, 
    v1, 
    v2, 
    field, 
    *, 
    size_interpolated=None, 
    xmin=None, 
    xmax=None, 
    ymin=None, 
    ymax=None, 
    niter_lic=5, 
    kernel_length=101, 
    lightsource=True, 
):
    yy, xx = np.meshgrid(y, x)
    xi, yi, v1i, v2i, fieldi = interpol(
            xx,
            yy,
            v1,
            v2,
            field,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            size_interpolated=size_interpolated,
    )
    Xi, Yi = np.meshgrid(xi, yi)
    licv = lick(v1i, v2i, niter_lic=niter_lic, kernel_length=kernel_length, lightsource=lightsource)
    return(Xi, Yi, v1i, v2i, fieldi, licv)

def lick_box_plot(
    ax,
    x, 
    y, 
    v1, 
    v2, 
    field, 
    *, 
    vmin=None, 
    vmax=None, 
    size_interpolated=None, 
    xmin=None, 
    xmax=None, 
    ymin=None, 
    ymax=None, 
    niter_lic=5, 
    kernel_length=101, 
    log=False, 
    cmap=None, 
    nbin=None, 
    density=1, 
    color_arrow="w",
    cmap_arrow="gray_r", 
    lightsource=True, 
    streamlines=False, 
    alpha_transparency=True,
    alpha=0.03,
    **kwargs, 
):
    Xi, Yi, v1i, v2i, fieldi, licv = lick_box(
        x, 
        y, 
        v1, 
        v2, 
        field, 
        size_interpolated=size_interpolated, 
        xmin=xmin, 
        xmax=xmax, 
        ymin=ymin, 
        ymax=ymax, 
        niter_lic=niter_lic, 
        kernel_length=kernel_length, 
        lightsource=lightsource, 
    )

    # print("lic function")
    if log:
        # print(f"{log=}")
        if not alpha_transparency:
            datalicv = np.log10(licv*fieldi)
        fieldi = np.log10(fieldi)
    else:
        # print(f"{log=}")
        if not alpha_transparency:
            f = 0.
            datalicv = (licv+f)*fieldi
        
    if vmin is None:
        vmin = fieldi.min()
        # print(f"{vmin=}")
    if vmax is None:
        vmax = fieldi.max()
        # print(f"{vmax=}")

    if alpha_transparency:
        im = ax.pcolormesh(Xi, Yi, fieldi, cmap=cmap, shading="nearest", vmin=vmin, vmax=vmax, rasterized=True)
        im.set_edgecolor('face')
        im2 = ax.pcolormesh(Xi, Yi, licv, cmap="gray", shading="nearest", alpha=alpha, rasterized=True)
        im2.set_edgecolor('face')
    else:
        im = ax.pcolormesh(Xi, Yi, datalicv, cmap=cmap, shading="nearest", vmin=vmin, vmax=vmax)

    # print("pcolormesh")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')#, format='%.0e')
    if streamlines:
        strm = ax.streamplot(Xi, Yi, v1i, v2i, density=density, arrowstyle="->", linewidth=0.8, color=color_arrow, cmap=cmap_arrow)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    #print("streamplot")


def test_lick_box_plot(cmap=None):
    plt.close("all")
    fig, ax = plt.subplots()
    x = np.geomspace(0.1,10,100)
    y = np.geomspace(0.1,5,100)
    a, b = np.meshgrid(x,y)
    v1 = np.cos(a)
    v2 = np.sin(b)
    field = v1**2+v2**2
    lick_box_plot(ax, x, y, v1, v2, field, cmap=cmap, refinement=5, kernel_length=100, streamlines=True)
    plt.show()
