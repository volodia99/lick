#!/usr/bin/env python

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LightSource
from scipy.interpolate import griddata
from skimage.util import random_noise
from skimage import exposure

from licplot import lic_internal

def interpol(
    xx:np.ndarray,
    yy:np.ndarray,
    v1:np.ndarray,
    v2:np.ndarray,
    field:np.ndarray,
    *,
    method:str="linear",
    xmin:Optional[float] = None,
    xmax:Optional[float] = None,
    ymin:Optional[float] = None,
    ymax:Optional[float] = None,
    size_interpolated:Optional[int] = None,
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

def lick(
    v1:np.ndarray, 
    v2:np.ndarray, 
    *, 
    niter_lic:int = 5, 
    kernel_length:int = 101, 
    lightsource:bool = True,
):
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

def lick_box(
    x:np.ndarray, 
    y:np.ndarray, 
    v1:np.ndarray, 
    v2:np.ndarray, 
    field:np.ndarray, 
    *, 
    size_interpolated:Optional[int] = None, 
    xmin:Optional[float] = None, 
    xmax:Optional[float] = None, 
    ymin:Optional[float] = None, 
    ymax:Optional[float] = None, 
    niter_lic:int = 5, 
    kernel_length:int = 101, 
    lightsource:bool = True, 
):
    if ((len(x.shape)==2) and (len(y.shape)==2)):
        yy = y
        xx = x
    elif ((len(x.shape)==1) and (len(y.shape)==1)):
        yy, xx = np.meshgrid(y, x)
    else:
        raise ValueError (
            f"Uncorrect shapes: for 'x' ({len(x.shape)}d array) and for 'y' ({len(y.shape)}d array). Should be 1d or 2d arrays."
        )
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
    fig,
    ax,
    x:np.ndarray, 
    y:np.ndarray, 
    v1:np.ndarray, 
    v2:np.ndarray, 
    field:np.ndarray, 
    *, 
    vmin:Optional[float] = None, 
    vmax:Optional[float] = None, 
    size_interpolated:Optional[int] = None, 
    xmin:Optional[float] = None, 
    xmax:Optional[float] = None, 
    ymin:Optional[float] = None, 
    ymax:Optional[float] = None, 
    niter_lic:int = 5, 
    kernel_length:int = 101, 
    log:bool = False, 
    cmap = None, 
    nbin:Optional[int] = None, 
    density:float = 1.0, 
    color_arrow:str = "w",
    cmap_arrow = None, 
    lightsource:bool = True, 
    streamlines:bool = False, 
    alpha_transparency:bool = True,
    alpha:float = 0.03,
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
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')#, format='%.0e')
    if streamlines:
        strm = ax.streamplot(Xi, Yi, v1i, v2i, density=density, arrowstyle="->", linewidth=0.8, color=color_arrow, cmap=cmap_arrow)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    #print("streamplot")

if __name__ == "__main__":
    cmap = "inferno"
    fig, ax = plt.subplots()
    x = np.geomspace(0.1,10,100)
    y = np.geomspace(0.1,5,100)
    a, b = np.meshgrid(x,y)
    v1 = np.cos(a)
    v2 = np.sin(b)
    field = v1**2+v2**2
    lick_box_plot(fig, ax, x, y, v1, v2, field, cmap=cmap, refinement=5, kernel_length=100, streamlines=True)
    plt.show()
