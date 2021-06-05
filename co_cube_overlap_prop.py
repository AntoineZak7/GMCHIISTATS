"""""
Python Script to check where a co cube overlaps with an hii region mask, and to investigate the co line values of the overlapping pixels
co_cube, co_mask and hii_mask must have same wcs
"""""

import numpy as np
from spectral_cube import SpectralCube, Projection
import scStackingRoutines as scr
from astropy.io import fits
from astropy import wcs
import astropy.units as u
import matplotlib.pyplot as plt
import aplpy


def co_cube_overlap_prop(co_cube, co_mask, hii_mask, galname, write, show):

    dir_co_cubes = '/home/antoine/data_cubes/shuffled/'
    dir_maps = '/home/antoine/data_cubes/maps/'

    # First make hii and co mask data = 1
    co_mask[0].data[co_mask[0].data != 0.] = 1
    hii_mask[0].data[hii_mask[0].data != 0.] = 1

    # then add mask and check where data == 2
    overlap_mask_data = co_mask[0].data + hii_mask[0].data
    overlap_mask_data[overlap_mask_data.astype(int) != 2] = 0
    overlap_mask_data[overlap_mask_data.astype(int) == 2 ] = 1

    # a) To deal with noise, extract pixels of co_cube that overlap with co_cube GMCs masks
    co_cube = co_cube.with_mask(co_mask[0].data.astype(int) == 1)

    # a) Plot moment 0 map (before removing noise ? in PIR 2.5 std before computing mom1 map) => Use CO mask to select same regions as for catalogs, so no need for a diff s/n treatment
    print('Computing moment 1 map')
    co_mom1_map =co_cube.moment(order=1)

    # b) Suffle cube with mom0 map to align velocities
    print('shuffling cube...')
    co_cube = scr.ShuffleCube(co_cube, co_mom1_map)
    if write:
        co_cube.write(dir_co_cubes+ galname + '_shuffled.fits', overwrite=True)

    # c) Compute co_cube moment maps
    print('Computing cube moment maps...')
    sigma_map_cube = co_cube.linewidth_sigma()

    co_cube.allow_huge_operations = True
    peak_map_cube = co_cube.max(axis = 0)
    peak_map_cube.quicklook()


    # d) Compute new cube by extracting the pixels overlapping with hii mask, and saving it as 'test.fits'
    overlap_cube = co_cube.with_mask(overlap_mask_data.astype(int) == 1)

    if write:
        overlap_cube.write(dir_co_cubes + galname+'_hii_overlap.fits', overwrite=True)

    # e) compute sigma and temperature at peak maps of original and extracted cube

    sigma_map_cube_overlap = overlap_cube.linewidth_sigma()
    overlap_cube.allow_huge_operations = True
    peak_map_overlap = overlap_cube.max(axis=0)
    peak_map_overlap.quicklook()




    print('plotting')

    left_align = 0.6
    right_align = 0.6
    down_align = 0.6
    text_size = 12
    offset = 0.05

    fig, axs = plt.subplots()

    axs.hist(sigma_map_cube.hdu.data.ravel(), bins=500, density=True, label = 'Total cube')
    axs.hist(sigma_map_cube_overlap.hdu.data.ravel(), bins=200, density=True, alpha = 0.5, label = 'overlapping')
    axs.axvline(x = np.nanmean(sigma_map_cube_overlap.hdu.data), color = 'orange', label = 'mean (overlapping pixels)')
    axs.axvline(x = np.nanmean(sigma_map_cube.hdu.data), color = 'blue', label = 'mean (all pixels)')
    axs.axvline(x = np.nanmedian(sigma_map_cube_overlap.hdu.data), color = 'orange', linestyle = '--', label = 'median (overlapping pixels)')
    axs.axvline(x = np.nanmedian(sigma_map_cube.hdu.data), color = 'blue', linestyle = '--', label = 'median (all pixels)')
    axs.legend()
    axs.set(xlim = [0,500])
    axs.text(left_align, down_align - offset, 'Mean all: %5.2f ' % (np.nanmean(sigma_map_cube.hdu.data)), fontsize=text_size,
             horizontalalignment='left', color = 'blue',
             verticalalignment='center', transform=axs.transAxes)
    axs.text(left_align, down_align, 'Mean overlapping: %5.2f ' % (np.nanmean(sigma_map_cube_overlap.hdu.data)), fontsize=text_size,
             horizontalalignment='left', color = 'orange',
             verticalalignment='center', transform=axs.transAxes)

    axs.text(left_align, down_align - 4*offset, 'Median all: %5.2f ' % (np.nanmedian(sigma_map_cube.hdu.data)), fontsize=text_size,
             horizontalalignment='left', color = 'blue',
             verticalalignment='center', transform=axs.transAxes)
    axs.text(left_align, down_align - 3*offset, 'Median overlapping: %5.2f ' % (np.nanmedian(sigma_map_cube_overlap.hdu.data)), fontsize=text_size,
             horizontalalignment='left', color = 'orange',
             verticalalignment='center', transform=axs.transAxes)

    fig, axs = plt.subplots()
    axs.hist(peak_map_cube.hdu.data.ravel(), bins=500, density=True,label='Total cube')
    axs.hist(peak_map_overlap.hdu.data.ravel(), bins=500, density=True, alpha=0.5, label='overlapping')
    axs.axvline(x=np.nanmean(peak_map_overlap.hdu.data), color='orange', label = 'mean (overlapping pixels)')
    axs.axvline(x=np.nanmean(peak_map_cube.hdu.data), color='blue', label = 'mean (all pixels)')
    axs.axvline(x=np.nanmedian(peak_map_overlap.hdu.data), color='orange', linestyle='--', label = 'median (overlapping pixels)')
    axs.axvline(x=np.nanmedian(peak_map_cube.hdu.data), color='blue', linestyle='--', label = 'median (all pixels)')
    axs.legend()
    axs.set(xlim = [0,2])

    axs.text(left_align, down_align - offset, 'Mean all: %5.2f ' % (np.nanmean(peak_map_cube.hdu.data)), fontsize=text_size,
             horizontalalignment='left', color = 'blue',
             verticalalignment='center', transform=axs.transAxes)
    axs.text(left_align, down_align, 'Mean overlapping: %5.2f ' % (np.nanmean(peak_map_overlap.hdu.data)), fontsize=text_size,
             horizontalalignment='left', color = 'orange',
             verticalalignment='center', transform=axs.transAxes)

    axs.text(left_align, down_align - 4*offset, 'Median all: %5.2f ' % (np.nanmedian(peak_map_cube.hdu.data)), fontsize=text_size,
             horizontalalignment='left', color = 'blue',
             verticalalignment='center', transform=axs.transAxes)
    axs.text(left_align, down_align - 3*offset, 'Median overlapping: %5.2f ' % (np.nanmedian(peak_map_overlap.hdu.data)), fontsize=text_size,
             horizontalalignment='left', color = 'orange',
             verticalalignment='center', transform=axs.transAxes)


    if show:
        plt.show()

    # add OPTION get hii region catalog and somehow check to which hii reg the overalp correspond to plot corr


co_cube = SpectralCube.read('/home/antoine/data_cubes/ngc7496_12m+7m+tp_co21_native_cube.fits')
co_cube = co_cube.with_spectral_unit(u.km/u.s, velocity_convention = 'radio')
print('reading spectral cube')
co_mask = fits.open('/home/antoine/Internship/gmc_masks/cats_native_amended/ngc7496_12m+7m+tp_co21_native_binmask2D.fits')
hii_mask = fits.open('/home/antoine/Internship/hii_masks_dr2/reprojected/Ngc7496HA6562_FLUXmask_reprojected_to_codatacube.fits')

co_cube_overlap_prop(co_cube, co_mask, hii_mask, galname='ngc7496', write = True, show = True)