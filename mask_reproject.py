"""""
Python script to reproject a galaxy wide hii region mask to a specified spectral_cube size/WCS
Can save hii region mask reprojected as fits with same header as CO spectral cube or
"""""

import gc
import warnings
import numpy as np
from reproject import reproject_interp
from astropy.io import fits
from astropy import wcs
import copy
import matplotlib.pyplot as plt

def mask_reproject(co_mask, hii_mask, save, galname, **kwargs):


    save_directory = kwargs.get('save_directory', None)
    #'/home/antoine/Internship/Galaxies/New_Muse/'


    #Load GMC datacube header and hii_mask_header
    co_header = co_mask[0].header
    CO_WCS = wcs.WCS(co_header)

    # Change WCS info in hii mask header


    #Check HII mask max value to give nb of hii regions
    region_nb = int(np.nanmax(hii_mask[0].data))

    #Create empty mask to add the other on
    new_mask = copy.deepcopy(hii_mask)
    new_mask_header = new_mask[0].header


    #new_mask_data = hii_mask[0].data.copy()


    new_mask[0].data[np.where(new_mask[0].data != 0.)] = 0 #to check if properly sets everything to 0
    new_mask[0].data[np.isnan(new_mask[0].data)] = 0


    #new_mask[0].data = new_mask_data

    new_mask_data, new_mask_data_footprint = reproject_interp(new_mask, co_header) #interp_repro is fastest, but not most precise ?
    new_mask_data[np.isnan(new_mask_data)] = 0



    #Loop over number of HII regions
    for region_id in range(1,region_nb+1):
        print("{} / {}".format(region_id, region_nb))

        #make copy of hii mask
        region_mask = copy.deepcopy(hii_mask)


        #keep only the desired hii region index values in array/mask

        region_mask[0].data[np.where(region_mask[0].data.astype(int) != region_id)] = 0

        #reproject using co datacube WCS with reproject_interp ?
        region_mask_data_repro, region_mask_repro_footprint = reproject_interp(region_mask, co_header)

        #change all non 0 values to hii region index


        region_mask_data_repro[np.isnan(region_mask_data_repro)] = 0
        region_mask_data_repro[np.where(region_mask_data_repro != 0. )] = region_id

        # fig = plt.figure()  # no figsize if zommed figsize=(10, 10)
        # axs = plt.subplot(1, 1, 1, projection=wcs.WCS(co_mask[0].header))
        # axs.imshow(region_mask_data_repro, cmap='Greys')
        # print(np.nanmax(region_mask_data_repro))
        # plt.show()

        #add region mask to gloabl mask
        new_mask_data += region_mask_data_repro

        #to check if overlap between hii region mask, check if max value > index. If so, change those values to max value (need to find better way to fix this, ie check with what it overlaps and which is greater)
        if np.nanmax(new_mask_data) > region_id:
            warnings.warn('HII region mask overlapping. Overlapping pixels are attributed to last region mask reprojected')
            new_mask_data[np.where(new_mask_data > region_id)] = region_id

        #del hii region masks and footprint for space
        del region_mask, region_mask_repro_footprint, region_mask_data_repro
        gc.collect()

    #either save file and return or just return


    if save == True:
        new_mask_header['NAXIS1'] = co_header['NAXIS1']
        new_mask_header['NAXIS2'] = co_header['NAXIS2']
        new_mask_header['CD1_1'] = co_header['CDELT1']
        new_mask_header['CD2_2'] = co_header['CDELT2']
        new_mask_header['CRVAL1'] = co_header['CRVAL1']
        new_mask_header['CRVAL2'] = co_header['CRVAL2']
        new_mask_header['CRPIX1'] = co_header['CRPIX1']
        new_mask_header['CRPIX2'] = co_header['CRPIX2']

        #save file as fits in specified directory
        file_name = save_directory + galname + hii_mask[0].header['EXTNAME']+'mask_reprojected_to_codatacube.fits'
        fits.writeto(file_name, new_mask_data, new_mask_header, overwrite=True)

    return new_mask_data




save_directory = '/home/antoine/Internship/hii_masks_dr2/reprojected/'

#galname = 'Ngc7496'
#
# galnames = ['Ic5332', 'Ngc1433', 'Ngc3627',
#             'Ngc0628', 'Ngc1512', 'Ngc4254',
#             'Ngc1087', 'Ngc1566', 'Ngc4303',
#             'Ngc1300', 'Ngc1672', 'Ngc4321',
#             'Ngc1365', 'Ngc2835', 'Ngc4535',
#             'Ngc1385', 'Ngc3351', 'Ngc5068',
#             'Ngc7496']

galnames = ['NGC3627']

for galname in galnames:
    co_mask = fits.open(
        '/home/antoine/Internship/gmc_masks/cats_native_amended/' + str.lower(galname)+'_12m+7m+tp_co21_native_binmask2D.fits')
    hii_mask = fits.open('/home/antoine/Internship/hii_masks_dr2/' + str.upper(galname)+'_HIIreg_mask.fits')

    test = mask_reproject(co_mask, hii_mask,save = True, save_directory = save_directory , galname=galname)

# co_mask = fits.open('/home/antoine/Internship/gmc_masks/cats_native_amended/' + str.lower(galname) + '_12m+7m+tp_co21_native_binmask2D.fits')
# hii_mask = fits.open('/home/antoine/Internship/hii_masks_dr2/reprojected/' + galname + 'HA6562_FLUXmask_reprojected_to_codatacube.fits')
#
# co_mask[0].data[co_mask[0].data != 0.] = 1
# hii_mask[0].data[hii_mask[0].data != 0.] = 1
#
#
# fig = plt.figure( ) #no figsize if zommed figsize=(10, 10)
# axs= plt.subplot(1, 1, 1, projection= wcs.WCS(co_mask[0].header))
# axs.imshow(co_mask[0].data*0.75+ hii_mask[0].data*10, cmap='Greys', vmax = 2)
# plt.show()


