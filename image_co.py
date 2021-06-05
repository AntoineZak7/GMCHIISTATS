import numpy as np
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf
from astropy.io import fits
import astropy.units as u
from astropy import wcs
from regions import read_ds9
import os
from regions import EllipseSkyRegion, CircleSkyRegion
from reproject import reproject_interp
import warnings
warnings. filterwarnings("ignore")

dir_script_data = os.getcwd() + "/script_data/"
dirhii, dirgmc, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots, dirhiimasks = pickle.load(open(dir_script_data + 'Directories_muse.pickle', "rb"))




name = '_12m+7m+tp_co21_broad_mom0_120pc.fits'  # "_Hasub_flux.fits"
galaxias = 'ngc1087'

dirmaps = '/home/antoine/Internship/Galaxies/forantoine_ALMA/'
#mask = fits.open(dirhiimasks + str.upper(galaxias) + '_HIIreg_mask.fits')
hdul = fits.open(dirmaps + galaxias + name)
hdr = hdul[0].header
image_data = hdul[0].data
#WC = wcs.WCS(mask[0].header)
#image_reprojected, footprint = reproject_interp(hdul, mask[0].header)
flat_imagedata = image_data.flatten()

max = np.nanmax(flat_imagedata)
min = np.nanmin(flat_imagedata)

fig = plt.figure()
fig.add_subplot(111)
plt.suptitle(' %s \n %s' % ( galaxias, 'Hasub_flux'))
plt.imshow(image_data, vmax=20, cmap='Greys')
plt.show()