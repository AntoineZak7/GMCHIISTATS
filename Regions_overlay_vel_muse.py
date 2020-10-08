import numpy as np
import pickle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from itertools import chain
import matplotlib.backends.backend_pdf as fpdf
from astropy.io import fits
import os
from astropy import wcs
from regions import read_ds9
from matplotlib import cm
from matplotlib import colors

def Vel_overlays(new_muse, gmc_catalog, overlap_matching, threshold_perc, outliers):

    def name(overperc, without_out, new_muse):
        name_append = ['perc_matching_', 'with_outliers', 'without_outliers', 'new_muse_', 'old_muse_', str(threshold_perc)]

        if new_muse == True:
            name_end = name_append[3]
            if overperc == True:
                name_end = name_end + name_append[0] +name_append[5]
                if without_out == True:
                    name_end = name_end + name_append[2]
                else:
                    name_end = name_end + name_append[1]

        else:
            name_end = name_append[4]
            if overperc == True:
                name_end = name_end + name_append[0] + name_append[5]
                if without_out == True:
                    name_end = name_end + name_append[2]
                else:
                    name_end = name_end + name_append[1]
        return name_end


    # ==============================================================================#
    typegmc = gmc_catalog  # '_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_
    overperc = overlap_matching  # True
    without_out = not outliers

    name_end = name(overperc, without_out, new_muse)
    namegmc = "_12m+7m+tp_co21%sprops" % typegmc

    dirhii, dirgmc, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots = pickle.load(
        open('Directories_muse.pickle', "rb"))  # retrieving the directories paths
    dirmuseproperties = os.path.dirname(os.path.realpath("Extract_info_plot_per_gal_muse.py")) + "/"

    if new_muse == False:
        dirplots = dirplots1
        dirregions = dirregions1
    else:
        dirplots = dirplots2
        dirregions = dirregions2

    galaxias, GMCprop, HIIprop, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay = pickle.load(
        open('%sGalaxies_variables_GMC%s%s.pickle' % (dirmuseproperties, namegmc, name_end), "rb"))
    SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, velHIIover, HIIminorover, HIImajorover, HIIangleover = HIIprop
    DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover = GMCprop

    SizepcHII, LumHacorr, sigmavHII, metaliHII, varmetHII, numGMConHII, MasscoGMC, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC = pickle.load(
        open('%sGalaxies_variables_notover_GMC%s%s.pickle' % (dirmuseproperties, namegmc, name_end), "rb"))

    name_gmc = "_12m+7m+tp_co21" + typegmc + "props-GMCs-matched" + name_end+".reg"
    name_hii = "_12m+7m+tp_co21" + typegmc + "props-HIIregions-matched"+ name_end + ".reg"

    # =======================================================================================================


    all_vel_offset = np.array(velHIIover) - np.array(velGMCover)
    all_vel_offset = list(chain.from_iterable(all_vel_offset))
    all_vel_offset = [n for n in all_vel_offset if n > -190]
    all_vel_offset = [n for n in all_vel_offset if n < 190]
    print(all_vel_offset)

    min_vel_c_all = np.nanmin(all_vel_offset)
    max_vel_c_all = np.nanmax(all_vel_offset)

    # norme =colors.Normalize(vmin=min_vel_c_all, vmax=max_vel_c_all, clip=False)
    RdBu = plt.get_cmap('rainbow')

    # ===========================Plot Vel regions HII=======================#
    pdf3 = fpdf.PdfPages("%sOverlays_HII_Vel_GMC%s%s.pdf" % (dirplots, namegmc, name_end))  # type: PdfPages

    name = "_ha.fits"  # "_Hasub_flux.fits"

    for j in range(len(galaxias)):

        gmc_regions = read_ds9(dirregions + galaxias[j] + name_gmc)
        hii_regions = read_ds9(dirregions + galaxias[j] + name_hii)

        fig = plt.figure()
        hdul = fits.open(dirmaps + str.upper(galaxias[j]) + name)
        hdr = hdul[0].header
        image_data = hdul[0].data
        WC = wcs.WCS(hdr)
        flat_imagedata = image_data.flatten()
        max = np.nanmax(flat_imagedata)
        min = np.nanmin(flat_imagedata)

        fig.add_subplot(111, projection=WC)
        plt.imshow(image_data, vmax=25000, cmap='Greys')

        regions_id = (5, 6)

        tot_vel_offset = velHIIover[j] - velGMCover[j]
        id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))

        tot_vel_offset = tot_vel_offset[(tot_vel_offset < 190) & (tot_vel_offset > -190)]
        max_vel = np.nanmax(abs(tot_vel_offset))
        min_vel_c = np.nanmin(tot_vel_offset)
        max_vel_c = np.nanmax(tot_vel_offset)

        if abs(min_vel_c) > max_vel_c:
            max = abs(min_vel_c)
            min = -max
        else:
            max = max_vel_c
            min = -max

        plt.imshow(image_data, vmax=10000, cmap='Greys')

        norme = colors.Normalize(vmin=min, vmax=max, clip=False)
        plt.title(galaxias[j])
        print(id_int[0])

        for i in id_int[0]:
            velhii = velHIIover[j][i]
            velgmc = velGMCover[j][i]
            vel_offset = velhii - velgmc

            # print(vel_offset)
            if vel_offset > 3 * 21 or vel_offset < -3 * 21:
                v = float((vel_offset - min) / (max - min))

                colour = RdBu(v)

                pixel_region = gmc_regions[i].to_pixel(wcs=WC)
                pixel_region1 = hii_regions[i].to_pixel(wcs=WC)
                pixel_region1.plot(color=(colour))
        print(max_vel_c)
        print(min_vel_c)
        fig.colorbar(cm.ScalarMappable(norm=norme, cmap=RdBu))

        plt.show()
        pdf3.savefig()
        #plt.close()
    pdf3.close()



