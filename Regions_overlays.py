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
from matplotlib import rc
#rc('text', usetex = True)

#plt.rcParams['text.usetex'] = True
#plt.rcParams['font.size'] = 18

def overlays(new_muse, gmc_catalog, matching, outliers, show, save, vel_limit, threshold_perc, randomize, *args, **kwargs):

    paired = kwargs.get('paired', None)
    unpaired = kwargs.get('unpaired', None)
    all = kwargs.get('all', None)


    def save_pdf(pdf):
        if save == True:
            pdf.savefig(fig)
        if show == True:
            #plt.xlim(250,400)
            #plt.ylim(320,470)
            plt.show()
        else:
            plt.close()

    def name(matching, without_out, new_muse, gmc_catalog,threshold_perc, vel_limit, randomize):
        name_append = ['', 'with_outliers', 'without_outliers', 'new_muse_', 'old_muse_',str(threshold_perc), 'vel_limit=', 'random']

        if new_muse == True:
            name_end = name_append[3] + gmc_catalog + matching + 'vel_limit:' + str(vel_limit)
            if matching != "distance":
                name_end = name_end + name_append[5]
                if without_out == True:
                    name_end = name_end + name_append[2]
                    if randomize == True:
                        name_end = name_end + name_append[7]
                else:
                    name_end = name_end + name_append[1]
                    if randomize == True:
                        name_end = name_end + name_append[7]
        return name_end


    def paired1():
        major_over = HIImajorover[j]

        minor_over = HIIminorover[j]

        major_gmc_over = majorGMCover[j]

        minor_gmc_over = minorGMCover[j]

        angle_gmc_over = angleGMCover[j]

        plt.title(galaxias[j])
        k = 0
        for i in range(len(hii_regions_paired)):
            center = hii_regions_paired[i].center
            circle_sky = CircleSkyRegion(center=center, radius=major_over[i] * u.deg)
            circle_pixel = circle_sky.to_pixel(wcs=WC)
            circle_pixel.plot(color='tab:blue', linestyle = '-.')

        for i in range(len(gmc_regions_paired)):
            center = gmc_regions_paired[i].center
            ellipse_sky_gmc = EllipseSkyRegion(center=center, height=major_gmc_over[i] * u.deg,
                                               width=minor_gmc_over[i] * u.deg,
                                               angle=(angle_gmc_over[i]) * u.deg)
            ellipse_pixel_gmc = ellipse_sky_gmc.to_pixel(wcs=WC)
            ellipse_pixel_gmc.plot(color='lightblue')


    def unpaired1():
        a = [hii_regions[i].center.to_pixel(wcs=WC) for i in range(len(hii_regions))]
        A = np.reshape(a, (-1, 2))

        b = [hii_regions_paired[i].center.to_pixel(wcs=WC) for i in range(len(hii_regions_paired))]
        B = np.reshape(b, (-1, 2))

        c = [gmc_regions[i].center.to_pixel(wcs=WC) for i in range(len(gmc_regions))]
        C = np.reshape(c, (-1, 2))

        d = [gmc_regions_paired[i].center.to_pixel(wcs=WC) for i in range(len(gmc_regions_paired))]
        D = np.reshape(d, (-1, 2))

        index_hii = [i for i, item in enumerate(A) if item not in B]
        hii_regions_unpaired = [hii_regions[i] for i in index_hii]
        index_gmc = [i for i, item in enumerate(C) if item not in D]
        gmc_regions_unpaired = [gmc_regions[i] for i in index_gmc]


        major = HIImajor[j]

        major_gmc = majorGMC[j]

        minor_gmc = minorGMC[j]

        angle_gmc = angleGMC[j]

        for i in index_hii:
            center = hii_regions[i].center
            circle_sky = CircleSkyRegion(center=center, radius=major[i] * u.deg)
            circle_pixel = circle_sky.to_pixel(wcs=WC)
            circle_pixel.plot(color='tab:red', linestyle= '-.')

        for i in index_gmc:
            center = gmc_regions[i].center
            ellipse_sky_gmc = EllipseSkyRegion(center=center, height=major_gmc[i] * u.deg,
                                               width=minor_gmc[i] * u.deg, angle=(angle_gmc[i]) * u.deg)
            ellipse_pixel_gmc = ellipse_sky_gmc.to_pixel(wcs=WC)
            ellipse_pixel_gmc.plot(color='red')

        # fig.colorbar(cm.ScalarMappable( norm = norme ,cmap=RdBu))

    def paired_and_unpaired():
        # ===== Check with HII regions and GMCs are not paired by differentiating all hii regions and gmcs and the one that are paired ===== #
        # ===== Certainly not the most straightforward method but the only one that works yet=============================================== #

        a = [hii_regions[i].center.to_pixel(wcs=WC) for i in range(len(hii_regions))]
        A = np.reshape(a, (-1, 2))

        b = [hii_regions_paired[i].center.to_pixel(wcs=WC) for i in range(len(hii_regions_paired))]
        B = np.reshape(b, (-1, 2))

        c = [gmc_regions[i].center.to_pixel(wcs=WC) for i in range(len(gmc_regions))]
        C = np.reshape(c, (-1, 2))

        d = [gmc_regions_paired[i].center.to_pixel(wcs=WC) for i in range(len(gmc_regions_paired))]
        D = np.reshape(d, (-1, 2))

        index_hii = [i for i, item in enumerate(A) if item not in B]
        hii_regions_unpaired = [hii_regions[i] for i in index_hii]
        index_gmc = [i for i, item in enumerate(C) if item not in D]
        gmc_regions_unpaired = [gmc_regions[i] for i in index_gmc]

        # ===== Stocking variables for each galaxy ======================================================================================== #

        major = HIImajor[j]
        major_over = HIImajorover[j]

        major_gmc = majorGMC[j]
        major_gmc_over = majorGMCover[j]

        minor_gmc = minorGMC[j]
        minor_gmc_over = minorGMCover[j]

        angle_gmc = angleGMC[j]
        angle_gmc_over = angleGMCover[j]

        for i in index_hii:
            center = hii_regions[i].center
            circle_sky = CircleSkyRegion(center=center, radius=major[i] * u.deg)
            circle_pixel = circle_sky.to_pixel(wcs=WC)
            circle_pixel.plot(color='tab:red', linestyle = '-.')

        for i in index_gmc:
            center = gmc_regions[i].center
            ellipse_sky_gmc = EllipseSkyRegion(center=center, height=major_gmc[i] * u.deg,
                                               width=minor_gmc[i] * u.deg, angle=(angle_gmc[i]) * u.deg)
            ellipse_pixel_gmc = ellipse_sky_gmc.to_pixel(wcs=WC)
            ellipse_pixel_gmc.plot(color='red')

        for i in range(len(gmc_regions_paired)):
            center = gmc_regions_paired[i].center
            ellipse_sky_gmc = EllipseSkyRegion(center=center, height=major_gmc_over[i] * u.deg,
                                               width=minor_gmc_over[i] * u.deg, angle=(angle_gmc_over[i]) * u.deg)
            ellipse_pixel_gmc = ellipse_sky_gmc.to_pixel(wcs=WC)
            ellipse_pixel_gmc.plot(color='lightblue')

        for i in range(len(hii_regions_paired)):
            center = hii_regions_paired[i].center
            circle_sky = CircleSkyRegion(center=center, radius=major_over[i] * u.deg)
            circle_pixel = circle_sky.to_pixel(wcs=WC)
            circle_pixel.plot(color='tab:blue', linestyle = '-.')



    typegmc = gmc_catalog#'_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_
    # ==============================================================================#

    without_out = not outliers

    name_end = name(matching, without_out, new_muse, gmc_catalog, threshold_perc, vel_limit, randomize)
    namegmc = "_12m+7m+tp_co21%sprops" % typegmc

    dir_script_data = os.getcwd() + "/script_data/"

    dirhii, dirgmc, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots, dirhiimasks = pickle.load(
        open(dir_script_data + 'Directories_muse.pickle', "rb"))  # retrieving the directories paths
    #dirmuseproperties = os.path.dirname(os.path.realpath("Extract_info_plot_per_gal_muse.py")) + "/"

    if new_muse == False:
        dirplots = dirplots1
        dirregions = dirregions1
    else:
        dirplots = dirplots2
        dirregions = dirregions2

    name_gmc = "_12m+7m+tp_co21" + typegmc + "props-GMCs-matched" + name_end+".reg"
    name_hii = "_12m+7m+tp_co21" + typegmc + "props-HIIregions-matched"+ name_end + ".reg"
    name_hii_all = "_12m+7m+tp_co21" + typegmc + "props-HIIregions-all_regions" + name_end + ".reg"
    name_gmc_all = "_12m+7m+tp_co21" + typegmc + "props-GMCs-all_regions" + name_end + ".reg"

    galaxias, GMCprop, HIIprop, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay, idoverhiis, idovergmcs = pickle.load(
        open( dir_script_data + 'Galaxies_variables_GMC%s%s.pickle' % ( namegmc, name_end), "rb"))
    SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, velHIIover, HIIminorover, HIImajorover, HIIangleover = HIIprop
    DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover,test = GMCprop

    SizepcHII, LumHacorr, sigmavHII, metaliHII, varmetHII, numGMConHII, MasscoGMC, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC = pickle.load(
        open( dir_script_data +'Galaxies_variables_notover_GMC%s%s.pickle' % ( namegmc, name_end), "rb"))

    



    # ==========================All Paired and unpaired=====================================#
    if paired == True or all == True:
        pdf1 = fpdf.PdfPages("%sOverlays_paired_Muse_%s%s.pdf" % (dirplots, namegmc, name_end))  # type: PdfPages
    if unpaired == True or all == True:
        pdf2 = fpdf.PdfPages("%sOverlays_unpaired_Muse_%s%s.pdf" % (dirplots, namegmc, name_end))  # type: PdfPages
    if (paired == True and unpaired == True) or all == True:
        pdf3 = fpdf.PdfPages("%sOverlays_paired_and_unpaired_Muse_%s%s.pdf" % (dirplots, namegmc, name_end))  # type: PdfPages

    name = "_ha.fits"  # "_Hasub_flux.fits"
    for j in range(len(galaxias)):


        if os.path.isfile("%s%s%s" % (dirmaps, str.upper(galaxias[j]), name)) == True:
            if os.path.isfile("%s%s%s" % (dirregions, galaxias[j], name_gmc)) == True and os.path.isfile(
                    "%s%s%s" % (dirregions, galaxias[j], name_hii)) == True:
                mask = fits.open(dirhiimasks + str.upper(galaxias[j]) + '_HIIreg_mask.fits')
                hdul = fits.open(dirmaps + str.upper(galaxias[j]) + name)
                hdr = hdul[0].header
                image_data = hdul[0].data
                WC = wcs.WCS(mask[0].header)
                image_reprojected, footprint = reproject_interp(hdul, mask[0].header)
                flat_imagedata = image_data.flatten()

                max = np.nanmax(flat_imagedata)
                min = np.nanmin(flat_imagedata)

                fig = plt.figure()
                fig.add_subplot(111, projection=WC)
                print(name_end)
                plt.suptitle('%s \n %s' %(name_end,galaxias[j]))
                plt.imshow(image_reprojected, vmax=6000, cmap='Greys')

                gmc_regions_paired = read_ds9(dirregions + galaxias[j] + name_gmc)
                hii_regions_paired = read_ds9(dirregions + galaxias[j] + name_hii)

                gmc_regions = read_ds9(dirregions + galaxias[j] + name_gmc_all)
                hii_regions = read_ds9(dirregions + galaxias[j] + name_hii_all)

                # ===============================#
                if paired == True:
                    if unpaired == True:
                        paired_and_unpaired()
                        save_pdf(pdf3)
                    else:
                        paired1()
                        save_pdf(pdf1)
                else:
                    if unpaired == True:
                        if paired == True:
                            paired_and_unpaired()
                            save_pdf(pdf3)
                        else:
                            unpaired1()
                            save_pdf(pdf2)
                if all == True:
                    paired1()
                    unpaired1()
                    paired_and_unpaired()
                    save_pdf(pdf1)
                    save_pdf(pdf2)
                    save_pdf(pdf3)



    if paired == True or all == True:
        pdf1.close()
    if unpaired == True or all == True:
        pdf2.close()
    if (paired == True and unpaired == True )or all == True:
        pdf3.close()

overlays(new_muse =True, gmc_catalog = "_native_", matching = "overlap_1om", outliers = True,show =True, save = True, threshold_perc = 0.9,  paired = True, unpaired = True, vel_limit = 10000, randomize = False)

