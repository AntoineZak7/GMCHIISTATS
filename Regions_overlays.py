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
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
from spectral_cube import Projection

def overlays(new_muse, gmc_catalog, overlap_matching, outliers, show, save, *args, **kwargs):


    paired = kwargs.get('paired', None)
    unpaired = kwargs.get('unpaired', None)
    all = kwargs.get('all', None)

    def name(overperc, without_out, new_muse):
        name_append = ['perc_matching_', 'with_outliers', 'without_outliers', 'new_muse_', 'old_muse_']

        if new_muse == True:
            name_end = name_append[3]
            if overperc == True:
                name_end = name_end + name_append[0]
                if without_out == True:
                    name_end = name_end + name_append[2]
                else:
                    name_end = name_end + name_append[1]

        else:
            name_end = name_append[4]
            if overperc == True:
                name_end = name_end + name_append[0]
                if without_out == True:
                    name_end = name_end + name_append[2]
                else:
                    name_end = name_end + name_append[1]
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
            print("major over =%i" %len(major_over))
            print("hii_regions_paired= %i" %len(hii_regions_paired))
            circle_sky = CircleSkyRegion(center=center, radius=major_over[i] * u.deg)
            circle_pixel = circle_sky.to_pixel(wcs=WC)
            circle_pixel.plot(color='green')

        for i in range(len(gmc_regions_paired)):
            center = gmc_regions_paired[i].center
            ellipse_sky_gmc = EllipseSkyRegion(center=center, height=major_gmc_over[i] * u.deg,
                                               width=minor_gmc_over[i] * u.deg,
                                               angle=(angle_gmc_over[i]) * u.deg)
            ellipse_pixel_gmc = ellipse_sky_gmc.to_pixel(wcs=WC)
            ellipse_pixel_gmc.plot(color='lime')
            # pdf1.savefig()
            # plt.close()

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

        #minor = HIIminor[j]

        #angle = HIIangle[j]

        major_gmc = majorGMC[j]

        minor_gmc = minorGMC[j]

        angle_gmc = angleGMC[j]

        for i in index_hii:
            center = hii_regions[i].center
            circle_sky = CircleSkyRegion(center=center, radius=major[i] * u.deg)
            circle_pixel = circle_sky.to_pixel(wcs=WC)
            circle_pixel.plot(color='blue')

        for i in index_gmc:
            center = gmc_regions[i].center
            ellipse_sky_gmc = EllipseSkyRegion(center=center, height=major_gmc[i] * u.deg,
                                               width=minor_gmc[i] * u.deg, angle=(angle_gmc[i]) * u.deg)
            ellipse_pixel_gmc = ellipse_sky_gmc.to_pixel(wcs=WC)
            ellipse_pixel_gmc.plot(color='red')

        # fig.colorbar(cm.ScalarMappable( norm = norme ,cmap=RdBu))

        #plt.show()
        #pdf2.savefig()
        #plt.close()

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

        #minor = HIIminor[j]
        #minor_over = HIIminorover[j]

        #angle = HIIangle[j]
        #angle_over = HIIangleover[j]

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
            circle_pixel.plot(color='blue')

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
            ellipse_pixel_gmc.plot(color='lightgreen')

        for i in range(len(hii_regions_paired)):
            center = hii_regions_paired[i].center
            circle_sky = CircleSkyRegion(center=center, radius=major_over[i] * u.deg)
            circle_pixel = circle_sky.to_pixel(wcs=WC)
            circle_pixel.plot(color='green')

        #plt.show()
        #pdf3.savefig()
        #plt.close()


    typegmc1 = ''  # match_, match_homogenized_ (nothing for native)
    typegmc = gmc_catalog#'_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_
    # ==============================================================================#

    overperc = overlap_matching  # True
    without_out = not outliers

    name_end = name(overperc, without_out, new_muse)
    namegmc = "_12m+7m+tp_co21%sprops" % typegmc

    dirhii, dirgmc, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots = pickle.load(
        open('Directories_muse.pickle', "rb"))  # retrieving the directories paths
    dirmuseproperties = os.path.dirname(os.path.realpath("Extract_info_plot_per_gal_muse.py")) + "/"

    if new_muse == False:
        dirplots = dirplots1
        name_gmc = "_12m+7m+tp_co21_native_props-GMCs-overlapped_old_muse.reg"
        name_hii = "_12m+7m+tp_co21_native_props-HIIregions-overlapped_old_muse.reg"
        name_hii_all = "_12m+7m+tp_co21_native_props-HIIregions-all_regions_old_muse.reg"
        name_gmc_all = "_12m+7m+tp_co21_native_props-GMCs-all_regions_old_muse.reg"

        dirregions = dirregions1
    else:
        dirplots = dirplots2
        name_gmc = "_12m+7m+tp_co21_native_props-GMCs-overlapped_new_muse.reg"
        name_hii = "_12m+7m+tp_co21_native_props-HIIregions-overlapped_new_muse.reg"
        name_hii_all = "_12m+7m+tp_co21_native_props-HIIregions-all_regions_new_muse.reg"
        name_gmc_all = "_12m+7m+tp_co21_native_props-GMCs-all_regions_new_muse.reg"

        dirregions = dirregions2

    galaxias, GMCprop, HIIprop, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay = pickle.load(
        open('%sGalaxies_variables_GMC%s%s.pickle' % (dirmuseproperties, namegmc, name_end), "rb"))
    SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, velHIIover, HIIminorover, HIImajorover, HIIangleover = HIIprop
    DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover = GMCprop

    SizepcHII, LumHacorr, sigmavHII, metaliHII, varmetHII, numGMConHII, MasscoGMC, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC = pickle.load(
        open('%sGalaxies_variables_notover_GMC%s%s.pickle' % (dirmuseproperties, namegmc, name_end), "rb"))

    # ==========================All Paired and unpaired=====================================#
    pdf1 = fpdf.PdfPages("%sOverlays_paired_Muse_%s.pdf" % (dirplots, namegmc))  # type: PdfPages
    pdf2 = fpdf.PdfPages("%sOverlays_unpaired_Muse_%s.pdf" % (dirplots, namegmc))  # type: PdfPages
    pdf3 = fpdf.PdfPages("%sOverlays_paired_and_unpaired_Muse_%s.pdf" % (dirplots, namegmc))  # type: PdfPages

    name = "_ha.fits"  # "_Hasub_flux.fits"
    for j in range(len(galaxias)):

        print("%s%s%s.fits" % (dirmaps, str.upper(galaxias[j]), name))
        print("%s%s%s" % (dirregions, galaxias[j], name_gmc))
        if os.path.isfile("%s%s%s" % (dirmaps, str.upper(galaxias[j]), name)) == True:
            if os.path.isfile("%s%s%s" % (dirregions, galaxias[j], name_gmc)) == True and os.path.isfile(
                    "%s%s%s" % (dirregions, galaxias[j], name_hii)) == True:
                hdul = fits.open(dirmaps + str.upper(galaxias[j]) + name)
                hdr = hdul[0].header
                image_data = hdul[0].data
                WC = wcs.WCS(hdr)
                flat_imagedata = image_data.flatten()
                # outline = Projection.from_hdu(hdulout[0])
                # outline_re = outline.reproject(hdr)

                max = np.nanmax(flat_imagedata)
                min = np.nanmin(flat_imagedata)

                fig = plt.figure()
                fig.add_subplot(111, projection=WC)
                plt.imshow(image_data, vmax=6000, cmap='Greys')

                gmc_regions_paired = read_ds9(dirregions + galaxias[j] + name_gmc)
                hii_regions_paired = read_ds9(dirregions + galaxias[j] + name_hii)

                gmc_regions = read_ds9(dirregions + galaxias[j] + name_gmc_all)
                hii_regions = read_ds9(dirregions + galaxias[j] + name_hii_all)

                # ===============================#
                if paired == True:
                    if unpaired == True:
                        paired_and_unpaired()
                    else:
                        paired1()
                else:
                    if unpaired == True:
                        if paired == True:
                            paired_and_unpaired()
                        else:
                            unpaired1()
                if all == True:
                    paired1()
                    unpaired1()
                    paired_and_unpaired()

                # fig.colorbar(cm.ScalarMappable( norm = norme ,cmap=RdBu))

                plt.show()

    pdf3.close()

overlays(new_muse = True, gmc_catalog = "_120pc_match_", overlap_matching = True, outliers = True,show = True, save = False, paired = True)
