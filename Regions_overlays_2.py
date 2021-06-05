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
plt.style.use('science')

warnings.filterwarnings("ignore")
from matplotlib import rc


# rc('text', usetex = True)

# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.size'] = 18

def overlays(muse, gmc_catalog, gmc_catalog_version, matching, outliers, show, save, vel_limit, threshold_perc,
             randomize, *args, **kwargs):
    paired = kwargs.get('paired', None)
    unpaired = kwargs.get('unpaired', None)
    all = kwargs.get('all', None)

    color_unmatch_gmc = 'deeppink'
    color_unmatch_hii = 'salmon'
    color_match_hii = 'yellow'
    color_match_gmc = 'orange'

    def save_pdf(pdf):
        # plt.xlim(482, 525)  # ngc4254, j = 13
        # plt.ylim(713, 744)
        #
        # axs.text((486), (722), 'A' ,color = color_unmatch_gmc,  fontsize=12, horizontalalignment='center',
        #          verticalalignment='center')
        #
        # axs.text((510), (721), 'B' ,color = color_match_gmc,  fontsize=12, horizontalalignment='center',
        #          verticalalignment='center')
        #
        # axs.text((522), (742), 'C' ,color = color_unmatch_gmc,  fontsize=12, horizontalalignment='center',
        #          verticalalignment='center')
        #
        #
        # axs.text((492), (728), '1' ,color = color_unmatch_hii,  fontsize=12, horizontalalignment='center',
        #          verticalalignment='center')
        #
        # axs.text((514.5), (717.5), '2' ,color = color_match_hii,  fontsize=12, horizontalalignment='center',
        #          verticalalignment='center')
        #
        # axs.text((511), (727), '3' ,color = color_unmatch_hii,  fontsize=12, horizontalalignment='center',
        #          verticalalignment='center')
        #
        # axs.text((518), (739), '4' ,color = color_unmatch_hii,  fontsize=12, horizontalalignment='center',
        #          verticalalignment='center')


        if save == True:
            pdf.savefig(figure=fig)
        if show == True:

            plt.show()
        else:
            plt.close()

    def name(matching, without_out, muse, gmc_catalog, gmc_catalog_version, threshold_perc, vel_limit, randomize):

        name_end = 'muse:' + muse + '_' + 'gmc:' + gmc_catalog + '(' + gmc_catalog_version + ')_' + 'vel_limit:' + str(
            vel_limit) + '_matching:' + matching +'_'+randomize+'_'

        if matching != "distance":
            name_end = name_end + '(' + str(threshold_perc).split(sep='.')[0] + str(threshold_perc).split(sep='.')[1]  + ')'
            if without_out == True:
                name_end = name_end + '_' + 'without_outliers'
            else:
                name_end = name_end + '_' + 'with_outliers'

        return name_end

    def paired1():
        major_over = HIImajorover[j]


        minor_over = HIIminorover[j]

        major_gmc_over = majorGMCover[j]

        minor_gmc_over = minorGMCover[j]

        angle_gmc_over = angleGMCover[j]

        #plt.title(galaxias[j])
        k = 0
        for i in range(len(hii_regions_paired)):
            center = hii_regions_paired[i].center
            circle_sky = CircleSkyRegion(center=center, radius=major_over[i] * u.deg)
            circle_pixel = circle_sky.to_pixel(wcs=WC)
            circle_pixel.plot(color=color_match_hii, linestyle='-.')

        for i in range(len(gmc_regions_paired)):
            center = gmc_regions_paired[i].center
            ellipse_sky_gmc = EllipseSkyRegion(center=center, height=major_gmc_over[i] * u.deg,
                                               width=minor_gmc_over[i] * u.deg,
                                               angle=(angle_gmc_over[i]) * u.deg)
            ellipse_pixel_gmc = ellipse_sky_gmc.to_pixel(wcs=WC)
            ellipse_pixel_gmc.plot(color=color_match_gmc)
            print('region')

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
            circle_pixel.plot(color=color_unmatch_hii, linestyle='-.')

        for i in index_gmc:
            center = gmc_regions[i].center
            ellipse_sky_gmc = EllipseSkyRegion(center=center, height=major_gmc[i] * u.deg,
                                               width=minor_gmc[i] * u.deg, angle=(angle_gmc[i]) * u.deg)
            ellipse_pixel_gmc = ellipse_sky_gmc.to_pixel(wcs=WC)
            ellipse_pixel_gmc.plot(color=color_unmatch_gmc)

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
            circle_pixel.plot(color=color_unmatch_hii, linestyle='-.')

        for i in index_gmc:
            center = gmc_regions[i].center
            ellipse_sky_gmc = EllipseSkyRegion(center=center, height=major_gmc[i] * u.deg, width = minor_gmc[i] * u.deg, angle = angle_gmc[i] * u.deg)
            # width=minor_gmc[i] * u.deg, angle=(angle_gmc[i]) * u.deg)
            ellipse_pixel_gmc = ellipse_sky_gmc.to_pixel(wcs=WC)
            ellipse_pixel_gmc.plot(color=color_unmatch_gmc)

        for i in range(len(gmc_regions_paired)):
            center = gmc_regions_paired[i].center
            ellipse_sky_gmc = EllipseSkyRegion(center=center, height=major_gmc_over[i] * u.deg, width = minor_gmc_over[i] * u.deg, angle=angle_gmc_over[i] * u.deg)
            # width=minor_gmc_over[i] * u.deg, angle=(angle_gmc_over[i]) * u.deg)
            ellipse_pixel_gmc = ellipse_sky_gmc.to_pixel(wcs=WC)
            ellipse_pixel_gmc.plot(color=color_match_gmc)

        for i in range(len(hii_regions_paired)):
            center = hii_regions_paired[i].center
            circle_sky = CircleSkyRegion(center=center, radius=major_over[i] * u.deg)
            circle_pixel = circle_sky.to_pixel(wcs=WC)
            circle_pixel.plot(color=color_match_hii, linestyle='-.')

    typegmc = gmc_catalog  # '_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_
    # ==============================================================================#

    without_out = not outliers

    name_end = name(without_out=without_out, matching=matching, muse=muse, gmc_catalog=gmc_catalog,
                    gmc_catalog_version=gmc_catalog_version, threshold_perc=threshold_perc, vel_limit=vel_limit, randomize= randomize)
    namegmc = "_12m+7m+tp_co21%sprops" % typegmc

    dir_script_data = os.getcwd() + "/script_data_dr2/"

    dirhii_dr1, dirhii_dr2, dirgmc_old, dirgmc_new, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots, dirhiimasks, dirgmcmasks,dirrr = pickle.load(
        open(dir_script_data + 'Directories_muse.pickle', "rb"))  # retrieving the directories paths
    # dirmuseproperties = os.path.dirname(os.path.realpath("Extract_info_plot_per_gal_muse.py")) + "/"

    if muse == 'dr1':
        dirplots = dirplots1
        dirregions = dirregions1
    elif muse == 'dr2':
        dirplots = dirplots2
        dirregions = dirregions2

    name_gmc = "_12m+7m+tp_co21" + typegmc + "props-GMCs-matched" + name_end + ".reg"
    name_hii = "_12m+7m+tp_co21" + typegmc + "props-HIIregions-matched" + name_end + ".reg"
    name_hii_all = "_12m+7m+tp_co21" + typegmc + "props-HIIregions-all_regions" + name_end + ".reg"
    name_gmc_all = "_12m+7m+tp_co21" + typegmc + "props-GMCs-all_regions" + name_end + ".reg"

    galaxias, GMCprop, HIIprop, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay, idoverhiis, idovergmcs, a,b = pickle.load(
        open(dir_script_data + 'Galaxies_variables_GMC%s%s.pickle' % (namegmc, name_end), "rb"))
    SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, velHIIover, HIIminorover, HIImajorover, HIIangleover,Rgal_hii = HIIprop
    DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover, test,Rgal_gmc = GMCprop

    SizepcHII, LumHacorr, sigmavHII, metaliHII, varmetHII, numGMConHII, FluxCOGMC, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC, MasscoGMC, SizepcGMC, SizepcHII, Sigmamole, sigmavGMC, TpeakGMC,a,b = pickle.load(
        open(dir_script_data + 'Galaxies_variables_notover_GMC%s%s.pickle' % (namegmc, name_end), "rb"))

    # ==========================All Paired and unpaired=====================================#
    if paired == True or all == True:
        pdf1 = fpdf.PdfPages("%sOverlays_paired_Muse_%s%s.pdf" % (dirplots, namegmc, name_end))  # type: PdfPages
    if unpaired == True or all == True:
        pdf2 = fpdf.PdfPages("%sOverlays_unpaired_Muse_%s%s.pdf" % (dirplots, namegmc, name_end))  # type: PdfPages
    if (paired == True and unpaired == True) or all == True:
        pdf3 = fpdf.PdfPages(
            "%sOverlays_paired_and_unpaired_Muse_%s%s.pdf" % (dirplots, namegmc, name_end))  # type: PdfPages

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

                fig = plt.figure( figsize=(10, 10)) #no figsize if zommed figsize=(10, 10)

                axs = plt.subplot(1, 1, 1, projection=WC)
                #plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.9)
                #axs.add_subplot(111, projection=WC)
                #axs.tick_params(axis="x", labelsize=30)
                #axs.tick_params(axis="y", labelsize=30)
                #axs.set_xlabel('RA' ,fontsize = '16')
                #axs.set_ylabel('DEC' ,fontsize = '16')


                #plt.suptitle('%s' % (galaxias[j]))

                axs.imshow(image_reprojected, vmax=1000, cmap='Greys') #6000
                axs.coords['ra'].set_axislabel('Right Ascension')#, fontsize = 30)
                axs.coords['dec'].set_axislabel(' ')#, fontsize = 30)
                #print(np.shape(image_reprojected))


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
    if (paired == True and unpaired == True) or all == True:
        pdf3.close()

overlays(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True,
         show=True, save=False,
         vel_limit=10000, threshold_perc=0.9, randomize='', paired = True, unpaired = True)


# overlays(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True,
#          show=False, save=True,
#          vel_limit=10000, threshold_perc=0.5, randomize='', unpaired=True, paired = True)
#
# overlays(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True,
#          show=False, save=True,
#          vel_limit=10000, threshold_perc=0.9, randomize='', unpaired=True, paired = True)


