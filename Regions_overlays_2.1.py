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
from astropy.table import Table
import gmchiistats_routines
import gmchiistats_routines as data_routines
import Stats_test_2_21 as stattests

warnings.filterwarnings("ignore")
from matplotlib import rc




def overlays(muse, gmc_catalog, gmc_catalog_version, matching, outliers, show, save, vel_limit, threshold_percs,
             randomize,sorting, *args, **kwargs):


    symmetrical = 'gmc'

    dir_script_data = os.getcwd() + "/script_data_dr22/"

    galaxias_sorted_r2 = ['ngc1433', 'ngc5068', 'ngc1365', 'ngc1385', 'ngc7496', 'ngc2835', 'ngc0628', 'ngc4303', 'ngc1300', 'ngc4254', 'ngc1087', 'ngc1512', 'ngc1566', 'ngc1672', 'ngc4321', 'ngc3627', 'ngc4535', 'ngc3351']
    sorted_r2 = [0.11304239520877529, 0.12497910081466236, 0.13541830078770825, 0.1881314669919857, 0.20650250969524178, 0.21157776080010318, 0.25013082036704287, 0.27192649394474144, 0.3092914675228949, 0.352161598138343, 0.35275177693334053, 0.3564571110126658, 0.41810048307630204, 0.47916417434553243, 0.5089082740461176, 0.51709886893152, 0.5911856678471191, 0.6236722593266394]

    galaxias_sorted_slope = ['ngc1433', 'ngc1365', 'ngc5068', 'ngc2835', 'ngc1385', 'ngc1512', 'ngc1087', 'ngc0628', 'ngc4303', 'ngc1300', 'ngc1566', 'ngc7496', 'ngc4321', 'ngc4254', 'ngc3351', 'ngc1672', 'ngc3627', 'ngc4535']
    sorted_slope = [0.13277640088168918, 0.13433365466411928, 0.14440995295291928, 0.1656561691956596, 0.20555275604752857, 0.2122238935644426, 0.23748479126056554, 0.2540454988991154, 0.2584701076899694, 0.2916059659515288, 0.3120747521303841, 0.3223247307827655, 0.32305434716550213, 0.33914556310580674, 0.3670393551330627, 0.3789973647210445, 0.393256635249105, 0.4205083182540185]

    galaxias = ['NGC0628', 'NGC1087', 'NGC1300', 'NGC1365', 'NGC1385', 'NGC1433', 'NGC1512', 'NGC1566', 'NGC1672', 'NGC2835', 'NGC3351', 'NGC3627', 'NGC4254', 'NGC4303', 'NGC4321', 'NGC4535', 'NGC5068', 'NGC7496']

    sorted_index_gal_r2 = []
    for value in galaxias_sorted_r2:
        for idd, item in enumerate(galaxias):
            if str.lower(item) == value:
                print(item)
                sorted_index_gal_r2.append(idd)

    sorted_index_gal_slope = []
    for value in galaxias_sorted_slope:
        for idd, item in enumerate(galaxias):
            if str.lower(item) == value:
                print(item)
                sorted_index_gal_slope.append(idd)


    paired = kwargs.get('paired', None)
    unpaired = kwargs.get('unpaired', None)
    all = kwargs.get('all', None)

    table = Table.read('/home/antoine/Internship/phangs_sample_table_v1p6.fits')
    table_muse = Table.read('/home/antoine/Internship/muse_hii_new_dr22/Nebulae_catalogue_v2.fits')

    w = 0
    galaxies_name = ['IC5332']

    for i in range(len(table_muse['gal_name']) - 1):
        if galaxies_name[w] != str(table_muse['gal_name'][i]):
            galaxies_name.append(str(table_muse['gal_name'][i]))
            w += 1

    galnames = galaxies_name

    galnames = [str.lower(x) for x in galnames]
    ids = [id for id, x in enumerate(table['name']) if x in galnames]

    SFR = table['props_sfr'][ids]
    sorted_SFR = table['props_sfr'][ids]
    stellar_Mass_1 = table['props_mstar'][ids]
    sorted_stellar_mass = table['props_mstar'][ids]

    sorted_stellar_mass.sort()

    sorted_stellar_mass.sort()

    sorted_SFR.sort()

    ind_value_mass = []
    for value in sorted_stellar_mass:
        for idd, item in enumerate(stellar_Mass_1):
            if item == value:
                ind_value_mass.append(idd)

    ind_value_sfr = []
    for value in sorted_SFR:
        for idd, item in enumerate(SFR):
            if item == value:
                ind_value_sfr.append(idd)

    ind_value_mass = []
    for value in sorted_stellar_mass:
        for idd, item in enumerate(stellar_Mass_1):
            if item == value:
                ind_value_mass.append(idd)

    ind_value_sfr = []
    for value in sorted_SFR:
        for idd, item in enumerate(SFR):
            if item == value:
                ind_value_sfr.append(idd)

    if sorting == 'mass':
        ind_value = ind_value_mass
        sorting = 'stellar mass'
    if sorting == 'sfr':
        ind_value = ind_value_sfr







    def save_pdf(pdf):
        if save == True:
            pdf.savefig(fig)
        if show == True:

            plt.show()
        else:
            plt.close()

    def name(matching, without_out, muse, gmc_catalog, gmc_catalog_version, threshold_perc, vel_limit, randomize,
             symmetrical):
        name_end = 'muse:' + muse + '_' + 'gmc:' + gmc_catalog + '(' + gmc_catalog_version + ')_' + 'vel_limit:' + str(
            vel_limit) + '_matching:' + matching + '_' + randomize + '_' + symmetrical

        if matching != "distance":
            name_end = name_end + '(' + str(threshold_perc).split(sep='.')[0] + str(threshold_perc).split(sep='.')[
                1] + ')'
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

        plt.title(galaxias[j])
        k = 0
        for i in range(len(hii_regions_paired)):
            center = hii_regions_paired[i].center
            circle_sky = CircleSkyRegion(center=center, radius=major_over[i] * u.deg)
            circle_pixel = circle_sky.to_pixel(wcs=WC)
            circle_pixel.plot(ax = axs[i1],color='blue', linestyle='-.')

        for i in range(len(gmc_regions_paired)):
            center = gmc_regions_paired[i].center
            ellipse_sky_gmc = EllipseSkyRegion(center=center, height=major_gmc_over[i] * u.deg,
                                               width=minor_gmc_over[i] * u.deg,
                                               angle=(angle_gmc_over[i]) * u.deg)
            ellipse_pixel_gmc = ellipse_sky_gmc.to_pixel(wcs=WC)
            ellipse_pixel_gmc.plot(ax = axs[i1],color='lightblue')

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
            circle_pixel.plot(ax = axs[i1],color='red', linestyle='-.')

        for i in index_gmc:
            center = gmc_regions[i].center
            ellipse_sky_gmc = EllipseSkyRegion(center=center, height=major_gmc[i] * u.deg,
                                               width=minor_gmc[i] * u.deg, angle=(angle_gmc[i]) * u.deg)
            ellipse_pixel_gmc = ellipse_sky_gmc.to_pixel(wcs=WC)
            ellipse_pixel_gmc.plot(ax = axs[i1],color='salmon')

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
            #circle_pixel.plot(ax = axs[i1],color='red', linestyle='-.')

        for i in index_gmc:
            center = gmc_regions[i].center
            ellipse_sky_gmc = EllipseSkyRegion(center=center, height=major_gmc[i] * u.deg, width = minor_gmc[i] * u.deg, angle = angle_gmc[i] * u.deg)
            # width=minor_gmc[i] * u.deg, angle=(angle_gmc[i]) * u.deg)
            ellipse_pixel_gmc = ellipse_sky_gmc.to_pixel(wcs=WC)
            #ellipse_pixel_gmc.plot(ax = axs[i1],color='salmon')

        for i in range(len(gmc_regions_paired)):
            center = gmc_regions_paired[i].center
            ellipse_sky_gmc = EllipseSkyRegion(center=center, height=major_gmc_over[i] * u.deg, width = minor_gmc_over[i] * u.deg, angle=angle_gmc_over[i] * u.deg)
            # width=minor_gmc_over[i] * u.deg, angle=(angle_gmc_over[i]) * u.deg)
            ellipse_pixel_gmc = ellipse_sky_gmc.to_pixel(wcs=WC)
            #ellipse_pixel_gmc.plot(ax = axs[i1],color='lightblue')

        for i in range(len(hii_regions_paired)):
            center = hii_regions_paired[i].center
            circle_sky = CircleSkyRegion(center=center, radius=major_over[i] * u.deg)
            circle_pixel = circle_sky.to_pixel(wcs=WC)
            #circle_pixel.plot(ax = axs[i1],color='blue', linestyle='-.')

    typegmc = gmc_catalog  # '_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_
    # ==============================================================================#

    without_out = not outliers

    name_end = name(without_out=without_out, matching=matching, muse=muse, gmc_catalog=gmc_catalog,
                    gmc_catalog_version=gmc_catalog_version, threshold_perc=threshold_percs[0], vel_limit=vel_limit, randomize= randomize, symmetrical=symmetrical)


    namegmc = "_12m+7m+tp_co21%sprops" % typegmc

    dir_script_data = os.getcwd() + "/script_data_dr22/"

    dirhii_dr1,dirhii_dr2, dirgmc_old,dirgmc_new, dirregions1, dirregions2, dirregions22, dirmaps, dirplots1, dirplots2, dirplots22, dirplots, dirhiimasks, dirgmcmasks, sample_table_dir = pickle.load(
        open(dir_script_data + 'Directories_muse.pickle', "rb"))  # retrieving the directories paths
    # dirmuseproperties = os.path.dirname(os.path.realpath("Extract_info_plot_per_gal_muse.py")) + "/"

    if muse == 'dr1':
        dirplots = dirplots1
        dirregions = dirregions1
    elif muse == 'dr2':
        dirplots = dirplots2
        dirregions = dirregions2
    elif muse == 'dr22':
        dirregions = dirregions22
        dirplots = dirplots22

    print(dirregions)



    name_gmc = "_12m+7m+tp_co21" + typegmc + "props-GMCs-matched" + name_end + ".reg"
    name_hii = "_12m+7m+tp_co21" + typegmc + "props-HIIregions-matched" + name_end + ".reg"
    name_hii_all = "_12m+7m+tp_co21" + typegmc + "props-HIIregions-all_regions" + name_end + ".reg"
    name_gmc_all = "_12m+7m+tp_co21" + typegmc + "props-GMCs-all_regions" + name_end + ".reg"

    galaxias, GMCprop, HIIprop, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay, idoverhiis, idovergmcs,a,b = pickle.load(
        open(dir_script_data + 'Galaxies_variables_GMC%s%s.pickle' % (namegmc, name_end), "rb"))
    SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, velHIIover, HIIminorover, HIImajorover, HIIangleover,Rgal_hii, hii_reff, hii_r25, hii_region_index = HIIprop
    DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover, test,Rgal_gmc = GMCprop

    SizepcHII, LumHacorrnot, sigmavHII, metaliHII, varmetHII, numGMConHII, \
    FluxCOGMCnot, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII, SigmaMol, Sigmav, COTpeak,VelHII,VelGMC = pickle.load(
        open(dir_script_data + 'Galaxies_variables_notover_GMC%s%s.pickle' % ( namegmc, name_end), "rb"))


    # ==========================All Paired and unpaired=====================================#
    if (paired == True and unpaired == False) or all == True:
        pdf1 = fpdf.PdfPages("%sOverlays_paired_Muse_%s%s.pdf" % (dirplots, namegmc, name_end))  # type: PdfPages
        print('pdf1')
    if (unpaired == True and paired == False) or all == True:
        print('pdf2')
        pdf2 = fpdf.PdfPages("%sOverlays_unpaired_Muse_%s%s.pdf" % (dirplots, namegmc, name_end))  # type: PdfPages
    if (paired == True and unpaired == True) or all == True:
        print('pdf3')
        pdf3 = fpdf.PdfPages(
            "%sOverlays_paired_and_unpaired_Muse_%s%s.pdf" % (dirplots, namegmc, name_end))  # type: PdfPages





    name_ha = "_ha.fits"  # "_Hasub_flux.fits"



    for thres in threshold_percs:
        threshold_perc = thres

        name_end = name(without_out=without_out, matching=matching, muse=muse, gmc_catalog=gmc_catalog,
                        gmc_catalog_version=gmc_catalog_version, threshold_perc=thres, vel_limit=vel_limit,
                        randomize=randomize, symmetrical=symmetrical)

        fig, axs = plt.subplots(4, 5, figsize=(12, 12))

        axs = axs.ravel()



        galaxias, GMCprop, HIIprop, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay, idoverhiis, idovergmcs,a,b = pickle.load(
            open(dir_script_data + 'Galaxies_variables_GMC%s%s.pickle' % (namegmc, name_end), "rb"))
        SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, velHIIover, HIIminorover, HIImajorover, HIIangleover, Rgal_hii, reff_hii, r25_hii, hii_region_index = HIIprop
        DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover, test, Rgal_gmc = GMCprop

        SizepcHII, LumHacorrnot, sigmavHII, metaliHII, varmetHII, numGMConHII, \
        FluxCOGMCnot, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, SigmaMol, Sigmav, COTpeak,VelHII,VelGMC = pickle.load(
            open(dir_script_data + 'Galaxies_variables_notover_GMC%s%s.pickle' % (namegmc, name_end), "rb"))

        galaxias = [str.lower(x) for x in galaxias]

        i1 = 0
        r2_slope_list = sorted_r2
        sorted_index_gal = sorted_index_gal_r2

        for j in sorted_index_gal:

            print("%i/%i" %(j,len(galaxias)))


            if os.path.isfile("%s%s%s" % (dirmaps, str.upper(galaxias[j]), name_ha)) == True:
                if os.path.isfile("%s%s%s" % (dirregions, galaxias[j], name_gmc)) == True and os.path.isfile(
                        "%s%s%s" % (dirregions, galaxias[j], name_hii)) == True:
                    if muse == 'dr2':
                        hii_suffix = '_HIIreg_mask.fits'
                    elif muse == 'dr22':
                        hii_suffix = '_nebulae_mask_V2.fits'
                    mask = fits.open(dirhiimasks + str.upper(galaxias[j]) + hii_suffix)
                    hdul = fits.open(dirmaps + str.upper(galaxias[j]) + name_ha)
                    hdr = hdul[0].header
                    image_data = hdul[0].data
                    WC = wcs.WCS(mask[0].header)
                    image_reprojected, footprint = reproject_interp(hdul, mask[0].header)
                    flat_imagedata = image_data.flatten()


                    axs[i1].imshow(image_reprojected, vmax=6000, cmap='Greys')
                    axs[i1].set_xlabel(galaxias[j] +'(r² = %1.2f)'%r2_slope_list[i1], fontsize = 12)




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

                    i1+=1

        if paired == True:
            if unpaired == True:
                save_pdf(pdf3)
            else:
                save_pdf(pdf1)
        else:
            if unpaired == True:
                if paired == True:
                    save_pdf(pdf3)
                else:
                    save_pdf(pdf2)
        if all == True:
            save_pdf(pdf1)
            save_pdf(pdf2)
            save_pdf(pdf3)

    if (paired == True and unpaired == False) or all == True:
        pdf1.close()
    if (unpaired == True and paired == False )or all == True:
        pdf2.close()
    if (paired == True and unpaired == True) or all == True:
        pdf3.close()

overlays(muse='dr22', gmc_catalog="_native_", gmc_catalog_version='v4', matching="overlap_1om", outliers=True,
         show=False, save=True,
             vel_limit=10000, threshold_percs=[0.9], randomize='', paired = True, unpaired = False, sorting = 'r2')


# overlays(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True,
#           show=False, save=True,
#              vel_limit=10000, threshold_percs=[0.5], randomize='', paired= True, sorting = 'sfr')
#
# overlays(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True,
#           show=False, save=True,
#              vel_limit=10000, threshold_percs=[0.5], randomize='', unpaired = True, sorting = 'sfr')

