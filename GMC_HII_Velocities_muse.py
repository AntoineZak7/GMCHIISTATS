import numpy as np
import pickle
import matplotlib.pyplot as plt
from itertools import chain
import matplotlib.backends.backend_pdf as fpdf
import os

def gmc_hii_vel(new_muse, gmc_catalog, overlap_matching, threshold_perc, outliers):


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

    # =======================================================================================================

    subplot_x = 4
    subplot_y = 2


    pdf1 = fpdf.PdfPages(
        "%sIndividual galaxy - GMCs & HII Velocities Histograms - GMC catalog: %s%s" % (dirplots, typegmc, name_end))


    fig, axs = plt.subplots(subplot_x, subplot_y, figsize=(8, 10))
    plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.9)
    axs = axs.ravel()
    plt.suptitle('- Histograms of HII and GMCs velocities - \n GMC catalog: %s%s' % (typegmc,name_end))

    for j in range(subplot_y * subplot_x):
        if j < len(galaxias):
            velhii = velHIIover[j]
            velgmc = velGMCover[j]
            # velhii = velhii[velhii != 0]
            # velgmc = velgmc[velgmc != 0]

            max_velhii = np.nanmax(velhii) + 1
            min_velhii = np.nanmin(velhii)
            max_velgmc = np.nanmax(velgmc) + 1
            min_velgmc = np.nanmin(velgmc)
            binresolution = 50
            bin_hii = np.arange(min_velhii, max_velhii, (max_velhii - min_velhii) / (binresolution))
            bin_gmc = np.arange(min_velgmc, max_velgmc, (max_velgmc - min_velgmc) / (binresolution))
            axs[j].hist(velhii, bins=bin_hii, histtype='stepfilled', label='%s' % galaxias[j],
                        alpha=0.5)
            axs[j].hist(velgmc, bins=bin_gmc, histtype='stepfilled', label='%s' % galaxias[j], alpha=0.5)
            axs[j].legend(['HII', 'GMCs'])
            axs[j].set_title('%s' % galaxias[j])
            axs[j].grid(alpha=0.3)
            axs[j].set_xlabel('Velocity (km/s)')

    pdf1.savefig(fig)
    plt.show()
    pdf1.close()

    for j in range(subplot_y * subplot_x):
        if j < len(galaxias):
            velhii = velHIIover[j]
            velgmc = velGMCover[j]

            dist = DisHIIGMCover[j]
            veloffset = velhii - velgmc
            veloffset[(veloffset > 2000)] = 0
            veloffset[(veloffset < -150)] = 0

            # plt.show()

    # pdf.savefig(fig)
    # plt.close()
    # pdf.close()

#================================================Individual galaxy===========================================================#

    pdf2 = fpdf.PdfPages("%sIndividual galaxy - Velocity Offset Histograms - GMC catalog: %s%s" % (dirplots, typegmc, name_end))

    fig, axs = plt.subplots(subplot_x, subplot_y, figsize=(8, 10))
    plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.9)
    axs = axs.ravel()
    plt.suptitle('- Histograms of offset velocity - \n GMC catalog: %s%s' % (typegmc, name_end))

    vel_offset_tot = []
    regions_id = (7, 8, 9)
    for j in range(subplot_y * subplot_x):
        if j < len(galaxias):
            id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
            if str(id_int) != '(array([], dtype=int64),)':
                velhii = velHIIover[j][id_int]
                velgmc = velGMCover[j][id_int]
            else:
                velhii = np.array([0])
                velgmc = np.array([0])
            vel_offset = velhii - velgmc
            vel_offset = vel_offset[(vel_offset < 2000) & (vel_offset > -2000)]

            max_veloffset = np.nanmax(vel_offset) + 1
            min_veloffset = np.nanmin(vel_offset)

            binresolution = 50
            bin = np.arange(min_veloffset, max_veloffset, abs(max_veloffset - min_veloffset) / (binresolution))
            axs[j].hist(vel_offset, bins=bin, histtype='stepfilled', label='%s' % galaxias[j])
            axs[j].set_title('%s' % galaxias[j])
            axs[j].grid(alpha=0.3)
            axs[j].set_xlabel('Offset Velocity (km/s) (Vel_HII - Vel_GMC)')
        vel_offset_tot.append(vel_offset)


    pdf2.savefig(fig)
    plt.show()
    pdf2.close()

    velhii_flat = list(chain.from_iterable(velHIIover))
    velgmc_flat = list(chain.from_iterable(velGMCover))

    vel_offset_tot = list(chain.from_iterable((vel_offset_tot)))
    flat_veloffset = vel_offset_tot
    max = np.nanmax(flat_veloffset) + 1
    min = np.nanmin(flat_veloffset)
    binresolution = 150



#=======================================ALL GALAXIES==================================================#

    pdf3 = fpdf.PdfPages("%sAll galaxies - Velocity Offset - GMC catalog: %s%s" % (dirplots, typegmc, name_end))


    fig = plt.figure()
    plt.hist(flat_veloffset, bins=np.arange(min, max, (max - min) / binresolution), histtype='stepfilled')
    plt.xlabel('Velocity offset (vel_HII - Vel_GMC) (km/s)')
    plt.grid(alpha=0.5, linestyle='-')
    plt.title('- Velocity offset all pairs - \n GMC catalog: %s%s' % (typegmc, name_end))


    pdf3.savefig(fig)
    plt.show()
    pdf3.close()

    print("std = %f km/s" %np.std(vel_offset_tot))
    print("mean = %f km/s" %np.mean(vel_offset_tot))
    print("median = %f km/s" %np.median(vel_offset_tot))