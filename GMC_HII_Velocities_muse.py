import numpy as np
import pickle
import matplotlib.pyplot as plt
from itertools import chain
import matplotlib.backends.backend_pdf as fpdf
import os

dir_script_data = os.getcwd() + "/script_data/"
dirhii, dirgmc, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots, dirhiimasks = pickle.load(
    open(dir_script_data + 'Directories_muse.pickle', "rb"))  # retrieving the directories paths


def name(matching, without_out, new_muse, gmc_catalog, threshold_perc, vel_limit):
    name_append = ['', 'with_outliers', 'without_outliers', 'new_muse_', 'old_muse_', str(threshold_perc), 'vel_limit=']

    if new_muse == True:
        name_end = name_append[3] + gmc_catalog + matching + 'vel_limit:' + str(vel_limit)
        if matching != "distance":
            name_end = name_end + name_append[5]
            if without_out == True:
                name_end = name_end + name_append[2]
            else:
                name_end = name_end + name_append[1]

    else:
        name_end = name_append[4] + gmc_catalog + matching + 'vel_limit:' + str(vel_limit)
        if matching != "distance":
            name_end = name_end + name_append[5]
            if without_out == True:
                name_end = name_end + name_append[2]
            else:
                name_end = name_end + name_append[1]
    return name_end

def get_data(new_muse, gmc_catalog, matching, outliers, threshold_perc, vel_limit):
    # ==============================================================================#

    typegmc = gmc_catalog  # '_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_

    without_out = not outliers
    name_end = name(matching, without_out, new_muse, gmc_catalog, threshold_perc, vel_limit)
    # ==============================================================================#

    namegmc = "_12m+7m+tp_co21%sprops" % typegmc

    # ====================================================================================================================#

    # =========================Getting all the GMC and HII properties from the pickle files===============================#


    galaxias, GMCprop1, HIIprop1, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay, idoverhii, idovergmc = pickle.load(
        open(dir_script_data + 'Galaxies_variables_GMC%s%s.pickle' % ( namegmc, name_end),
             "rb"))  # retrieving the regions properties

    SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, \
    velHIIover, HIIminorover, HIImajorover, HIIangleover = HIIprop1

    HIIprop = SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover

    DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, \
    tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover = GMCprop1

    GMCprop = DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover

    SizepcHII, LumHacorr, sigmavHII, metaliHII, varmetHII, numGMConHII, \
    MasscoGMC, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC = pickle.load(
        open(dir_script_data + 'Galaxies_variables_notover_GMC%s%s.pickle' % ( namegmc, name_end), "rb"))

    shortlab = ['HIIGMCdist', 'Mco', 'GMCsize', 'Smol', 'sigmav', 'avir', 'TpeakCO', 'tauff']
    MassesCO = [1e5 * i for i in MasscoGMCover]  #

    labsyay = labsyay[0:len(labsyay) - 5]  # removing  vel, major axis, minor axis and PA, no need to plot them
    labsxax = labsxax[0:len(labsxax) - 4]

    arrayyay = GMCprop
    arrayxax = HIIprop

    # Limits in the properties of HIIR and GMCs
    xlim, ylim, xx, yy = pickle.load(open(    dir_script_data + 'limits_properties.pickle', "rb"))

    return labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, velGMCover, velHIIover

def gmc_hii_vel(new_muse, gmc_catalog, matching, threshold_perc, outliers, vel_limit):


    # ==============================================================================#
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, velGMCover, velHIIover = get_data(new_muse, gmc_catalog,
                                                                                            matching, outliers,
                                                                                            threshold_perc, vel_limit)

    # =======================================================================================================

    subplot_x = 4
    subplot_y = 2


    pdf1 = fpdf.PdfPages(
        "%sIndividual galaxy - GMCs & HII Velocities Histograms - GMC catalog: %s%s" % (dirplots, namegmc, name_end))


    fig, axs = plt.subplots(subplot_x, subplot_y, figsize=(8, 10))
    plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.9)
    axs = axs.ravel()
    plt.suptitle('- Histograms of HII and GMCs velocities - \n GMC catalog: %s%s' % (namegmc,name_end))

    for j in range(subplot_y * subplot_x):
        if j < len(galaxias):
            velhii = velHIIover[j]
            velgmc = velGMCover[j]

            print(velhii)
            print(velgmc)
            # velhii = velhii[velhii != 0]
            # velgmc = velgmc[velgmc != 0]

            print(np.size(velhii))
            if np.size(velhii) > 0:
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

            veloffset = velhii - velgmc
            veloffset[(veloffset > 2000)] = 0
            veloffset[(veloffset < -150)] = 0

            plt.show()

    # pdf.savefig(fig)
    # plt.close()
    # pdf.close()

#================================================Individual galaxy===========================================================#

    pdf2 = fpdf.PdfPages("%sIndividual galaxy - Velocity Offset Histograms - GMC catalog: %s%s" % (dirplots, namegmc, name_end))

    fig, axs = plt.subplots(subplot_x, subplot_y, figsize=(8, 10))
    plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.9)
    axs = axs.ravel()
    plt.suptitle('- Histograms of offset velocity - \n GMC catalog: %s%s' % (namegmc, name_end))

    vel_offset_tot = []
    regions_id = (7, 8, 9)
    for j in range(subplot_y * subplot_x):
        if j < len(galaxias):
            #id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
            #if str(id_int) != '(array([], dtype=int64),)':
                #velhii = velHIIover[j][id_int]
                #velgmc = velGMCover[j][id_int]
            #else:
                #velhii = np.array([0])
                #velgmc = np.array([0])

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

    pdf3 = fpdf.PdfPages("%sAll galaxies - Velocity Offset - GMC catalog: %s%s" % (dirplots, namegmc, name_end))

    flat_veloffset = np.array(flat_veloffset)
    fig = plt.figure()
    plt.hist(flat_veloffset, bins=np.arange(min, max, (max - min) / binresolution), histtype='stepfilled')
    plt.xlabel('Velocity offset (vel_HII - Vel_GMC) (km/s)')
    plt.grid(alpha=0.5, linestyle='-')
    plt.title('- Velocity offset all pairs - \n GMC catalog: %s%s' % (namegmc, name_end))


    pdf3.savefig(fig)
    plt.show()
    pdf3.close()

    print("std = %f km/s" %np.std(vel_offset_tot))
    print("mean = %f km/s" %np.mean(vel_offset_tot))
    print("median = %f km/s" %np.median(vel_offset_tot))

gmc_hii_vel(new_muse=True, gmc_catalog= "_150pc_homogenized_", matching="overlap_1o1",threshold_perc=0.1,outliers=True, vel_limit = 1000)