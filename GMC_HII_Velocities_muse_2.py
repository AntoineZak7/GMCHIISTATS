import numpy as np
import pickle
import matplotlib.pyplot as plt
from itertools import chain
import matplotlib.backends.backend_pdf as fpdf
import os

dir_script_data = os.getcwd() + "/script_data_dr2/"
dirhii_dr1, dirhii_dr2, dirgmc_old, dirgmc_new, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots, dirhiimasks, dirgmcmasks, dir_sample_table = pickle.load(
    open(dir_script_data + 'Directories_muse.pickle', "rb"))  # retrieving the directories paths


def name(matching, without_out, muse, gmc_catalog, gmc_catalog_version, threshold_perc, vel_limit, randomize):
    name_end = 'muse:' + muse + '_' + 'gmc:' + gmc_catalog + '(' + gmc_catalog_version + ')_' + 'vel_limit:' + str(
        vel_limit) + '_matching:' + matching + '_' + randomize + '_'

    if matching != "distance":
        name_end = name_end + '(' + str(threshold_perc).split(sep='.')[0] + str(threshold_perc).split(sep='.')[1] + ')'
        if without_out == True:
            name_end = name_end + '_' + 'without_outliers'
        else:
            name_end = name_end + '_' + 'with_outliers'

    return name_end

def save_pdf(pdf, fig, save, show):
    if save == True:
        pdf.savefig(fig)
    if show == True:
        plt.show()
    else:
        plt.close()

    def checknaninf(v1, v2, lim1, lim2):
        v1n = np.array(v1)
        v2n = np.array(v2)
        indok = np.where((np.absolute(v1n) < lim1) & (np.absolute(v2n) < lim2))[0].tolist()
        # print indok
        nv1n = v1n[indok].tolist()
        nv2n = v2n[indok].tolist()
        return nv1n, nv2n


def get_data(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, threshold_perc, vel):
    # ==============================================================================#

    typegmc = gmc_catalog  # '_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_

    without_out = not outliers
    name_end = name(without_out=without_out, matching=matching, muse=muse, gmc_catalog=gmc_catalog,
                    gmc_catalog_version=gmc_catalog_version, threshold_perc=threshold_perc, vel_limit=vel, randomize=randomize)
    # ==============================================================================#

    namegmc = "_12m+7m+tp_co21%sprops" % typegmc

    # ====================================================================================================================#

    # =========================Getting all the GMC and HII properties from the pickle files===============================#


    galaxias, GMCprop1, HIIprop1, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay, idoverhii, idovergmc = pickle.load(
        open(dir_script_data + 'Galaxies_variables_GMC%s%s.pickle' % ( namegmc, name_end), "rb"))  # retrieving the regions properties

    SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, \
    velHIIover, HIIminorover, HIImajorover, HIIangleover, Rgal_hii = HIIprop1

    HIIprop = SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, Rgal_hii

    DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, \
    tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover, FluxCOGMCover, Rgal_gmc = GMCprop1

    GMCprop = DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover, Rgal_gmc

    SizepcHII, LumHacorrnot, sigmavHII, metaliHII, varmetHII, numGMConHII, \
    FluxCOGMCnot, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII, SigmaMol, Sigmav, COTpeak , VelHII, VelGMC = pickle.load(
        open(dir_script_data + 'Galaxies_variables_notover_GMC%s%s.pickle' % ( namegmc, name_end), "rb"))

    shortlab = ['HIIGMCdist', 'Mco', 'GMCsize', 'Smol', 'sigmav', 'avir', 'TpeakCO', 'tauff']
    MassesCO = [1e5 * i for i in MasscoGMCover]  #

    labsyay = labsyay[0:len(labsyay) - 5]  # removing  vel, major axis, minor axis and PA, no need to plot them
    labsxax = labsxax[0:len(labsxax) - 4]

    arrayyay = GMCprop
    arrayxax = HIIprop

    # Limits in the properties of HIIR and GMCs
    #xlim, ylim, xx, yy = pickle.load(open(    dir_script_data + 'limits_properties.pickle', "rb"))

    return labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, velGMCover, velHIIover, VelHII, VelGMC


def gmc_hii_vel_offset_single_galaxy(gmc_catalog, gmc_catalog_version, muse, matching, threshold_perc, outliers, vel_limit, randomize, show, save):
    # ==============================================================================#
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, velGMCover, velHIIover = get_data(
        muse=muse, gmc_catalog=gmc_catalog, outliers=outliers, matching=matching, threshold_perc=threshold_perc,
        vel=vel_limit, randomize=randomize, gmc_catalog_version=gmc_catalog_version)


    subplot_x = 4
    subplot_y = 2

    # ================================================Individual galaxy===========================================================#
    if save == True:
        pdf_name ="%sIndividual_galaxy_vel_off%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf2 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf2 = fpdf.PdfPages("blank")



    for page in range(3):

        fig, axs = plt.subplots(subplot_x, subplot_y, figsize=(8, 12), sharex='col')
        plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.9)

        axes = plt.gca()
        axs = axs.ravel()

        # plt.suptitle('- Histograms of offset velocity - \n GMC catalog: %s%s' % (namegmc, name_end))

        regions_id = (7, 8, 9)
        for l in range(subplot_y * subplot_x):
            j = l + page * subplot_x * subplot_y

            plt.tick_params(axis="x", labelsize=16)
            plt.tick_params(axis="y", labelsize=16)

            for tick in axs[l].xaxis.get_major_ticks():
                tick.label.set_fontsize(14)

            for tick in axs[l].yaxis.get_major_ticks():
                tick.label.set_fontsize(14)

            if j < len(galaxias):
                velhii = velHIIover[j]
                velgmc = velGMCover[j]

                vel_offset = velhii - velgmc
                vel_offset = vel_offset[(vel_offset < 2000) & (vel_offset > -2000)]

                if len(vel_offset) > 0:
                    max_veloffset = np.nanmax(vel_offset)
                    min_veloffset = np.nanmin(vel_offset)
                    mean_veloffset = np.nanmean(vel_offset)
                    median_veloffset = np.nanmedian(vel_offset)
                    std_veloffset = np.nanstd(vel_offset)

                    binresolution = 50
                    bin = np.arange(min_veloffset, max_veloffset, abs(max_veloffset - min_veloffset) / (binresolution))
                    axs[l].hist(vel_offset, bins=bin, histtype='stepfilled', label='%s' % galaxias[j])
                    axs[l].set_title('%s' % galaxias[j], fontsize=14)
                    axs[l].grid(alpha=0.3)
                    axs[6].set_xlabel('Velocity Offset (km/s)', fontsize=14)
                    axs[7].set_xlabel('Velocity Offset (km/s)', fontsize=14)

                    axs[l].text(0.98, 0.9, 'Mean vel offset:%5.1f' % ((mean_veloffset)), fontsize=9,
                                horizontalalignment='right',
                                verticalalignment='center', transform=axs[l].transAxes)

                    axs[l].text(0.98, 0.8, 'Median vel offset:%5.1f' % ((median_veloffset)), fontsize=9,
                                horizontalalignment='right',
                                verticalalignment='center', transform=axs[l].transAxes)

                    axs[l].text(0.98, 0.7, 'Std vel offset:%5.1f' % ((std_veloffset)), fontsize=9,
                                horizontalalignment='right',
                                verticalalignment='center', transform=axs[l].transAxes)

                    axs[l].set(xlim=(-vel_limit, vel_limit))



        save_pdf(pdf2, fig, save, show)

    pdf2.close()




def gmc_hii_vel_offset_allgal(gmc_catalog, gmc_catalog_version, muse, matching, threshold_perc, outliers, vel_limit, randomize, save, show):


    # ==============================================================================#
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, velGMCover, velHIIover = get_data(muse = muse, gmc_catalog=gmc_catalog, outliers=outliers, matching=matching, threshold_perc=threshold_perc, vel=vel_limit, randomize=randomize,gmc_catalog_version=gmc_catalog_version)

    # =======================================================================================================

    subplot_x = 4
    subplot_y = 2
    binresolution = 100

#=======================================ALL GALAXIES==================================================#

    if save == True:
        pdf_name ="%sAll_galaxies_vel_off:%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")



    vel_offset_tot = [velHIIover[j] - velGMCover[j] for j in range(len(galaxias))  ]
    vel_offset_tot = list(chain.from_iterable((vel_offset_tot)))
    vel_offset_tot = np.array(vel_offset_tot)
    fig = plt.figure()
    plt.hist(vel_offset_tot, bins=binresolution, histtype='stepfilled', density = True)
    plt.xlabel('Velocity offset (vel_HII - Vel_GMC) (km/s)')
    plt.grid(alpha=0.5, linestyle='-')
    #plt.title('- Velocity offset all pairs - \n GMC catalog: %s%s' % (namegmc, name_end))

    plt.text(30, 0.08, 'Mean vel offset: %5.1f' % (np.mean(vel_offset_tot)), fontsize=10, horizontalalignment='left',
                verticalalignment='center')

    plt.text(30, 0.07, 'Median vel offset: %5.1f' % (np.median(vel_offset_tot)), fontsize=10, horizontalalignment='left',
                verticalalignment='center')

    plt.text(30, 0.06, 'Std vel offset: %5.1f' % (np.std(vel_offset_tot)), fontsize=10, horizontalalignment='left',
                verticalalignment='center')

    plt.xlim(-vel_limit, vel_limit)

    save_pdf(pdf3, fig, save, show)
    pdf3.close()

    print("std = %f km/s" %np.std(vel_offset_tot))
    print("mean = %f km/s" %np.mean(vel_offset_tot))
    print("median = %f km/s" %np.median(vel_offset_tot))

    print(len(vel_offset_tot))

    print(np.mean(vel_offset_tot))



def gmc_hii_vel(gmc_catalog, gmc_catalog_version, muse, matching, threshold_perc, outliers, vel_limit, randomize,save,show):


    # ==============================================================================#
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, velGMCover, velHIIover = get_data(muse = muse, gmc_catalog=gmc_catalog, outliers=outliers, matching=matching, threshold_perc=threshold_perc, vel=vel_limit, randomize=randomize,gmc_catalog_version=gmc_catalog_version)

    # =======================================================================================================

    subplot_x = 4
    subplot_y = 2


    if save == True:
        pdf_name ="%sIndividual_galaxy_vels_hist:%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf1 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf1 = fpdf.PdfPages("blank")



    for page in range(3):

        fig, axs = plt.subplots(subplot_x, subplot_y, figsize=(8, 12))
        plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.9)
        axs = axs.ravel()
        #plt.suptitle('- Histograms of HII and GMCs velocities - \n GMC catalog: %s%s' % (namegmc, name_end))


        for l in range(subplot_y * subplot_x):
            j = l + page*subplot_x*subplot_y


            for tick in axs[l].xaxis.get_major_ticks():
                tick.label.set_fontsize(14)

            for tick in axs[l].yaxis.get_major_ticks():
                tick.label.set_fontsize(14)


            if j < len(galaxias):
                velhii = velHIIover[j]
                velgmc = velGMCover[j]

                print(len(arrayyay[1][j]))



                if np.size(velhii) > 0:
                    max_velhii = np.nanmax(velhii) + 1
                    min_velhii = np.nanmin(velhii)
                    max_velgmc = np.nanmax(velgmc) + 1
                    min_velgmc = np.nanmin(velgmc)
                    binresolution = 50
                    bin_hii = np.arange(min_velhii, max_velhii, (max_velhii - min_velhii) / (binresolution))
                    bin_gmc = np.arange(min_velgmc, max_velgmc, (max_velgmc - min_velgmc) / (binresolution))
                    axs[l].hist(velhii, bins=bin_hii, histtype='stepfilled', label='%s' % galaxias[j],
                                alpha=0.5)
                    axs[l].hist(velgmc, bins=bin_gmc, histtype='stepfilled', label='%s' % galaxias[j], alpha=0.5)
                    axs[0].legend(['HII', 'GMCs'])
                    axs[l].set_title('%s' % galaxias[j], fontsize = 16)
                    axs[l].grid(alpha=0.3)
                    axs[6].set_xlabel('Velocity (km/s)', fontsize = 14)
                    axs[7].set_xlabel('Velocity (km/s)', fontsize = 14)

        save_pdf(pdf1, fig, save, show)
    pdf1.close()


def gmc_vel(gmc_catalog, gmc_catalog_version, muse, matching, threshold_perc, outliers, vel_limit, randomize,save,show):


    # ==============================================================================#
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, velGMCover, velHIIover, VelHII, VelGMC = get_data(muse = muse, gmc_catalog=gmc_catalog, outliers=outliers, matching=matching, threshold_perc=threshold_perc, vel=vel_limit, randomize=randomize,gmc_catalog_version=gmc_catalog_version)

    # =======================================================================================================


    gal_incl = [17.35,21,47,55.78, 53.86,43.9, 48.63, 59.15,50,87,33.77,51.91,52.91,63.47,29.38,26.94,29.37,26.84,55.93]
    gal_incl_sorted = [17.35,21,47,55.78, 53.86,43.9, 48.63, 59.15,50,87,33.77,51.91,52.91,63.47,29.38,26.94,29.37,26.84,55.93]
    gal_incl_sorted.sort()


    ind_value_incl = []
    for value in gal_incl_sorted:
        for idd, item in enumerate(gal_incl):
            if item == value:
                ind_value_incl.append(idd)


    print(len(gal_incl))
    velHIIover = [velHIIover[ind] for ind in ind_value_incl]
    velGMCover = [velGMCover[ind] for ind in ind_value_incl]
    VelGMC = [VelGMC[ind] for ind in ind_value_incl]
    VelHII = [VelHII[ind] for ind in ind_value_incl]
    galaxias = [galaxias[ind] for ind in ind_value_incl]

    if save == True:
        pdf_name ="%sIndividual_galaxy_gmc_vels_hist:%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf1 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf1 = fpdf.PdfPages("blank")




    fig, axs = plt.subplots(5, 4, figsize=(17, 15))
    plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.9)
    axs = axs.ravel()
    #plt.suptitle('- Histograms of HII and GMCs velocities - \n GMC catalog: %s%s' % (namegmc, name_end))

    for j in range(20):

        if j < len(galaxias):
            velhii = velHIIover[j]
            velgmc = velGMCover[j]
            velgmcall = VelGMC[j]

            max_velhii = np.nanmax(velhii) + 1
            min_velhii = np.nanmin(velhii)
            max_velgmc = np.nanmax(velgmc) + 1
            min_velgmc = np.nanmin(velgmc)
            binresolution = 50
            bin_gmc = np.arange(min_velgmc, max_velgmc, (max_velgmc - min_velgmc) / (binresolution))

            axs[j].hist(velgmcall, bins=bin_gmc, histtype='stepfilled', label='%s' % galaxias[j], alpha=0.5, density = True, color = 'red')
            axs[j].hist(velgmc, bins=bin_gmc, histtype='stepfilled', label='%s' % galaxias[j], alpha=0.5, density = True, color = 'blue')
            axs[0].legend(['All GMCs', 'Matched GMCs'])
            axs[j].set_title('%s' % galaxias[j], fontsize = 12)
            axs[j].grid(alpha=0.3)
            axs[j].set_xlabel('Velocity (km/s)', fontsize = 10)

    save_pdf(pdf1, fig, save, show)
    pdf1.close()

def hii_vel(gmc_catalog, gmc_catalog_version, muse, matching, threshold_perc, outliers, vel_limit, randomize,save,show):


    # ==============================================================================#
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, velGMCover, velHIIover, VelHII, VelGMC = get_data(muse = muse, gmc_catalog=gmc_catalog, outliers=outliers, matching=matching, threshold_perc=threshold_perc, vel=vel_limit, randomize=randomize,gmc_catalog_version=gmc_catalog_version)

    # =======================================================================================================


    gal_incl = [17.35,21,47,55.78, 53.86,43.9, 48.63, 59.15,50,87,33.77,51.91,52.91,63.47,29.38,26.94,29.37,26.84,55.93]
    gal_incl_sorted = [17.35,21,47,55.78, 53.86,43.9, 48.63, 59.15,50,87,33.77,51.91,52.91,63.47,29.38,26.94,29.37,26.84,55.93]
    gal_incl_sorted.sort()


    ind_value_incl = []
    for value in gal_incl_sorted:
        for idd, item in enumerate(gal_incl):
            if item == value:
                ind_value_incl.append(idd)


    print(len(gal_incl))
    velHIIover = [velHIIover[ind] for ind in ind_value_incl]
    velGMCover = [velGMCover[ind] for ind in ind_value_incl]
    VelGMC = [VelGMC[ind] for ind in ind_value_incl]
    VelHII = [VelHII[ind] for ind in ind_value_incl]

    galaxias = [galaxias[ind] for ind in ind_value_incl]

    if save == True:
        pdf_name ="%sIndividual_galaxy_hii_vels_hist:%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf1 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf1 = fpdf.PdfPages("blank")




    fig, axs = plt.subplots(5, 4, figsize=(17, 15))
    plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.9)
    axs = axs.ravel()
    #plt.suptitle('- Histograms of HII and GMCs velocities - \n GMC catalog: %s%s' % (namegmc, name_end))

    for j in range(20):

        if j < len(galaxias):
            velhii = velHIIover[j]
            velgmc = velGMCover[j]
            velgmcall = VelGMC[j]
            velhiiall = VelHII[j]

            max_velhii = np.nanmax(velhii) + 1
            min_velhii = np.nanmin(velhii)
            max_velgmc = np.nanmax(velgmc) + 1
            min_velgmc = np.nanmin(velgmc)
            binresolution = 50
            bin_gmc = np.arange(min_velgmc, max_velgmc, (max_velgmc - min_velgmc) / (binresolution))

            axs[j].hist(velhiiall, bins=bin_gmc, histtype='stepfilled', label='%s' % galaxias[j], alpha=0.5, density = True, color = 'red')
            axs[j].hist(velhii, bins=bin_gmc, histtype='stepfilled', label='%s' % galaxias[j], alpha=0.5, density = True, color = 'blue')
            axs[0].legend(['All HII regions', 'Matched HII regions'])
            axs[j].set_title('%s' % galaxias[j], fontsize = 12)
            axs[j].grid(alpha=0.3)
            axs[j].set_xlabel('Velocity (km/s)', fontsize = 10)

    save_pdf(pdf1, fig, save, show)
    pdf1.close()

#

#gmc_hii_vel(new_muse=True, gmc_catalog= "_150pc_homogenized_", matching="overlap_1o1",threshold_perc=0.1,outliers=True, vel_limit = 1000)
#gmc_hii_vel(gmc_catalog="_native_", gmc_catalog_version='new', muse='dr2', matching="overlap_1om", outliers=True,threshold_perc=0.1 , vel_limit= 100, randomize = '',save=True,show = True)
#gmc_hii_vel_offset_single_galaxy(gmc_catalog="_native_", gmc_catalog_version='new', muse='dr2', matching="overlap_1om", outliers=True,threshold_perc=0.5 , vel_limit= 100, randomize = '',save=True,show = True)
#gmc_hii_vel_offset_allgal(gmc_catalog="_native_", gmc_catalog_version='new', muse='dr2', matching="overlap_1om", outliers=True,threshold_perc=0.9 , vel_limit= 100, randomize = '',save=True,show = True)
hii_vel(gmc_catalog="_native_", gmc_catalog_version='new', muse='dr2', matching="overlap_1om", outliers=True,threshold_perc=0.5 , vel_limit= 10000, randomize = '',save=True,show = False)
