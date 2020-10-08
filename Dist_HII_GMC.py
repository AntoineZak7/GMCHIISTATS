import os
import numpy as np
from itertools import chain
import pickle
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf


def name(overperc, without_out, new_muse):
    name_append = ['perc_matching_', 'with_outliers', 'without_outliers', 'new_muse_', 'old_muse_', str(threshold_perc)]

    if new_muse == True:
        name_end = name_append[3]
        if overperc == True:
            name_end = name_end + name_append[0] + name_append[5]
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


def dif_hii_gmc(new_muse, gmc_catalog, overlap_matching, outliers, show, save, *args, **kwargs):

    paired = kwargs.get('paired', None)
    unpaired = kwargs.get('unpaired', None)

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





    flat_dist = list(chain.from_iterable(DisHIIGMCover))
    max_dist = np.nanmax(flat_dist)
    min_dist = np.nanmin(flat_dist)

    pdf = fpdf.PdfPages("All galaxies - Distance GMC-HII Histogram ")

    fig = plt.figure()
    plt.hist(flat_dist, bins=np.arange(min_dist, max_dist, (max_dist - min_dist) / 250), histtype='stepfilled')
    plt.xlabel('Distance between pairs (pc)')
    plt.grid(alpha=0.5, linewidth=2, linestyle='-')
    plt.title('Histogram of minimal distance between pairs - All galaxies, MUSE catalog')
    if save == True:
        pdf.savefig(fig)
    if show == True:
        plt.show()
    else:
        plt.close()

    pdf.close()

    pages = int(len(galaxias) / 9) + 1
    print('pages = %s' % pages)
    x = 0
    subplot_x = 3
    subplot_y = 3
    while x < pages:
        pdf = fpdf.PdfPages("Individual galaxy - Distance GMC-HII Histogram - Page%i.pdf" % (x + 1))

        fig, axs = plt.subplots(subplot_x, subplot_y, figsize=(8, 10), gridspec_kw={'hspace': 0})
        axs = axs.ravel()
        for j in range(subplot_y * subplot_x):
            if x * 9 + j < len(galaxias) and len(DisHIIGMCover[x * 9 + j]) > 0:
                dist = DisHIIGMCover[x * 9 + j]
                max_dist = np.nanmax(dist) + 1
                min_dist = np.nanmin(dist)
                axs[j].hist(dist, bins=np.arange(min_dist, max_dist, (max_dist - min_dist) / (150)),
                            histtype='stepfilled')

        if save == True:
            pdf.savefig(fig)
        if show == True:
            plt.show()
        else:
            plt.close()
        x = x + 1
        pdf.close()

    pdf = fpdf.PdfPages("All galaxies - GMC-HII Distance Histogram - Narrowband & Muse ")

    fig = plt.figure()

    bin = np.arange(min_dist, max_dist, (max_dist - min_dist) / 100)
    #binm = np.arange(min_dist_m, max_dist_m, (max_dist_m - min_dist_m) / 100)

    plt.hist(flat_dist, bins=bin, histtype='stepfilled', alpha=0.5)

    plt.legend('Muse')

    plt.xlabel('Distance between pairs (pc)')
    plt.grid(alpha=0.5, linewidth=1, linestyle='-')
    plt.title('Histogram of minimal distance between pairs - All galaxies')
    if save == True:
        pdf.savefig(fig)
    if show == True:
        plt.show()
    else:
        plt.close()

    pdf.close()


