
'''
V2.22
This module provides different routines to perform statistical tests, analyses, correlations etc...
of data/quantities from matched HII regions and GMCs

Antoine Zakardjian
'''

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf
import seaborn as sns
import os
import warnings
import scipy.stats
import statsmodels.stats.stattools
import copy
from matplotlib.ticker import FormatStrFormatter
from astropy.table import Table
import matplotlib.colors as colors
import matplotlib as mpl

import gmchiistats_routines
import gmchiistats_routines as data_routines
from typing import Union

warnings.filterwarnings("ignore")
plt.style.use('science')

# ========================================================#

dir_script_data = os.getcwd() + "/script_data_dr22_v3/"
dirhii_dr1,dirhii_dr2, dirgmc_old,dirgmc_new, dirregions1, dirregions2, dirregions22, dirmaps, dirplots1, dirplots2, dirplots22, dirplots, dirhiimasks, dirgmcmasks, sample_table_dir = pickle.load(
    open(dir_script_data + 'Directories_muse.pickle', "rb"))  # retrieving the directories paths


##########################################################
#               RECURRING ROUTINES                      #
##########################################################

def linear_regression(x: list, y: list) -> Union[float, float, float]:
    '''
    Compute a linear regression

    :param x: The x-axis values used to perform the linear regression
    :param y: The y-axis values used to perform the linear regression
    :return: slope, intercept and correlation coefficient of the linear regression
    '''

    n = len(x)
    xm = np.mean(x)
    ym = np.mean(y)

    cov = (1 / n) * sum([xi * yi for xi, yi in zip(x, y)]) - xm * ym
    vx = (1 / n) * sum([xi ** 2 for xi in x]) - xm ** 2
    vy = (1 / n) * sum([yi ** 2 for yi in y]) - ym ** 2
    r = cov / (np.sqrt(vx * vy))

    slope = cov / vx
    b = ym - slope * xm
    r_sq = r ** 2

    return slope, b, r_sq


def conf_intervals(x: list, y: list, alpha: float) -> Union[list, float]:
    '''
    Compute the confidence and prediction intervals associated to a linear regression, along with other parameters.

    :param x: The x-axis values used to perform the linear regression
    :param y: The y-axis values used to perform the linear regression
    :param alpha: The significance level used
    :return: Confidence intervals + other parameters
    '''

    n = len(x)
    xm = np.mean(x)
    ym = np.mean(y)

    cov = (1 / n) * sum([xi * yi for xi, yi in zip(x, y)]) - xm * ym
    vx = np.var(x)
    vy = np.var(y)
    r = cov / (np.sqrt(vx * vy))

    slope = cov / vx
    intercept = ym - slope * xm  # intercept

    Sxx = sum([(xi - xm) ** 2 for xi in x])
    Syy = sum([(yi - ym) ** 2 for yi in y])
    Err = y - (x * slope + intercept)
    SSe = sum([err ** 2 for err in Err])  # Sum squared errors
    SSt = Syy
    MSt = SSt / (n - 2)
    SSr = sum([(xi * slope + intercept - ym) ** 2 for xi in x])
    sigma_sq = SSe / (n - 2)
    MSe = sigma_sq
    MSr = SSr

    Syx2 = SSe / (n - 2)

    x0 = np.sort(x)

    k = n - 2
    ta = scipy.stats.t.ppf(1 - alpha / 2, df=k)

    Sm2 = Syx2 / Sxx  # Variance of slope
    Sm = np.sqrt(Sm2)  # Standard deviation of slope

    Sb2 = Syx2 * (1 / n + (xm ** 2) / Sxx)
    Sb = np.sqrt(Sb2)

    # slope and intercept confidnce interval

    conf_a = Sm * ta
    conf_b = Sb * ta

    # =====intervalle confiance droite régression=====#

    y_conf_sup = ta * np.sqrt(MSe * (1 / n + np.array([(x - xm) ** 2 for x in x0]) / Sxx))
    y_conf_inf = -ta * np.sqrt(MSe * (1 / n + [(x - xm) ** 2 for x in x0] / Sxx))

    # ========intervalles prévision de y en x0========#

    int_prev_y = ta * np.sqrt(MSe * (1 + 1 / n + [(x - xm) ** 2 for x in x0] / Sxx))  # vector

    # =======Tests H1================#

    dw = statsmodels.stats.stattools.durbin_watson(Err)
    rms_err = np.std(np.array(Err))
    rms_tot = np.std(y)

    # =======test chi2==============#

    mybin = 20
    xbin, ybin, ebin, nbin = data_routines.bindata(x, y, mybin=mybin)
    y_prev = intercept + slope * x
    xbin_prev, ybin_prev, ebin_prev, nbin_prev = data_routines.bindata(x, y_prev, mybin=mybin)

    bin_err = ybin - (xbin * slope + intercept)
    bin_err = np.array(bin_err)
    xbin = np.array(xbin)
    ybin = np.array(ybin)
    ebin = np.array(ebin)
    nbin = np.array(nbin)

    bin_err = bin_err[np.where(nbin != 0)[0]]
    xbin = xbin[np.where(nbin != 0)[0]]
    ybin = ybin[np.where(nbin != 0)[0]]
    ebin = ebin[np.where(nbin != 0)[0]]

    bin_err = bin_err[np.where(ebin != 0)[0]]

    ebin = ebin[np.where(ebin != 0)[0]]

    slope1, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(x, y)
    mean_error = np.nanmean(abs(Err))

    return conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr, rms_tot, rms_err, dw, mean_error, SSt, SSe, MSe, MSt


def num_gmcs(idovergmc: list) -> float:
    '''
    Compute the number of unique gmc indexes in a list

    :param idovergmc: gmc indexes list
    :return: Number of unique gmc indexes
    '''

    ids = idovergmc
    n_gmcs = 0
    for i in range(len(ids)):

        if not isinstance(ids[0], list):
            ids = [id.tolist() for id in ids]
        id = [ids[i].count(x) for x in ids[i]]
        n_gmcs += len(id) - sum([x - 1 for x in id])

    n_gmcs = int(n_gmcs)
    return n_gmcs


def num_hiis(idoverhii: list) -> float:
    '''
    Compute the number of unique hii region indexes in a list

    :param idoverhii: hii region indexes list
    :return: Number of unique hii region indexes
    '''

    n_hiis = 0
    for i in range(len(idoverhii)):

        if not isinstance(idoverhii[0], list):
            idoverhii = [id.tolist() for id in idoverhii]
        id = [idoverhii[i].count(x) for x in idoverhii[i]]
        n_hiis += len(id)

    n_hiis = int(n_hiis)
    return n_hiis

###########################################################################
#               STAT TESTS AND CORRELATIONS ROUTINES                      #
###########################################################################


def plot_single_correlations(muse: str, gmc_catalog: str, gmc_catalog_version: str, randomize: str, matching: str, outliers: str, show: bool, save: bool,
                             threshold_percs: float, vel: float, gmcprop: list, rgal_color: bool, symmetrical: str) -> None:
    '''
    Compute global (i.e using data from all the galaxies in the sample) linear regressions of a gmc quantities against an hii region

            GMC properties index
            # 1 : MCO
            # 3 : Sigma mol
            # 4 : Sigma v
            # 5 : alpha vir
            # 6 : CO Tpeak
            # 7 : Tauff

    :param muse: 'dr1' or 'dr2' Version of the Muse nebulae catalog used
    :param gmc_catalog: '_native_', '_150pc_', '150pc_homogenized_' ...  GMC catalog resolution used
    :param gmc_catalog_version: 'old' or 'new' Version of the gmc catalog (PHANGS ALMA survey) used
    :param randomize: 'po', 'prop' or '' Shuffle hii regions positions, Halpha luminosity or no shuffle
    :param matching: '1om' or '1o1' Matching method used
    :param outliers: 'with' or 'without' To remove or keep outliers, i.e. points with a distance to the linear regression > 3 sigma
    :param show: 'True' or 'False' To show the linear regressions using pyplot
    :param save: 'True' or 'False' To save the linear regression in a pdf file
    :param threshold_percs: list of the minimal overlap percentage used during the matching, values between 0 and 1.
    :param vel: Maximal valocity offset accepted during the matching
    :param gmcprop: List of floats correspondig to the gmc properties to plot against the Halpha luminosity
    :param rgal_color: 'False' Obsolete
    :param symmetrical: 'hii', 'gmc' or 'sym' Matching symmetry used
    :return:
    '''

    threshold_perc = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        symmetrical=symmetrical, matching=matching, muse=muse, gmc_catalog=gmc_catalog,
        gmc_catalog_version=gmc_catalog_version, outliers=outliers, randomize=randomize, threshold_perc=threshold_perc,
        vel=vel, dir_script_data=dir_script_data)

    xlimmin, xlimmax = 35.416676089337685, 41.356351026116506 #Plot limits

    z = 0
    for i in gmcprop:
        name_gmc_prop = ['', 'MCO', '', 'Sigma_mol', 'Sigma_v', '', 'CO_Tpeak']
        pdf_name = "%sCor_HA_%s_%s%s.pdf" % (dirplots, name_gmc_prop[i], name_end, str(z))
        pdf3 = fpdf.PdfPages(pdf_name)

        for threshold_perc in threshold_percs:
            z += 1
            labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
                matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
                outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel, symmetrical=symmetrical)

            rgal_gmc = arrayyay[len(arrayyay) - 1]
            rgal_hii = arrayxax[len(arrayxax) - 1]
            labsyay = labsyay[0:len(labsyay) - 5]  # removing  vel, major axis, minor axis and PA, no need to plot them
            labsxax = labsxax[0:len(labsxax) - 4]

            k = 1
            l = 0

            fig, axs = plt.subplots(1, 1, sharex='col', figsize=(9, 11), dpi=80, gridspec_kw={'hspace': 0})
            fig.gca().set_aspect(1.5, adjustable='box')
            delta = 2

            if i == 4 or i == 6:
                fig.gca().set_aspect(2, adjustable='box')
                delta = delta / (2 / 1.5)

            label_fontsize = 24
            tick_size = 24

            axs.set_ylabel(labsyay[i], fontsize=label_fontsize)
            axs.set_xlabel(labsxax[1], fontsize=label_fontsize)

            axs.tick_params(axis="x", labelsize=tick_size)
            axs.tick_params(axis="y", labelsize=tick_size)
            # color = 'black'

            xaxall = gmchiistats_routines.prepdata(arrayxax[k])
            yayall = gmchiistats_routines.prepdata(arrayyay[i])
            rgal_all = gmchiistats_routines.prepdata(rgal_gmc)

            if i == 3 or i == 6:
                color = "tab:red"
            elif i == 1 or i == 4:
                color = "tab:blue"

            axs.plot(xaxall, yayall, 'o', markerfacecolor='None', color=color, markersize=2,
                     label='Minimal Overlap Percentage (MOP) = %2.0f \%%' % (threshold_perc * 100), linestyle='None')

            if xaxall.any() != 0 and yayall.any != ():
                xmin = np.amin(xaxall)
                xmax = np.amax(xaxall)

                LumHacorrover = arrayxax[1]
                LumHacorrovertot = sum([x for sublist in LumHacorrover for x in sublist])
                LumHacorrtot = sum([x for sublist in LumHacorrnot for x in sublist])

                FluxCOGMCovertot = sum([x for sublist in FluxCOGMCover for x in sublist])
                FluxCOGMCtot = sum([x for sublist in FluxCOGMCnot for x in sublist])

                MassCOGMC = sum([x for sublist in MassCOGMC for x in sublist])
                MasscoGMCover = sum([x for sublist in MasscoGMCover for x in sublist])

                n_gmcs_tot = sum([len(x) for x in FluxCOGMCnot])
                n_hiis_tot = sum([len(x) for x in LumHacorrnot])

                n_gmcs = num_gmcs(idovergmc)
                n_hiis = num_hiis(idoverhii)

                alpha = 0.05
                conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr, rms_tot, rms_err, dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(
                    xaxall, yayall, alpha)

                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * x0

                axs.plot(x0, y_pred + y_conf_sup, '--', color='black', label='Power law confidence interval')
                axs.plot(x0, y_pred + y_conf_inf, '--', color='black')

                axs.plot(x0, y_pred + 3 * rms_err, '-.', color='grey', label='$\pm 3 \sigma$ Interval')
                # axs.plot(x0, y_pred-3*rms_err, '-.', color = 'grey')

                axs.fill_between(x0, y_pred + 3 * rms_err, y_pred - 3 * rms_err, color='grey', alpha=0.05)

                if i == 1:
                    index_val = 37.5
                    index_val2 = 37.5
                    index_val3 = 37.5

                elif i == 3:
                    index_val = 36.7
                    index_val2 = 36.7
                    index_val3 = 36.7

                elif i == 4:
                    index_val = 0
                    index_val2 = 0
                    index_val3 = 0

                elif i == 6:
                    index_val = 36.6
                    index_val2 = 36.6
                    index_val3 = 36.6

                else:
                    index_val = 0
                    index_val2 = 0
                    index_val3 = 0

                if threshold_perc == 0.1:
                    print(index_val)
                    indexes = [i for i, n in enumerate(x0) if n > index_val]
                    indexes_1 = [i for i, n in enumerate(x0) if
                                 n <= index_val2]  # 37 for tpeak, 37.3 for MCO, nothing for sigmamol
                    axs.plot(x0[indexes], y_pred[indexes] - 3 * rms_err, '-.', color='grey')
                    axs.plot(x0[indexes_1], y_pred[indexes_1] - 3 * rms_err, '-.', color='grey', alpha=0.3)

                elif threshold_perc == 0.5:
                    indexes = [i for i, n in enumerate(x0) if n > index_val]
                    indexes_1 = [i for i, n in enumerate(x0) if n <= index_val2]
                    axs.plot(x0[indexes], y_pred[indexes] - 3 * rms_err, '-.', color='grey')
                    axs.plot(x0[indexes_1], y_pred[indexes_1] - 3 * rms_err, '-.', color='grey', alpha=0.3)

                elif threshold_perc == 0.9:
                    indexes = [i for i, n in enumerate(x0) if n > index_val]
                    indexes_1 = [i for i, n in enumerate(x0) if n <= index_val2]
                    axs.plot(x0[indexes], y_pred[indexes] - 3 * rms_err, '-.', color='grey')
                    axs.plot(x0[indexes_1], y_pred[indexes_1] - 3 * rms_err, '-.', color='grey', alpha=0.3)

                axs.plot(x0, y_pred, color='navy', label='Power law fit')

                r = np.sqrt(r_sq)

                right_align = 0.53
                left_align = 0.05
                upper_shift = 0.5
                down_align = 0.04
                offset = 0.06

                text_size = 14
                text_size_r2slope = 18
                legend_size = 14
                label_size = 25
                tick_size = 25

                axs.tick_params(axis="x", labelsize=tick_size)
                axs.tick_params(axis="y", labelsize=tick_size)

                axs.text(left_align, down_align + 3 * offset, 'R²: %5.3f' % (r_sq),
                         fontsize=text_size_r2slope, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(left_align, down_align + offset, 'Intercept: %5.2f $\pm$ %5.2f ' % (b, conf_b),
                         fontsize=text_size_r2slope, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(left_align, down_align, 'Power law index: %5.2f $\pm$ %5.2f ' % (slope, conf_a),
                         fontsize=text_size_r2slope, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(left_align, down_align + 2 * offset, 'Std Dev: %5.3f ' % (np.sqrt(MSe)),
                         fontsize=text_size_r2slope, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(right_align, down_align,
                         'Matched GMCs: %5.0f (%5.1f \%%) ' % (n_gmcs, (n_gmcs * 100 / n_gmcs_tot)),
                         fontsize=text_size, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(right_align, down_align + 1 * offset,
                         'Matched Hii regions: %5.0f (%5.1f \%%) ' % (n_hiis, (n_hiis * 100 / n_hiis_tot)),
                         fontsize=text_size, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                # axs.text(right_align, down_align + 2 * offset,
                #             'matched GMCs CO flux: %5.2f \%% ' % (FluxCOGMCovertot * 100 / FluxCOGMCtot),
                #             fontsize=text_size, horizontalalignment='left',
                #             verticalalignment='center', transform=axs.transAxes)

                axs.text(right_align, down_align + 2 * offset,
                         'Matched GMCs Mol. Mass: %5.1f \%% ' % (MasscoGMCover * 100 / MassCOGMC),
                         fontsize=text_size, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(right_align, down_align + 3 * offset,
                         'Matched Hii regions Ha Lum: %5.1f \%% ' % (LumHacorrovertot * 100 / LumHacorrtot),
                         fontsize=text_size, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                if threshold_perc == 0.1:
                    if i == 1:
                        med_mco = np.nanmedian(y_pred)
                    elif i == 3:
                        med_sigmamol = np.nanmedian(y_pred)

                    elif i == 4:
                        med_sigmav = np.nanmedian(y_pred)

                    elif i == 6:
                        med_tpeak = np.nanmedian(y_pred)

                if i == 1:
                    y0 = med_mco - delta - 0.35
                    yf = med_mco + delta - 0.35

                elif i == 3:
                    y0 = med_sigmamol - delta - 0.3
                    yf = med_sigmamol + delta - 0.3

                elif i == 4:
                    y0 = med_sigmav - delta - 0.45
                    yf = med_sigmav + delta - 0.45

                elif i == 6:
                    y0 = med_tpeak - delta - 0.2
                    yf = med_tpeak + delta - 0.2

                axs.set(ylim=(y0, yf))
                axs.set(xlim=(35.4, 41.2))
                axs.legend(prop={'size': legend_size}, loc=2)
                data_routines.save_pdf(pdf3, fig, save, show)
                l += 1

        pdf3.close()


def plot_correlations(muse: str, gmc_catalog: str, gmc_catalog_version: str, randomize: str, matching: str, outliers: str, show: bool, save: bool,
                             threshold_percs: float, vel: float, gmcprop: list, rgal_color: bool, symmetrical: str):
    '''
    Compute global (i.e using data from all the galaxies in the sample) linear regressions of different gmc quantities against an hii region quantity

            GMC properties index
            # 1 : MCO
            # 3 : Sigma mol
            # 4 : Sigma v
            # 5 : alpha vir
            # 6 : CO Tpeak
            # 7 : Tauff

    :param muse: 'dr1' or 'dr2' Version of the Muse nebulae catalog used
    :param gmc_catalog: '_native_', '_150pc_', '150pc_homogenized_' ...  GMC catalog resolution used
    :param gmc_catalog_version: 'old' or 'new' Version of the gmc catalog (PHANGS ALMA survey) used
    :param randomize: 'po', 'prop' or '' Shuffle hii regions positions, Halpha luminosity or no shuffle
    :param matching: '1om' or '1o1' Matching method used
    :param outliers: 'with' or 'without' To remove or keep outliers, i.e. points with a distance to the linear regression > 3 sigma
    :param show: 'True' or 'False' To show the linear regressions using pyplot
    :param save: 'True' or 'False' To save the linear regression in a pdf file
    :param threshold_percs: list of the minimal overlap percentage used during the matching, values between 0 and 1.
    :param vel: Maximal valocity offset accepted during the matching
    :param gmcprop: List of floats correspondig to the gmc properties to plot against the Halpha luminosity
    :param rgal_color: 'False' Obsolete
    :param symmetrical: 'hii', 'gmc' or 'sym' Matching symmetry used
    :return:
    '''

    threshold_perc = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel, symmetrical=symmetrical, dir_script_data=dir_script_data)
    z = 0

    xlimmin, xlimmax = 35.416676089337685, 41.356351026116506
    pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s%s.pdf" % (dirplots, namegmc, name_end, str(z))
    pdf3 = fpdf.PdfPages(pdf_name)

    for threshold_perc in threshold_percs:
        z += 1

        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
            matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
            outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel, symmetrical=symmetrical, dir_script_data=dir_script_data)

        rgal_gmc = arrayyay[len(arrayyay) - 1]
        rgal_hii = arrayxax[len(arrayxax) - 1]
        labsyay = labsyay[0:len(labsyay) - 5]  # removing  vel, major axis, minor axis and PA, no need to plot them
        labsxax = labsxax[0:len(labsxax) - 4]

        k = 1

        plt.rc('xtick')
        plt.rc('ytick')

        fig, axs = plt.subplots(2, 1, sharex='col', figsize=(9, 13),
                                gridspec_kw={'hspace': 0.05})  # , dpi=80, gridspec_kw={'hspace': 0})
        axs = axs.ravel()

        LumHacorrover = arrayxax[1]
        LumHacorrovertot = sum([x for sublist in LumHacorrover for x in sublist])
        FluxCOGMCovertot = sum([x for sublist in FluxCOGMCover for x in sublist])

        FluxCOGMCtot = sum([x for sublist in FluxCOGMCnot for x in sublist])
        LumHacorrtot = sum([x for sublist in LumHacorrnot for x in sublist])

        MassCOGMC = sum([x for sublist in MassCOGMC for x in sublist])
        MasscoGMCover = sum([x for sublist in MasscoGMCover for x in sublist])

        l = 0

        for i in gmcprop:

            delta = 2
            # if i == 4 or i == 6:
            #     fig.gca().set_aspect(2, adjustable='box')
            #     delta = delta / (2 / 1.5)

            label_fontsize = 28
            tick_size = 28

            axs[l].set_ylabel(labsyay[i], fontsize=label_fontsize)
            axs[1].set_xlabel(labsxax[1], fontsize=label_fontsize)

            axs[l].tick_params(axis="x", labelsize=tick_size)
            axs[l].tick_params(axis="y", labelsize=tick_size)
            axs[l].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # color = 'black'

            xaxall, yayall, rgal_all = gmchiistats_routines.prepdata3(arrayxax[k],arrayyay[i], rgal_gmc)



            if i == 3 or i == 6:
                color = "tab:red"
            elif i == 1 or i == 4:
                color = "tab:blue"

            axs[l].plot(xaxall, yayall, 'o', markerfacecolor='None', color=color, markersize=1.7,
                        label='Minimal Overlap Percentage (MOP) = %1.1f' % (threshold_perc*100), linestyle='None')

            if xaxall.any() != 0 and yayall.any != ():
                xmin = np.amin(xaxall)
                xmax = np.amax(xaxall)

                n_gmcs_tot = sum([len(x) for x in FluxCOGMCnot])
                n_hiis_tot = sum([len(x) for x in LumHacorrnot])

                n_gmcs = num_gmcs(idovergmc)
                n_hiis = num_hiis(idoverhii)

                alpha = 0.05
                conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr, rms_tot, rms_err, dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(
                    xaxall, yayall, alpha)

                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * x0

                axs[l].plot(x0, y_pred + y_conf_sup, '--', color='black', label='Power law confidence interval')
                axs[l].plot(x0, y_pred + y_conf_inf, '--', color='black')

                axs[l].plot(x0, y_pred + 3 * rms_err, '-.', color='grey', label=' $\pm 3 \sigma$ Interval')
                axs[l].plot(x0, y_pred - 3 * rms_err, '-.', color='grey')

                axs[l].fill_between(x0, y_pred + 3 * rms_err, y_pred - 3 * rms_err, color='grey', alpha=0.05)

                # if i == 1:
                #     index_val = 37
                #     index_val2 = 37
                #     index_val3 = 0
                #
                # elif i == 3 or gmcprop[0] == 4:
                #     index_val = 0
                #     index_val2 = 0
                #     index_val3 = 0
                #
                # elif i == 6:
                #     index_val = 0
                #     index_val2 = 0
                #     index_val3 = 0
                # else:
                #     index_val = 0
                #     index_val2 = 0
                #     index_val3 = 0
                #
                # if threshold_perc == 0.1:
                #     print(index_val)
                #     indexes = [i for i, n in enumerate(x0) if n > index_val]
                #     indexes_1 = [i for i, n in enumerate(x0) if
                #                  n <= index_val2]  # 37 for tpeak, 37.3 for MCO, nothing for sigmamol
                #     axs[l].plot(x0[indexes], y_pred[indexes] - int_prev_y[indexes], '-.', color='grey')
                #     axs[l].plot(x0[indexes_1], y_pred[indexes_1] - int_prev_y[indexes_1], '-.', color='grey', alpha=0.3)
                #
                # elif threshold_perc == 0.5:
                #     indexes = [i for i, n in enumerate(x0) if n > index_val]
                #     indexes_1 = [i for i, n in enumerate(x0) if n <= index_val2]
                #     axs[l].plot(x0[indexes], y_pred[indexes] - int_prev_y[indexes], '-.', color='grey')
                #     axs[l].plot(x0[indexes_1], y_pred[indexes_1] - int_prev_y[indexes_1], '-.', color='grey', alpha=0.3)
                #
                # elif threshold_perc == 0.9:
                #     indexes = [i for i, n in enumerate(x0) if n > index_val]
                #     indexes_1 = [i for i, n in enumerate(x0) if n <= index_val2]
                #     axs[l].plot(x0[indexes], y_pred[indexes] - int_prev_y[indexes], '-.', color='grey')
                #     axs[l].plot(x0[indexes_1], y_pred[indexes_1] - int_prev_y[indexes_1], '-.', color='grey', alpha=0.3)

                axs[l].plot(x0, y_pred, color='navy', label='Power law fit')

                r = np.sqrt(r_sq)

                right_align = 0.55
                left_align = 0.05
                upper_shift = 0.5
                down_align = 0.06
                offset = 0.06

                text_size = 14
                text_size_r2slope = 18
                legend_size = 14
                label_size = 25
                tick_size = 25

                axs[l].tick_params(axis="x", labelsize=tick_size)
                axs[l].tick_params(axis="y", labelsize=tick_size)

                axs[l].text(right_align, down_align + 3 * offset, 'R²: %5.3f' % (r_sq),
                            fontsize=text_size_r2slope, horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes)

                axs[l].text(right_align, down_align + offset, 'Intercept: %5.2f $\pm$ %5.2f ' % (b, conf_b),
                            fontsize=text_size_r2slope, horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes)

                axs[l].text(right_align, down_align, 'Power law index: %5.2f $\pm$ %5.2f ' % (slope, conf_a),
                            fontsize=text_size_r2slope, horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes)

                axs[l].text(right_align, down_align + 2 * offset, 'Std Dev: %5.2f ' % (np.sqrt(MSe)),
                            fontsize=text_size_r2slope, horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes)
                #
                # axs[l].text(right_align, down_align,
                #          'Matched GMCs: %5.0f (%5.1f \%%) ' % (n_gmcs, (n_gmcs * 100 / n_gmcs_tot)),
                #          fontsize=text_size, horizontalalignment='left',
                #          verticalalignment='center', transform=axs[l].transAxes)
                #
                # axs[l].text(right_align, down_align + 1 * offset,
                #          'Matched Hii regions: %5.0f (%5.1f \%%) ' % (n_hiis, (n_hiis * 100 / n_hiis_tot)),
                #          fontsize=text_size, horizontalalignment='left',
                #          verticalalignment='center', transform=axs[l].transAxes)
                #
                # # axs.text(right_align, down_align + 2 * offset,
                # #             'matched GMCs CO flux: %5.2f \%% ' % (FluxCOGMCovertot * 100 / FluxCOGMCtot),
                # #             fontsize=text_size, horizontalalignment='left',
                # #             verticalalignment='center', transform=axs.transAxes)
                #
                # axs[l].text(right_align, down_align + 2 * offset,
                #          'Matched GMCs Mol. Mass: %5.1f \%% ' % (MasscoGMCover * 100 / MassCOGMC),
                #          fontsize=text_size, horizontalalignment='left',
                #          verticalalignment='center', transform=axs[l].transAxes)
                #
                # axs[l].text(right_align, down_align + 3 * offset,
                #          'Matched Hii regions Ha Lum: %5.1f \%% ' % (LumHacorrovertot * 100 / LumHacorrtot),
                #          fontsize=text_size, horizontalalignment='left',
                #          verticalalignment='center', transform=axs[l].transAxes)

                if threshold_perc == 0.1:
                    if i == 1:
                        med_mco = np.nanmedian(y_pred)
                    elif i == 3:
                        med_sigmamol = np.nanmedian(y_pred)

                    elif i == 4:
                        med_sigmav = np.nanmedian(y_pred)

                    elif i == 6:
                        med_tpeak = np.nanmedian(y_pred)

                if i == 1:
                    delta = 3
                    y0 = med_mco - delta - 0.35
                    yf = med_mco + delta - 0.35

                elif i == 3:
                    delta = 2
                    y0 = med_sigmamol - delta - 0.1
                    yf = med_sigmamol + delta - 0.1

                elif i == 4:
                    delta = 1.5
                    y0 = med_sigmav - delta - 0.2
                    yf = med_sigmav + delta - 0.2

                elif i == 6:
                    delta = 1.5
                    y0 = med_tpeak - delta - 0.1
                    yf = med_tpeak + delta - 0.1

                axs[l].set(ylim=(y0, yf))
                axs[l].legend(prop={'size': legend_size}, loc=2)
                axs[l].set(xlim = (xlimmin, xlimmax))

                l += 1
        data_routines.save_pdf(pdf3, fig, save, show)

    pdf3.close()


def plot_correlations_regions(muse: str, gmc_catalog: str, gmc_catalog_version: str, randomize: str, matching: str, outliers: str, show: bool, save: bool,
                             threshold_percs: float, vel: float, gmcprop: list, rgal_color: bool, symmetrical: str, regions: list):

    '''
    Compute global (i.e using data from all the galaxies in the sample) linear regressions of different gmc quantities against an hii region quantity from matched hii regions
    and gmcs in particular galactic environments

            GMC properties index
            # 1 : MCO
            # 3 : Sigma mol
            # 4 : Sigma v
            # 5 : alpha vir
            # 6 : CO Tpeak
            # 7 : Tauff

            Region indexes

            1 --> center (small bulge or nucleus)
            2 = bar (excluding bar ends)
            3 = bar ends (overlap of bar and spiral)
            4 = interbar (R_gal < R_bar but outside bar footprint)
            5 = spiral arms inside interbar (R_gal < R_bar)
            6 = spiral arms (R_gal > R_bar)
            7 = interarm (only R_gal spanned by spiral arms, and R_gal > R_bar)
            8 = outer disc (R_gal > spiral arm ends, only for galaxies with identif
            9 = disc (R_gal > R_bar) where no spiral arms were identified

    :param muse: 'dr1' or 'dr2' Version of the Muse nebulae catalog used
    :param gmc_catalog: '_native_', '_150pc_', '150pc_homogenized_' ...  GMC catalog resolution used
    :param gmc_catalog_version: 'old' or 'new' Version of the gmc catalog (PHANGS ALMA survey) used
    :param randomize: 'po', 'prop' or '' Shuffle hii regions positions, Halpha luminosity or no shuffle
    :param matching: '1om' or '1o1' Matching method used
    :param outliers: 'with' or 'without' To remove or keep outliers, i.e. points with a distance to the linear regression > 3 sigma
    :param show: 'True' or 'False' To show the linear regressions using pyplot
    :param save: 'True' or 'False' To save the linear regression in a pdf file
    :param threshold_percs: list of the minimal overlap percentage used during the matching, values between 0 and 1.
    :param vel: Maximal valocity offset accepted during the matching
    :param gmcprop: List of floats correspondig to the gmc properties to plot against the Halpha luminosity
    :param rgal_color: 'False' Obsolete
    :param symmetrical: 'hii', 'gmc' or 'sym' Matching symmetry used
    :param region: List of the region indexes used
    :return:
    '''



    region_name = ['Center', 'Arms', 'Disc']
    gmc_names = ['_', 'Mass', '_', 'Sigmamol', 'sigmav', '_', 'Tpeak']

    threshold_perc = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel, symmetrical=symmetrical)

    if save == True:
        print(gmc_names[gmcprop[0]])
        pdf_name = "%sCorrelations_HA_%s_threshold%s.pdf" % (dirplots, gmc_names[gmcprop[0]], name_end)
        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    reg = 0

    fig, axs = plt.subplots(1, 1, sharex='col', figsize=(9, 11), dpi=80, gridspec_kw={'hspace': 0})
    fig.gca().set_aspect(1.5, adjustable='box')

    for i in range(len(regions)):
        ii = i
        fig, axs = plt.subplots(1, 1, sharex='col', figsize=(9, 11), dpi=80, gridspec_kw={'hspace': 0})
        fig.gca().set_aspect(1.5, adjustable='box')
        delta = 2

        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
            matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
            outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel, symmetrical=symmetrical)

        for e in gmcprop:
            k = 1

            yayall = gmchiistats_routines.prepdata(arrayyay[e])
            xaxall = gmchiistats_routines.prepdata(arrayxax[k])

            alpha = 0.05
            conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr, rms_tot, rms_err, dw, mean_error, SSt, SSe, MSe, d = conf_intervals(
                xaxall, yayall, alpha)

            slope, b, r_sq = linear_regression(xaxall, yayall)
            y_pred = b + slope * x0

            if e == 1:
                med_mco = np.nanmedian(y_pred)
            elif e == 3:
                med_sigmamol = np.nanmedian(y_pred)

            elif e == 4:
                med_sigmav = np.nanmedian(y_pred)

            elif e == 6:
                med_tpeak = np.nanmedian(y_pred)

        regions_id = regions[i]

        for i in range(np.shape(arrayxax)[0]):
            for j in range(np.shape(arrayxax)[1]):
                id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
                arrayxax[i][j] = arrayxax[i][j][id_int]

        for i in range(np.shape(arrayyay)[0]):
            for j in range(np.shape(arrayyay)[1]):
                id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
                arrayyay[i][j] = arrayyay[i][j][id_int]

        for j in range(len(FluxCOGMCover)):
            id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
            FluxCOGMCover[j] = FluxCOGMCover[j][id_int]

        idoverhii = [np.array(x) for x in idoverhii]
        idovergmc = [np.array(x) for x in idovergmc]

        for j in range(len(idoverhii)):
            id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
            idoverhii[j] = idoverhii[j][id_int]

        for j in range(len(idovergmc)):
            id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
            idovergmc[j] = idovergmc[j][id_int]

        l = 0
        for i in gmcprop:
            if i == 4 or i == 6:
                fig.gca().set_aspect(2, adjustable='box')
                delta = delta / (2 / 1.5)

            xaxall = gmchiistats_routines.prepdata(arrayxax[k])
            yayall = gmchiistats_routines.prepdata(arrayyay[i])

            colors = ['blue', 'red', 'red']
            color = colors[reg]

            axs.plot(xaxall, yayall, 'o', markerfacecolor='None', markersize=2, markeredgecolor='black',
                     label='Minimal Overlap Percentage (MOP) = %2.0f \%%' % (threshold_perc * 100), linestyle='None')

            if xaxall.any() != 0 and yayall.any != ():
                xmin = np.amin(xaxall)
                xmax = np.amax(xaxall)

                LumHacorrover = arrayxax[1]
                LumHacorrovertot = sum([x for sublist in LumHacorrover for x in sublist])
                FluxCOGMCovertot = sum([x for sublist in FluxCOGMCover for x in sublist])

                FluxCOGMCtot = sum([x for sublist in FluxCOGMCnot for x in sublist])
                LumHacorrtot = sum([x for sublist in LumHacorrnot for x in sublist])

                MassCOGMC = sum([x for sublist in MassCOGMC for x in sublist])
                MasscoGMCover = sum([x for sublist in MasscoGMCover for x in sublist])

                n_gmcs_tot = sum([len(x) for x in FluxCOGMCnot])
                n_hiis_tot = sum([len(x) for x in LumHacorrnot])

                n_gmcs = num_gmcs(idovergmc)
                n_hiis = num_hiis(idoverhii)

                conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr, rms_tot, rms_err, dw, mean_error, SSt, SSe, MSe, d = conf_intervals(
                    xaxall, yayall, alpha)

                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * x0

                r = np.sqrt(r_sq)

                axs.plot(x0, y_pred + y_conf_sup, '--', color='black', label='Power law confidence interval')
                axs.plot(x0, y_pred + y_conf_inf, '--', color='black')

                axs.plot(x0, y_pred + 3 * rms_err, '-.', color='grey', label='$\pm 3 \sigma$ Interval')

                axs.fill_between(x0, y_pred + 3 * rms_err, y_pred - 3 * rms_err, color='grey', alpha=0.05)

                if gmcprop[0] == 1:
                    index_val = 37.6
                    index_val2 = 38
                    index_val3 = 37.6

                elif gmcprop[0] == 3 or gmcprop[0] == 4:
                    index_val = 0
                    index_val2 = 0
                    index_val3 = 0

                elif gmcprop[0] == 6:
                    index_val = 37.3
                    index_val2 = 37.5
                    index_val3 = 0
                else:
                    index_val = 0
                    index_val2 = 0
                    index_val3 = 0

                if regions_id != [7, 8, 9]:
                    print(index_val)
                    indexes = [i for i, n in enumerate(x0) if n > index_val]
                    indexes_1 = [i for i, n in enumerate(x0) if
                                 n <= index_val2]  # 37 for tpeak, 37.3 for MCO, nothing for sigmamol
                    axs.plot(x0[indexes], y_pred[indexes] - 3 * rms_err, '-.', color='grey')
                    axs.plot(x0[indexes_1], y_pred[indexes_1] - 3 * rms_err, '-.', color='grey', alpha=0.3)

                else:
                    indexes = [i for i, n in enumerate(x0) if n > index_val3]
                    indexes_1 = [i for i, n in enumerate(x0) if n <= index_val3]
                    axs.plot(x0[indexes], y_pred[indexes] - 3 * rms_err, '-.', color='grey')
                    axs.plot(x0[indexes_1], y_pred[indexes_1] - 3 * rms_err, '-.', color='grey', alpha=0.3)

                axs.plot(x0, y_pred, color='navy', label='Power law fit')

                right_align = 0.53
                left_align = 0.05
                upper_shift = 0.5
                down_align = 0.04
                offset = 0.06

                text_size = 14
                text_size_r2slope = 18
                legend_size = 14
                label_size = 25
                tick_size = 25

                label_fontsize = 24
                tick_size = 24
                #
                axs.set_ylabel(labsyay[i], fontsize=label_fontsize)
                axs.set_xlabel(labsxax[1], fontsize=label_fontsize)
                #
                axs.tick_params(axis="x", labelsize=tick_size)
                axs.tick_params(axis="y", labelsize=tick_size)
                axs.text(left_align, down_align + 3 * offset, 'R²: %5.3f' % ((r_sq)),
                         fontsize=text_size_r2slope, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(left_align, down_align + offset, 'Intercept: %5.2f $\pm$ %5.2f ' % (b, conf_b),
                         fontsize=text_size_r2slope, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(left_align, down_align, 'Power law index: %5.2f $\pm$ %5.2f ' % (slope, conf_a),
                         fontsize=text_size_r2slope, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(left_align, down_align + 2 * offset, 'Std Dev: %5.2f ' % (np.sqrt(MSe)),
                         fontsize=text_size_r2slope, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(right_align, down_align,
                         'Matched GMCs: %5.0f (%5.1f \%%) ' % (n_gmcs, (n_gmcs * 100 / n_gmcs_tot)),
                         fontsize=text_size, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(right_align, down_align + 1 * offset,
                         'Matched Hii regions: %5.0f (%5.1f \%%) ' % (n_hiis, (n_hiis * 100 / n_hiis_tot)),
                         fontsize=text_size, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                # axs.text(right_align, down_align + 2 * offset,
                #             'matched GMCs CO flux: %5.2f \%% ' % (FluxCOGMCovertot * 100 / FluxCOGMCtot),
                #             fontsize=text_size, horizontalalignment='left',
                #             verticalalignment='center', transform=axs.transAxes)

                axs.text(right_align, down_align + 2 * offset,
                         'Matched GMCs Mol. Mass: %5.1f \%% ' % (MasscoGMCover * 100 / MassCOGMC),
                         fontsize=text_size, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(right_align, down_align + 3 * offset,
                         'Matched Hii regions Ha Lum: %5.1f \%% ' % (LumHacorrovertot * 100 / LumHacorrtot),
                         fontsize=text_size, horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes)

                if i == 1:
                    y0 = med_mco - delta - 0.35
                    yf = med_mco + delta - 0.35
                elif i == 3:
                    y0 = med_sigmamol - delta - 0.3
                    yf = med_sigmamol + delta - 0.3
                elif i == 4:
                    y0 = med_sigmav - delta - 0.45
                    yf = med_sigmav + delta - 0.45
                    print(delta)
                elif i == 6:
                    y0 = med_tpeak - delta - 0.2
                    yf = med_tpeak + delta - 0.2

                axs.set(ylim=(y0, yf))
                axs.set(xlim=(35.4, 41.2))

                axs.legend(prop={'size': legend_size}, loc=2)

            l += 1
        axs.set_ylabel(labsyay[i], fontsize=label_size)
        axs.set_xlabel(labsxax[k], fontsize=label_size)
        reg += 1
        data_routines.save_pdf(pdf3, fig, save, show)

    pdf3.close()


def plot_correlations_color_rgal(muse: str, gmc_catalog: str, gmc_catalog_version: str, randomize: str, matching: str, outliers: str, show: bool, save: bool,
                             threshold_percs: float, vel: float, gmcprop: list, rgal_color: bool, symmetrical: str):
    '''
       Compute global (i.e using data from all the galaxies in the sample) linear regressions of different gmc quantities against an hii region quantity

               GMC properties index
               # 1 : MCO
               # 3 : Sigma mol
               # 4 : Sigma v
               # 5 : alpha vir
               # 6 : CO Tpeak
               # 7 : Tauff

       :param muse: 'dr1' or 'dr2' Version of the Muse nebulae catalog used
       :param gmc_catalog: '_native_', '_150pc_', '150pc_homogenized_' ...  GMC catalog resolution used
       :param gmc_catalog_version: 'old' or 'new' Version of the gmc catalog (PHANGS ALMA survey) used
       :param randomize: 'po', 'prop' or '' Shuffle hii regions positions, Halpha luminosity or no shuffle
       :param matching: '1om' or '1o1' Matching method used
       :param outliers: 'with' or 'without' To remove or keep outliers, i.e. points with a distance to the linear regression > 3 sigma
       :param show: 'True' or 'False' To show the linear regressions using pyplot
       :param save: 'True' or 'False' To save the linear regression in a pdf file
       :param threshold_percs: list of the minimal overlap percentage used during the matching, values between 0 and 1.
       :param vel: Maximal valocity offset accepted during the matching
       :param gmcprop: List of floats correspondig to the gmc properties to plot against the Halpha luminosity
       :param rgal_color: 'False' or 'True'
       :param symmetrical: 'hii', 'gmc' or 'sym' Matching symmetry used
       :return:
       '''


    threshold_perc = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)

    print("Plots of all galaxies together")

    xlimmin, xlimmax = data_routines.get_min(arrayxax)
    z = 0
    print("Starting loop to create figures of all galaxies together - points")

    pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s%s.pdf" % (dirplots, namegmc, name_end, str(z))

    pdf3 = fpdf.PdfPages(pdf_name)
    for threshold_perc in threshold_percs:

        z += 1

        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = data_routines.get_data(
            matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
            outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)

        rgal_gmc = arrayyay[len(arrayyay) - 1]
        rgal_hii = arrayxax[len(arrayxax) - 1]

        # labsyay = labsyay[0:len(labsyay) - 5]  # removing  vel, major axis, minor axis and PA, no need to plot them
        # labsxax = labsxax[0:len(labsxax) - 4]

        k = len(arrayxax) - 1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(1, 1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        plt.subplots_adjust(hspace=0.0)

        # fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s' % name_end, fontsize=18, va='top')
        #        axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0

        for i in gmcprop:

            axs.set(ylabel=labsxax[i])
            axs.grid()
            axs.set(xlabel=labsxax[len(labsxax) - 1])

            if threshold_perc == 0.1:
                color = "black"
            elif i == 3 or i == 6:
                color = "tab:red"
            elif i == 1 or i == 4:
                color = "tab:blue"

            color = "black"

            xaxall = gmchiistats_routines.prepdata(arrayxax[k])
            yayall = gmchiistats_routines.prepdata(arrayyay[i])
            rgal_all = gmchiistats_routines.prepdata(rgal_hii)

            if rgal_color == False:
                axs.plot(xaxall, yayall, 'o', markerfacecolor='None', markersize=2, markeredgecolor=color,
                         label='threshold = %f' % threshold_perc, linestyle='None')
            else:
                for xi in range(len(xaxall)):
                    colormap = plt.get_cmap('viridis')
                    colour = colormap(1 - 2 * rgal_all[xi] / np.nanmax(rgal_all))
                    axs.plot(xaxall[xi], yayall[xi], 'o', markeredgecolor=colour, markersize=2, linestyle='None',
                             markerfacecolor='None')

            if xaxall.any() != 0 and yayall.any != ():
                xmin = np.amin(xaxall)
                xmax = np.amax(xaxall)

                LumHacorrover = arrayxax[1]
                LumHacorrovertot = sum([x for sublist in LumHacorrover for x in sublist])
                FluxCOGMCovertot = sum([x for sublist in FluxCOGMCover for x in sublist])

                FluxCOGMCtot = sum([x for sublist in FluxCOGMCnot for x in sublist])
                LumHacorrtot = sum([x for sublist in LumHacorrnot for x in sublist])

                n_gmcs_tot = sum([len(x) for x in FluxCOGMCnot])
                n_hiis_tot = sum([len(x) for x in LumHacorrnot])

                n_gmcs = num_gmcs(idovergmc)
                n_hiis = num_hiis(idoverhii)
                alpha = 0.05
                conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr,rms_tot,rms_err,dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(xaxall, yayall, alpha)

                slope, b, r_sq = linear_regression(xaxall, yayall)
                # y_pred = b + slope * x0

                # axs.plot(x0, y_pred+y_conf_sup, '--', color = 'black', label='confidence interval')
                # axs.plot(x0, y_pred + y_conf_inf, '--', color = 'black')

                # axs.plot(x0, y_pred+int_prev_y, '-.', color = 'grey', label = 'prediction interval')
                # axs.plot(x0, y_pred - int_prev_y, '-.', color = 'grey')

                # axs.plot(x0, y_pred, color = 'navy', label = 'linear regression')

                r = np.sqrt(r_sq)

                # axs.text(0.8, 0.15, 'P-Value: %6.2f' % (pvalue), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                # axs.text(0.8, 0.07, 'Error RMS: %5.3f' % (rms_err), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                # axs.text(0.8, 0.15, 'Standard error of estimate: %5.3f' % (stderr), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.03, 'R²: %5.3f' % ((r_sq)), fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                # axs.text(0.8, 0.19, 'Durbin-Watson stat: %6.2f' % (dw), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                # axs.text(0.2, 0.03, 'Slope %5.3f $\pm$ %5.3f ' % (slope, conf_a), fontsize=8, horizontalalignment='center',
                # verticalalignment='center', transform=axs.transAxes)
                #
                # axs.text(0.8, 0.07, 'Standard deviation %5.3f ' % (np.sqrt(MSe)), fontsize=8, horizontalalignment='center',
                # verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.11, 'paired gmcs %5.3f (%5.2f %%) ' % (n_gmcs, n_gmcs * 100 / n_gmcs_tot), fontsize=8,
                         horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.15, 'paired hii regions %5.3f (%5.2f %%) ' % (n_hiis, n_hiis * 100 / n_hiis_tot),
                         fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.19, 'paired gmcs CO flux: %5.2f %% ' % (FluxCOGMCovertot * 100 / FluxCOGMCtot),
                         fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.23, 'paired hii regions Ha Lum: %5.2f %% ' % (LumHacorrovertot * 100 / LumHacorrtot),
                         fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                # axs.set(ylim=(y0, yf))
                if l == 0:
                    axs.legend(prop={'size': 8}, loc=2)

                l += 1
        data_routines.save_pdf(pdf3, fig, save, show)

    pdf3.close()


def plot_correlations_fct_rgal(muse, gmc_catalog, gmc_catalog_version, randomize, matching, outliers, show, save,
                               threshold_percs, vel, gmcprop, rgal_color):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair
    threshold_perc = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)

    print("Plots of all galaxies together")

    xlimmin, xlimmax = data_routines.get_min(arrayxax)
    z = 0
    print("Starting loop to create figures of all galaxies together - points")

    pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s%s.pdf" % (dirplots, namegmc, name_end, str(z))

    pdf3 = fpdf.PdfPages(pdf_name)

    threshold_perc = threshold_percs[0]

    z += 1

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)

    rgal_gmc = arrayyay[len(arrayyay) - 1]
    rgal_hii = arrayxax[len(arrayxax) - 1]

    labsyay = labsyay[0:len(labsyay) - 5]  # removing  vel, major axis, minor axis and PA, no need to plot them
    labsxax = labsxax[0:len(labsxax) - 4]

    k = 1

    numbers = 4

    # Galactic distance vs: Mco, avir, sigmav,Sigmamol

    xaxall_gal = []
    yayall_gal = []

    test_ids = []

    for gal in range(len(galaxias)):

        # gmcprop
        # 1 : MCO
        # 3 : Sigma mol
        # 4 : Sigma v
        # 5 : alpha vir
        # 6 : CO Tpeak
        # 7 : Tauff

        yayall = arrayyay[gmcprop[0]][gal]
        xaxall = arrayxax[k][gal]

        xaxall_rgal = []
        yayall_rgal = []

        # ========

        rgal_all = rgal_hii[gal]

        rgal_max = np.nanmax(rgal_all)
        rgal_min = np.nanmin(rgal_all)
        rgal_bins = (rgal_max) / (numbers - 1)
        print(rgal_bins)
        print(rgal_max)
        print('\n')

        plt.hist(rgal_all)
        plt.title(galaxias[gal])
        plt.show()

        for ri in range(numbers):
            ids_rgal = np.where((rgal_all > ri * rgal_bins) & (rgal_all < (ri + 1) * rgal_bins))[0]
            print("%5.2f --- %5.2f" % (ri * rgal_bins, ((ri + 1) * rgal_bins)))
            print(len(ids_rgal))
            test_ids.append(len(ids_rgal))

            xaxall = np.array(xaxall)
            yayall = np.array(yayall)
            xaxall = xaxall[ids_rgal]
            yayall = yayall[ids_rgal]

            LumHacorrover_rgal = arrayxax[1][gal]
            LumHacorrover_rgal = LumHacorrover_rgal[ids_rgal]

            FluxCOGMCover_rgal = FluxCOGMCover[gal]

            LumHacorrovertot_rgal = sum(LumHacorrover_rgal)
            FluxCOGMCovertot_rgal = sum(FluxCOGMCover_rgal)

            FluxCOGMCtot = sum([x for sublist in FluxCOGMCnot for x in sublist])
            LumHacorrtot = sum([x for sublist in LumHacorrnot for x in sublist])

            n_gmcs_tot = sum([len(x) for x in FluxCOGMCnot])
            n_hiis_tot = sum([len(x) for x in LumHacorrnot])

            idoverhii_rgal = np.array(idoverhii[gal])[ids_rgal]
            idovergmc_rgal = np.array(idovergmc[gal])[ids_rgal]

            n_gmcs = len(set((idovergmc_rgal)))
            n_hiis = len(set((idoverhii_rgal)))

            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)
            idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
            xaxall = xaxall[idok]
            yayall = yayall[idok]
            rgal_all = rgal_all[idok]
            lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
            lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
            indlim = np.where((xaxall < lim2) & (xaxall > lim1))
            xaxall = xaxall[indlim]
            yayall = yayall[indlim]

            xaxall_rgal.append(xaxall)
            yayall_rgal.append(yayall)

        xaxall_gal.append(xaxall_rgal)
        yayall_gal.append(yayall_rgal)

    xaxall_all = [np.concatenate(xaxall_gal[i]) for i in range(np.shape(xaxall_gal)[1])]
    yayall_all = [np.concatenate(yayall_gal[i]) for i in range(np.shape(yayall_gal)[1])]

    print('TEEEEEEST')
    print(np.sum(test_ids))
    sns.set(style='white', color_codes=True)
    fig, axs = plt.subplots(int(np.sqrt(numbers)), int(np.sqrt(numbers)), sharex='col', figsize=(9, 10), dpi=80,
                            gridspec_kw={'hspace': 0})
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.0)

    # fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s' % name_end, fontsize=18, va='top')
    axs = axs.ravel()

    l = 0

    r2_list = []
    slope_list = []
    error_slope_list = []
    error_r2_list = []

    for rad in range(len(xaxall_all)):
        xaxall = xaxall_all[rad]
        yayall = yayall_all[rad]
        i = gmcprop[0]

        axs[l].plot(xaxall, yayall, 'o', markerfacecolor='None', markersize=2,
                    label='threshold = %f' % threshold_perc, linestyle='None')

        axs[l].grid()
        # axs[0].set(ylabel=labsyay[i])
        # axs[4].set(ylabel=labsyay[i])
        # axs[8].set(ylabel=labsyay[i])
        # axs[12].set(ylabel=labsyay[i])

        if xaxall.any() != 0 and yayall.any != ():
            xmin = np.amin(xaxall)
            xmax = np.amax(xaxall)
            xprang = (xmax - xmin) * 0.1
            x = xaxall.reshape((-1, 1))
            y = yayall

            conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr, rms_tot, rms_err, dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(
                xaxall, yayall, alpha)

            slope, b, r_sq = linear_regression(xaxall, yayall)
            y_pred = b + slope * x0

            slope_list.append(slope)
            r2_list.append(r_sq)
            error_slope_list.append(conf_a)
            error_r2_list.append(np.sqrt(MSe))

            axs[l].plot(x0, y_pred + y_conf_sup, '--', color='black', label='confidence interval')
            axs[l].plot(x0, y_pred + y_conf_inf, '--', color='black')

            # axs[l].plot(x0, y_pred+int_prev_y, '-.', color = 'grey', label = 'prediction interval')
            # axs[l].plot(x0, y_pred - int_prev_y, '-.', color = 'grey')

            axs[l].plot(x0, y_pred, color='navy', label='linear regression')

            r_sq = np.sqrt(r_sq)

            x0, xf = xlim[k]
            y0, yf = ylim[i]
            # axs.text(0.8, 0.15, 'P-Value: %6.2f' % (pvalue), fontsize=8, horizontalalignment='center',
            #             verticalalignment='center', transform=axs.transAxes)

            # =======
            x = xaxall
            y = yayall
            # p,V = np.polyfit(x, y, 1, cov=True)

            # axs.text(0.8, 0.07, 'Error RMS: %5.3f' % (rms_err), fontsize=8, horizontalalignment='center',
            #             verticalalignment='center', transform=axs.transAxes)

            # axs.text(0.8, 0.15, 'Standard error of estimate: %5.3f' % (stderr), fontsize=8, horizontalalignment='center',
            #             verticalalignment='center', transform=axs.transAxes)

            axs[l].text(0.8, 0.03, 'r²: %5.3f' % ((r_sq) ** 2), fontsize=10, horizontalalignment='center',
                        verticalalignment='center', transform=axs[l].transAxes, fontweight='bold')

            # axs.text(0.8, 0.19, 'Durbin-Watson stat: %6.2f' % (dw), fontsize=8, horizontalalignment='center',
            #             verticalalignment='center', transform=axs.transAxes)

            axs[l].text(0.23, 0.03, 'Slope %5.3f $\pm$ %5.3f ' % (slope, conf_a), fontsize=10,
                        horizontalalignment='center',
                        verticalalignment='center', transform=axs[l].transAxes, fontweight='bold')
            #
            # axs.text(0.8, 0.07, 'Standard deviation %5.3f ' % (np.sqrt(MSe)), fontsize=8, horizontalalignment='center',
            # verticalalignment='center', transform=axs.transAxes)

            axs[l].text(0.72, 0.11, 'paired gmcs %5.3f (%5.2f %%) ' % (n_gmcs, n_gmcs * 100 / n_gmcs_tot),
                        fontsize=10, horizontalalignment='center',
                        verticalalignment='center', transform=axs[l].transAxes)

            axs[l].text(0.72, 0.17, 'paired hiis %5.3f (%5.2f %%) ' % (n_hiis, n_hiis * 100 / n_hiis_tot),
                        fontsize=10, horizontalalignment='center',
                        verticalalignment='center', transform=axs[l].transAxes)
            #
            # axs[l].text(0.74, 0.19, 'paired gmcs CO flux: %5.2f %% ' % (FluxCOGMCovertot_rgal*100/FluxCOGMCtot),
            #          fontsize=10, horizontalalignment='center',
            #          verticalalignment='center', transform=axs[l].transAxes)
            #
            # axs[l].text(0.74, 0.23, 'paired hii regions Ha Lum: %5.2f %% ' % (LumHacorrovertot_rgal*100/LumHacorrtot),
            #          fontsize=10, horizontalalignment='center',
            #          verticalalignment='center', transform=axs[l].transAxes)

            # axs.set(ylim=(y0, yf))
            if l == 0:
                axs[l].legend(prop={'size': 8}, loc=2)

            l += 1

    # axs[12].set(xlabel=labsxax[k])
    # axs[13].set(xlabel=labsxax[k])
    # axs[14].set(xlabel=labsxax[k])
    # axs[15].set(xlabel=labsxax[k])

    data_routines.save_pdf(pdf3, fig, save, show)

    pdf3.close()

    slope_list = slope_list[0:len(slope_list)]
    error_slope_list = error_slope_list[0:len(error_slope_list)]

    rgal_list = [rgal_bins * i for i in range(numbers)]
    # rgal_list = rgal_list[0:len(rgal_list)]
    # r2_list = r2_list[0:len(r2_list) ]

    # plt.errorbar( x =rgal_list,y =slope_list, yerr=error_slope_list, capsize=5,
    # elinewidth=2,
    # markeredgewidth=2,  marker = '+', markersize = 12, color = 'black')
    plt.plot(rgal_list, slope_list, marker='+', markersize=12, color='black')

    plt.ylabel('Slope', fontsize=40)
    plt.xlabel('R_gal (pc)', fontsize=40)
    plt.grid()

    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)

    plt.show()

    plt.plot(rgal_list, r2_list, marker='+', markersize=12, color='black')

    plt.ylabel('Correlation coefficient r²', fontsize=40)
    plt.xlabel('R_gal (pc)', fontsize=40)
    plt.grid()

    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)

    plt.show()


def plot_correlations_randomized(muse, gmc_catalog, gmc_catalog_version, randomize, matching, outliers, show, save,
                                 threshold_percs, vel, gmcprop, random):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair
    threshold_perc = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)

    print("Plots of all galaxies together")

    xlimmin, xlimmax = data_routines.get_min(arrayyay)
    z = 0
    print("Starting loop to create figures of all galaxies together - points")

    pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold_random_gmc_prop%s%s%s.pdf" % (
    dirplots, namegmc, name_end, str(z))

    pdf3 = fpdf.PdfPages(pdf_name)

    yaytmp = arrayyay[1]
    yayall = np.concatenate([f.tolist() for f in yaytmp])

    # rand_id = np.where(yayall >= -1*1e20)
    all_ids = np.linspace(0, len(yayall) - 1, len(yayall), dtype=int)
    rand_ids = all_ids
    np.random.shuffle(rand_ids)

    for threshold_perc in threshold_percs:

        z += 1

        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = data_routines.get_data(
            matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
            outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)

        labsyay = labsyay[0:len(labsyay) - 5]  # removing  vel, major axis, minor axis and PA, no need to plot them
        labsxax = labsxax[0:len(labsxax) - 4]

        k = 1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(1, 1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        plt.subplots_adjust(hspace=0.0)

        # fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s' % name_end, fontsize=18, va='top')
        #        axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0

        for i in gmcprop:

            axs.set(ylabel=labsyay[i])
            axs.grid()
            axs.set(xlabel=labsxax[1])
            axs.set(xlabel=labsxax[1])

            if threshold_perc == 0.1:
                color = "black"
            elif i == 3 or i == 6:
                color = "tab:red"
            elif i == 1 or i == 4:
                color = "tab:blue"

            color = "black"

            # gmcprop
            # 1 : MCO
            # 3 : Sigma mol
            # 4 : Sigma v
            # 5 : alpha vir
            # 6 : CO Tpeak
            # 7 : Tauff

            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]

            # ========

            # for l in range(len(yaytmp)):
            #   yaytmp[l] = yaytmp[l] - np.nanmean(yaytmp[l])
            #  print(np.log10(np.nanmean(yaytmp[l])))

            # ========

            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            # if k < 5:
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)

            axs.plot(xaxall, yayall, 'o', markerfacecolor='None', markersize=2, markeredgecolor=color,
                     label='threshold = %f' % threshold_perc, linestyle='None')

            axs.set_xlim(xlimmin, xlimmax)

            idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
            xaxall = xaxall[idok]
            yayall = yayall[idok]
            lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
            lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
            indlim = np.where((xaxall < lim2) & (xaxall > lim1))
            xaxall = xaxall[indlim]
            yayall = yayall[indlim]

            if random == True:
                ids_kept = 0

                # yayall = yayall[rand_ids]

            if xaxall.any() != 0 and yayall.any != ():
                xmin = np.amin(xaxall)
                xmax = np.amax(xaxall)
                xprang = (xmax - xmin) * 0.1
                x = xaxall.reshape((-1, 1))
                y = yayall

                LumHacorrover = arrayxax[1]
                LumHacorrovertot = sum([x for sublist in LumHacorrover for x in sublist])
                FluxCOGMCovertot = sum([x for sublist in FluxCOGMCover for x in sublist])

                FluxCOGMCtot = sum([x for sublist in FluxCOGMCnot for x in sublist])
                LumHacorrtot = sum([x for sublist in LumHacorrnot for x in sublist])

                n_gmcs_tot = sum([len(x) for x in FluxCOGMCnot])
                n_hiis_tot = sum([len(x) for x in LumHacorrnot])

                n_gmcs = num_gmcs(idovergmc)
                n_hiis = num_hiis(idoverhii)

                conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr, rms_tot, rms_err, dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(
                    xaxall, yayall, alpha)

                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * x0

                # axs.plot(x0, y_pred, '-')
                # sns.regplot(x,y, scatter_kws={'s':2} )
                y_pred_sup = b + conf_b + (slope + conf_a) * x0
                y_pred_inf = b - conf_b + (slope - conf_a) * x0

                # axs.plot(x0, y_pred_sup, color = "lightgreen")
                # axs.plot(x0, y_pred_inf, color = "lightgreen")

                axs.plot(x0, y_pred + y_conf_sup, '--', color='black', label='confidence interval')
                axs.plot(x0, y_pred + y_conf_inf, '--', color='black')

                axs.plot(x0, y_pred + int_prev_y, '-.', color='grey', label='prediction interval')
                axs.plot(x0, y_pred - int_prev_y, '-.', color='grey')

                axs.plot(x0, y_pred, color='navy', label='linear regression')

                r_sq = np.sqrt(r_sq)

                x0, xf = xlim[k]
                y0, yf = ylim[i]
                # axs.text(0.8, 0.15, 'P-Value: %6.2f' % (pvalue), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                # =======
                x = xaxall
                y = yayall
                p, V = np.polyfit(x, y, 1, cov=True)

                # axs.text(0.8, 0.07, 'Error RMS: %5.3f' % (rms_err), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                # axs.text(0.8, 0.15, 'Standard error of estimate: %5.3f' % (stderr), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.03, 'R²: %5.3f' % ((r_sq) ** 2), fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                # axs.text(0.8, 0.19, 'Durbin-Watson stat: %6.2f' % (dw), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                axs.text(0.2, 0.03, 'Slope %5.3f $\pm$ %5.3f ' % (slope, conf_a), fontsize=8,
                         horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)
                #
                axs.text(0.8, 0.07, 'Standard deviation %5.3f ' % (np.sqrt(MSe)), fontsize=8,
                         horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.11, 'paired gmcs %5.3f (%5.2f %%) ' % (n_gmcs, n_gmcs * 100 / n_gmcs_tot), fontsize=8,
                         horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.15, 'paired hii regions %5.3f (%5.2f %%) ' % (n_hiis, n_hiis * 100 / n_hiis_tot),
                         fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.19, 'paired gmcs CO flux: %5.2f %% ' % (FluxCOGMCovertot * 100 / FluxCOGMCtot),
                         fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.23, 'paired hii regions Ha Lum: %5.2f %% ' % (LumHacorrovertot * 100 / LumHacorrtot),
                         fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.set(ylim=(y0, yf))
                if l == 0:
                    axs.legend(prop={'size': 8}, loc=2)

                l += 1
        # axs[l].set(xlabel=labsxax[k])
        data_routines.save_pdf(pdf3, fig, save, show)

    pdf3.close()


def plot_residus(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, show, save, threshold_percs):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair
    threshold_perc = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover = data_routines.get_data(
        muse, gmc_catalog, matching, outliers, threshold_perc, )

    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    xlimmin, xlimmax = data_routines.get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    for threshold_perc in threshold_percs:
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover = data_routines.get_data(
            muse, gmc_catalog, matching, outliers, threshold_perc, )
        idoverhiis = [item for sublist in idoverhii for item in sublist]

        k = 1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(3, 1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n Distance to line histogram \n%s' % name_end,
                     fontsize=18, va='top')
        # axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0
        for i in [1, 3, 6]:

            # axs[l].set(ylabel=labsyay[i])
            axs[2].set(ylabel='line distance / error')

            axs[l].grid()
            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            # if k < 5:
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)

            idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
            xaxall = xaxall[idok]
            yayall = yayall[idok]
            lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
            lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
            indlim = np.where((xaxall < lim2) & (xaxall > lim1))
            xaxall = xaxall[indlim]
            yayall = yayall[indlim]
            if xaxall.any() != 0 and yayall.any != ():
                xmin = np.amin(xaxall)
                xmax = np.amax(xaxall)
                xprang = (xmax - xmin) * 0.1
                x = xaxall.reshape((-1, 1))
                y = yayall

                # conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf = conf_intervals(xaxall, yayall, alpha)

                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * xaxall

                err = (y_pred - yayall)
                # axs.scatter(y_pred, err)
                print(scipy.stats.shapiro(err))
                axs[l].hist(err, bins=70, label=labsyay[i])

                x0, xf = xlim[k]
                y0, yf = ylim[i]

                axs[l].legend(prop={'size': 8})
                l += 1
        data_routines.save_pdf(pdf3, fig, save, show)

        plt.show()

    pdf3.close()


def hist_std_gmc_props(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, vel, show, save,
                   threshold_percs, bin, gmc_props, symmetrical):
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    threshold_perc = threshold_percs[0]

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel, symmetrical=symmetrical)
    if save == True:
        pdf_name = "%sHist_gmc_propst%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    xlimmin, xlimmax = data_routines.get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_percs[0], vel=vel, symmetrical=symmetrical)

    arrayyay1 = [SizepcGMCover, FluxCOGMCover, arrayyay[1], arrayyay[3], arrayyay[4], arrayyay[6]]
    labsyay1 = [labsyay[2], labsyay[13], labsyay[1], labsyay[3], labsyay[4], labsyay[6]]

    arrayyay_all = [SizepcGMC, FluxCOGMCnot, MassCOGMC, SigmaMol, Sigmav, COTpeak]

    for i in range(len(arrayyay1)):

        fig, axs = plt.subplots(3, 1, figsize=(4, 4), dpi=80, gridspec_kw={'hspace': 0.10}, sharex=True)
        # fig.suptitle(
        #    'All galaxies - Overlapping HIIregions and GMCs - GMCproperties histograms \n threshold = %s' % threshold_perc,
        #   fontsize=18, va='top')

        axs = axs.ravel()

        l = 0
        print("\n")
        print(" student tests results for: %s" % labsyay1[i])

        for thres in threshold_percs:

            labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
                matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
                outliers=outliers, randomize=randomize, threshold_perc=thres, vel=vel, symmetrical=symmetrical)

            arrayyay1 = [SizepcGMCover, FluxCOGMCover, arrayyay[1], arrayyay[3], arrayyay[4], arrayyay[6]]

            labsyay1 = [labsyay[2], labsyay[13], labsyay[1], labsyay[3], labsyay[4], labsyay[6]]

            # axs[l].set(xlabel=labsyay1[i])
            axs[2].set_xlabel(labsyay1[i])  # , fontsize = 23)
            axs[l].tick_params(axis="both")  # , labelsize=18, reset=False)

            yaytmp = arrayyay1[i]
            yaytmp_all = arrayyay_all[i]
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            yayall_all = np.concatenate([f.tolist() for f in yaytmp_all])

            yayall = np.log10(yayall)
            yayall_all = np.log10(yayall_all)

            idok = np.where((abs(yayall) < 100000))
            yayall = yayall[idok]

            idok = np.where((abs(yayall_all) < 100000))
            yayall_all = yayall_all[idok]

            if yayall_all.any != ():
                axs[l].hist(yayall_all, bins=100, color='red', label='All GMCs', histtype='stepfilled', alpha=0.70,
                            stacked=True, density=True)

                mean = np.nanmean(yayall_all)
                median = np.nanmedian(yayall_all)
                standard_dev = np.nanstd(yayall_all)

                fontsize = 5.5
                left_align = 0.03

            if i == 3 or i == 5:

                axs[l].text(0.65, 0.90, 'Mean all %5.2f ' % (mean), fontsize=fontsize,
                            horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes, weight='bold')

                axs[l].text(0.65, 0.62, 'Median all %5.2f ' % (median), fontsize=fontsize,
                            horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes, weight='bold')

                axs[l].text(0.65, 0.34, 'Std all %5.2f ' % (standard_dev), fontsize=fontsize,
                            horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes, weight='bold')
            else:

                axs[l].text(left_align, 0.90, 'Mean all %5.2f ' % (mean), fontsize=fontsize,
                            horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes, weight='bold')

                axs[l].text(left_align, 0.62, 'Median all %5.2f ' % (median), fontsize=fontsize,
                            horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes, weight='bold')

                axs[l].text(left_align, 0.34, 'Std all %5.2f ' % (standard_dev), fontsize=fontsize,
                            horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes, weight='bold')

            if yayall.any != ():
                axs[l].hist(yayall, bins=100, color='blue', label='Matched GMCs \n (MOP = %2.0f\%%)' % (100 * thres),
                            histtype='stepfilled', alpha=0.65, stacked=True, density=True)

                mean = np.nanmean(yayall)
                median = np.nanmedian(yayall)
                standard_dev = np.nanstd(yayall)

            if i == 3 or i == 5:

                axs[l].text(0.65, 0.78, 'Mean matched %5.2f ' % (mean), fontsize=fontsize,
                            horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes, weight='bold')

                axs[l].text(0.65, 0.50, 'Median matched %5.2f ' % (median), fontsize=fontsize,
                            horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes, weight='bold')

                axs[l].text(0.65, 0.22, 'Std matched %5.2f ' % (standard_dev), fontsize=fontsize,
                            horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes, weight='bold')

                axs[l].legend(prop={'size': 5}, loc=2)


            else:

                axs[l].text(left_align, 0.78, 'Mean matched %5.2f ' % (mean), fontsize=fontsize,
                            horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes, weight='bold')

                axs[l].text(left_align, 0.50, 'Median matched %5.2f ' % (median), fontsize=fontsize,
                            horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes, weight='bold')

                axs[l].text(left_align, 0.22, 'Std matched %5.2f ' % (standard_dev), fontsize=fontsize,
                            horizontalalignment='left',
                            verticalalignment='center', transform=axs[l].transAxes, weight='bold')

                if i == 0:
                    axs[l].legend(prop={'size': 3.5}, loc=1)  #
                else:
                    axs[l].legend(prop={'size': 5}, loc=1)  #

            if i == 0:
                axs[l].set(xlim=[0.5, 2.6])

            # =======STAT TESTS FOR MEAN AND STD==============#

            # ttest_stat, p_value = scipy.stats.ttest_ind(yayall_all, yayall, equal_var = False)
            #
            #
            # alpha = 0.05  # Or whatever you want your alpha to be.
            # F = np.std(yayall) / np.std(yayall_all)
            # df1 = len(yayall) - 1
            # df2 = len(yayall_all) - 1
            # p_value_f =  1-scipy.stats.f.cdf(F, df1, df2)
            #
            # #print("f test p-value (%5.1f%%) = %f" %((thres*100),p_value_f))
            # if p_value_f < alpha:
            #     a=0
            #     print("null hypothesis rejected, DIFFERENT STD")
            # #else:
            #     #print("null hypothesis not rejected, IDENTICAL STD")
            #
            #
            # #print("student test p-value (%5.1f%%) = %f" %((thres*100),p_value))
            # if p_value > 0.05:
            #     print("null hypothesis not rejected, IDENTICAL AVERAGES")
            # #else:
            #     #print("null hypothesis rejected, DIFFERENT AVERAGES")

            # barlett_stat, p_value_barlett = scipy.stats.bartlett(yayall_all, yayall) #std
            # leven_stat, p_value_levene = scipy.stats.levene(yayall_all, yayall, center = 'mean') #std
            #
            # print("levene test p-value (%5.1f%%) = %f" %((thres*100),p_value_levene))
            # if p_value_levene > 0.05:
            #     print("null hypothesis not rejected, IDENTICAL STD")
            # else:
            #     print("null hypothesis rejected, DIFFERENT STD")
            #
            #
            # print("bartlett test p-value (%5.1f%%) = %f" %((thres*100),p_value_barlett))
            # if p_value_barlett > 0.05:
            #     print("null hypothesis not rejected, IDENTICAL STD")
            # else:
            #     print("null hypothesis rejected, DIFFERENT STD")
            # print('\n')

            kruskal_stat, p_value_kruskal = scipy.stats.kruskal(yayall_all, yayall)  # std

            print("levene test p-value (%5.1f%%) = %f" % ((thres * 100), p_value_kruskal))
            if p_value_kruskal > 0.05:
                print("null hypothesis not rejected, IDENTICAL MEDIAN")
            else:
                print("null hypothesis rejected, DIFFERENT MEDIAN")

            l += 1

        data_routines.save_pdf(pdf3, fig, save, show)

    pdf3.close()


def hist_std_hii_props(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, vel, show, save,
                       threshold_percs, bin, gmc_props, symmetrical):
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    threshold_perc = threshold_percs[0]

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel, symmetrical=symmetrical)
    if save == True:
        pdf_name = "%shist_hii_props_%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    xlimmin, xlimmax = data_routines.get_min(arrayxax)

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_percs[0], vel=vel, symmetrical=symmetrical)

    SizepcHII_not = [SizepcHII[i][j] for i in range(len(SizepcHII)) for j in range(len(galaxias)) if
                     i not in idoverhii[j]]
    LumHacorrnot_not = [LumHacorrnot[i][j] for i in range(len(LumHacorrnot)) for j in range(len(galaxias)) if
                        i not in idoverhii[j]]

    arrayyay1 = [arrayxax[0], arrayxax[1]]
    arrayyay_all = [SizepcHII, LumHacorrnot]
    arrayyay_not = [SizepcHII_not, LumHacorrnot_not]

    for i in range(len(arrayyay1)):

        fig, axs = plt.subplots(3, 1, figsize=(4, 4), dpi=80, gridspec_kw={'hspace': 0.10}, sharex=True)
        # fig.suptitle(
        #   'All galaxies - Overlapping HIIregions and GMCs - GMCproperties histograms \n threshold = %s' % threshold_perc,
        #  fontsize=18, va='top')

        axs = axs.ravel()
        l = 0

        print("\n")
        print(" student tests results for: %s" % labsyay[i])

        for thres in threshold_percs:

            labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
                matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
                outliers=outliers, randomize=randomize, threshold_perc=thres, vel=vel, symmetrical=symmetrical)

            arrayyay1 = [arrayxax[0], arrayxax[1]]
            print(labsxax)

            axs[2].set_xlabel(labsxax[i])  # , fontsize = 20)
            # axs[l].tick_params(axis="both", labelsize=18, reset=False)
            # axs[l].grid()
            yaytmp = arrayyay1[i]
            yaytmp_not = arrayyay_not[i]
            yaytmp_all = arrayyay_all[i]
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            yayall_all = np.concatenate([f.tolist() for f in yaytmp_all])

            # yayall_not = np.concatenate([f.tolist() for f in yaytmp_not])
            yayall_not = yaytmp_not

            yayall = np.log10(yayall)
            yayall_all = np.log10(yayall_all)

            yayall_not = np.log10(yayall_not)

            idok = np.where((abs(yayall) < 100000))
            yayall = yayall[idok]

            idok = np.where((abs(yayall_all) < 100000))
            yayall_all = yayall_all[idok]

            idok = np.where((abs(yayall_not) < 100000))
            yayall_not = yayall_not[idok]

            binn = bin
            fontsize = 5

            if yayall_all.any != ():
                axs[l].hist(yayall_all, bins=binn, color='red', label='All Hii regions', histtype='stepfilled',
                            density=True, alpha=0.7)

                mean = np.nanmean(yayall_all)
                median = np.nanmedian(yayall_all)
                standard_dev = np.nanstd(yayall_all)

            #
            # if yayall_not.any != ():
            #      axs[l].hist(yayall_not, bins = binn, color = 'blue', label = 'unmatched hii',histtype='stepfilled', density = True)
            #      axs[l].legend(prop={'size': 8})
            #
            #      mean = np.nanmean(yayall_not)
            #      median = np.nanmedian(yayall_not)
            #
            # axs[l].text(0.15, 0.90, 'Mean unmatched %5.3f ' % (mean), fontsize=10,
            #             horizontalalignment='center',
            #             verticalalignment='center', transform=axs[l].transAxes)
            #
            # axs[l].text(0.15, 0.70, 'Median unmatched %5.3f ' % (median), fontsize=10,
            #             horizontalalignment='center',
            #             verticalalignment='center', transform=axs[l].transAxes)

            if yayall.any != ():
                axs[l].hist(yayall, bins=binn, color='blue',
                            label='Matched Hii regions  \n (MOP = %2.0f\%%)' % (thres * 100), density=True,
                            histtype='stepfilled', alpha=0.65)

                mean = np.nanmean(yayall)
                median = np.nanmedian(yayall)
                standard_deviation = np.nanstd(yayall)

            if i == 0:
                axs[l].legend(prop={'size': 4}, loc=1)
                left_align = 0.05
                right_align = 0.6
                fontsize = 5.5
                axs[l].set(xlim=[0.5, 2.6])
            else:
                left_align = 0.63
                fontsize = 5.5
                axs[l].set(xlim=[34.7, 43.2])

                axs[l].legend(prop={'size': 4}, loc=2)

            axs[l].text(left_align, 0.90 - 0.12, 'Mean matched %5.3f ' % (mean), fontsize=fontsize,
                        horizontalalignment='left',
                        verticalalignment='center', transform=axs[l].transAxes, weight='bold')

            axs[l].text(left_align, 0.90 - 0.12 - 0.12 - 0.14, 'Median matched %5.3f ' % (median),
                        fontsize=fontsize,
                        horizontalalignment='left',
                        verticalalignment='center', transform=axs[l].transAxes, weight='bold')

            axs[l].text(left_align, (0.90 - 0.12 - 0.12 - 0.14 - 0.14 - 0.12),
                        'Std matched %5.2f ' % (standard_dev), fontsize=fontsize,
                        horizontalalignment='left',
                        verticalalignment='center', transform=axs[l].transAxes, weight='bold')

            axs[l].text(left_align, 0.90, 'Mean all %5.3f ' % (mean), fontsize=fontsize,
                        horizontalalignment='left',
                        verticalalignment='center', transform=axs[l].transAxes, weight='bold')

            axs[l].text(left_align, 0.90 - 0.12 - 0.14, 'Median all %5.3f ' % (median), fontsize=fontsize,
                        horizontalalignment='left',
                        verticalalignment='center', transform=axs[l].transAxes, weight='bold')

            axs[l].text(left_align, 0.90 - 0.12 * 2 - 0.14 * 2, 'Std all %5.2f ' % (standard_dev),
                        fontsize=fontsize,
                        horizontalalignment='left',
                        verticalalignment='center', transform=axs[l].transAxes, weight='bold')

            # ttest_stat, p_value = scipy.stats.ttest_ind(yayall_all, yayall, equal_var = True)
            # alpha = 0.05  # Or whatever you want your alpha to be.
            # F = np.std(yayall) / np.std(yayall_all)
            # df1 = len(yayall) - 1
            # df2 = len(yayall_all) - 1
            # p_value_f = scipy.stats.f.cdf(F, df1, df2)
            #
            #
            # print("f test p-value (%5.1f%%) = %f" %((thres*100),p_value_f))
            # if p_value_f > alpha:
            #     print("null hypothesis rejected, DIFFERENT STD")
            # else:
            #     print("null hypothesis not rejected, IDENTICAL STD")
            #
            #
            #
            #
            # print("student test p-value (%5.1f%%) = %f" %((thres*100),p_value))
            # if p_value > 0.05:
            #     print("null hypothesis not rejected, IDENTICAL AVERAGES")
            # else:
            #     print("null hypothesis rejected, DIFFERENT AVERAGES")

            kruskal_stat, p_value_kruskal = scipy.stats.kruskal(yayall_all, yayall)  # std

            print("levene test p-value (%5.1f%%) = %f" % ((thres * 100), p_value_kruskal))
            if p_value_kruskal > 0.05:
                print("null hypothesis not rejected, IDENTICAL MEDIAN")
            else:
                print("null hypothesis rejected, DIFFERENT MEDIAN")

            l += 1

        data_routines.save_pdf(pdf3, fig, save, show)

    pdf3.close()


def cdf_std_gmc_props(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, vel, show, save,
                  threshold_percs, bin, gmc_props, symmetrical):
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    threshold_perc = threshold_percs[0]

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel, symmetrical=symmetrical)
    if save == True:
        pdf_name = "%sCDF_gmc_propst%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    xlimmin, xlimmax = data_routines.get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_percs[0], vel=vel, symmetrical=symmetrical)

    arrayyay1 = [SizepcGMCover, FluxCOGMCover, arrayyay[1], arrayyay[3], arrayyay[4], arrayyay[6]]

    arrayyay_all = [SizepcGMC, FluxCOGMCnot, MassCOGMC, SigmaMol, Sigmav, COTpeak]

    for i in range(len(arrayyay1)):

        fig, axs = plt.subplots(1, 1, figsize=(4, 3), dpi=80)
        # fig.subplots_adjust(bottom = 0.15, top = 0.92,  right = 0.93)

        fontsize = 6
        # fig.suptitle(
        #    'All galaxies - Overlapping HIIregions and GMCs - GMCproperties histograms \n threshold = %s' % threshold_perc,
        #   fontsize=18, va='top')
        # axs = axs.ravel()

        l = 0

        yaytmp_all = arrayyay_all[i]
        yayall_all = np.concatenate([f.tolist() for f in yaytmp_all])
        yayall_all = np.log10(yayall_all)
        idok = np.where((abs(yayall_all) < 100000))
        yayall_all = yayall_all[idok]

        if yayall_all.any != ():
            Y = yayall_all
            stepp = ((np.nanmax(yayall_all) - np.nanmin(yayall_all)) / len(yayall_all))
            X = [i * stepp for i in range(len(yayall_all))]
            X = X / np.nanmax(X)
            axs.plot(sorted(Y), X, color='red', label='All GMCss', linewidth=0.6)

            mean = np.nanmean(yayall_all)
            median = np.nanmedian(yayall_all)
            std_all = np.nanstd(yayall_all)
            mean_err = std_all / np.sqrt(len(yayall_all))

            axs.text(0.05, 0.95, 'Mean all = %5.2f  ' % (mean), fontsize=fontsize,
                     horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes, weight='bold')

            axs.text(0.05, (0.68),
                     'Std all = %5.2f ' % (std_all), fontsize=fontsize,
                     horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes, weight='bold')

            # axs.text(0.30, 0.95, 'Median all %5.2f ' % (median), fontsize=8,
            #          horizontalalignment='left',
            #          verticalalignment='center', transform=axs.transAxes, weight='bold')

        thres_i = 1
        for thres in threshold_percs:

            labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
                matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
                outliers=outliers, randomize=randomize, threshold_perc=thres, vel=vel, symmetrical=symmetrical)

            arrayyay1 = [SizepcGMCover, FluxCOGMCover, arrayyay[1], arrayyay[3], arrayyay[4], arrayyay[6]]

            labsyay1 = [labsyay[2], labsyay[13], labsyay[1], labsyay[3], labsyay[4], labsyay[6]]

            axs.set_xlabel(labsyay1[i])  # , fontsize = 20
            # axs.tick_params(axis="both", labelsize=18, reset=False)
            # axs.grid()

            yaytmp = arrayyay1[i]
            yayall = np.concatenate([f.tolist() for f in yaytmp])

            yayall = np.log10(yayall)

            idok = np.where((abs(yayall) < 100000))
            yayall = yayall[idok]

            if thres == 0.1:
                colorr = 'orange'
            if thres == 0.5:
                colorr = 'green'
            if thres == 0.9:
                colorr = 'blue'

            if yayall.any != ():

                Y = yayall

                stepp = ((np.nanmax(yayall) - np.nanmin(yayall)) / len(yayall))
                X = [i * stepp for i in range(len(yayall))]
                X = X / np.nanmax(X)
                axs.plot(sorted(Y), X, color=colorr, label='Paired GMCs (%2.0f\%%)' % (100 * thres), linewidth=0.6)
                axs.legend(prop={'size': 6}, loc=4)

                mean = np.nanmean(yayall)
                median = np.nanmedian(yayall)
                stdd = np.nanstd(yayall)
                mean_err1 = stdd / np.sqrt(len(yayall))

                axs.text(0.05, (0.95 - thres_i * 0.05), 'Mean (%1.0f \%%) = %5.2f' % ((thres * 100), mean),
                         fontsize=fontsize,  # $\pm$ %5.2f   , mean_err1
                         horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes, weight='bold')

                axs.text(0.05, (0.68 - thres_i * 0.05), 'Std (%1.0f \%%) = %5.2f ' % ((thres * 100), stdd,),
                         fontsize=fontsize,
                         horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes, weight='bold')

                # axs.text(0.30, (0.95-thres_i*0.05), 'Median matched (%1.1f) %5.2f ' % (thres,median), fontsize=8,
                #         horizontalalignment='left',
                #         verticalalignment='center', transform=axs.transAxes, weight = 'bold')
                if i == 0:
                    axs.set(xlim=[0.5, 2.6])

                thres_i += 1
            l += 1

        data_routines.save_pdf(pdf3, fig, save, show)

    pdf3.close()


def cdf_std_hii_props(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, vel, show, save,
                      threshold_percs, bin, gmc_props, symmetrical):
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    threshold_perc = threshold_percs[0]

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel, symmetrical=symmetrical)
    if save == True:
        pdf_name = "%sCDF_hii_propst%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    xlimmin, xlimmax = data_routines.get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_percs[0], vel=vel, symmetrical=symmetrical)

    arrayyay1 = [arrayxax[0], arrayxax[1]]
    arrayyay_all = [SizepcHII, LumHacorrnot]

    labsyay1 = [labsxax[0], labsxax[1]]

    for i in range(len(arrayyay1)):

        fig, axs = plt.subplots(1, 1, figsize=(4, 3), dpi=80)
        # fig.subplots_adjust(bottom = 0.15, top = 0.92,  right = 0.93)
        fontsize = 6

        axs.set_xlabel(labsyay1[i])  # , fontsize=20)
        # axs.tick_params(axis="both", labelsize=18, reset=False)
        # fig.suptitle(
        #    'All galaxies - Overlapping HIIregions and GMCs - GMCproperties histograms \n threshold = %s' % threshold_perc,
        #   fontsize=18, va='top')
        # axs = axs.ravel()

        l = 0

        yaytmp_all = arrayyay_all[i]
        yayall_all = np.concatenate([f.tolist() for f in yaytmp_all])
        yayall_all = np.log10(yayall_all)
        idok = np.where((abs(yayall_all) < 100000))
        yayall_all = yayall_all[idok]

        if yayall_all.any != ():
            Y = yayall_all
            stepp = ((np.nanmax(yayall_all) - np.nanmin(yayall_all)) / len(yayall_all))
            X = [i * stepp for i in range(len(yayall_all))]
            X = X / np.nanmax(X)
            axs.plot(sorted(Y), X, color='red', label='All Hii regions', linewidth=0.6)

            mean = np.nanmean(yayall_all)
            median = np.nanmedian(yayall_all)
            std_all = np.nanstd(yayall_all)
            mean_err = std_all / np.sqrt(len(yayall_all))

            axs.text(0.05, 0.95, 'Mean all = %5.2f  ' % (mean), fontsize=fontsize,  # $\pm$ %5.2f   , mean_err
                     horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes, weight='bold')

            axs.text(0.05, (0.68),
                     'Std all = %5.2f ' % (std_all), fontsize=fontsize,
                     horizontalalignment='left',
                     verticalalignment='center', transform=axs.transAxes, weight='bold')

            # axs.text(0.30, 0.95, 'Median all %5.2f ' % (median), fontsize=8,
            #          horizontalalignment='left',
            #          verticalalignment='center', transform=axs.transAxes, weight='bold')

        thres_i = 1
        for thres in threshold_percs:

            labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
                matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
                outliers=outliers, randomize=randomize, threshold_perc=thres, vel=vel, symmetrical=symmetrical)

            arrayyay1 = [arrayxax[0], arrayxax[1]]

            labsyay1 = [labsxax[0], labsxax[1]]

            axs.set(xlabel=labsyay1[i])
            # axs.grid()
            yaytmp = arrayyay1[i]
            yayall = np.concatenate([f.tolist() for f in yaytmp])

            yayall = np.log10(yayall)

            idok = np.where((abs(yayall) < 100000))
            yayall = yayall[idok]

            if thres == 0.1:
                colorr = 'orange'
            if thres == 0.5:
                colorr = 'green'
            if thres == 0.9:
                colorr = 'blue'

            if yayall.any != ():
                Y = yayall

                stepp = ((np.nanmax(yayall) - np.nanmin(yayall)) / len(yayall))
                X = [i * stepp for i in range(len(yayall))]
                X = X / np.nanmax(X)
                axs.plot(sorted(Y), X, color=colorr, label='Matched Hii reg. (%2.0f\%%)' % (100 * thres), linewidth=0.6)

                if i == 0:
                    axs.legend(prop={'size': 5}, loc=4)
                else:
                    axs.legend(prop={'size': 6}, loc=4)

                mean = np.nanmean(yayall)
                median = np.nanmedian(yayall)
                std_all = np.nanstd(yayall)
                mean_err1 = std_all / np.sqrt(len(yayall))

                axs.text(0.05, (0.95 - thres_i * 0.05), 'Mean (%1.0f\%%) = %5.2f ' % ((thres * 100), mean),
                         fontsize=fontsize,  # $\pm$ %5.2f   , mean_err1
                         horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes, weight='bold')

                axs.text(0.05, (0.68 - thres_i * 0.05), 'Std (%1.0f\%%) = %5.2f ' % ((thres * 100), std_all,),
                         fontsize=fontsize,
                         horizontalalignment='left',
                         verticalalignment='center', transform=axs.transAxes, weight='bold')

                # axs.text(0.30, (0.95-thres_i*0.05), 'Median matched (%1.1f) %5.2f ' % (thres,median), fontsize=8,
                #         horizontalalignment='left',
                #         verticalalignment='center', transform=axs.transAxes, weight = 'bold')

                if i == 0:
                    axs.set(xlim=[0.5, 2.6])
                else:
                    axs.set(xlim=[34.7, 43.2])

                thres_i += 1
            l += 1

        data_routines.save_pdf(pdf3, fig, save, show)

    pdf3.close()


def plot_gmcprop_regions(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, show, save,
                         threshold_perc, vel, gmcprop, regions):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair

    region_name = ['Center', 'Arms', 'Disc']

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)
    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    xlimmin, xlimmax = data_routines.get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    reg = 0
    for i in range(len(regions)):

        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = data_routines.get_data(
            matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
            outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)

        regions_id = regions[i]

        if regions[i] != [1, 2, 3]:
            color = 'red'

        else:
            color = 'black'

        regions_colors = regions[i]

        for i in range(np.shape(arrayxax)[0]):
            for j in range(np.shape(arrayxax)[1]):
                id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
                arrayxax[i][j] = arrayxax[i][j][id_int]

        for i in range(np.shape(arrayyay)[0]):
            for j in range(np.shape(arrayyay)[1]):
                id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
                arrayyay[i][j] = arrayyay[i][j][id_int]

        for j in range(len(FluxCOGMCover)):
            id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
            FluxCOGMCover[j] = FluxCOGMCover[j][id_int]

        idoverhii = [np.array(x) for x in idoverhii]
        idovergmc = [np.array(x) for x in idovergmc]
        for j in range(len(idoverhii)):
            id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
            idoverhii[j] = idoverhii[j][id_int]

        for j in range(len(idovergmc)):
            id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
            idovergmc[j] = idovergmc[j][id_int]

        k = 1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(2, 2, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        # fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s \n %s' % (name_end, region_name[reg]), fontsize=18, va='top')
        axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0
        for i in gmcprop:

            if regions_colors == [4, 5, 6] and i == 3:
                color = 'tab:blue'
            elif regions_colors == [1, 2, 3]:
                color = 'black'
            else:
                color = 'tab:red'

            axs[l].set(ylabel=labsyay[i])
            axs[l].grid()
            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            # if k < 5:
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)

            axs[l].plot(xaxall, yayall, 'o', markerfacecolor='None', markersize=2, markeredgecolor=color,
                        label='threshold = %f' % threshold_perc, linestyle='None')

            axs[l].set_xlim(xlimmin, xlimmax)

            idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
            xaxall = xaxall[idok]
            yayall = yayall[idok]
            lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
            lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
            indlim = np.where((xaxall < lim2) & (xaxall > lim1))
            xaxall = xaxall[indlim]
            yayall = yayall[indlim]
            if xaxall.any() != 0 and yayall.any != ():
                xmin = np.amin(xaxall)
                xmax = np.amax(xaxall)
                xprang = (xmax - xmin) * 0.1
                x = xaxall.reshape((-1, 1))
                y = yayall

                LumHacorrover = arrayxax[1]
                LumHacorrovertot = sum([x for sublist in LumHacorrover for x in sublist])
                FluxCOGMCovertot = sum([x for sublist in FluxCOGMCover for x in sublist])

                FluxCOGMCtot = sum([x for sublist in FluxCOGMCnot for x in sublist])
                LumHacorrtot = sum([x for sublist in LumHacorrnot for x in sublist])

                n_gmcs_tot = sum([len(x) for x in FluxCOGMCnot])
                n_hiis_tot = sum([len(x) for x in LumHacorrnot])

                n_gmcs = num_gmcs(idovergmc)
                n_hiis = num_hiis(idoverhii)

                conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr, rms_tot, rms_err, dw, mean_error, SSt, SSe, MSe, d = conf_intervals(
                    xaxall, yayall, alpha)

                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * x0

                # axs.plot(x0, y_pred, '-')
                # sns.regplot(x,y, scatter_kws={'s':2} )
                y_pred_sup = b + conf_b + (slope + conf_a) * x0
                y_pred_inf = b - conf_b + (slope - conf_a) * x0

                # axs.plot(x0, y_pred_sup, color = "lightgreen")
                # axs.plot(x0, y_pred_inf, color = "lightgreen")

                axs[l].plot(x0, y_pred + y_conf_sup, '--', color='black', label='confidence interval')
                axs[l].plot(x0, y_pred + y_conf_inf, '--', color='black')

                axs[l].plot(x0, y_pred + int_prev_y, '-.', color='grey', label='prediction interval')
                axs[l].plot(x0, y_pred - int_prev_y, '-.', color='grey')

                axs[l].plot(x0, y_pred, color='navy', label='linear regression')

                r_sq = np.sqrt(r_sq)

                x0, xf = xlim[k]
                y0, yf = ylim[i]
                # axs.text(0.8, 0.15, 'P-Value: %6.2f' % (pvalue), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                # =======
                x = xaxall
                y = yayall
                p, V = np.polyfit(x, y, 1, cov=True)

                axs[l].text(0.8, 0.05, 'R²: %5.3f' % ((r_sq) ** 2), fontsize=10, horizontalalignment='center',
                            verticalalignment='center', transform=axs[l].transAxes)

                # axs.text(0.8, 0.19, 'Durbin-Watson stat: %6.2f' % (dw), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                axs[l].text(0.8, 0.1, 'Slope %5.3f  ' % (slope), fontsize=10, horizontalalignment='center',
                            verticalalignment='center', transform=axs[l].transAxes)

                # axs.text(0.8, 0.07, 'Standard deviation %5.3f ' % (np.sqrt(MSe)), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)
                #
                #
                # axs.text(0.8, 0.11, 'paired gmcs %5.3f (%5.2f %%) ' % (n_gmcs, n_gmcs*100/n_gmcs_tot), fontsize=8, horizontalalignment='center',
                #          verticalalignment='center', transform=axs.transAxes)
                #
                # axs.text(0.8, 0.15, 'paired hii regions %5.3f (%5.2f %%) ' % (n_hiis, n_hiis*100/n_hiis_tot), fontsize=8, horizontalalignment='center',
                #          verticalalignment='center', transform=axs.transAxes)
                #
                # axs.text(0.8, 0.19, 'paired gmcs CO flux: %5.2f %% ' % (FluxCOGMCovertot*100/FluxCOGMCtot),
                #          fontsize=8, horizontalalignment='center',
                #          verticalalignment='center', transform=axs.transAxes)
                #
                # axs.text(0.8, 0.23, 'paired hii regions Ha Lum: %5.2f %% ' % (LumHacorrovertot*100/LumHacorrtot),
                #          fontsize=8, horizontalalignment='center',
                #          verticalalignment='center', transform=axs.transAxes)

                axs[l].set(ylim=(y0, yf))
                axs[l].legend(prop={'size': 8}, loc=2)

                axs[l].set(xlabel=labsxax[k])

                l += 1
        # axs[2].set(xlabel=labsxax[k])

        data_routines.save_pdf(pdf3, fig, save, show)
        reg += 1

    pdf3.close()


def plotallgals(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, show, save, threshold_perc, vel,
                gmcprop):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)
    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    xlimmin, xlimmax = data_routines.get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    for j in range(len(galaxias)):
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = data_routines.get_data(
            matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
            outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)

        k = 1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(1, 1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s \n %s' % (name_end, galaxias[j]),
                     fontsize=18, va='top')
        # axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0
        for i in gmcprop:

            axs.set(ylabel=labsyay[i])
            axs.grid()
            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
            xaxall = arrayxax[k][j]
            yayall = arrayyay[i][j]
            # if k < 5:
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)

            axs.plot(xaxall, yayall, 'o', markerfacecolor='None', markersize=2, markeredgecolor='black',
                     label='threshold = %f' % threshold_perc, linestyle='None')

            idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
            xaxall = xaxall[idok]
            yayall = yayall[idok]
            lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
            lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
            indlim = np.where((xaxall < lim2) & (xaxall > lim1))
            # xaxall = xaxall[indlim]
            # yayall = yayall[indlim]
            if xaxall.any() != 0 and yayall.any != ():
                xmin = np.amin(xaxall)
                xmax = np.amax(xaxall)
                xprang = (xmax - xmin) * 0.1
                x = xaxall.reshape((-1, 1))
                y = yayall

                LumHacorrover = arrayxax[1]
                LumHacorrovertot = sum([x for sublist in LumHacorrover for x in sublist])
                FluxCOGMCovertot = sum([x for sublist in FluxCOGMCover for x in sublist])

                FluxCOGMCtot = sum([x for sublist in FluxCOGMCnot for x in sublist])
                LumHacorrtot = sum([x for sublist in LumHacorrnot for x in sublist])

                n_gmcs_tot = sum([len(x) for x in FluxCOGMCnot])
                n_hiis_tot = sum([len(x) for x in LumHacorrnot])

                id = [idovergmc[j].count(x) for x in idovergmc[j]]
                n_gmcs = len(idovergmc[j]) - sum([x - 1 for x in id])  # num_gmcs(idovergmc[j])
                n_hiis = len(idoverhii[j])

                conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr, rms_tot, rms_err, dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(
                    xaxall, yayall, alpha)

                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * x0

                # axs.plot(x0, y_pred, '-')
                # sns.regplot(x,y, scatter_kws={'s':2} )
                y_pred_sup = b + conf_b + (slope + conf_a) * x0
                y_pred_inf = b - conf_b + (slope - conf_a) * x0

                # axs.plot(x0, y_pred_sup, color = "lightgreen")
                # axs.plot(x0, y_pred_inf, color = "lightgreen")

                axs.plot(x0, y_pred + y_conf_sup, '--', color='black', label='confidence interval')
                axs.plot(x0, y_pred + y_conf_inf, '--', color='black')

                axs.plot(x0, y_pred + int_prev_y, '-.', color='grey', label='prediction interval')
                axs.plot(x0, y_pred - int_prev_y, '-.', color='grey')

                axs.plot(x0, y_pred, color='navy', label='linear regression')

                r_sq = np.sqrt(r_sq)

                x0, xf = xlim[k]
                y0, yf = ylim[i]
                # axs.text(0.8, 0.15, 'P-Value: %6.2f' % (pvalue), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                # =======
                x = xaxall
                y = yayall
                # p,V = np.polyfit(x, y, 1, cov=True)

                axs.text(0.8, 0.03, 'R²: %5.3f' % ((r_sq) ** 2), fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                # axs.text(0.8, 0.19, 'Durbin-Watson stat: %6.2f' % (dw), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                axs.text(0.15, 0.05, 'Slope %5.3f $\pm$ %5.3f ' % (slope, conf_a), fontsize=8,
                         horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.07, 'Standard error of estimate %5.3f ' % (np.sqrt(MSe)), fontsize=8,
                         horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.11, 'paired gmcs %5.3f (%5.2f %%) ' % (n_gmcs, n_gmcs * 100 / n_gmcs_tot), fontsize=8,
                         horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.15, 'paired hii regions %5.3f (%5.2f %%) ' % (n_hiis, n_hiis * 100 / n_hiis_tot),
                         fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.19, 'paired gmcs CO flux: %5.2f %% ' % (FluxCOGMCovertot * 100 / FluxCOGMCtot),
                         fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.23, 'paired hii regions Ha Lum: %5.2f %% ' % (LumHacorrovertot * 100 / LumHacorrtot),
                         fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.set(ylim=(y0, yf))
                axs.legend(prop={'size': 8})

                l += 1
        axs.set(xlabel=labsxax[k])
        data_routines.save_pdf(pdf3, fig, save, show)

    pdf3.close()


def hist_all(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, vel, show, save, threshold_percs,
             bin, gmc_props, symmetrical):
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    threshold_perc = threshold_percs[0]

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel, symmetrical=symmetrical)
    if save == True:
        pdf_name = "%shist_hii_props_all_%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    fig, axs = plt.subplots(1, 1, figsize=(4, 4), dpi=80, gridspec_kw={'hspace': 0.10}, sharex=True)
    # fig.suptitle(
    #   'All galaxies - Overlapping HIIregions and GMCs - GMCproperties histograms \n threshold = %s' % threshold_perc,
    #  fontsize=18, va='top')

    SizepcGMC = np.concatenate([f.tolist() for f in SizepcGMC])
    SizepcHII = np.concatenate([f.tolist() for f in SizepcHII])

    SizepcGMC = np.log10(SizepcGMC)
    SizepcHII = np.log10(SizepcHII)

    idok = np.where((abs(SizepcGMC) < 100000))
    SizepcGMC = SizepcGMC[idok]

    idok = np.where((abs(SizepcHII) < 100000))
    SizepcHII = SizepcHII[idok]

    binn = bin
    fontsize = 5

    if SizepcHII.any != ():
        axs.hist(SizepcHII, bins=binn, color='red', label='Hii regions', histtype='stepfilled',
                 alpha=0.7)

        mean = np.nanmean(SizepcHII)
        median = np.nanmedian(SizepcHII)
        standard_dev = np.nanstd(SizepcHII)

    if SizepcGMC.any != ():
        axs.hist(SizepcGMC, bins=binn, color='blue',
                 label='GMCs ',
                 histtype='stepfilled', alpha=0.65)

        mean = np.nanmean(SizepcGMC)
        median = np.nanmedian(SizepcGMC)
        standard_deviation = np.nanstd(SizepcGMC)

    # axs[l].text(left_align, 0.90 - 0.12, 'Mean matched %5.3f ' % (mean), fontsize=fontsize,
    #             horizontalalignment='left',
    #             verticalalignment='center', transform=axs[l].transAxes, weight='bold')
    #
    # axs[l].text(left_align, 0.90 - 0.12 - 0.12 - 0.14, 'Median matched %5.3f ' % (median),
    #             fontsize=fontsize,
    #             horizontalalignment='left',
    #             verticalalignment='center', transform=axs[l].transAxes, weight='bold')
    #
    # axs[l].text(left_align, (0.90 - 0.12 - 0.12 - 0.14 - 0.14 - 0.12),
    #             'Std matched %5.2f ' % (standard_dev), fontsize=fontsize,
    #             horizontalalignment='left',
    #             verticalalignment='center', transform=axs[l].transAxes, weight='bold')
    #
    # axs[l].text(left_align, 0.90, 'Mean all %5.3f ' % (mean), fontsize=fontsize,
    #             horizontalalignment='left',
    #             verticalalignment='center', transform=axs[l].transAxes, weight='bold')
    #
    # axs[l].text(left_align, 0.90 - 0.12 - 0.14, 'Median all %5.3f ' % (median), fontsize=fontsize,
    #             horizontalalignment='left',
    #             verticalalignment='center', transform=axs[l].transAxes, weight='bold')
    #
    # axs[l].text(left_align, 0.90 - 0.12 * 2 - 0.14 * 2, 'Std all %5.2f ' % (standard_dev),
    #             fontsize=fontsize,
    #             horizontalalignment='left',
    #             verticalalignment='center', transform=axs[l].transAxes, weight='bold')

    # ttest_stat, p_value = scipy.stats.ttest_ind(yayall_all, yayall, equal_var = True)
    # alpha = 0.05  # Or whatever you want your alpha to be.
    # F = np.std(yayall) / np.std(yayall_all)
    # df1 = len(yayall) - 1
    # df2 = len(yayall_all) - 1
    # p_value_f = scipy.stats.f.cdf(F, df1, df2)
    #
    #
    # print("f test p-value (%5.1f%%) = %f" %((thres*100),p_value_f))
    # if p_value_f > alpha:
    #     print("null hypothesis rejected, DIFFERENT STD")
    # else:
    #     print("null hypothesis not rejected, IDENTICAL STD")
    #
    #
    #
    #
    # print("student test p-value (%5.1f%%) = %f" %((thres*100),p_value))
    # if p_value > 0.05:
    #     print("null hypothesis not rejected, IDENTICAL AVERAGES")
    # else:
    #     print("null hypothesis rejected, DIFFERENT AVERAGES")

    data_routines.save_pdf(pdf3, fig, save, show)

    pdf3.close()


def plotallgals_galprop(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, show, save,
                        threshold_perc, vel, gmcprop, sorting, symmetrical):
    # ===============================================================

    # Plots of correlations with dots for each pair
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel, symmetrical=symmetrical,
        dir_script_data=dir_script_data)
    if save == True:
        pdf_name = "%sCorrelations_AllGals%s%s%s.pdf" % (dirplots, sorting, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    table = Table.read('/home/antoine/Internship/phangs_sample_table_v1p6.fits')

    galnames = galaxias

    galnames = [str.lower(x) for x in galnames]
    ids = [id for id, x in enumerate(table['name']) if x in galnames]
    SFR = table['props_sfr'][ids]
    sorted_SFR = table['props_sfr'][ids]

    stellar_Mass_1 = table['props_mstar'][ids]
    sorted_stellar_mass = table['props_mstar'][ids]

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

    if sorting == 'mass':
        ind_value = ind_value_mass
        sorting = 'stellar mass'
    if sorting == 'sfr':
        ind_value = ind_value_sfr

    print(ind_value)

    # arrayxax_c = arrayxax
    # arrayyay_c = arrayyay
    #
    # for i in range(np.shape(arrayxax)[0]):
    #     for j in range(np.shape(arrayxax)[1]):
    #         arrayxax[i][j] = arrayxax_c[i][ind_value[j]]
    #         arrayyay[i][j] = arrayyay_c[i][ind_value[j]]

    r2_list = []
    slope_list = []
    mse_list = []
    conf_a_list = []

    xlimmin, xlimmax = data_routines.get_min(arrayxax)
    num_graphs = 20

    pages = int(len(galaxias) / num_graphs) + 1

    print("Starting loop to create figures of all galaxies together - points")
    fig, axs = plt.subplots(5, 4, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
    plt.subplots_adjust(wspace=0.3)

    axs = axs.ravel()
    l = 0
    k = 1
    i = gmcprop[0]

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel, symmetrical=symmetrical,
        dir_script_data=dir_script_data)

    xaxall = copy.copy(arrayxax[k])
    yayall = copy.copy(arrayyay[i])

    xaxall, yayall = gmchiistats_routines.prepdata2(xaxall, yayall)

    slope_glob, b, r_sq_glob = linear_regression(xaxall, yayall)

    for j in range(len(galaxias)):

        axs[0].set(ylabel=labsyay[i])
        axs[4].set(ylabel=labsyay[i])
        axs[8].set(ylabel=labsyay[i])
        axs[12].set(ylabel=labsyay[i])
        axs[16].set(ylabel=labsyay[i])

        axs[l].grid()
        # axs[l].text(.5,.9,'%s' %galaxias[j], fontsize=18, ha='center')



        xaxall = copy.copy(arrayxax[k][ind_value[j]])
        yayall = copy.copy(arrayyay[i][ind_value[j]])
        # if k < 5:
        xaxall = np.log10(xaxall)
        yayall = np.log10(yayall)
        print(j)

        if len(xaxall) <= 1:
            xaxall = np.array([40, 41])
            yayall = np.array([1, 2])

        axs[l].plot(xaxall, yayall, 'o', markerfacecolor='None', markersize=2, markeredgecolor='black',
                    label=galaxias[ind_value[j]], linestyle='None')
        axs[l].legend(prop={'size': 8})

        idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))

        xaxall = xaxall[idok]
        yayall = yayall[idok]
        lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
        lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
        indlim = np.where((xaxall < lim2) & (xaxall > lim1))
        xaxall = xaxall[indlim]
        yayall = yayall[indlim]
        xmin = np.amin(xaxall)
        xmax = np.amax(xaxall)
        xprang = (xmax - xmin) * 0.1
        x = xaxall.reshape((-1, 1))
        y = yayall

        LumHacorrover = arrayxax[1]
        # LumHacorrovertot = sum([x for sublist in LumHacorrover for x in sublist])
        # FluxCOGMCovertot = sum([x for sublist in FluxCOGMCover for x in sublist])
        #
        # FluxCOGMCtot = sum([x for sublist in FluxCOGMCnot for x in sublist])
        # LumHacorrtot = sum([x for sublist in LumHacorrnot for x in sublist])
        #
        #
        #
        # n_gmcs_tot = sum([len(x) for x in FluxCOGMCnot])
        # n_hiis_tot = sum([len(x) for x in LumHacorrnot])

        id = [idovergmc[j].count(x) for x in idovergmc[j]]
        n_gmcs = len(idovergmc[j]) - sum([x - 1 for x in id])  # num_gmcs(idovergmc[j])
        n_hiis = len(idoverhii[j])
        alpha = 0.05

        conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr, rms_tot, rms_err, dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(
            xaxall, yayall, alpha)


        slope, b, r_sq = linear_regression(xaxall, yayall)
        y_pred = b + slope * x0

        print(galaxias[j])
        print(r_sq)
        print(len(xaxall))
        print('\n')

        # axs.plot(x0, y_pred, '-')
        # sns.regplot(x,y, scatter_kws={'s':2} )
        y_pred_sup = b + conf_b + (slope + conf_a) * x0
        y_pred_inf = b - conf_b + (slope - conf_a) * x0

        # axs.plot(x0, y_pred_sup, color = "lightgreen")
        # axs.plot(x0, y_pred_inf, color = "lightgreen")

        axs[l].plot(x0, y_pred + y_conf_sup, '--', color='black')
        axs[l].plot(x0, y_pred + y_conf_inf, '--', color='black')

        axs[l].plot(x0, y_pred + int_prev_y, '-.', color='grey')
        axs[l].plot(x0, y_pred - int_prev_y, '-.', color='grey')

        axs[l].plot(x0, y_pred, color='navy')

        if galaxias[ind_value[j]] == 'ic5332':
            conf_a = 0

        r2_list.append(r_sq)
        print(r_sq)
        slope_list.append(slope)
        conf_a_list.append(conf_a)
        mse_list.append(rms_err)

        r = np.sqrt(r_sq)


        axs[l].text(0.75, 0.15, 'r²: %5.3f' % ((r_sq)), fontsize=8, horizontalalignment='center',
                    verticalalignment='center', transform=axs[l].transAxes, fontweight="bold")

        # axs.text(0.8, 0.19, 'Durbin-Watson stat: %6.2f' % (dw), fontsize=8, horizontalalignment='center',
        #             verticalalignment='center', transform=axs.transAxes)

        axs[l].text(0.75, 0.05, 'Slope %5.3f ' % (slope), fontsize=8, horizontalalignment='center',
                    verticalalignment='center', transform=axs[l].transAxes, fontweight="bold")

        # axs[l].text(0.85, 0.15, 'Intercept %5.3f ' % (b), fontsize=10, horizontalalignment='center',
        #             verticalalignment='center', transform=axs[l].transAxes, fontweight = "bold")

        # axs[l].text(0.8, 0.07, 'Standard error of estimate %5.3f ' % (np.sqrt(MSe)), fontsize=8, horizontalalignment='center',
        #             verticalalignment='center', transform=axs[l].transAxes)
        #
        #
        # axs[l].text(0.8, 0.11, 'paired gmcs %5.3f (%5.2f %%) ' % (n_gmcs, n_gmcs*100/n_gmcs_tot), fontsize=8, horizontalalignment='center',
        #          verticalalignment='center', transform=axs[l].transAxes)
        #
        # axs[l].text(0.8, 0.15, 'paired hii regions %5.3f (%5.2f %%) ' % (n_hiis, n_hiis*100/n_hiis_tot), fontsize=8, horizontalalignment='center',
        #          verticalalignment='center', transform=axs[l].transAxes)
        #
        # axs[l].text(0.8, 0.19, 'paired gmcs CO flux: %5.2f %% ' % (FluxCOGMCovertot*100/FluxCOGMCtot),
        #          fontsize=8, horizontalalignment='center',
        #          verticalalignment='center', transform=axs[l].transAxes)
        #
        # axs[l].text(0.8, 0.23, 'paired hii regions Ha Lum: %5.2f %% ' % (LumHacorrovertot*100/LumHacorrtot),
        #          fontsize=8, horizontalalignment='center',
        #          verticalalignment='center', transform=axs[l].transAxes)


        l += 1

    axs[16].set(xlabel=labsxax[k])
    axs[17].set(xlabel=labsxax[k])
    axs[18].set(xlabel=labsxax[k])
    axs[19].set(xlabel=labsxax[k])

    data_routines.save_pdf(pdf3, fig, save, show)

    pdf3.close()

    if sorting == 'stellar mass':
        sorted_prop = sorted_stellar_mass
        sorting = sorting + ' (log10(M$_{\odot}$))'
    if sorting == 'sfr':
        sorting = str.upper(sorting) + ' log10(M$_{\odot}$/yr)'
        sorted_prop = sorted_SFR

    sorted_prop = np.array(sorted_prop)
    slope_list = np.array(slope_list)
    r2_list = np.array(r2_list)
    conf_a_list = np.array(conf_a_list)
    mse_list = np.array(mse_list)



    print(r2_list)

    sorted_prop = np.log10(sorted_prop)


    #cdf
    xlabel=['','Molecular Mass', '' ,'Velocity Dispersion', '', 'Temperature at Peak']

    fig, axs = plt.subplots(1, 1, figsize=(4, 4))

    Y = r2_list
    stepp = ((np.nanmax(r2_list) - np.nanmin(r2_list)) / len(r2_list))
    X = [i * stepp for i in range(len(r2_list))]
    X = X / np.nanmax(X)
    axs.plot(sorted(Y), X, color='red', label='All GMCss', linewidth=0.6)
    axs.set_xlabel("R² (%s)"%xlabel[gmcprop[0]], fontsize=30)

    plt.tick_params(axis="x", labelsize=25)
    plt.tick_params(axis="y", labelsize=25)


    xlabel=['','Molecular Mass', '' ,'Velocity Dispersion', '', 'Temperature at Peak']

    fig, axs = plt.subplots(1, 1, figsize=(4, 4))

    Y = slope_list
    stepp = ((np.nanmax(slope_list) - np.nanmin(slope_list)) / len(slope_list))
    X = [i * stepp for i in range(len(slope_list))]
    X = X / np.nanmax(X)
    axs.plot(sorted(Y), X, color='red', label='All GMCss', linewidth=0.6)
    axs.set_xlabel("Slope (%s)"%xlabel[gmcprop[0]], fontsize=30)

    plt.tick_params(axis="x", labelsize=25)
    plt.tick_params(axis="y", labelsize=25)
    plt.show()


    #
    plt.errorbar((sorted_prop), r2_list, mse_list, capsize=1.5, capthick=0.5, elinewidth=0.5, color='black')
    plt.hlines(r_sq_glob, color='red', xmin=np.nanmin(sorted_prop), xmax=np.nanmax(sorted_prop))

    plt.ylabel('Correlation coefficient r² (%s)' %labsyay[gmcprop[0]], fontsize=30)
    plt.xlabel(sorting, fontsize=30)
    plt.grid()

    plt.tick_params(axis="x", labelsize=25)
    plt.tick_params(axis="y", labelsize=25)

    plt.show()

    #
    plt.errorbar((sorted_prop), slope_list, conf_a_list, capsize=1.5, capthick=0.5, elinewidth=0.5, color='black')
    plt.hlines(slope_glob, color='red', xmin=np.nanmin(sorted_prop), xmax=np.nanmax(sorted_prop))

    plt.ylabel('Slope (%s)' %labsyay[gmcprop[0]], fontsize=30)
    plt.xlabel(sorting, fontsize=30)
    plt.grid()

    plt.tick_params(axis="x", labelsize=25)
    plt.tick_params(axis="y", labelsize=25)

    plt.show()


def plotallgals_covariance(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, show, save,
                           threshold_perc, vel, gmcprop, symmetrical):

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel, symmetrical=symmetrical, dir_script_data=dir_script_data)



    table = Table.read('/home/antoine/Internship/phangs_sample_table_v1p6.fits')
    table_muse = Table.read('/home/antoine/Internship/muse_hii_new_dr22/Nebulae_catalogue_v2.fits')
    region_name = ['Center', 'Arms', 'Disc']

    galnames = galaxias

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


    r2_list_gals = []
    slope_list_gals = []

    k = 1

    for gmc_prop in gmcprop:

        r2_list = []
        slope_list = []

        galaxias = galnames

        for j in range(len(galaxias)):
            labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1 = data_routines.get_data(
                matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
                outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel, symmetrical=symmetrical, dir_script_data=dir_script_data)
            galaxias = galnames




            i = gmc_prop

            xaxall = arrayxax[k][j]
            yayall = arrayyay[i][j]

            regions_id = regions[i]

            for i in range(np.shape(arrayxax)[0]):
                for j in range(np.shape(arrayxax)[1]):
                    id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
                    arrayxax[i][j] = arrayxax[i][j][id_int]

            for i in range(np.shape(arrayyay)[0]):
                for j in range(np.shape(arrayyay)[1]):
                    id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
                    arrayyay[i][j] = arrayyay[i][j][id_int]

            for j in range(len(FluxCOGMCover)):
                id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
                FluxCOGMCover[j] = FluxCOGMCover[j][id_int]

            idoverhii = [np.array(x) for x in idoverhii]
            idovergmc = [np.array(x) for x in idovergmc]

            for j in range(len(idoverhii)):
                id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
                idoverhii[j] = idoverhii[j][id_int]

            for j in range(len(idovergmc)):
                id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
                idovergmc[j] = idovergmc[j][id_int]















            xaxall, yayall = gmchiistats_routines.prepdata_single2(xaxall, yayall)




            alpha = 0.05
            if xaxall.any() != 0 and yayall.any != ():

                conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr, rms_tot, rms_err, dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(
                    xaxall, yayall, alpha)
                slope, b, r_sq = linear_regression(xaxall, yayall)
                r2_list.append(r_sq)

                slope_list.append(slope)
            else:
                print('bug')
                print(j)

        r2_list_gals.append(r2_list)
        slope_list_gals.append(slope_list)


    fig = plt.figure()
    fig.suptitle('Correlation Coefficient')
    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel('Moelcular Mass r²')
    ax.set_ylabel('Velocity dispersion r²')
    ax.set_zlabel('Temperature at peak r²')

    ax.set_xlim3d(0, np.nanmax(r2_list_gals))
    ax.set_ylim3d(0, np.nanmax(r2_list_gals))
    ax.set_zlim3d(0, np.nanmax(r2_list_gals))

    lco_phangs = table['lco_phangs'][ids]
    mh1 = table['props_mhi'][ids]
    mh2 = table['mh2_phangs'][ids]

    color_param = SFR


    # cmap = plt.get_cmap('viridis')#)
    norm = mpl.colors.Normalize(vmin=np.nanmin(color_param), vmax=np.nanmax(color_param))
    for j in range(len(galaxias) ):



        ax.scatter(r2_list_gals[0][j], r2_list_gals[1][j], r2_list_gals[2][j], c=color_param[j], norm=norm)
        ax.text(r2_list_gals[0][j], r2_list_gals[1][j], r2_list_gals[2][j] - 0.02, galaxias[j ])

    fig = plt.figure()
    fig.suptitle('Slope')
    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel('Moelcular Mass slope')
    ax.set_ylabel('Velocity dispersion slope')
    ax.set_zlabel('Temperature at peak slope')
    ax.set_xlim3d(0, np.nanmax(slope_list_gals))
    ax.set_ylim3d(0, np.nanmax(slope_list_gals))
    ax.set_zlim3d(0, np.nanmax(slope_list_gals))

    for j in range(len(galaxias)):
        ax.scatter(slope_list_gals[0][j], slope_list_gals[1][j], slope_list_gals[2][j], c=color_param[j], norm=norm)
        ax.text(slope_list_gals[0][j], slope_list_gals[1][j], slope_list_gals[2][j] - 0.02, galaxias[j])




    labels_r2 = ['Moelcular Mass r²','Velocity dispersion r²','Temperature at peak r²']
    labels_slope = ['Moelcular Mass slope','Velocity dispersion slope','Temperature at peak slope']



    pdf_name = "%sCovariance_r2%s%s.pdf" % (dirplots, namegmc, name_end)
    if save == True:
        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    txt_size = 5
    point_size = 0.25

    fig = plt.figure()
    ax = fig.add_subplot()

    for j in range(len(galaxias) ):
        ax.scatter(r2_list_gals[0][j], r2_list_gals[1][j],  c=color_param[j], linewidth = point_size)
        ax.text(r2_list_gals[0][j], (r2_list_gals[1][j]-0.02),  galaxias[j ], size = txt_size)
        ax.set_xlabel(labels_r2[0])
        ax.set_ylabel(labels_r2[1])


    data_routines.save_pdf(pdf3, fig, save, show)

    fig = plt.figure()
    ax = fig.add_subplot()

    for j in range(len(galaxias) ):
        ax.scatter(r2_list_gals[0][j], r2_list_gals[2][j],  c=color_param[j], linewidth = point_size)
        ax.text(r2_list_gals[0][j], r2_list_gals[2][j]-0.02,  galaxias[j ], size = txt_size)
        ax.set_xlabel(labels_r2[0])
        ax.set_ylabel(labels_r2[2])

    data_routines.save_pdf(pdf3, fig, save, show)

    fig = plt.figure()
    ax = fig.add_subplot()

    for j in range(len(galaxias) ):
        ax.scatter(r2_list_gals[1][j], r2_list_gals[2][j],  c=color_param[j], linewidth = point_size)
        ax.text(r2_list_gals[1][j], r2_list_gals[2][j]-0.02,  galaxias[j ], size = txt_size)
        ax.set_xlabel(labels_r2[1])
        ax.set_ylabel(labels_r2[2])

    data_routines.save_pdf(pdf3, fig, save, show)

    pdf3.close()



    pdf_name = "%sCovariance_slope%s%s.pdf" % (dirplots, namegmc, name_end)
    if save == True:
        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    fig = plt.figure()
    ax = fig.add_subplot()

    for j in range(len(galaxias) ):
        ax.scatter(slope_list_gals[0][j], slope_list_gals[1][j],  c=color_param[j], linewidth = point_size)
        ax.text(slope_list_gals[0][j], slope_list_gals[1][j]-0.02,  galaxias[j ], size = txt_size)
        ax.set_xlabel(labels_slope[0])
        ax.set_ylabel(labels_slope[1])

    data_routines.save_pdf(pdf3, fig, save, show)

    fig = plt.figure()
    ax = fig.add_subplot()

    for j in range(len(galaxias) ):
        ax.scatter(slope_list_gals[0][j], slope_list_gals[2][j],  c=color_param[j], linewidth = point_size)
        ax.text(slope_list_gals[0][j], slope_list_gals[2][j]-0.02,  galaxias[j ], size = txt_size)
        ax.set_xlabel(labels_slope[0])
        ax.set_ylabel(labels_slope[2])

    data_routines.save_pdf(pdf3, fig, save, show)

    fig = plt.figure()
    ax = fig.add_subplot()

    for j in range(len(galaxias) ):
        ax.scatter(slope_list_gals[1][j], slope_list_gals[2][j],  c=color_param[j], linewidth = point_size)
        ax.text(slope_list_gals[1][j], slope_list_gals[2][j]-0.02,  galaxias[j ], size = txt_size)
        ax.set_xlabel(labels_slope[1])
        ax.set_ylabel(labels_slope[2])

    data_routines.save_pdf(pdf3, fig, save, show)

    pdf3.close()
    mean_r2_list = [(r2_list_gals[0][j] + r2_list_gals[1][j] + r2_list_gals[2][j])/3 for j in range(len(r2_list_gals[1]))]
    mean_slope_list = [(slope_list_gals[0][j] + slope_list_gals[1][j] + slope_list_gals[2][j])/3 for j in range(len(slope_list_gals[1]))]

    galnames = table['name'][ids]
    sorted_galnames = table['name'][ids]
    SFR = table['props_sfr'][ids]
    sorted_SFR = table['props_sfr'][ids]
    stellar_Mass_1 = table['props_mstar'][ids]
    sorted_stellar_mass = table['props_mstar'][ids]

    sorted_mean_r2_list = copy.deepcopy(mean_r2_list)
    sorted_mean_r2_list.sort()

    sorted_mean_slope_list = copy.deepcopy(mean_slope_list)
    sorted_mean_slope_list.sort()


    sorted_mass_r2 = []
    sorted_galnames_r2 = []
    for value in sorted_mean_r2_list:
        for idd, item in enumerate(mean_r2_list):
            if item == value:
                print(item)
                sorted_mass_r2.append(stellar_Mass_1[idd])
                sorted_galnames_r2.append(galnames[idd])

    sorted_mass_slope = []
    sorted_galnames_slope = []
    for value in sorted_mean_slope_list:
        for idd, item in enumerate(mean_slope_list):
            if item == value:
                print(item)
                sorted_mass_slope.append(stellar_Mass_1[idd])
                sorted_galnames_slope.append(galnames[idd])

    for i in range(len(mean_r2_list)):
        print('%s   %1.4f   %1.2f' %(sorted_galnames_r2[i], np.log10(sorted_mass_r2[i]), sorted_mean_r2_list[i]))

    print(sorted_galnames_r2)
    print(sorted_mean_r2_list)

    print('\n')

    for i in range(len(mean_slope_list)):
        print('%s   %1.4f   %1.2f' %(sorted_galnames_slope[i], np.log10(sorted_mass_slope[i]), sorted_mean_slope_list[i]))

    print(sorted_galnames_slope)
    print(sorted_mean_slope_list)






