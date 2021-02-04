import sys
import numpy as np
import math
import pickle

from astropy import constants as ct
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
import seaborn as sns
import os

import warnings
warnings. filterwarnings("ignore")

np.set_printoptions(threshold=sys.maxsize)
sns.set(style="white", color_codes=False)
# ===================================================================================
from typing import List




def binned_linear_regression(new_muse, gmc_catalog, matching, outliers, show, save, threshold_perc,  *args, **kwargs):

    new_muse2 = kwargs.get('new_muse2', None)
    matching2 = kwargs.get('matching2', None)
    outliers2 = kwargs.get('outliers2', None)
    threshold_perc2 = kwargs.get('thresold_perc2', None)
    gmc_catalog2 = kwargs.get('gmc_catalog2', None)
    arm = kwargs.get('arm', None)
    out = kwargs.get('out', None)
    cent = kwargs.get('cent', None)
    region = kwargs.get('region', None)
    plot_lr = kwargs.get('plot_lr', None)
    plot_double_lr = kwargs.get('plot_double_lr', None)



    def save_pdf(pdf, fig):
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



    def plot_binned_regions(labsxax, labsyay, arrayxax1, arrayyay1, pdf_name, regions_id):
        if save == True:
            pdf4 = fpdf.PdfPages(pdf_name)  # type: PdfPages
        else:
            pdf4 = fpdf.PdfPages("blank")
        print("Starting loop to create figures of all galaxies together - binned")

        arrayxax = arrayxax1
        arrayyay = arrayyay1

        for i in range(np.shape(arrayxax)[0]):
            for j in range(np.shape(arrayxax)[1]):
                id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
                arrayxax[i][j] = arrayxax[i][j][id_int]

        for i in range(np.shape(arrayyay)[0]):
            for j in range(np.shape(arrayyay)[1]):
                id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
                arrayyay[i][j] = arrayyay[i][j][id_int]

        for k in range(len(arrayxax)):
            #    print "starting for k"
            sns.set(style='white', color_codes=False)
            fig, axs = plt.subplots(4, 2, sharex='col', figsize=(8, 10), gridspec_kw={'hspace': 0})
            plt.subplots_adjust(wspace=0.3)
            fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s%s \n regions = %s' %(name_end,typegmc,str(regions_id)), fontsize=18, va='top')
            axs = axs.ravel()
            #    print "starting for i"
            # Galactic distance vs: Mco, avir, sigmav,Sigmamol
            for i in range(len(labsyay)):
                yaytmp = arrayyay[i]
                xaxtmp = arrayxax[k]
                xaxall = np.concatenate([f.tolist() for f in xaxtmp])
                yayall = np.concatenate([f.tolist() for f in yaytmp])
                yayall = np.log10(yayall)
                if "Metallicity" in labsxax[k]:
                    xaxall = xaxall
                else:
                    xaxall = np.log10(xaxall)
                idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
                xaxall = xaxall[idok];
                yayall = yayall[idok]
                lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
                lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
                indlim = np.where((xaxall < lim2) & (xaxall > lim1))
                xaxall = xaxall[indlim];
                yayall = yayall[indlim]
                mybin = 20
                if xaxall.any() != 0 and yayall.any() != 0 and mybin != 0:
                    xbinned, ybinned, eybinned, nybinned = bindata(xaxall, yayall, mybin)
                    # if there is any nan inside
                    ido = np.where(np.array(nybinned) != 0)
                    xbinned = xbinned[ido]
                    ybinned = [g for g in np.array(ybinned)[ido]]
                    eybinned = [g for g in np.array(eybinned)[ido]]
                    nybinned = [g for g in np.array(nybinned)[ido]]
                    # Plot binned data
                    mysize = np.array(nybinned).astype(float)
                    mysize = (mysize - np.min(mysize)) / (np.max(mysize) - np.min(mysize)) * 9 + 3
                    mylims = [np.argmin(mysize), np.argmax(mysize)]
                    mylabs = ["Num of pairs: %s" % min(nybinned), "Num of pairs: %s" % max(nybinned)]
                    for j in range(len(xbinned)):
                        if j == np.argmin(mysize) or j == np.argmax(mysize):
                            axs[i].plot(xbinned[j], ybinned[j], linestyle="None", alpha=0.5, marker="o",
                                        markersize=mysize[j],
                                        color="red", label="Num of pairs: %s" % nybinned[j])
                        else:
                            axs[i].plot(xbinned[j], ybinned[j], linestyle="None", alpha=0.5, marker="o",
                                        markersize=mysize[j],
                                        color="red")
                    axs[i].errorbar(xbinned, ybinned, eybinned, capsize=5)
                    axs[i].set(ylabel=labsyay[i])
                    axs[i].grid()
                    # Computing the linear fit to the data, using the amount of
                    x = xbinned.reshape((-1, 1))
                    y = ybinned
                    model = LinearRegression().fit(x, y, nybinned)
                    r_sq = model.score(x, y)
                    slope = model.coef_
                    y_pred = model.intercept_ + model.coef_ * x.ravel()
                    axs[i].plot(xbinned, y_pred, '-')
                    # sn.regplot(x=xaxall, y=yayall, ax=axs[i])
                    if i == 0:
                        xmin = np.amin(xbinned)
                        xmax = np.amax(xbinned)
                        xprang = (xmax - xmin) * 0.03
                        x0 = xmin + xprang
                    #        y0, yf = axs[i].get_ylim()
                    #        my0 = y0-(yf-y0)*0.13
                    # new!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    x0, xf = xlim[k]
                    y0, yf = ylim[i]
                    xprang = xf - x0
                    yprang = yf - y0
                    my0 = y0 - (yf - y0) * 0.13
                    # axs[i].set(xlim=(x0,xf))
                    axs[i].set(ylim=(y0, yf))
                    axs[i].text(0.8, 0.1, 'R^2: %6.2f' % (r_sq), fontsize=8, horizontalalignment='center',
                                verticalalignment='center', transform=axs[i].transAxes)

                    axs[i].text(0.15, 0.1, 'Slope %5.2f' % (slope), fontsize=8, horizontalalignment='center',
                                verticalalignment='center', transform=axs[i].transAxes)

            #        axs[i].set(ylim=(y0-(yf-y0)*0.15,yf+(yf-y0)*0.15))
            #        axs[i].set(xlim=(xmin - xprang*3, xmax + xprang*3))
            axs[0].legend(prop={'size': 9})
            axs[6].set(xlabel=labsxax[k])
            axs[7].set(xlabel=labsxax[k])
            save_pdf(pdf4, fig)
        if save == True:
            pdf4.close()

    def plot_double_binned(labsxax, labsyay, arrayxax, arrayxax_no, arrayyay, arrayyay_no, pdf_name):
        # Plot binned
        if save == True:
            pdf4 = fpdf.PdfPages(pdf_name)  # type: PdfPages
        else:
            pdf4 = fpdf.PdfPages("blank")
        print("Starting loop to create figures of all galaxies together - binned")
        for k in range(len(arrayxax)):
            #    print "starting for k"
            sns.set(style='white', color_codes=False)
            fig, axs = plt.subplots(4, 2, sharex='col', figsize=(8, 10), gridspec_kw={'hspace': 0})
            plt.subplots_adjust(wspace=0.3)
            fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s%s \n %s%s' %(name_end,typegmc, name_end1,typegmc1), fontsize=18, va='top')
            axs = axs.ravel()
            #    print "starting for i"
            # Galactic distance vs: Mco, avir, sigmav,Sigmamol
            for i in range(len(labsyay)):
                yaytmp = arrayyay[i]
                yaytmp_no = arrayyay_no[i]

                xaxtmp = arrayxax[k]
                xaxtmp_no = arrayxax_no[k]

                xaxall = np.concatenate([f.tolist() for f in xaxtmp])
                xaxall_no = np.concatenate([f.tolist() for f in xaxtmp_no])

                yayall = np.concatenate([f.tolist() for f in yaytmp])
                yayall_no = np.concatenate([f.tolist() for f in yaytmp_no])

                yayall = np.log10(yayall)
                yayall_no = np.log10(yayall_no)

                if "Metallicity" in labsxax[k]:
                    xaxall = xaxall
                else:
                    xaxall = np.log10(xaxall)

                if "Metallicity" in labsxax_no[k]:
                    xaxall_no = xaxall_no
                else:
                    xaxall_no = np.log10(xaxall_no)

                idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
                idok_no = np.where((abs(yayall_no) < 100000) & (abs(xaxall_no) < 100000))

                xaxall = xaxall[idok]
                xaxall_no = xaxall_no[idok_no]

                yayall = yayall[idok]
                yayall_no = yayall_no[idok_no]

                lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
                lim1_no = np.nanmedian(xaxall_no) - np.nanstd(xaxall_no) * 4

                lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
                lim2_no = np.nanmedian(xaxall_no) + np.nanstd(xaxall_no) * 4

                indlim = np.where((xaxall < lim2) & (xaxall > lim1))
                indlim_no = np.where((xaxall_no < lim2_no) & (xaxall_no > lim1_no))

                xaxall = xaxall[indlim]
                xaxall_no = xaxall_no[indlim_no]

                yayall = yayall[indlim]
                yayall_no = yayall_no[indlim_no]

                mybin = 20
                if (xaxall.any() != 0 and yayall.any() != 0 and mybin != 0) and (
                        xaxall_no.any() != 0 and yayall_no.any() != 0 and mybin != 0):
                    xbinned, ybinned, eybinned, nybinned = bindata(xaxall, yayall, mybin)
                    xbinned_no, ybinned_no, eybinned_no, nybinned_no = bindata(xaxall_no, yayall_no, mybin)

                    # if there is any nan inside
                    ido = np.where(np.array(nybinned) != 0)
                    ido_no = np.where(np.array(nybinned_no) != 0)

                    xbinned = xbinned[ido]
                    xbinned_no = xbinned_no[ido_no]

                    ybinned = [g for g in np.array(ybinned)[ido]]
                    ybinned_no = [g for g in np.array(ybinned_no)[ido_no]]

                    eybinned = [g for g in np.array(eybinned)[ido]]
                    eybinned_no = [g for g in np.array(eybinned_no)[ido_no]]

                    nybinned = [g for g in np.array(nybinned)[ido]]
                    nybinned_no = [g for g in np.array(nybinned_no)[ido_no]]

                    # Plot binned data
                    mysize = np.array(nybinned).astype(float)
                    mysize_no = np.array(nybinned_no).astype(float)

                    mysize = (mysize - np.min(mysize)) / (np.max(mysize) - np.min(mysize)) * 9 + 3
                    mysize_no = (mysize_no - np.min(mysize_no)) / (np.max(mysize_no) - np.min(mysize_no)) * 9 + 3

                    mylims = [np.argmin(mysize), np.argmax(mysize)]
                    mylims_no = [np.argmin(mysize_no), np.argmax(mysize_no)]

                    mylabs = ["Num of pairs: %s" % min(nybinned), "Num of pairs: %s" % max(nybinned)]
                    mylabs_no = ["Num of pairs: %s" % min(nybinned_no), "Num of pairs: %s" % max(nybinned_no)]

                    for j in range(len(xbinned)):
                        if j == np.argmin(mysize) or j == np.argmax(mysize):
                            axs[i].plot(xbinned[j], ybinned[j], linestyle="None", alpha=0.5, marker="o",
                                        markersize=mysize[j],
                                        color="red", label="Num of pairs: %s" % nybinned[j])
                        else:
                            axs[i].plot(xbinned[j], ybinned[j], linestyle="None", alpha=0.5, marker="o",
                                        markersize=mysize[j],
                                        color="red")
                    axs[i].errorbar(xbinned, ybinned, eybinned, capsize=5)
                    axs[i].set(ylabel=labsyay[i])
                    axs[i].grid()
                    # Computing the linear fit to the data, using the amount of
                    x = xbinned.reshape((-1, 1))
                    y = ybinned
                    model = LinearRegression().fit(x, y, nybinned)
                    r_sq = model.score(x, y)
                    slope = model.coef_
                    y_pred = model.intercept_ + model.coef_ * x.ravel()
                    axs[i].plot(xbinned, y_pred, '-')
                    # sn.regplot(x=xaxall, y=yayall, ax=axs[i])
                    if i == 0:
                        xmin = np.amin(xbinned)
                        xmax = np.amax(xbinned)
                        xprang = (xmax - xmin) * 0.03
                        x0 = xmin + xprang
                    #        y0, yf = axs[i].get_ylim()
                    #        my0 = y0-(yf-y0)*0.13
                    # new!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    x0, xf = xlim[k]
                    y0, yf = ylim[i]
                    xprang = xf - x0
                    yprang = yf - y0
                    my0 = y0 - (yf - y0) * 0.13
                    # axs[i].set(xlim=(x0,xf))
                    axs[i].set(ylim=(y0, yf))
                    axs[i].text(0.8, 0.1, 'R^2: %6.2f' % (r_sq), fontsize=8, horizontalalignment='center',
                                verticalalignment='center', transform=axs[i].transAxes)

                    axs[i].text(0.15, 0.1, 'Slope %5.2f' % (slope) , fontsize=8, horizontalalignment='center',
                                verticalalignment='center', transform=axs[i].transAxes)
                    #        axs[i].set(ylim=(y0-(yf-y0)*0.15,yf+(yf-y0)*0.15))
                    #        axs[i].set(xlim=(xmin - xprang*3, xmax + xprang*3))

                    # ====================================================================================================#

                    for j in range(len(xbinned_no)):
                        if j == np.argmin(mysize_no) or j == np.argmax(mysize_no):
                            axs[i].plot(xbinned_no[j], ybinned_no[j], linestyle="None", alpha=0.5, marker="o",
                                        markersize=mysize_no[j],
                                        color="peru", label="Num of pairs: %s" % nybinned_no[j])
                        else:
                            axs[i].plot(xbinned_no[j], ybinned_no[j], linestyle="None", alpha=0.5, marker="o",
                                        markersize=mysize_no[j],
                                        color="peru")
                    axs[i].errorbar(xbinned_no, ybinned_no, eybinned_no, capsize=5)
                    axs[i].set(ylabel=labsyay_no[i])
                    axs[i].grid()
                    # Computing the linear fit to the data, using the amount of
                    x = xbinned_no.reshape((-1, 1))
                    y = ybinned_no
                    model = LinearRegression().fit(x, y, nybinned_no)
                    r_sq_no = model.score(x, y)
                    slope_no = model.coef_
                    y_pred = model.intercept_ + model.coef_ * x.ravel()
                    axs[i].plot(xbinned_no, y_pred, '-')
                    # sn.regplot(x=xaxall, y=yayall, ax=axs[i])
                    if i == 0:
                        xmin = np.amin(xbinned_no)
                        xmax = np.amax(xbinned_no)
                        xprang = (xmax - xmin) * 0.03
                        x0 = xmin + xprang
                    #        y0, yf = axs[i].get_ylim()
                    #        my0 = y0-(yf-y0)*0.13
                    # new!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    x0, xf = xlim[k]
                    y0, yf = ylim[i]
                    xprang = xf - x0
                    yprang = yf - y0
                    my0 = y0 - (yf - y0) * 0.13
                    # axs[i].set(xlim=(x0,xf))
                    axs[i].set(ylim=(y0, yf))
                    axs[i].text(0.8, 0.04, 'R^2_other : %6.2f' % (r_sq_no), fontsize=8, horizontalalignment='center',
                                verticalalignment='center', transform=axs[i].transAxes)

                    axs[i].text(0.20, 0.04, 'Slope_other : %5.2f' %(slope_no), fontsize=8, horizontalalignment='center',
                                verticalalignment='center', transform=axs[i].transAxes)
            #        axs[i].set(ylim=(y0-(yf-y0)*0.15,yf+(yf-y0)*0.15))
            #        axs[i].set(xlim=(xmin - xprang*3, xmax + xprang*3))

            axs[0].legend(prop={'size': 9})
            axs[6].set(xlabel=labsxax[k])
            axs[7].set(xlabel=labsxax[k])
            save_pdf(pdf4, fig)
        if save == True:
            pdf4.close()

    def bindata(xaxall, yayall, mybin):
        xran = np.amax(xaxall) - np.amin(xaxall)
        xspa = xran / mybin
        xsta = np.amin(xaxall) + xspa / 2
        xfin = np.amax(xaxall) - xspa / 2
        xbinned = np.linspace(xsta, xfin, mybin)
        xbinned1 = []
        ybinned = []
        eybinned = []
        nybinned = []
        for t in range(mybin):
            idxs = np.where(abs(xaxall - xbinned[t]) < xspa / 2)
            yayin = yayall[idxs]
            nyayin = len(yayin)
            xbinned1.append(np.mean(xaxall[idxs]))
            myayin = np.nanmean(yayin)
            syayin = np.nanstd(yayin)
            ybinned.append(myayin)
            eybinned.append(syayin)
            nybinned.append(nyayin)

        return xbinned, ybinned, eybinned, nybinned



    def plot_binned(labsxax, labsyay, arrayxax1, arrayyay1, pdf_name):

        if save == True:
            pdf4 = fpdf.PdfPages(pdf_name)  # type: PdfPages
        else:
            pdf4 = fpdf.PdfPages("blank")
        print("Starting loop to create figures of all galaxies together - binned")

        arrayxax = arrayxax1
        arrayyay = arrayyay1

        for k in range(len(arrayxax)):
            #    print "starting for k"
            sns.set(style='white', color_codes=False)
            fig, axs = plt.subplots(4, 2, sharex='col', figsize=(8, 10), gridspec_kw={'hspace': 0})
            plt.subplots_adjust(wspace=0.3)
            fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s%s' %(name_end, typegmc), fontsize=18, va='top')
            axs = axs.ravel()
            #    print "starting for i"
            # Galactic distance vs: Mco, avir, sigmav,Sigmamol
            for i in range(len(labsyay)):
                yaytmp = arrayyay[i]
                xaxtmp = arrayxax[k]
                xaxall = np.concatenate([f.tolist() for f in xaxtmp])
                yayall = np.concatenate([f.tolist() for f in yaytmp])
                yayall = np.log10(yayall)
                if "Metallicity" in labsxax[k]:
                    xaxall = xaxall
                else:
                    xaxall = np.log10(xaxall)
                idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
                xaxall = xaxall[idok];
                yayall = yayall[idok]
                lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
                lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
                indlim = np.where((xaxall < lim2) & (xaxall > lim1))
                xaxall = xaxall[indlim];
                yayall = yayall[indlim]
                mybin = 20
                if xaxall.any() != 0 and yayall.any() != 0 and mybin != 0:
                    xbinned, ybinned, eybinned, nybinned = bindata(xaxall, yayall, mybin)
                    # if there is any nan inside
                    ido = np.where(np.array(nybinned) != 0)
                    xbinned = xbinned[ido]
                    ybinned = [g for g in np.array(ybinned)[ido]]
                    eybinned = [g for g in np.array(eybinned)[ido]]
                    nybinned = [g for g in np.array(nybinned)[ido]]
                    # Plot binned data
                    mysize = np.array(nybinned).astype(float)
                    mysize = (mysize - np.min(mysize)) / (np.max(mysize) - np.min(mysize)) * 9 + 3
                    mylims = [np.argmin(mysize), np.argmax(mysize)]
                    mylabs = ["Num of pairs: %s" % min(nybinned), "Num of pairs: %s" % max(nybinned)]
                    for j in range(len(xbinned)):
                        if j == np.argmin(mysize) or j == np.argmax(mysize):
                            axs[i].plot(xbinned[j], ybinned[j], linestyle="None", alpha=0.5, marker="o",
                                        markersize=mysize[j],
                                        color="red", label="Num of pairs: %s" % nybinned[j])
                        else:
                            axs[i].plot(xbinned[j], ybinned[j], linestyle="None", alpha=0.5, marker="o",
                                        markersize=mysize[j],
                                        color="red")
                    axs[i].errorbar(xbinned, ybinned, eybinned, capsize=5)
                    axs[i].set(ylabel=labsyay[i])
                    axs[i].grid()
                    # Computing the linear fit to the data, using the amount of
                    x = xbinned.reshape((-1, 1))
                    y = ybinned
                    model = LinearRegression().fit(x, y, nybinned)
                    r_sq = model.score(x, y)
                    slope = model.coef_
                    y_pred = model.intercept_ + model.coef_ * x.ravel()
                    axs[i].plot(xbinned, y_pred, '-')
                    # sn.regplot(x=xaxall, y=yayall, ax=axs[i])
                    if i == 0:
                        xmin = np.amin(xbinned)
                        xmax = np.amax(xbinned)
                        xprang = (xmax - xmin) * 0.03
                        x0 = xmin + xprang
                    #        y0, yf = axs[i].get_ylim()
                    #        my0 = y0-(yf-y0)*0.13
                    # new!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    x0, xf = xlim[k]
                    y0, yf = ylim[i]
                    xprang = xf - x0
                    yprang = yf - y0
                    my0 = y0 - (yf - y0) * 0.13
                    # axs[i].set(xlim=(x0,xf))
                    axs[i].set(ylim=(y0, yf))
                    axs[i].text(0.8, 0.1, 'R^2: %6.2f' % (r_sq), fontsize=8, horizontalalignment='center',
                                verticalalignment='center', transform=axs[i].transAxes)

                    axs[i].text(0.15, 0.1, 'Slope %5.2f' % (slope), fontsize=8, horizontalalignment='center',
                                verticalalignment='center', transform=axs[i].transAxes)

            #        axs[i].set(ylim=(y0-(yf-y0)*0.15,yf+(yf-y0)*0.15))
            #        axs[i].set(xlim=(xmin - xprang*3, xmax + xprang*3))
            axs[0].legend(prop={'size': 9})
            axs[6].set(xlabel=labsxax[k])
            axs[7].set(xlabel=labsxax[k])
            save_pdf(pdf4, fig)
        if save == True:
            pdf4.close()

    def name(matching, without_out, new_muse, gmc_catalog):
        name_append = ['', 'with_outliers', 'without_outliers', 'new_muse_', 'old_muse_',
                       str(threshold_perc)]

        if new_muse == True:
            name_end = name_append[3] + gmc_catalog + matching
            if matching != "distance":
                name_end = name_end + name_append[5]
                if without_out == True:
                    name_end = name_end + name_append[2]
                else:
                    name_end = name_end + name_append[1]

        else:
            name_end = name_append[4] + gmc_catalog + matching
            if matching != "distance":
                name_end = name_end + name_append[5]
                if without_out == True:
                    name_end = name_end + name_append[2]
                else:
                    name_end = name_end + name_append[1]
        return name_end

    # ==============================================================================#
    typegmc = gmc_catalog#'_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_

    without_out = not outliers
    name_end = name(matching, without_out, new_muse, gmc_catalog)
    # ==============================================================================#

    without_out1 = not outliers2
    new_muse1 = new_muse2
    name_end1 = name(matching2, without_out1, new_muse1, gmc_catalog2)
    typegmc1 = gmc_catalog2


    namegmc = "_12m+7m+tp_co21%sprops" % typegmc
    namegmc1 = "_12m+7m+tp_co21%sprops" % typegmc1
    print(namegmc1)
    print(name_end1)

    # ====================================================================================================================#

    dirhii, dirgmc, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots, dirhiimasks = pickle.load(
        open('Directories_muse.pickle', "rb"))  # retrieving the directories paths
    dirmuseproperties = os.path.dirname(os.path.realpath("Extract_info_plot_per_gal_muse.py")) + "/"

    # =========================Getting all the GMC and HII properties from the pickle files===============================#

    galaxias, GMCprop1, HIIprop1, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay = pickle.load(
        open('%sGalaxies_variables_GMC%s%s.pickle' % (dirmuseproperties, namegmc, name_end),
             "rb"))  # retrieving the regions properties

    SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, \
    velHIIover, HIIminorover, HIImajorover, HIIangleover = HIIprop1

    HIIprop = SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover

    DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, \
    tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover = GMCprop1

    GMCprop = DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover

    SizepcHII, LumHacorr, sigmavHII, metaliHII, varmetHII, numGMConHII, \
    MasscoGMC, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC = pickle.load(
        open('%sGalaxies_variables_notover_GMC%s%s.pickle' % (dirmuseproperties, namegmc, name_end), "rb"))

    shortlab = ['HIIGMCdist', 'Mco', 'GMCsize', 'Smol', 'sigmav', 'avir', 'TpeakCO', 'tauff']
    MassesCO = [1e5 * i for i in MasscoGMCover]  #

    labsyay = labsyay[0:len(labsyay) - 5]  # removing  vel, major axis, minor axis and PA, no need to plot them
    labsxax = labsxax[0:len(labsxax) - 4]

    arrayyay = GMCprop
    arrayxax = HIIprop

    # ======================================other file if need to compare two sets of data==========================================================================================#
    if plot_double_lr == True:
        galaxias_no, GMCprop1_no, HIIprop1_no, RAgmc_no, DECgmc_no, RAhii_no, DEChii_no, labsxax_no, labsyay_no = pickle.load(
            open('%sGalaxies_variables_GMC%s%s.pickle' % (dirmuseproperties, namegmc1, name_end1), "rb"))

        SizepcHIIover_no, LumHacorrover_no, sigmavHIIover_no, ratlin_no, metaliHIIover_no, varmetHIIover_no, \
        velHIIover_no, HIIminorover_no, HIImajorover_no, HIIangleover_no = HIIprop1_no

        HIIprop_no = SizepcHIIover_no, LumHacorrover_no, sigmavHIIover_no, ratlin_no, metaliHIIover_no, varmetHIIover_no

        DisHIIGMCover_no, MasscoGMCover_no, SizepcGMCover_no, Sigmamoleover_no, sigmavGMCover_no, aviriaGMCover_no, TpeakGMCover_no, \
        tauffGMCover_no, velGMCover_no, angleGMCover_no, majorGMCover_no, minorGMCover_no, regionindexGMCover_no = GMCprop1_no

        GMCprop_no = DisHIIGMCover_no, MasscoGMCover_no, SizepcGMCover_no, Sigmamoleover_no, sigmavGMCover_no, aviriaGMCover_no, TpeakGMCover_no, tauffGMCover_no

        SizepcHII_no, LumHacorr_no, sigmavHII_no, metaliHII_no, varmetHII_no, numGMConHII_no, \
        MasscoGMC_no, HIIminor_no, HIImajor_no, HIIangle_no, angleGMC_no, majorGMC_no, minorGMC_no = pickle.load(
            open('%sGalaxies_variables_notover_GMC%s%s.pickle' % (dirmuseproperties, namegmc1, name_end1), "rb"))

        shortlab_no = ['HIIGMCdist', 'Mco', 'GMCsize', 'Smol', 'sigmav', 'avir', 'TpeakCO', 'tauff']
        MassesCO_no = [1e5 * i for i in MasscoGMCover_no]  #

        labsyay_no = labsyay_no[
                     0:len(labsyay_no) - 5]  # removing  vel, major axis, minor axis and PA, no need to plot them
        labsxax_no = labsxax_no[0:len(labsxax_no) - 4]

        arrayyay_no = GMCprop_no
        arrayxax_no = HIIprop_no

    # =======================================================================================================================================#

    count = []

    for i in range(np.shape(arrayxax)[1]):
        count.append(len(arrayxax[1][i]))
    print("total number of HII regions that are matched : %i   (%s)"%(np.sum(count), name_end))  # total number of hii regions

    # =====================================================================================================================================#

    # Limits in the properties of HIIR and GMCs
    xlim, ylim, xx, yy = pickle.load(open('limits_properties.pickle', "rb"))  # comment this line if the file limit has not been run yet

    # =====================================================================================================================================#

    # ==================================diff regions=============================================================#

    if cent == True :
        regions_id = (1, 2, 3)
        pdf_name = "%sCorrelations_allgals_GMC_binned_int%s%s.pdf" % (dirplots, namegmc, name_end)
        plot_binned_regions(labsxax,labsyay,arrayxax,arrayyay,pdf_name, regions_id)

    if arm == True:
        regions_id = (5, 6)
        pdf_name = "%sCorrelations_allgals_GMC_binned_arm%s%s.pdf" % (dirplots, namegmc, name_end)
        plot_binned_regions(labsxax,labsyay,arrayxax,arrayyay,pdf_name, regions_id)

    if out == True:
        regions_id = (7, 8, 9)
        pdf_name = "%sCorrelations_allgals_GMC_binned_out%s%s.pdf" % (dirplots, namegmc, name_end)
        plot_binned_regions(labsxax,labsyay,arrayxax,arrayyay,pdf_name, regions_id)

    if type(region) != type(None):
        region = list(region)
        regions_id = region
        print(type(regions_id))
        pdf_name = "%sCorrelations_allgals_GMC_binned_regions%s%s.pdf" % (dirplots, namegmc, name_end)
        plot_binned_regions(labsxax, labsyay, arrayxax, arrayyay, pdf_name, regions_id)

    if plot_lr == True:
        pdf_name = "%sCorrelations_allgals_GMC_binned%s%s.pdf" % (dirplots, namegmc, name_end)
        plot_binned(labsxax, labsyay, arrayxax, arrayyay, pdf_name)

    if plot_double_lr == True:
        pdf_name = "%sCorrelations_allgals_GMC_binned%s%s%s.pdf" % (dirplots, namegmc, name_end, name_end1)
        plot_double_binned(labsxax, labsyay, arrayxax, arrayxax_no, arrayyay, arrayyay_no, pdf_name)



i = 5
threshold_perc = (i+1) * 0.1
#binned_linear_regression(new_muse = True, gmc_catalog = "_native_", matching = "overlap1o1", outliers = True, show = True, save =False, threshold_perc = 0.5, new_muse2= True, matching2= "overlap_1om", outliers2= True, threshold_perc2 = 0.5, plot_double_lr= True, gmc_catalog2 = "_native_")
binned_linear_regression(new_muse = True, gmc_catalog = "_native_", matching = "overlap_1o1", outliers = True, show = True, save = False, threshold_perc = threshold_perc, plot_lr= True, new_muse2= True, matching2= "overlap_1om", outliers2= True, threshold_perc2 = 0.5, gmc_catalog2 = "_native_")


#0.6000000000000001
#0.30000000000000004

#new_muse2 = kwargs.get('new_muse2', None)
#overlap_matching2 = kwargs.get('overlap_matching2', None)
#outliers2 = kwargs.get('outliers2', None)
#arm = kwargs.get('arm', None)
#out = kwargs.get('out', None)
#cent = kwargs.get('cent', None)
#region = kwargs.get('region', None)
#plot_lr = kwargs.get('plot_lr', None)
