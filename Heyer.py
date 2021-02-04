import sys
import numpy as np
import numpy as np
import math
import pickle
from astropy import constants as ct
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf
from sklearn.linear_model import LinearRegression
import seaborn as sns
import os
import warnings

np.set_printoptions(threshold=sys.maxsize)
warnings. filterwarnings("ignore")
sns.set(style="white", color_codes=True)

#===================================================================================




def Heyer(new_muse, gmc_catalog, matching, outliers, show, save, threshold_perc,  *args, **kwargs):

    def checknaninf(v1, v2, lim1, lim2):
        v1n = np.array(v1)
        v2n = np.array(v2)
        indok = np.where((np.absolute(v1n) < lim1) & (np.absolute(v2n) < lim2))[0].tolist()
        # print indok
        nv1n = v1n[indok].tolist()
        nv2n = v2n[indok].tolist()
        return nv1n, nv2n

    def bindata(xaxall, yayall, mybin):
        xran = np.amax(xaxall) - np.amin(xaxall)
        xspa = xran / mybin
        xsta = np.amin(xaxall) + xspa / 2
        xfin = np.amax(xaxall) - xspa / 2
        xbinned = np.linspace(xsta, xfin, mybin)
        ybinned = []
        eybinned = []
        nybinned = []
        for t in range(mybin):
            idxs = np.where(abs(xaxall - xbinned[t]) < xspa / 2)
            yayin = yayall[idxs]
            nyayin = len(yayin)
            myayin = np.nanmean(yayin)
            syayin = np.nanstd(yayin)
            ybinned.append(myayin)
            eybinned.append(syayin)
            nybinned.append(nyayin)
        return xbinned, ybinned, eybinned, nybinned

    def save_pdf(pdf, fig):
        if save == True:
            pdf.savefig(fig)
        if show == True:
            plt.show()
        else:
            plt.close()

    def name(matching, without_out, new_muse, gmc_catalog):
        name_append = ['', 'with_outliers', 'without_outliers', 'new_muse_', 'old_muse_', str(threshold_perc)]

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
    typegmc = gmc_catalog  # '_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_
    without_out = not outliers
    name_end = name(matching, without_out, new_muse, gmc_catalog)
    # ==============================================================================#



    namegmc = "_12m+7m+tp_co21%sprops" % typegmc


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

    # Heyerr relationship
    rootR = [np.sqrt(f) for f in SizepcGMCover]
    #rootR = [np.sqrt(np.sqrt(np.pi*a*b)) for a,b in zip(majorGMCover,minorGMCover)]
    arrayyay = np.divide(sigmavGMCover, rootR)
    arrayxax = Sigmamoleover  #

    labsyay = r'log($\sigma_v$/R$^{0.5}$) [km/s pc$^{-1/2}$]'
    labsxax = r'log($\Sigma_{mol}$[M$_{\odot}$/pc$^2$])'  #

    pdf6 = fpdf.PdfPages("%sCorrelations_Heyer_allgals_GMC_%s%s.pdf" % (dirplots, typegmc, name_end))  # type: PdfPages

    sns.set(style='white', color_codes=True)
    fig, axs = plt.subplots(1, 1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
    plt.subplots_adjust(wspace=0.3)
    fig.suptitle('All galaxies - Overlapping HIIregions and GMCs', fontsize=18, va='top')
    yaytmp = arrayyay
    xaxtmp = arrayxax
    xaxall = np.concatenate([f.tolist() for f in xaxtmp])
    yayall = np.concatenate([f.tolist() for f in yaytmp])
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

    for j in range(len(galaxias)):
        xax2 = [h for h in arrayxax[j]]
        yay2 = [h for h in arrayyay[j]]
        xax = np.log10(xax2)
        yay = np.log10(yay2)
        axs.plot(xax, yay, '8', label='%s' % galaxias[j], alpha=0.7, markersize=5)

    axs.set(ylabel=labsyay)
    # axs.set_yscale('log')
    # axs.set_xscale('log')
    axs.grid()

    ybc = np.log10(
        math.sqrt(math.pi * ct.G.cgs.value / 5 * ct.M_sun.cgs.value / ct.pc.cgs.value * 10 ** -10)) + 0.5 * xaxall
    axs.plot(xaxall, ybc)

    xmin = np.amin(xaxall)
    xmax = np.amax(xaxall)
    xprang = (xmax - xmin) * 0.1
    x = xaxall.reshape((-1, 1))
    y = yayall
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    slope = model.coef_
    y_pred = model.intercept_ + model.coef_ * x.ravel()

    axs.plot(xaxall, y_pred, '-')
    x0 = xmin + xprang
    x0, xf = axs.get_xlim()
    y0, yf = axs.get_ylim()

    axs.text(0.15, 0.1, 'R^2: %6.2f' % (r_sq), fontsize=8, horizontalalignment='center',
                verticalalignment='center', transform=axs.transAxes)

    axs.text(0.15, 0.05, 'Slope %5.2f' % (slope), fontsize=8, horizontalalignment='center',
                verticalalignment='center', transform=axs.transAxes)


    axs.set(xlim=(x0, xf))
    axs.set(ylim=(y0, yf))
    axs.legend(prop={'size': 14})
    axs.set(xlabel=labsxax)
    pdf6.savefig(fig)
    plt.show()

    pdf6.close()

i = 3
threshold_perc = (i+1)*0.1
Heyer(new_muse = True, gmc_catalog = "_native_", matching = "overlap_1o1", outliers = True, show = True, save = False, threshold_perc = threshold_perc)
