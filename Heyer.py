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

plt.style.use('science')


#===================================================================================


dir_script_data = os.getcwd() + "/script_data_dr2/"
dirhii_dr1, dirhii_dr2, dirgmc_old, dirgmc_new, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots, dirhiimasks, dirgmcmasks, dir_sample_table = pickle.load(
    open(dir_script_data + 'Directories_muse.pickle', "rb"))  # retrieving the directories paths



def Heyer(muse, gmc_catalog, gmc_catalog_version, matching, outliers, symmetrical, randomize, show, save, threshold_percs, vel_limit,  *args, **kwargs):

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

    def get_data(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, threshold_perc, vel,
                 symmetrical):
        # ==============================================================================#

        typegmc = gmc_catalog  # '_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_

        without_out = not outliers
        name_end = name(without_out=without_out, matching=matching, muse=muse, gmc_catalog=gmc_catalog,
                        gmc_catalog_version=gmc_catalog_version, threshold_perc=threshold_perc, vel_limit=vel,
                        randomize=randomize, symmetrical=symmetrical)
        # ==============================================================================#

        namegmc = "_12m+7m+tp_co21%sprops" % typegmc

        # ====================================================================================================================#

        # =========================Getting all the GMC and HII properties from the pickle files===============================#

        galaxias, GMCprop1, HIIprop1, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay, idoverhii, idovergmc = pickle.load(
            open(dir_script_data + 'Galaxies_variables_GMC%s%s.pickle' % (namegmc, name_end),
                 "rb"))  # retrieving the regions properties

        SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, \
        velHIIover, HIIminorover, HIImajorover, HIIangleover, Rgal_hii = HIIprop1

        HIIprop = SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, Rgal_hii

        DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, \
        tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover, FluxCOGMCover, Rgal_gmc = GMCprop1

        GMCprop = DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover, Rgal_gmc

        SizepcHII, LumHacorrnot, sigmavHII, metaliHII, varmetHII, numGMConHII, \
        FluxCOGMCnot, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, SigmaMol, Sigmav, COTpeak, a, b = pickle.load(
            open(dir_script_data + 'Galaxies_variables_notover_GMC%s%s.pickle' % (namegmc, name_end), "rb"))

        shortlab = ['HIIGMCdist', 'Mco', 'GMCsize', 'Smol', 'sigmav', 'avir', 'TpeakCO', 'tauff']
        MassesCO = [1e5 * i for i in MasscoGMCover]  #

        labsyay = labsyay  # removing  vel, major axis, minor axis and PA, no need to plot them
        labsxax = labsxax

        arrayyay = GMCprop
        arrayxax = HIIprop

        # Limits in the properties of HIIR and GMCs
        xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

        return labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak

    pdf6 = fpdf.PdfPages("%sCorrelations_Heyer_allgals_GMC.pdf" % (dirplots))  # type: PdfPages


    for threshold_perc in threshold_percs:
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, sigmav_not_over, COTpeak = get_data(
            matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
            outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel_limit, symmetrical=symmetrical)

        # Heyerr relationship

        SigmaMol = arrayyay[3]
        Sigmav = arrayxax[2]
        SizepcGMCover = arrayyay[2]

        # ========

        # ========

        # SigmaMol = np.concatenate([f.tolist() for f in SigmaMol])
        # Sigmav = np.concatenate([f.tolist() for f in Sigmav])
        # SizepcGMCover = np.concatenate([f.tolist() for f in SizepcGMCover])

        rootR = [np.sqrt(f) for f in SizepcGMCover]
        #rootR = [np.sqrt(np.sqrt(np.pi*a*b)) for a,b in zip(majorGMCover,minorGMCover)]
        arrayyay = np.divide(Sigmav, rootR)
        arrayxax = SigmaMol  #

        labsyay = 'log($\sigma_v$/R$^{0.5}$) [km/s pc$^{-1/2}$]'
        labsxax = 'log($\Sigma_{mol}$[M$_{\odot}$/pc$^2$])'


        fig, axs = plt.subplots(1, 1, figsize=(5, 4))
        #fig.suptitle('All galaxies - Overlapping HIIregions and GMCs', va='top')
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
            axs.plot(xax, yay, '8',  alpha=0.7, markersize=1)

        axs.set(ylabel=labsyay)
        # axs.set_yscale('log')
        # axs.set_xscale('log')

        ybc = np.log10(
            math.sqrt(math.pi * ct.G.cgs.value / 5 * ct.M_sun.cgs.value / ct.pc.cgs.value * 10 ** -10)) + 0.5 * xaxall
        #axs.plot(xaxall, ybc)
        #axs.plot(xaxall,yayall)

        xmin = np.amin(xaxall)
        xmax = np.amax(xaxall)
        xprang = (xmax - xmin) * 0.1
        x = xaxall.reshape((-1, 1))
        y = yayall
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        slope = model.coef_
        y_pred = model.intercept_ + model.coef_ * x.ravel()

        axs.plot(xaxall, y_pred, '-', color = 'black', label = 'Threshold = %5.0f \%%'%(threshold_perc*100))
        x0 = xmin + xprang


        axs.text(0.7, 0.1, 'r²: %6.2f' % (r_sq), horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        axs.text(0.7, 0.05, 'Slope %5.2f' % (slope), horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        axs.set(xlim = (0.8,3.7))
        axs.set(ylim = (-1,2))

        axs.legend(prop={'size': 10})
        axs.set(xlabel=labsxax)
        #plt.show()

        pdf6.savefig(fig)

    pdf6.close()

i = 3
threshold_percs = [0.1,0.5,0.9]
Heyer(muse = 'dr2', gmc_catalog = "_native_",gmc_catalog_version='new', matching = "overlap_1om", outliers = True, show = True, save = False, threshold_percs = threshold_percs, symmetrical='', randomize = '', vel_limit = 10000)
