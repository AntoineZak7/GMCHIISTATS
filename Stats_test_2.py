import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf
from sklearn.linear_model import LinearRegression
import seaborn as sns
import os
import warnings
import statistics
import scipy.stats
import statsmodels.stats.stattools
from astropy.table import Table
warnings. filterwarnings("ignore")
from matplotlib import rc
#import latex
import matplotlib
#rc('text', usetex = True)


np.set_printoptions(threshold=sys.maxsize)
sns.set(style="white", color_codes=False)
# ===================================================================================

dir_script_data = os.getcwd() + "/script_data_dr2/"
dirhii_dr1, dirhii_dr2, dirgmc_old, dirgmc_new, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots, dirhiimasks, dirgmcmasks, dir_sample_table = pickle.load(
    open(dir_script_data + 'Directories_muse.pickle', "rb"))  # retrieving the directories paths


def linear_regression(x, y):
    n = len(x)
    xm = np.mean(x)
    ym = np.mean(y)

    cov = (1/n)*sum([xi*yi for xi, yi in zip(x,y)]) - xm*ym
    vx = (1/n)*sum([xi**2 for xi in x]) - xm**2
    vy = (1/n)*sum([yi**2 for yi in y]) - ym**2
    r = cov/(np.sqrt(vx*vy))

    slope = cov/vx
    b = ym - slope*xm
    r_sq = r**2

    return slope,b,r_sq

def extract_ind(list1,  value):
    ind_val = [[idx,item] for [idx, item] in enumerate(list1) if item == value]
    indexes = [item[0] for item in ind_val]
    return indexes

def make_list(a):
    if not isinstance(a, list):
        a = [a]
    return a

def do_list_indexes(list1, indexes):
    #list1 = make_list(list1)
    list1 = [[idx,item] for [idx, item] in enumerate(list1) if idx in indexes]
    list1 = [item[1] for item in list1]

    return list1

def dup_lists(arrayx, arrayy, ha, idoverhiis):
    duplicates = len([item for item in idoverhiis if item == ha])
    indexes = extract_ind(idoverhiis, ha)
    xvalues = do_list_indexes(arrayx, indexes)
    yvalues = do_list_indexes(arrayy, indexes)
    return duplicates, xvalues, yvalues

def get_min(arrayxax):
    xaxtmp = arrayxax[1]
    yayall = np.concatenate([f.tolist() for f in xaxtmp])
    yayall = np.log10(yayall)
    xlimmin=(np.nanmin(yayall) - 0.2)
    xlimmax=(np.nanmax(yayall) + 0.2)

    return xlimmin, xlimmax

def clean_data(xarray, yarray):
    xaxall = xarray
    yayall = yarray


    idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
    xaxall = xaxall[idok]
    yayall = yayall[idok]
    lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
    lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
    indlim = np.where((xaxall < lim2) & (xaxall > lim1))
    xaxall = xaxall[indlim]
    yayall = yayall[indlim]

    return xaxall, yayall

def checknaninf(v1, v2, lim1, lim2):
    v1n = np.array(v1)
    v2n = np.array(v2)
    indok = np.where((np.absolute(v1n) < lim1) & (np.absolute(v2n) < lim2))[0].tolist()
    # print indok
    nv1n = v1n[indok].tolist()
    nv2n = v2n[indok].tolist()
    return nv1n, nv2n

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

def bindata(xaxall, yayall, mybin):
    xran = np.amax(xaxall) - np.amin(xaxall)
    xspa = xran / mybin
    xsta = np.amin(xaxall) + xspa / 2
    xfin = np.amax(xaxall) - xspa / 2
    xbinned = np.linspace(xsta, xfin, mybin)
    ybinned = []
    eybinned = []
    nybinned = []
    stdbin = []
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

def name(matching, without_out, muse, gmc_catalog, gmc_catalog_version, threshold_perc, vel_limit, randomize):
    name_end = 'muse:' + muse + '_' + 'gmc:' + gmc_catalog + '(' + gmc_catalog_version + ')_' + 'vel_limit:' + str(
        vel_limit) + '_matching:' + matching + '_' + randomize + '_'

    if matching != "distance":
        name_end = name_end + '(' + str(threshold_perc) + ')'
        if without_out == True:
            name_end = name_end + '_' + 'without_outliers'
        else:
            name_end = name_end + '_' + 'with_outliers'

    return name_end

def conf_intervals(x, y, alpha):
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
    a = slope

    Sxy = sum(np.array([xi - xm for xi in x ])*np.array([yi - ym for yi in y]))
    Sxx = sum([(xi - xm)**2 for xi in x])
    Syy = sum([(yi - ym)**2 for yi in y])
    Err = y - (x*slope + b)
    Res = Err
    SSe = sum([err**2 for err in Err])
    SSt = Syy
    MSt = SSt/(n-2)
    SSr = sum([(xi*slope + b - ym)**2 for xi in x])
    sigma_sq = SSe / (n-2)
    MSe = sigma_sq
    MSr = SSr

    Syx2 = SSe/(n-2)

    max = np.nanmax(x)
    min =  np.nanmin(x)
    step = (max-min)/(n)
    #x0 = np.arange(start =min, stop = max, step = step)
    x0 = np.sort(x)

    #========intervalles confiance a et b========#
    k= n-2
    ta = scipy.stats.t.ppf(1-alpha/2, df = k)



    Sm2 = Syx2/Sxx # Variance of slope
    Sm = np.sqrt(Sm2) # Standard deviation of slope

    Sb2 = Syx2*(1/n+(xm**2)/Sxx)
    Sb = np.sqrt(Sb2)
    print('conf intervals')

    #conf_b = ta*np.sqrt(MSe*(1/n + (xm**2)/Sxx)) #scalar
    #conf_a = ta*np.sqrt(MSe/Sxx) #scalar, conf interval of slope
    conf_a = Sm*ta
    conf_b = Sb*ta


    #=====intervalle confiance droite régression=====#

    y_conf_sup = ta * np.sqrt(MSe*(1/n+ np.array([(x - xm)**2 for x in x0])/Sxx))
    y_conf_inf = -ta * np.sqrt(MSe*(1/n+ [(x - xm)**2 for x in x0]/Sxx))

    #========intervalles prévision de y en x0========#

    int_prev_y = ta*np.sqrt(MSe*(1+1/n+ [(x - xm)**2 for x in x0]/Sxx   )) #vector

    #=======Tests H1================#

    F0 = MSr/MSe
    fa = scipy.stats.f(1,k)
    dw = statsmodels.stats.stattools.durbin_watson(Res)
    rms_err = np.std(np.array(Err))
    rms_tot = np.std(y)

    #=======test chi2==============#

    mybin = 20
    xbin, ybin, ebin, nbin = bindata(x,y,mybin= mybin)
    y_prev = b + slope*x
    xbin_prev, ybin_prev,ebin_prev,nbin_prev = bindata(x,y_prev, mybin = mybin)

    bin_err = ybin-(xbin*slope + b)
    bin_err = np.array(bin_err)
    xbin = np.array(xbin)
    ybin = np.array(ybin)
    ebin = np.array(ebin)
    nbin = np.array(nbin)

    bin_err = bin_err[np.where(nbin !=0)[0]]
    xbin = xbin[np.where(nbin !=0)[0]]
    ybin = ybin[np.where(nbin !=0)[0]]
    ebin = ebin[np.where(nbin !=0)[0]]

    bin_err = bin_err[np.where(ebin !=0)[0]]
    xbin = xbin[np.where(ebin !=0)[0]]
    ybin = ybin[np.where(ebin !=0)[0]]
    ebin = ebin[np.where(ebin !=0)[0]]

    chi2list = np.nan_to_num(np.array([(bin_erri**2)/sigmai**2 for bin_erri, sigmai in zip(bin_err,ebin) ]))
    chi2 = sum(chi2list)
    #print(1-scipy.stats.chi2.cdf(chi2,k)) # seems ok
    nbin = np.array(nbin)
    nbin_prev = np.array(nbin_prev)
    test = np.bincount(np.array(x*100).astype(int))

    slope1, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(x,y)
    mean_error = np.nanmean(abs(Err))



    return conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr,rms_tot,rms_err,dw, mean_error, SSt, SSe, MSe, MSt

def num_gmcs(idovergmc):
    ids = idovergmc
    n_gmcs = 0
    for i in range(len(ids)):

        if not isinstance(ids[0], list):
            ids = [id.tolist() for id in ids]
        id = [ids[i].count(x) for x in ids[i]]
        n_gmcs += len(id) - sum([x-1 for x in id])

    n_gmcs = int(n_gmcs)
    return n_gmcs

def num_hiis(idoverhii):
    n_hiis = 0
    for i in range(len(idoverhii)):

        if not isinstance(idoverhii[0], list):
            idoverhii = [id.tolist() for id in idoverhii]
        id = [idoverhii[i].count(x) for x in idoverhii[i]]
        n_hiis += len(id)

    n_hiis = int(n_hiis)
    return n_hiis

def prepdata(datax, datay):
    # ========
    xaxtmp = datax
    yaytmp = datay
    # ========

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

    return xaxall, yayall







def plot_correlations(muse, gmc_catalog, gmc_catalog_version, randomize, matching, outliers,  show, save, threshold_percs, vel, gmcprop, rgal_color):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    print(xlim)
    alpha = 0.05

    # Plots of correlations with dots for each pair
    threshold_perc  = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(matching=matching, muse = muse, gmc_catalog= gmc_catalog, gmc_catalog_version=gmc_catalog_version,outliers=outliers,randomize=randomize,threshold_perc=threshold_perc,vel=vel)


    print("Plots of all galaxies together")


    #xlimmin, xlimmax = get_min(arrayxax)
    xlimmin, xlimmax = 35.416676089337685 ,41.356351026116506
    z=0
    print("Starting loop to create figures of all galaxies together - points")

    pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s%s.pdf" % (dirplots, namegmc, name_end, str(z))

    pdf3 = fpdf.PdfPages(pdf_name)
    for threshold_perc in threshold_percs:

        z+=1

        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover,HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak= get_data(
            matching=matching, muse = muse, gmc_catalog= gmc_catalog, gmc_catalog_version=gmc_catalog_version,outliers=outliers,randomize=randomize,threshold_perc=threshold_perc,vel=vel)


        rgal_gmc = arrayyay[len(arrayyay)-1]
        rgal_hii = arrayxax[len(arrayxax)-1]
        labsyay = labsyay[0:len(labsyay) - 5]  # removing  vel, major axis, minor axis and PA, no need to plot them
        labsxax = labsxax[0:len(labsxax) - 4]

        k=1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(2,2, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        plt.subplots_adjust(hspace=0.0)

        #fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s' % name_end, fontsize=18, va='top')
        axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0

        for i in gmcprop:

            axs[l].set(ylabel=labsyay[i])
            axs[l].grid()
            axs[l].set(xlabel=labsxax[1])
            axs[l].set(xlabel=labsxax[1])


            if threshold_perc == 0.1 :
                color = "black"
            elif i == 3 or i==6:
                color = "tab:red"
            elif i == 1 or i ==4:
                color = "tab:blue"

            #color = 'black'


            # gmcprop
            # 1 : MCO
            # 3 : Sigma mol
            # 4 : Sigma v
            # 5 : alpha vir
            # 6 : CO Tpeak
            # 7 : Tauff

            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]

            #========






            #========


            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            yayall = np.concatenate([f.tolist() for f in yaytmp])

            rgal_all = np.concatenate([f.tolist() for f in rgal_gmc])
            #if k < 5:
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)

            axs[l].set_xlim(xlimmin, xlimmax)

            idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
            xaxall = xaxall[idok]
            yayall = yayall[idok]
            rgal_all = rgal_all[idok]
            lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
            lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
            indlim = np.where((xaxall < lim2) & (xaxall > lim1))
            xaxall = xaxall[indlim]
            yayall = yayall[indlim]
            rgal_all = rgal_all[indlim]


            if rgal_color == False:
                axs[l].plot(xaxall, yayall,'o', markerfacecolor = 'None' , markersize = 2, markeredgecolor = color, label='threshold = %f' % threshold_perc ,linestyle = 'None')
            else:
                for xi in range(len(xaxall)):

                    colormap =  plt.get_cmap('viridis')
                    colour = colormap(1 - 2*rgal_all[xi]/np.nanmax(rgal_all))
                    axs[l].plot(xaxall[xi], yayall[xi], 'o',markeredgecolor = colour, markersize = 2, linestyle = 'None',markerfacecolor = 'None' )


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



                conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr,rms_tot,rms_err,dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(xaxall, yayall, alpha)

                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * x0


                axs[l].plot(x0, y_pred+y_conf_sup, '--', color = 'black', label='confidence interval')
                axs[l].plot(x0, y_pred + y_conf_inf, '--', color = 'black')

                axs[l].plot(x0, y_pred+int_prev_y, '-.', color = 'grey', label = 'prediction interval')
                axs[l].plot(x0, y_pred - int_prev_y, '-.', color = 'grey')

                axs[l].plot(x0, y_pred, color = 'navy', label = 'linear regression')

                r_sq = np.sqrt(r_sq)

                x0, xf = xlim[k]
                y0, yf = ylim[i]
                # axs.text(0.8, 0.15, 'P-Value: %6.2f' % (pvalue), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)


                #=======
                x = xaxall
                y = yayall
                p,V = np.polyfit(x, y, 1, cov=True)




                # axs.text(0.8, 0.07, 'Error RMS: %5.3f' % (rms_err), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                # axs.text(0.8, 0.15, 'Standard error of estimate: %5.3f' % (stderr), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                axs[l].text(0.8, 0.1, 'R²: %5.3f' % ((r_sq)**2), fontsize=10, horizontalalignment='center',
                            verticalalignment='center', transform=axs[l].transAxes)

                # axs.text(0.8, 0.19, 'Durbin-Watson stat: %6.2f' % (dw), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                axs[l].text(0.8, 0.05, 'Slope %5.3f ' % (slope), fontsize=10, horizontalalignment='center',
                            verticalalignment='center', transform=axs[l].transAxes)
                # #
                # axs[l].text(0.8, 0.07, 'Standard deviation %5.3f ' % (np.sqrt(MSe)), fontsize=8, horizontalalignment='center',
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




                axs[l].set(ylim=(y0, yf))
                if l == 0:
                    axs[l].legend(prop={'size': 8}, loc = 2)





                l+=1
        #axs[l].set(xlabel=labsxax[k])
        save_pdf(pdf3, fig, save, show)

    pdf3.close()


def plot_correlations_regions(muse, gmc_catalog, gmc_catalog_version, randomize, matching, outliers,  show, save, threshold_percs, vel, gmcprop, rgal_color, regions):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair
    threshold_perc  = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(matching=matching, muse = muse, gmc_catalog= gmc_catalog, gmc_catalog_version=gmc_catalog_version,outliers=outliers,randomize=randomize,threshold_perc=threshold_perc,vel=vel)


    print("Plots of all galaxies together")


    xlimmin, xlimmax = get_min(arrayxax)
    z=0
    print("Starting loop to create figures of all galaxies together - points")

    pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s%s.pdf" % (dirplots, namegmc, name_end, str(z))

    pdf3 = fpdf.PdfPages(pdf_name)
    for threshold_perc in threshold_percs:

        z+=1

        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover,HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak= get_data(
            matching=matching, muse = muse, gmc_catalog= gmc_catalog, gmc_catalog_version=gmc_catalog_version,outliers=outliers,randomize=randomize,threshold_perc=threshold_perc,vel=vel)


        rgal_gmc = arrayyay[len(arrayyay)-1]
        rgal_hii = arrayxax[len(arrayxax)-1]
        labsyay = labsyay[0:len(labsyay) - 5]  # removing  vel, major axis, minor axis and PA, no need to plot them
        labsxax = labsxax[0:len(labsxax) - 4]

        arrayxax_regions = [arrayxax, arrayxax, arrayxax]
        arrayyay_regions = [arrayyay, arrayyay, arrayyay]
        arrayxax_test = arrayxax

        reg = 0
        for i in range(len(regions)):
            regions_id = regions[i]

            for i in range(np.shape(arrayxax)[0]):
                for j in range(np.shape(arrayxax)[1]):
                    id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
                    arrayxax[i][j] = arrayxax[i][j][id_int]

            for i in range(np.shape(arrayyay)[0]):
                for j in range(np.shape(arrayyay)[1]):
                    id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
                    arrayyay_regions[reg][i][j] = arrayyay[i][j][id_int]

            reg+=1

            # for j in range(len(FluxCOGMCover)):
            #     id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
            #     FluxCOGMCover[j] = FluxCOGMCover[j][id_int]
            #
            # idoverhii = [np.array(x) for x in idoverhii]
            # idovergmc = [np.array(x) for x in idovergmc]
            # for j in range(len(idoverhii)):
            #     id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
            #     idoverhii[j] = idoverhii[j][id_int]
            #
            # for j in range(len(idovergmc)):
            #     id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
            #     idovergmc[j] = idovergmc[j][id_int]

        k=1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(1,1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        plt.subplots_adjust(hspace=0.0)

        l = 0

        for i in gmcprop:

            for reg in len(regions):

                axs.set(ylabel=labsyay[i])
                axs.grid()
                axs.set(xlabel=labsxax[1])
                axs.set(xlabel=labsxax[1])


                if threshold_perc == 0.1 :
                    color = "black"
                elif i == 3 or i==6:
                    color = "tab:red"
                elif i == 1 or i ==4:
                    color = "tab:blue"

                color = "black"


                yaytmp1 = arrayyay[reg][i]
                xaxtmp1 = arrayxax[reg][k]

                xaxall, yayall = prepdata(xaxtmp1, yaytmp1)




                #========

                axs.set_xlim(xlimmin, xlimmax)


                #========




                axs.plot(xaxall, yayall,'o', markerfacecolor = 'None' , markersize = 2, markeredgecolor = color, label='threshold = %f' % threshold_perc ,linestyle = 'None')


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



                    conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr,rms_tot,rms_err,dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(xaxall, yayall, alpha)

                    slope, b, r_sq = linear_regression(xaxall, yayall)
                    y_pred = b + slope * x0


                    axs.plot(x0, y_pred+y_conf_sup, '--', color = 'black', label='confidence interval')
                    axs.plot(x0, y_pred + y_conf_inf, '--', color = 'black')

                    axs.plot(x0, y_pred+int_prev_y, '-.', color = 'grey', label = 'prediction interval')
                    axs.plot(x0, y_pred - int_prev_y, '-.', color = 'grey')

                    axs.plot(x0, y_pred, color = 'navy', label = 'linear regression')

                    r_sq = np.sqrt(r_sq)

                    x0, xf = xlim[k]
                    y0, yf = ylim[i]
                    # axs.text(0.8, 0.15, 'P-Value: %6.2f' % (pvalue), fontsize=8, horizontalalignment='center',
                    #             verticalalignment='center', transform=axs.transAxes)


                    #=======
                    x = xaxall
                    y = yayall
                    p,V = np.polyfit(x, y, 1, cov=True)




                    # axs.text(0.8, 0.07, 'Error RMS: %5.3f' % (rms_err), fontsize=8, horizontalalignment='center',
                    #             verticalalignment='center', transform=axs.transAxes)

                    # axs.text(0.8, 0.15, 'Standard error of estimate: %5.3f' % (stderr), fontsize=8, horizontalalignment='center',
                    #             verticalalignment='center', transform=axs.transAxes)

                    axs.text(0.8, 0.03, 'R²: %5.3f' % ((r_sq)**2), fontsize=8, horizontalalignment='center',
                                verticalalignment='center', transform=axs.transAxes)

                    # axs.text(0.8, 0.19, 'Durbin-Watson stat: %6.2f' % (dw), fontsize=8, horizontalalignment='center',
                    #             verticalalignment='center', transform=axs.transAxes)

                    axs.text(0.2, 0.03, 'Slope %5.3f $\pm$ %5.3f ' % (slope, conf_a), fontsize=8, horizontalalignment='center',
                                verticalalignment='center', transform=axs.transAxes)
                    #
                    axs.text(0.8, 0.07, 'Standard deviation %5.3f ' % (np.sqrt(MSe)), fontsize=8, horizontalalignment='center',
                                verticalalignment='center', transform=axs.transAxes)


                    axs.text(0.8, 0.11, 'paired gmcs %5.3f (%5.2f %%) ' % (n_gmcs, n_gmcs*100/n_gmcs_tot), fontsize=8, horizontalalignment='center',
                             verticalalignment='center', transform=axs.transAxes)

                    axs.text(0.8, 0.15, 'paired hii regions %5.3f (%5.2f %%) ' % (n_hiis, n_hiis*100/n_hiis_tot), fontsize=8, horizontalalignment='center',
                             verticalalignment='center', transform=axs.transAxes)

                    axs.text(0.8, 0.19, 'paired gmcs CO flux: %5.2f %% ' % (FluxCOGMCovertot*100/FluxCOGMCtot),
                             fontsize=8, horizontalalignment='center',
                             verticalalignment='center', transform=axs.transAxes)

                    axs.text(0.8, 0.23, 'paired hii regions Ha Lum: %5.2f %% ' % (LumHacorrovertot*100/LumHacorrtot),
                             fontsize=8, horizontalalignment='center',
                             verticalalignment='center', transform=axs.transAxes)




                    axs.set(ylim=(y0, yf))
                    if l == 0:
                        axs.legend(prop={'size': 8}, loc = 2)





                    l+=1
        #axs[l].set(xlabel=labsxax[k])
        save_pdf(pdf3, fig, save, show)

    pdf3.close()

    # ===============================================================
    # xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    # alpha = 0.05
    #
    # # Plots of correlations with dots for each pair
    #
    # region_name = ['Center','Arms','Disc']
    # threshold_perc = threshold_percs[0]
    #
    # labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(
    #     matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
    #     outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)
    # if save == True:
    #     pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s.pdf" % (dirplots, namegmc, name_end)
    #
    #     pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    # else:
    #     pdf3 = fpdf.PdfPages("blank")
    #
    # print("Plots of all galaxies together")
    #
    # xlimmin, xlimmax = get_min(arrayxax)
    #
    # print("Starting loop to create figures of all galaxies together - points")
    #
    # reg = 0
    #
    #
    # sns.set(style='white', color_codes=True)
    # fig, axs = plt.subplots(1, 1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
    # plt.subplots_adjust(wspace=0.3)
    # # fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s ' % (name_end),
    # #              fontsize=18, va='top')
    # for i in range(len(regions)):
    #
    #     labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(
    #         matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
    #         outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)
    #
    #     regions_id = regions[i]
    #     for i in range(np.shape(arrayxax)[0]):
    #         for j in range(np.shape(arrayxax)[1]):
    #             id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
    #             arrayxax[i][j] = arrayxax[i][j][id_int]
    #
    #     for i in range(np.shape(arrayyay)[0]):
    #         for j in range(np.shape(arrayyay)[1]):
    #             id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
    #             arrayyay[i][j] = arrayyay[i][j][id_int]
    #
    #     for j in range(len(FluxCOGMCover)):
    #         id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
    #         FluxCOGMCover[j] = FluxCOGMCover[j][id_int]
    #
    #     idoverhii = [np.array(x) for x in idoverhii]
    #     idovergmc = [np.array(x) for x in idovergmc]
    #     for j in range(len(idoverhii)):
    #         id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
    #         idoverhii[j] = idoverhii[j][id_int]
    #
    #     for j in range(len(idovergmc)):
    #         id_int = np.where(np.isin(np.array(regionindexGMCover[j]), (regions_id)))
    #         idovergmc[j] = idovergmc[j][id_int]
    #
    #
    #     k=1
    #
    #     #axs = axs.ravel()
    #     # Galactic distance vs: Mco, avir, sigmav,Sigmamol
    #     l = 0
    #     for i in gmcprop:
    #
    #         axs.set(ylabel=labsyay[i])
    #         axs.grid()
    #         yaytmp = arrayyay[i]
    #         xaxtmp = arrayxax[k]
    #         xaxall = np.concatenate([f.tolist() for f in xaxtmp])
    #         yayall = np.concatenate([f.tolist() for f in yaytmp])
    #         #if k < 5:
    #         xaxall = np.log10(xaxall)
    #         yayall = np.log10(yayall)
    #
    #         colors = ['blue','red','red']
    #         color = colors[reg]
    #
    #         axs.plot(xaxall, yayall,'o', markerfacecolor = 'None' , markersize = 2, markeredgecolor = color, label='threshold = %f' % threshold_perc ,linestyle = 'None')
    #
    #         axs.set_xlim(xlimmin, xlimmax)
    #
    #         idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
    #         xaxall = xaxall[idok]
    #         yayall = yayall[idok]
    #         lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
    #         lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
    #         indlim = np.where((xaxall < lim2) & (xaxall > lim1))
    #         xaxall = xaxall[indlim]
    #         yayall = yayall[indlim]
    #         if xaxall.any() != 0 and yayall.any != ():
    #             xmin = np.amin(xaxall)
    #             xmax = np.amax(xaxall)
    #             xprang = (xmax - xmin) * 0.1
    #             x = xaxall.reshape((-1, 1))
    #             y = yayall
    #
    #             LumHacorrover = arrayxax[1]
    #             LumHacorrovertot = sum([x for sublist in LumHacorrover for x in sublist])
    #             FluxCOGMCovertot = sum([x for sublist in FluxCOGMCover for x in sublist])
    #
    #             FluxCOGMCtot = sum([x for sublist in FluxCOGMCnot for x in sublist])
    #             LumHacorrtot = sum([x for sublist in LumHacorrnot for x in sublist])
    #
    #
    #
    #             n_gmcs_tot = sum([len(x) for x in FluxCOGMCnot])
    #             n_hiis_tot = sum([len(x) for x in LumHacorrnot])
    #
    #             n_gmcs = num_gmcs(idovergmc)
    #             n_hiis = num_hiis(idoverhii)
    #
    #
    #
    #             conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr,rms_tot,rms_err,dw, mean_error,SSt,SSe,MSe,d = conf_intervals(xaxall, yayall, alpha)
    #
    #             slope, b, r_sq = linear_regression(xaxall, yayall)
    #             y_pred = b + slope * x0
    #
    #             axs.plot(x0, y_pred, color = color, label = 'linear regression')
    #
    #             r_sq = np.sqrt(r_sq)
    #
    #             x0, xf = xlim[k]
    #             y0, yf = ylim[i]
    #             # axs.text(0.8, 0.15, 'P-Value: %6.2f' % (pvalue), fontsize=8, horizontalalignment='center',
    #             #             verticalalignment='center', transform=axs.transAxes)
    #
    #
    #             #=======
    #             x = xaxall
    #             y = yayall
    #             print(y)
    #             p,V = np.polyfit(x, y, 1, cov=True)
    #             print(p[0])
    #             print(np.sqrt(V[0][0]))
    #             print('\n')
    #             print(p[1])
    #             print(np.sqrt(V[1][1]))
    #
    #
    #
    #             axs.text(0.8, 0.03, 'R²: %5.3f' % ((r_sq)**2), fontsize=8, horizontalalignment='center',
    #                         verticalalignment='center', transform=axs.transAxes)
    #
    #             # axs.text(0.8, 0.19, 'Durbin-Watson stat: %6.2f' % (dw), fontsize=8, horizontalalignment='center',
    #             #             verticalalignment='center', transform=axs.transAxes)
    #
    #             axs.text(0.15, 0.03, 'Slope %5.3f $\pm$ %5.3f ' % (slope, conf_a), fontsize=8, horizontalalignment='center',
    #                         verticalalignment='center', transform=axs.transAxes)
    #
    #             axs.text(0.8, 0.07, 'Standard deviation %5.3f ' % (np.sqrt(MSe)), fontsize=8, horizontalalignment='center',
    #                         verticalalignment='center', transform=axs.transAxes)
    #
    #
    #             axs.text(0.8, 0.11, 'paired gmcs %5.3f (%5.2f %%) ' % (n_gmcs, n_gmcs*100/n_gmcs_tot), fontsize=8, horizontalalignment='center',
    #                      verticalalignment='center', transform=axs.transAxes)
    #
    #             axs.text(0.8, 0.15, 'paired hii regions %5.3f (%5.2f %%) ' % (n_hiis, n_hiis*100/n_hiis_tot), fontsize=8, horizontalalignment='center',
    #                      verticalalignment='center', transform=axs.transAxes)
    #
    #             axs.text(0.8, 0.19, 'paired gmcs CO flux: %5.2f %% ' % (FluxCOGMCovertot*100/FluxCOGMCtot),
    #                      fontsize=8, horizontalalignment='center',
    #                      verticalalignment='center', transform=axs.transAxes)
    #
    #             axs.text(0.8, 0.23, 'paired hii regions Ha Lum: %5.2f %% ' % (LumHacorrovertot*100/LumHacorrtot),
    #                      fontsize=8, horizontalalignment='center',
    #                      verticalalignment='center', transform=axs.transAxes)
    #
    #
    #             axs.set(ylim=(y0, yf))
    #             axs.legend(prop={'size': 8}, loc = 2)
    #
    #
    #
    #
    #
    #             l+=1
    #     axs.set(xlabel=labsxax[k])
    #     #axs[2].set(xlabel=labsxax[k])
    #
    #     reg +=1
    # save_pdf(pdf3, fig, save, show)
    #
    #
    # pdf3.close()


def plot_correlations_rgal(muse, gmc_catalog, gmc_catalog_version, randomize, matching, outliers,  show, save, threshold_percs, vel, gmcprop, rgal_color):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair
    threshold_perc  = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(matching=matching, muse = muse, gmc_catalog= gmc_catalog, gmc_catalog_version=gmc_catalog_version,outliers=outliers,randomize=randomize,threshold_perc=threshold_perc,vel=vel)


    print("Plots of all galaxies together")


    xlimmin, xlimmax = get_min(arrayxax)
    z=0
    print("Starting loop to create figures of all galaxies together - points")

    pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s%s.pdf" % (dirplots, namegmc, name_end, str(z))

    pdf3 = fpdf.PdfPages(pdf_name)
    for threshold_perc in threshold_percs:

        z+=1

        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover,HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak= get_data(
            matching=matching, muse = muse, gmc_catalog= gmc_catalog, gmc_catalog_version=gmc_catalog_version,outliers=outliers,randomize=randomize,threshold_perc=threshold_perc,vel=vel)


        rgal_gmc = arrayyay[len(arrayyay)-1]
        rgal_hii = arrayxax[len(arrayxax)-1]


        #labsyay = labsyay[0:len(labsyay) - 5]  # removing  vel, major axis, minor axis and PA, no need to plot them
        #labsxax = labsxax[0:len(labsxax) - 4]

        k=len(arrayxax)-1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(1,1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        plt.subplots_adjust(hspace=0.0)

        #fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s' % name_end, fontsize=18, va='top')
#        axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0

        for i in gmcprop:

            axs.set(ylabel=labsxax[i])
            axs.grid()
            axs.set(xlabel=labsxax[len(labsxax)-1])


            if threshold_perc == 0.1 :
                color = "black"
            elif i == 3 or i==6:
                color = "tab:red"
            elif i == 1 or i ==4:
                color = "tab:blue"

            color = "black"

            # gmcprop
            # 1 : MCO
            # 3 : Sigma mol
            # 4 : Sigma v
            # 5 : alpha vir
            # 6 : CO Tpeak
            # 7 : Tauff

            yaytmp = arrayxax[i]
            xaxtmp = arrayxax[k]

            #========






            #========


            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            rgal_all = np.concatenate([f.tolist() for f in rgal_hii])

            #if k < 5:
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)

            #axs.set_xlim(xlimmin, xlimmax)

            idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
            xaxall = xaxall[idok]
            yayall = yayall[idok]
            rgal_all = rgal_all[idok]
            lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
            lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
            indlim = np.where((xaxall < lim2) & (xaxall > lim1))
            xaxall = xaxall[indlim]
            yayall = yayall[indlim]
            rgal_all = rgal_all[indlim]


            if rgal_color == False:
                axs.plot(xaxall, yayall,'o', markerfacecolor = 'None' , markersize = 2, markeredgecolor = color, label='threshold = %f' % threshold_perc ,linestyle = 'None')
            else:
                for xi in range(len(xaxall)):

                    colormap =  plt.get_cmap('viridis')
                    colour = colormap(1 - 2*rgal_all[xi]/np.nanmax(rgal_all))
                    axs.plot(xaxall[xi], yayall[xi], 'o',markeredgecolor = colour, markersize = 2, linestyle = 'None',markerfacecolor = 'None' )


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



                #conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr,rms_tot,rms_err,dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(xaxall, yayall, alpha)

                slope, b, r_sq = linear_regression(xaxall, yayall)
               # y_pred = b + slope * x0


                #axs.plot(x0, y_pred+y_conf_sup, '--', color = 'black', label='confidence interval')
                #axs.plot(x0, y_pred + y_conf_inf, '--', color = 'black')

                #axs.plot(x0, y_pred+int_prev_y, '-.', color = 'grey', label = 'prediction interval')
                #axs.plot(x0, y_pred - int_prev_y, '-.', color = 'grey')

               # axs.plot(x0, y_pred, color = 'navy', label = 'linear regression')

                r_sq = np.sqrt(r_sq)

                x0, xf = xlim[k]
                y0, yf = ylim[i]
                # axs.text(0.8, 0.15, 'P-Value: %6.2f' % (pvalue), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)


                #=======
                x = xaxall
                y = yayall
                p,V = np.polyfit(x, y, 1, cov=True)




                # axs.text(0.8, 0.07, 'Error RMS: %5.3f' % (rms_err), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                # axs.text(0.8, 0.15, 'Standard error of estimate: %5.3f' % (stderr), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.03, 'R²: %5.3f' % ((r_sq)**2), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs.transAxes)

                # axs.text(0.8, 0.19, 'Durbin-Watson stat: %6.2f' % (dw), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                #axs.text(0.2, 0.03, 'Slope %5.3f $\pm$ %5.3f ' % (slope, conf_a), fontsize=8, horizontalalignment='center',
                           # verticalalignment='center', transform=axs.transAxes)
                #
               # axs.text(0.8, 0.07, 'Standard deviation %5.3f ' % (np.sqrt(MSe)), fontsize=8, horizontalalignment='center',
                            #verticalalignment='center', transform=axs.transAxes)


                axs.text(0.8, 0.11, 'paired gmcs %5.3f (%5.2f %%) ' % (n_gmcs, n_gmcs*100/n_gmcs_tot), fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.15, 'paired hii regions %5.3f (%5.2f %%) ' % (n_hiis, n_hiis*100/n_hiis_tot), fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.19, 'paired gmcs CO flux: %5.2f %% ' % (FluxCOGMCovertot*100/FluxCOGMCtot),
                         fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.23, 'paired hii regions Ha Lum: %5.2f %% ' % (LumHacorrovertot*100/LumHacorrtot),
                         fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)




                #axs.set(ylim=(y0, yf))
                if l == 0:
                    axs.legend(prop={'size': 8}, loc = 2)





                l+=1
        #axs[l].set(xlabel=labsxax[k])
        save_pdf(pdf3, fig, save, show)

    pdf3.close()


def plot_correlations_fct_rgal(muse, gmc_catalog, gmc_catalog_version, randomize, matching, outliers,  show, save, threshold_percs, vel, gmcprop, rgal_color):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair
    threshold_perc  = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(matching=matching, muse = muse, gmc_catalog= gmc_catalog, gmc_catalog_version=gmc_catalog_version,outliers=outliers,randomize=randomize,threshold_perc=threshold_perc,vel=vel)


    print("Plots of all galaxies together")


    xlimmin, xlimmax = get_min(arrayxax)
    z=0
    print("Starting loop to create figures of all galaxies together - points")

    pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s%s.pdf" % (dirplots, namegmc, name_end, str(z))

    pdf3 = fpdf.PdfPages(pdf_name)

    threshold_perc = threshold_percs[0]

    z+=1

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover,HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak= get_data(
        matching=matching, muse = muse, gmc_catalog= gmc_catalog, gmc_catalog_version=gmc_catalog_version,outliers=outliers,randomize=randomize,threshold_perc=threshold_perc,vel=vel)


    rgal_gmc = arrayyay[len(arrayyay)-1]
    rgal_hii = arrayxax[len(arrayxax)-1]


    labsyay = labsyay[0:len(labsyay) - 5]  # removing  vel, major axis, minor axis and PA, no need to plot them
    labsxax = labsxax[0:len(labsxax) - 4]

    k=1

    numbers = 16

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

        #========

        rgal_all = rgal_hii[gal]


        rgal_max = np.nanmax(rgal_all)
        rgal_min = np.nanmin(rgal_all)
        rgal_bins = (rgal_max-rgal_min)/(numbers)




        for ri in range(numbers):




            ids_rgal = np.where((rgal_all > ri*rgal_bins) & (rgal_all < (ri+1) * rgal_bins))[0]
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



    xaxall_all = [np.concatenate(xaxall_gal[i]) for i in range(np.shape(xaxall_gal)[1]) ]
    yayall_all = [np.concatenate(yayall_gal[i]) for i in range(np.shape(yayall_gal)[1]) ]




    print('TEEEEEEST')
    print(np.sum(test_ids))
    sns.set(style='white', color_codes=True)
    fig, axs = plt.subplots(int(np.sqrt(numbers)), int(np.sqrt(numbers)), sharex='col', figsize=(9, 10), dpi=80,
                            gridspec_kw={'hspace': 0})
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.0)

    # fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s' % name_end, fontsize=18, va='top')
    axs = axs.ravel()

    l=0

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
        axs[0].set(ylabel=labsyay[i])
        axs[4].set(ylabel=labsyay[i])
        axs[8].set(ylabel=labsyay[i])
        axs[12].set(ylabel=labsyay[i])

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


    axs[12].set(xlabel=labsxax[k])
    axs[13].set(xlabel=labsxax[k])
    axs[14].set(xlabel=labsxax[k])
    axs[15].set(xlabel=labsxax[k])


    save_pdf(pdf3, fig, save, show)

    pdf3.close()


    slope_list = slope_list[0:len(slope_list)-4]
    error_slope_list = error_slope_list[0:len(error_slope_list)-4]


    rgal_list = [rgal_bins*i for i in range(numbers)]
    rgal_list = rgal_list[0:len(rgal_list)-4]
    r2_list = r2_list[0:len(r2_list) -4]


    # plt.errorbar( x =rgal_list,y =slope_list, yerr=error_slope_list, capsize=5,
    # elinewidth=2,
    # markeredgewidth=2,  marker = '+', markersize = 12, color = 'black')
    plt.plot( rgal_list,slope_list, marker = '+', markersize = 12, color = 'black')


    plt.ylabel('Slope', fontsize = 40)
    plt.xlabel('R_gal (pc)', fontsize = 40)
    plt.grid()

    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)


    plt.show()

    plt.plot( rgal_list,r2_list, marker = '+', markersize = 12, color = 'black')

    plt.ylabel('Correlation coefficient r²', fontsize = 40)
    plt.xlabel('R_gal (pc)', fontsize = 40)
    plt.grid()



    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)

    plt.show()







def plot_correlations_randomized(muse, gmc_catalog, gmc_catalog_version, randomize, matching, outliers,  show, save, threshold_percs, vel, gmcprop, random):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair
    threshold_perc = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)

    print("Plots of all galaxies together")

    xlimmin, xlimmax = get_min(arrayyay)
    z = 0
    print("Starting loop to create figures of all galaxies together - points")

    pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold_random_gmc_prop%s%s%s.pdf" % (dirplots, namegmc, name_end, str(z))

    pdf3 = fpdf.PdfPages(pdf_name)

    yaytmp = arrayyay[1]
    yayall = np.concatenate([f.tolist() for f in yaytmp])

    #rand_id = np.where(yayall >= -1*1e20)
    all_ids = np.linspace(0,len(yayall)-1, len(yayall), dtype = int)
    rand_ids = all_ids
    np.random.shuffle(rand_ids)



    for threshold_perc in threshold_percs:

        z += 1

        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC, SizepcGMC, SizepcHII, MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(
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

                #yayall = yayall[rand_ids]


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
        save_pdf(pdf3, fig, save, show)

    pdf3.close()


def plot_residus(muse, gmc_catalog, matching, outliers,gmc_catalog_version, randomize,  show, save, threshold_percs):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair
    threshold_perc  = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(
        muse, gmc_catalog, matching, outliers, threshold_perc, )

    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")


    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    for threshold_perc in threshold_percs:
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(
            muse, gmc_catalog, matching, outliers, threshold_perc, )
        idoverhiis = [item for sublist in idoverhii for item in sublist]

        k=1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(3,1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n Distance to line histogram \n%s' % name_end, fontsize=18, va='top')
        #axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0
        for i in [1,3,6]:

            #axs[l].set(ylabel=labsyay[i])
            axs[2].set(ylabel='line distance / error')

            axs[l].grid()
            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            #if k < 5:
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

                #conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf = conf_intervals(xaxall, yayall, alpha)



                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope*xaxall

                err = (y_pred - yayall)
                #axs.scatter(y_pred, err)
                print(scipy.stats.shapiro(err))
                axs[l].hist(err, bins=70, label = labsyay[i])

                x0, xf = xlim[k]
                y0, yf = ylim[i]


                axs[l].legend(prop={'size': 8})
                l+=1
        save_pdf(pdf3, fig, save, show)

        plt.show()

    pdf3.close()


def hist_std_props(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize,vel,  show, save, threshold_percs, bin, gmc_props):
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    threshold_perc = threshold_percs[0]

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover,HIImajor, majorGMC, minorGMC,MassCOGMC,SizepcGMC,SizepcHII ,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak= get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)
    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold_dist%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(dirplots + 'Hist_gmc_properties')  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover,HIImajor, majorGMC, minorGMC,MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_percs[0], vel=vel)

    labsxax2, labsyay2, arrayxax2, arrayyay2, name_end2, namegmc2, galaxias2, idoverhii2, idovergmc2, LumHacorrnot2, FluxCOGMCnot2, FluxCOGMCover2, regionindexGMCover2,HIImajor2, majorGMC2, minorGMC2,MassCOGMC2,SizepcGMC2,SizepcHII2,MasscoGMCover2, SizepcGMCover2, SigmaMol2, Sigmav2, COTpeak2 = get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_percs[1], vel=vel)


    arrayyay1 = [ SizepcGMCover,FluxCOGMCover, arrayyay[1],arrayyay[3],arrayyay[4],arrayyay[6] ]
    arrayyay_all = [SizepcGMC, FluxCOGMCnot, MassCOGMC, SigmaMol, Sigmav, COTpeak ]

    labsyay1 = [ labsyay[2], labsyay[13], labsyay[1],labsyay[3], labsyay[4], labsyay[6]]

    arrayyay2 = [ SizepcGMCover2,FluxCOGMCover2, arrayyay2[1],arrayyay2[3],arrayyay2[4],arrayyay2[6] ]
    arrayyay_all2 = [SizepcGMC2, FluxCOGMCnot2, MassCOGMC2, SigmaMol2, Sigmav2, COTpeak2 ]

    l = 0
    for i in range(len(arrayyay1)):

        fig, axs = plt.subplots(1, 1, figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0.25})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle(
            'All galaxies - Overlapping HIIregions and GMCs - GMCproperties histograms \n threshold = %s' % threshold_perc,
            fontsize=18, va='top')

        axs.set(xlabel=labsyay1[i])
        axs.grid()
        yaytmp = arrayyay1[i]
        yaytmp_all = arrayyay_all[i]
        yayall = np.concatenate([f.tolist() for f in yaytmp])
        yayall_all = np.concatenate([f.tolist() for f in yaytmp_all])

        yayall = np.log10(yayall)
        yayall_all = np.log10(yayall_all)


        idok = np.where((abs(yayall) < 100000) )
        yayall = yayall[idok]

        idok = np.where((abs(yayall_all) < 100000) )
        yayall_all = yayall_all[idok]


        if yayall_all.any != ():
             axs.hist(yayall_all, bins = 100, color = 'red', label = 'all mgcs', density = True)
             axs.legend(prop={'size': 8})

             mean = np.nanmean(yayall_all)
             median = np.nanmedian(yayall_all)

        axs.text(0.15, 0.90, 'Mean all %5.3f ' % (mean), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        axs.text(0.15, 0.75, 'Median all %5.3f ' % (median), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)




        if  yayall.any != ():

            axs.hist(yayall,  bins = 100, color = 'blue', label = 'paired gmcs', density = True)
            axs.legend(prop={'size': 8})

            mean = np.nanmean(yayall)
            median = np.nanmedian(yayall)


        axs.text(0.15, 0.85, 'Mean matched %5.3f ' % (mean), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        axs.text(0.15, 0.70, 'Median matched %5.3f ' % (median), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)
        l += 1

        save_pdf(pdf3, fig, save, show)




        #=============Other threshold=====================#

        fig, axs = plt.subplots(1, 1, figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0.25})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle(
            'All galaxies - Overlapping HIIregions and GMCs - standard deviation histogram \n threshold = 0.9',
            fontsize=18, va='top')

        axs.set(xlabel=labsyay1[i])
        axs.grid()
        yaytmp = arrayyay2[i]
        yaytmp_all = arrayyay_all2[i]
        yayall = np.concatenate([f.tolist() for f in yaytmp])
        yayall_all = np.concatenate([f.tolist() for f in yaytmp_all])

        yayall = np.log10(yayall)
        yayall_all = np.log10(yayall_all)


        idok = np.where((abs(yayall) < 100000) )
        yayall = yayall[idok]

        idok = np.where((abs(yayall_all) < 100000) )
        yayall_all = yayall_all[idok]


        if yayall_all.any != ():
             axs.hist(yayall_all, bins = 100, color = 'red', label = 'all mgcs', density = True)
             axs.legend(prop={'size': 8})

             mean = np.nanmean(yayall_all)
             median = np.nanmedian(yayall_all)

        axs.text(0.15, 0.90, 'Mean all %5.3f ' % (mean), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        axs.text(0.15, 0.75, 'Median all %5.3f ' % (median), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)




        if  yayall.any != ():

            axs.hist(yayall,  bins = 100, color = 'green', label = 'paired gmcs', density = True)
            axs.legend(prop={'size': 8})

            mean = np.nanmean(yayall)
            median = np.nanmedian(yayall)


        axs.text(0.15, 0.85, 'Mean matched %5.3f ' % (mean), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        axs.text(0.15, 0.70, 'Median matched %5.3f ' % (median), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)
        l += 1

        save_pdf(pdf3, fig, save, show)

    pdf3.close()



    #===============================Hist superposés========================================#

    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold_dist%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(dirplots + 'Hist_gmc_properties_overlayed')  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover,HIImajor, majorGMC, minorGMC,MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_percs[0], vel=vel)

    labsxax2, labsyay2, arrayxax2, arrayyay2, name_end2, namegmc2, galaxias2, idoverhii2, idovergmc2, LumHacorrnot2, FluxCOGMCnot2, FluxCOGMCover2, regionindexGMCover2,HIImajor2, majorGMC2, minorGMC2,MassCOGMC2,SizepcGMC2,SizepcHII2,MasscoGMCover2, SizepcGMCover2, SigmaMol2, Sigmav2, COTpeak2 = get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_percs[1], vel=vel)


    arrayyay1 = [ SizepcGMCover,FluxCOGMCover, arrayyay[1],arrayyay[3],arrayyay[4],arrayyay[6] ]
    arrayyay_all = [SizepcGMC, FluxCOGMCnot, MassCOGMC, SigmaMol, Sigmav, COTpeak ]

    labsyay1 = [ labsyay[2], labsyay[13], labsyay[1],labsyay[3], labsyay[4], labsyay[6]]

    arrayyay2 = [ SizepcGMCover2,FluxCOGMCover2, arrayyay2[1],arrayyay2[3],arrayyay2[4],arrayyay2[6] ]
    arrayyay_all2 = [SizepcGMC2, FluxCOGMCnot2, MassCOGMC2, SigmaMol2, Sigmav2, COTpeak2 ]

    l = 0
    for i in range(len(arrayyay1)):

        fig, axs = plt.subplots(1, 1, figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0.25})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle(
            'All galaxies - Overlapping HIIregions and GMCs - GMCproperties histograms \n threshold = %s' % threshold_perc,
            fontsize=18, va='top')

        axs.set(xlabel=labsyay1[i])
        axs.grid()

        yaytmp = arrayyay1[i]
        yaytmp_all = arrayyay_all[i]
        yaytmp2 = arrayyay2[i]


        yayall = np.concatenate([f.tolist() for f in yaytmp])
        yayall_all = np.concatenate([f.tolist() for f in yaytmp_all])
        yayall2 = np.concatenate([f.tolist() for f in yaytmp2])

        yayall = np.log10(yayall)
        yayall_all = np.log10(yayall_all)
        yayall2 = np.log10(yayall2)


        idok = np.where((abs(yayall) < 100000) )
        yayall = yayall[idok]

        idok = np.where((abs(yayall_all) < 100000) )
        yayall_all = yayall_all[idok]

        idok = np.where((abs(yayall2) < 100000) )
        yayall2 = yayall2[idok]


        if yayall_all.any != ():
             axs.hist(yayall_all, bins = 100, color = 'red', label = 'all mgcs', density = True)
             axs.legend(prop={'size': 8})

             mean = np.nanmean(yayall_all)
             median = np.nanmedian(yayall_all)

        axs.text(0.15, 0.90, 'Mean all %5.3f ' % (mean), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        axs.text(0.15, 0.75, 'Median all %5.3f ' % (median), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)


        if  yayall.any != ():

            axs.hist(yayall,  bins = 100, color = 'blue', label = 'paired gmcs (0.1)', density = True)
            axs.legend(prop={'size': 8})

            mean = np.nanmean(yayall)
            median = np.nanmedian(yayall)


        axs.text(0.15, 0.85, 'Mean matched (0.1) %5.3f ' % (mean), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        axs.text(0.15, 0.70, 'Median matched (0.1) %5.3f ' % (median), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        if  yayall2.any != ():

            axs.hist(yayall2,  bins = 100, color = 'green', label = 'paired gmcs (0.9)', density = True)
            axs.legend(prop={'size': 8})

            mean = np.nanmean(yayall2)
            median = np.nanmedian(yayall2)


        axs.text(0.15, 0.80, 'Mean matched (0.9) %5.3f ' % (mean), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='baseline', transform=axs.transAxes)

        axs.text(0.15, 0.65, 'Median matched (0.9) %5.3f ' % (median), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)
        l += 1

        save_pdf(pdf3, fig, save, show)

    pdf3.close()


def hist_std_hii_props(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize,vel,  show, save, threshold_percs, bin, gmc_props):
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    threshold_perc = threshold_percs[0]

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover,HIImajor, majorGMC, minorGMC,MassCOGMC,SizepcGMC,SizepcHII ,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak= get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)
    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold_dist%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(dirplots + 'Hist_hii_properties')  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover,HIImajor, majorGMC, minorGMC,MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_percs[0], vel=vel)

    labsxax2, labsyay2, arrayxax2, arrayyay2, name_end2, namegmc2, galaxias2, idoverhii2, idovergmc2, LumHacorrnot2, FluxCOGMCnot2, FluxCOGMCover2, regionindexGMCover2,HIImajor2, majorGMC2, minorGMC2,MassCOGMC2,SizepcGMC2,SizepcHII2,MasscoGMCover2, SizepcGMCover2, SigmaMol2, Sigmav2, COTpeak2 = get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_percs[1], vel=vel)



    SizepcHII_not = [SizepcHII[i][j] for i in range(len(SizepcHII)) for j in range(len(galaxias)) if i not in idoverhii[j]]
    LumHacorrnot_not = [LumHacorrnot[i][j] for i in range(len(LumHacorrnot)) for j in range(len(galaxias)) if i not in idoverhii[j]]

    SizepcHII_not2 = [SizepcHII2[i][j] for i in range(len(SizepcHII2)) for j in range(len(galaxias)) if i not in idoverhii2[j]]
    LumHacorrnot_not2 = [LumHacorrnot2[i][j] for i in range(len(LumHacorrnot2)) for j in range(len(galaxias)) if i not in idoverhii2[j]]



    arrayyay1 = [ arrayxax[0], arrayxax[1] ]
    arrayyay_all = [SizepcHII, LumHacorrnot]
    arrayyay_not = [SizepcHII_not, LumHacorrnot_not]


    labsyay1 = [ labsxax[0], labsxax[1]]

    arrayyay2 = [ arrayxax2[0], arrayxax2[1]]
    arrayyay_all2 = [SizepcHII2, LumHacorrnot2 ]
    arrayyay_not2 = [SizepcHII_not2, LumHacorrnot_not2]

    l = 0
    for i in range(len(arrayyay1)):

        fig, axs = plt.subplots(1, 1, figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0.25})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle(
            'All galaxies - Overlapping HIIregions and GMCs - GMCproperties histograms \n threshold = %s' % threshold_perc,
            fontsize=18, va='top')

        axs.set(xlabel=labsyay1[i])
        axs.grid()
        yaytmp = arrayyay1[i]
        yaytmp_not = arrayyay_not[i]
        yaytmp_all = arrayyay_all[i]
        yayall = np.concatenate([f.tolist() for f in yaytmp])
        yayall_all = np.concatenate([f.tolist() for f in yaytmp_all])

        #yayall_not = np.concatenate([f.tolist() for f in yaytmp_not])
        yayall_not = yaytmp_not


        yayall = np.log10(yayall)
        yayall_all = np.log10(yayall_all)

        yayall_not = np.log10(yayall_not)

        idok = np.where((abs(yayall) < 100000) )
        yayall = yayall[idok]

        idok = np.where((abs(yayall_all) < 100000) )
        yayall_all = yayall_all[idok]

        idok = np.where((abs(yayall_not) < 100000) )
        yayall_not = yayall_not[idok]

        binn = 25


        if yayall_all.any != ():
             axs.hist(yayall_all, bins = binn, color = 'red', label = 'all hii', histtype='stepfilled',density = True)
             axs.legend(prop={'size': 8})

             mean = np.nanmean(yayall_all)
             median = np.nanmedian(yayall_all)

        axs.text(0.15, 0.95, 'Mean all %5.3f ' % (mean), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        axs.text(0.15, 0.75, 'Median all %5.3f ' % (median), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)


        if yayall_not.any != ():
             axs.hist(yayall_not, bins = binn, color = 'blue', label = 'unmatched hii',histtype='stepfilled', density = True)
             axs.legend(prop={'size': 8})

             mean = np.nanmean(yayall_not)
             median = np.nanmedian(yayall_not)

        axs.text(0.15, 0.90, 'Mean unmatched %5.3f ' % (mean), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        axs.text(0.15, 0.70, 'Median unmatched %5.3f ' % (median), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)


        if  yayall.any != ():

            axs.hist(yayall,  bins = binn, color = 'green', label = 'paired hii', histtype='stepfilled',density = True)
            axs.legend(prop={'size': 8})

            mean = np.nanmean(yayall)
            median = np.nanmedian(yayall)


        axs.text(0.15, 0.85, 'Mean matched %5.3f ' % (mean), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        axs.text(0.15, 0.65, 'Median matched %5.3f ' % (median), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)
        l += 1

        save_pdf(pdf3, fig, save, show)




        #=============Other threshold=====================#

        fig, axs = plt.subplots(1, 1, figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0.25})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle(
            'All galaxies - Overlapping HIIregions and GMCs - standard deviation histogram \n threshold = 0.9',
            fontsize=18, va='top')

        axs.set(xlabel=labsyay1[i])
        axs.grid()
        yaytmp = arrayyay2[i]
        yaytmp_not = arrayyay_not2[i]
        yaytmp_all = arrayyay_all2[i]
        yayall = np.concatenate([f.tolist() for f in yaytmp])
        yayall_all = np.concatenate([f.tolist() for f in yaytmp_all])
        #yayall_not = np.concatenate([f.tolist() for f in yaytmp_not])
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

        if yayall_all.any != ():
            axs.hist(yayall_all, bins=binn, color='red', label='all hii',histtype='stepfilled', density=True)
            axs.legend(prop={'size': 8})

            mean = np.nanmean(yayall_all)
            median = np.nanmedian(yayall_all)

        axs.text(0.15, 0.95, 'Mean all %5.3f ' % (mean), fontsize=10,
                 horizontalalignment='center',
                 verticalalignment='center', transform=axs.transAxes)

        axs.text(0.15, 0.75, 'Median all %5.3f ' % (median), fontsize=10,
                 horizontalalignment='center',
                 verticalalignment='center', transform=axs.transAxes)

        if yayall_not.any != ():
            axs.hist(yayall_not, bins=binn, color='blue', label='unmatched hii',histtype='stepfilled', density=True)
            axs.legend(prop={'size': 8})

            mean = np.nanmean(yayall_not)
            median = np.nanmedian(yayall_not)

        axs.text(0.15, 0.90, 'Mean unmatched %5.3f ' % (mean), fontsize=10,
                 horizontalalignment='center',
                 verticalalignment='center', transform=axs.transAxes)

        axs.text(0.15, 0.70, 'Median unmatched %5.3f ' % (median), fontsize=10,
                 horizontalalignment='center',
                 verticalalignment='center', transform=axs.transAxes)

        if yayall.any != ():
            axs.hist(yayall, bins=binn, color='green', label='paired hii',histtype='stepfilled', density=True)
            axs.legend(prop={'size': 8})

            mean = np.nanmean(yayall)
            median = np.nanmedian(yayall)

        axs.text(0.15, 0.85, 'Mean matched %5.3f ' % (mean), fontsize=10,
                 horizontalalignment='center',
                 verticalalignment='center', transform=axs.transAxes)

        axs.text(0.15, 0.65, 'Median matched %5.3f ' % (median), fontsize=10,
                 horizontalalignment='center',
                 verticalalignment='center', transform=axs.transAxes)
        l += 1

        save_pdf(pdf3, fig, save, show)

    pdf3.close()



    #===============================Hist superposés========================================#

    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold_dist%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(dirplots + 'Hist_hii_properties_overlayed')  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover,HIImajor, majorGMC, minorGMC,MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_percs[0], vel=vel)

    labsxax2, labsyay2, arrayxax2, arrayyay2, name_end2, namegmc2, galaxias2, idoverhii2, idovergmc2, LumHacorrnot2, FluxCOGMCnot2, FluxCOGMCover2, regionindexGMCover2,HIImajor2, majorGMC2, minorGMC2,MassCOGMC2,SizepcGMC2,SizepcHII2,MasscoGMCover2, SizepcGMCover2, SigmaMol2, Sigmav2, COTpeak2 = get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_percs[1], vel=vel)


    arrayyay1 = [ arrayxax[0], arrayxax[1] ]
    arrayyay_all = [SizepcHII, LumHacorrnot]

    labsyay1 = [ labsxax[0], labsxax[1]]

    arrayyay2 = [ arrayxax2[0], arrayxax2[1]]
    arrayyay_all2 = [SizepcHII2, LumHacorrnot2 ]


    l = 0
    for i in range(len(arrayyay1)):

        fig, axs = plt.subplots(1, 1, figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0.25})
        plt.subplots_adjust(wspace=0.3)
        # fig.suptitle(
        #     'All galaxies - Overlapping HIIregions and GMCs - GMCproperties histograms \n threshold = %s' % threshold_perc,
        #     fontsize=18, va='top')

        axs.set(xlabel=labsyay1[i])
        axs.grid()

        yaytmp = arrayyay1[i]
        yaytmp_all = arrayyay_all[i]
        yaytmp2 = arrayyay2[i]


        yayall = np.concatenate([f.tolist() for f in yaytmp])
        yayall_all = np.concatenate([f.tolist() for f in yaytmp_all])
        yayall2 = np.concatenate([f.tolist() for f in yaytmp2])

        yayall = np.log10(yayall)
        yayall_all = np.log10(yayall_all)
        yayall2 = np.log10(yayall2)


        idok = np.where((abs(yayall) < 100000) )
        yayall = yayall[idok]

        idok = np.where((abs(yayall_all) < 100000) )
        yayall_all = yayall_all[idok]

        idok = np.where((abs(yayall2) < 100000) )
        yayall2 = yayall2[idok]


        if yayall_all.any != ():
             axs.hist(yayall_all, bins = binn, color = 'red', label = 'all mgcs',histtype='stepfilled', density = True)
             axs.legend(prop={'size': 8})

             mean = np.nanmean(yayall_all)
             median = np.nanmedian(yayall_all)

        axs.text(0.15, 0.90, 'Mean all %5.3f ' % (mean), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        axs.text(0.15, 0.75, 'Median all %5.3f ' % (median), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)


        if  yayall.any != ():

            axs.hist(yayall,  bins = binn, color = 'blue', label = 'paired gmcs (0.1)', histtype='stepfilled',density = True)
            axs.legend(prop={'size': 8})

            mean = np.nanmean(yayall)
            median = np.nanmedian(yayall)


        axs.text(0.15, 0.85, 'Mean matched (0.1) %5.3f ' % (mean), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        axs.text(0.15, 0.70, 'Median matched (0.1) %5.3f ' % (median), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)

        if  yayall2.any != ():

            axs.hist(yayall2,  bins = binn, color = 'green', label = 'paired gmcs (0.9)',histtype='stepfilled', density = True)
            axs.legend(prop={'size': 8})

            mean = np.nanmean(yayall2)
            median = np.nanmedian(yayall2)


        axs.text(0.15, 0.80, 'Mean matched (0.9) %5.3f ' % (mean), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='baseline', transform=axs.transAxes)

        axs.text(0.15, 0.65, 'Median matched (0.9) %5.3f ' % (median), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='center', transform=axs.transAxes)



        l += 1

        save_pdf(pdf3, fig, save, show)

    pdf3.close()


def hist_std(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize,  show, save, threshold_perc, bin):
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(
        muse, gmc_catalog, matching, outliers, threshold_perc, )
    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold_dist%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    for threshold_perc in threshold_percs:
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(
            muse, gmc_catalog, matching, outliers, threshold_perc, )
        idoverhiis = [item for sublist in idoverhii for item in sublist]

        k = 1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(3, 1, figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0.25})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle('All galaxies - Overlapping HIIregions and GMCs - standard deviation histogram \n %s' % name_end, fontsize=18, va='top')
        axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0
        for i in [1, 3, 6]:
            axs[l].set(xlabel=labsyay[i])
            axs[l].grid()
            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
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


            xaxbin, yaybin, ebin, nbin = bindata(xaxall, yayall, bin)

            if xaxall.any() != 0 and yayall.any != ():
                xmin = np.amin(xaxall)
                xmax = np.amax(xaxall)
                x = xaxall.reshape((-1, 1))
                y = yayall
                slope, b, r_sq = linear_regression(xaxall, yayall)
                x0, xf = xlim[k]
                y0, yf = ylim[i]
                ebin = np.nan_to_num(ebin)
                ebin = np.array(ebin)[np.where(ebin !=0)[0]]
                print(ebin)
                axs[l].hist(ebin, bins = 100)
                #axs[l].plot(xaxall, yayall_dist, '8', label='threshold = %f' % threshold_perc, alpha=0.7, markersize=2)

                # axs[l].set(ylim=(y0, yf))
                axs[l].legend(prop={'size': 8})
                l += 1
        save_pdf(pdf3, fig, save, show)

    pdf3.close()


def plot_dist_prop(muse, gmc_catalog, matching, outliers,gmc_catalog_version, randomize,  show, save, threshold_percs):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    threshold_perc  = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(
        muse, gmc_catalog, matching, outliers, threshold_perc, )

    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold_dist%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")


    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    for threshold_perc in threshold_percs:
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(
            muse, gmc_catalog, matching, outliers, threshold_perc, )
        idoverhiis = [item for sublist in idoverhii for item in sublist]

        k=1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(3, 1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s' % name_end, fontsize=18, va='top')
        axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0
        for i in [1,3,6]:
            axs[l].set(ylabel=labsyay[i])
            axs[l].grid()
            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)



            # for id in idoverhiis:
            #     duplicates, xvalues, yvalues = dup_lists(xaxall, yayall, id, idoverhiis)
            #     symbol = ['o','s','^', 'P','*','h','X']
            #     RdBu = plt.get_cmap('inferno')
            #
            #     axs[l].plot(xvalues, yvalues, '8', alpha=0.7, marker = symbol[duplicates-1], markersize = 2.5) #, color = RdBu((1-(duplicates/10))**2)
            #     axs[l].set_xlim(xlimmin, xlimmax)

            idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
            xaxall = xaxall[idok]
            yayall = yayall[idok]
            lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
            lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
            indlim = np.where((xaxall < lim2) & (xaxall > lim1))
            xaxall = xaxall[indlim]
            yayall = yayall[indlim]

            #axs[l].plot(xaxall, yayall, '8', label='threshold = %f' % threshold_perc, alpha=0.7, markersize=2)
            axs[l].set_xlim(xlimmin, xlimmax)

            if xaxall.any() != 0 and yayall.any != ():
                xmin = np.amin(xaxall)
                xmax = np.amax(xaxall)
                xprang = (xmax - xmin) * 0.1
                x = xaxall.reshape((-1, 1))
                y = yayall
                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * x.ravel()
                #axs[l].plot(xaxall, y_pred, '-')
                x0, xf = xlim[k]
                y0, yf = ylim[i]

                yayall_dist = (yayall - (b + slope * xaxall))
                axs[l].plot(xaxall, yayall_dist, '8', label='threshold = %f' % threshold_perc, alpha=0.7, markersize=2)
                axs[l].plot(xaxall, np.zeros(shape = np.shape(xaxall)))




                axs[l].text(0.8, 0.05, 'R^2: %6.2f' % (r_sq), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[l].transAxes)

                axs[l].text(0.15, 0.05, 'Slope %5.2f' % (slope), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[l].transAxes)

                #axs[l].set(ylim=(y0, yf))
                axs[l].legend(prop={'size': 8})
                axs[0].set(ylim=(-1.5, 1.5))
                axs[1].set(ylim=(-1.5, 1.5))
                axs[2].set(ylim=(y0, yf))


                l+=1
        axs[2].set(xlabel=labsxax[k])
        save_pdf(pdf3, fig, save, show)

    pdf3.close()


def plot_abs_dist_prop(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, show, save, threshold_percs):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    threshold_perc  = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(
        muse, gmc_catalog, matching, outliers, threshold_perc, )

    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold_dist%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")


    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    for threshold_perc in threshold_percs:
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc,LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(
            muse, gmc_catalog, matching, outliers, threshold_perc, )
        idoverhiis = [item for sublist in idoverhii for item in sublist]

        k=1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(3, 1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s' % name_end, fontsize=18, va='top')
        axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0
        for i in [1,3,6]:
            axs[l].set(ylabel=labsyay[i])
            axs[l].grid()
            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)



            # for id in idoverhiis:
            #     duplicates, xvalues, yvalues = dup_lists(xaxall, yayall, id, idoverhiis)
            #     symbol = ['o','s','^', 'P','*','h','X']
            #     RdBu = plt.get_cmap('inferno')
            #
            #     axs[l].plot(xvalues, yvalues, '8', alpha=0.7, marker = symbol[duplicates-1], markersize = 2.5) #, color = RdBu((1-(duplicates/10))**2)
            #     axs[l].set_xlim(xlimmin, xlimmax)

            idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
            xaxall = xaxall[idok]
            yayall = yayall[idok]
            lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
            lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
            indlim = np.where((xaxall < lim2) & (xaxall > lim1))
            xaxall = xaxall[indlim]
            yayall = yayall[indlim]

            #axs[l].plot(xaxall, yayall, '8', label='threshold = %f' % threshold_perc, alpha=0.7, markersize=2)
            axs[l].set_xlim(xlimmin, xlimmax)

            if xaxall.any() != 0 and yayall.any != ():
                xmin = np.amin(xaxall)
                xmax = np.amax(xaxall)
                xprang = (xmax - xmin) * 0.1
                x = xaxall.reshape((-1, 1))
                y = yayall
                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * x.ravel()
                #axs[l].plot(xaxall, y_pred, '-')
                x0, xf = xlim[k]
                y0, yf = ylim[i]

                yayall_dist = abs(yayall - (b + slope * xaxall))
                axs[l].plot(xaxall, yayall_dist, '8', label='threshold = %f' % threshold_perc, alpha=0.7, markersize=2)



                axs[l].text(0.8, 0.05, 'R^2: %6.2f' % (r_sq), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[l].transAxes)

                axs[l].text(0.15, 0.05, 'Slope %5.2f' % (slope), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[l].transAxes)

                #axs[l].set(ylim=(y0, yf))
                axs[l].legend(prop={'size': 8})
                l+=1
        axs[2].set(xlabel=labsxax[k])
        save_pdf(pdf3, fig, save, show)

    pdf3.close()


def plot_gmcprop_regions(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, show, save, threshold_perc, vel, gmcprop, regions):


    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair

    region_name = ['Center','Arms','Disc']

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)
    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    reg = 0
    for i in range(len(regions)):

        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(
            matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
            outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)

        regions_id = regions[i]


        if regions[i] != [1, 2, 3]:
            color = 'red'

        else:
            color = 'black'

        regions_colors= regions[i]

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


        k=1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(2, 2, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        #fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s \n %s' % (name_end, region_name[reg]), fontsize=18, va='top')
        axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0
        for i in gmcprop:

            if regions_colors == [4,5,6] and i == 3:
                color = 'tab:blue'
            elif regions_colors == [1,2,3]:
                color = 'black'
            else:
                color = 'tab:red'

            axs[l].set(ylabel=labsyay[i])
            axs[l].grid()
            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            #if k < 5:
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)



            axs[l].plot(xaxall, yayall,'o', markerfacecolor = 'None' , markersize = 2, markeredgecolor = color, label='threshold = %f' % threshold_perc ,linestyle = 'None')

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



                conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr,rms_tot,rms_err,dw, mean_error,SSt,SSe,MSe,d = conf_intervals(xaxall, yayall, alpha)

                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * x0

                #axs.plot(x0, y_pred, '-')
                #sns.regplot(x,y, scatter_kws={'s':2} )
                y_pred_sup = b+conf_b + (slope + conf_a)*x0
                y_pred_inf = b-conf_b + (slope - conf_a)*x0

                #axs.plot(x0, y_pred_sup, color = "lightgreen")
                #axs.plot(x0, y_pred_inf, color = "lightgreen")

                axs[l].plot(x0, y_pred+y_conf_sup, '--', color = 'black', label='confidence interval')
                axs[l].plot(x0, y_pred + y_conf_inf, '--', color = 'black')

                axs[l].plot(x0, y_pred+int_prev_y, '-.', color = 'grey', label = 'prediction interval')
                axs[l].plot(x0, y_pred - int_prev_y, '-.', color = 'grey')

                axs[l].plot(x0, y_pred, color = 'navy', label = 'linear regression')

                r_sq = np.sqrt(r_sq)

                x0, xf = xlim[k]
                y0, yf = ylim[i]
                # axs.text(0.8, 0.15, 'P-Value: %6.2f' % (pvalue), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)


                #=======
                x = xaxall
                y = yayall
                p,V = np.polyfit(x, y, 1, cov=True)




                axs[l].text(0.8, 0.05, 'R²: %5.3f' % ((r_sq)**2), fontsize=10, horizontalalignment='center',
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
                axs[l].legend(prop={'size': 8}, loc = 2)




                axs[l].set(xlabel=labsxax[k])

                l+=1
        #axs[2].set(xlabel=labsxax[k])

        save_pdf(pdf3, fig, save, show)
        reg +=1

    pdf3.close()


def plotallgals(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, show, save, threshold_perc, vel, gmcprop):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)
    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")


    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    for j in range(len(galaxias)):
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(
            matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
            outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)

        k=1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(1, 1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s \n %s' % (name_end,galaxias[j]), fontsize=18, va='top')
        #axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0
        for i in gmcprop:

            axs.set(ylabel=labsyay[i])
            axs.grid()
            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
            xaxall = arrayxax[k][j]
            yayall = arrayyay[i][j]
            #if k < 5:
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)


            axs.plot(xaxall, yayall,'o', markerfacecolor = 'None' , markersize = 2, markeredgecolor = 'black', label='threshold = %f' % threshold_perc ,linestyle = 'None')




            idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
            xaxall = xaxall[idok]
            yayall = yayall[idok]
            lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
            lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
            indlim = np.where((xaxall < lim2) & (xaxall > lim1))
            #xaxall = xaxall[indlim]
            #yayall = yayall[indlim]
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
                n_gmcs = len(idovergmc[j]) - sum([x-1 for x in id]) #num_gmcs(idovergmc[j])
                n_hiis = len(idoverhii[j])



                conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr,rms_tot,rms_err,dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(xaxall, yayall, alpha)

                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * x0

                #axs.plot(x0, y_pred, '-')
                #sns.regplot(x,y, scatter_kws={'s':2} )
                y_pred_sup = b+conf_b + (slope + conf_a)*x0
                y_pred_inf = b-conf_b + (slope - conf_a)*x0

                #axs.plot(x0, y_pred_sup, color = "lightgreen")
                #axs.plot(x0, y_pred_inf, color = "lightgreen")

                axs.plot(x0, y_pred+y_conf_sup, '--', color = 'black', label='confidence interval')
                axs.plot(x0, y_pred + y_conf_inf, '--', color = 'black')

                axs.plot(x0, y_pred+int_prev_y, '-.', color = 'grey', label = 'prediction interval')
                axs.plot(x0, y_pred - int_prev_y, '-.', color = 'grey')

                axs.plot(x0, y_pred, color = 'navy', label = 'linear regression')

                r_sq = np.sqrt(r_sq)

                x0, xf = xlim[k]
                y0, yf = ylim[i]
                # axs.text(0.8, 0.15, 'P-Value: %6.2f' % (pvalue), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)


                #=======
                x = xaxall
                y = yayall
                #p,V = np.polyfit(x, y, 1, cov=True)




                axs.text(0.8, 0.03, 'R²: %5.3f' % ((r_sq)**2), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs.transAxes)

                # axs.text(0.8, 0.19, 'Durbin-Watson stat: %6.2f' % (dw), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                axs.text(0.15, 0.05, 'Slope %5.3f $\pm$ %5.3f ' % (slope, conf_a), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.07, 'Standard error of estimate %5.3f ' % (np.sqrt(MSe)), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs.transAxes)


                axs.text(0.8, 0.11, 'paired gmcs %5.3f (%5.2f %%) ' % (n_gmcs, n_gmcs*100/n_gmcs_tot), fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.15, 'paired hii regions %5.3f (%5.2f %%) ' % (n_hiis, n_hiis*100/n_hiis_tot), fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.19, 'paired gmcs CO flux: %5.2f %% ' % (FluxCOGMCovertot*100/FluxCOGMCtot),
                         fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)

                axs.text(0.8, 0.23, 'paired hii regions Ha Lum: %5.2f %% ' % (LumHacorrovertot*100/LumHacorrtot),
                         fontsize=8, horizontalalignment='center',
                         verticalalignment='center', transform=axs.transAxes)





                axs.set(ylim=(y0, yf))
                axs.legend(prop={'size': 8})





                l+=1
        axs.set(xlabel=labsxax[k])
        save_pdf(pdf3, fig, save, show)

    pdf3.close()


def plotallgals_galprop(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, show, save, threshold_perc, vel, gmcprop, sorting):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(
        matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
        outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)
    if save == True:
        pdf_name = "%sCorrelations_AllGals%s%s%s.pdf" % (dirplots,sorting, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")



    stellar_Mass_1 = [9.69, 10.34, 9.81, 10.86, 10.97, 10.22, 10.71, 10.60, 10.79, 10.36, 9.87, 10.37, 10.61, 10.57,
                      10.72, 10.73, 10.56, 9.40, 10.04]


    print(len(stellar_Mass_1) - len(galaxias))
    sorted_stellar_mass = [9.69, 10.34, 9.81, 10.86, 10.97, 10.22, 10.71, 10.60, 10.79, 10.36, 9.87, 10.37, 10.61,
                           10.57, 10.72, 10.73, 10.56, 9.40, 10.04]
    sorted_stellar_mass.sort()

    stellar_Mass_1 = [9.69, 10.34, 9.81, 10.86, 10.97, 10.22, 10.71, 10.60, 10.79, 10.36, 9.87, 10.37, 10.61,
                           10.57, 10.72, 10.73, 10.56, 9.40, 10.04]
    sorted_stellar_mass = [9.69, 10.34, 9.81, 10.86, 10.97, 10.22, 10.71, 10.60, 10.79, 10.36, 9.87, 10.37, 10.61,
                           10.57, 10.72, 10.73, 10.56, 9.40, 10.04]
    sorted_stellar_mass.sort()

    SFR = [-0.37, 0.29, 0.01, 0.37, 0.86, 0.49, -0.02, -0.09, 0.63, 0.30, -0.08, -0.11, 0.22, 0.72, 0.77, 0.55, 0.34,
           -0.44, 0.02]
    sorted_SFR = [-0.37, 0.29, 0.01, 0.37, 0.86, 0.49, -0.02, -0.09, 0.63, 0.30, -0.08, -0.11, 0.22, 0.72, 0.77, 0.55,
                  0.34, -0.44, 0.02]
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

    # arrayxax_c = arrayxax
    # arrayyay_c = arrayyay
    #
    # for i in range(np.shape(arrayxax)[0]):
    #     for j in range(np.shape(arrayxax)[1]):
    #         arrayxax[i][j] = arrayxax_c[i][ind_value[j]]
    #         arrayyay[i][j] = arrayyay_c[i][ind_value[j]]


    r2_list = []
    slope_list = []

    xlimmin, xlimmax = get_min(arrayxax)
    num_graphs = 20

    pages = int(len(galaxias) / num_graphs) + 1

    print("Starting loop to create figures of all galaxies together - points")
    for page in range((pages)):

        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(5,4, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle('All galaxies - GMC velocity dispersion vs Halpha Luminosity \n Sorted by increasing %s (threshold = %1.1f)' %(str.upper(sorting) , threshold_perc),
                     fontsize=18,
                     va='top')
        axs = axs.ravel()
        for l in range(num_graphs):
            print(l)
            j = l+num_graphs*page

            if j < len(galaxias):
                labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak = get_data(
                    matching=matching, muse=muse, gmc_catalog=gmc_catalog, gmc_catalog_version=gmc_catalog_version,
                    outliers=outliers, randomize=randomize, threshold_perc=threshold_perc, vel=vel)

                k=1

                i = gmcprop[0]

                axs[0].set(ylabel=labsyay[i])
                axs[4].set(ylabel=labsyay[i])
                axs[8].set(ylabel=labsyay[i])
                axs[12].set(ylabel=labsyay[i])
                axs[16].set(ylabel=labsyay[i])




                axs[l].grid()
                #axs[l].text(.5,.9,'%s' %galaxias[j], fontsize=18, ha='center')
                yaytmp = arrayyay[i]
                xaxtmp = arrayxax[k]
                xaxall = arrayxax[k][ind_value[j]]
                yayall = arrayyay[i][ind_value[j]]
                #if k < 5:
                xaxall = np.log10(xaxall)
                yayall = np.log10(yayall)

                axs[l].plot(xaxall, yayall,'o', markerfacecolor = 'None' , markersize = 2, markeredgecolor = 'black', label=galaxias[ind_value[j]] ,linestyle = 'None')
                print(galaxias[ind_value[j]])
                axs[l].legend(prop={'size': 8})


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

                    id = [idovergmc[j].count(x) for x in idovergmc[j]]
                    n_gmcs = len(idovergmc[j]) - sum([x-1 for x in id]) #num_gmcs(idovergmc[j])
                    n_hiis = len(idoverhii[j])



                    conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr,rms_tot,rms_err,dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(xaxall, yayall, alpha)

                    slope, b, r_sq = linear_regression(xaxall, yayall)
                    y_pred = b + slope * x0

                    #axs.plot(x0, y_pred, '-')
                    #sns.regplot(x,y, scatter_kws={'s':2} )
                    y_pred_sup = b+conf_b + (slope + conf_a)*x0
                    y_pred_inf = b-conf_b + (slope - conf_a)*x0

                    #axs.plot(x0, y_pred_sup, color = "lightgreen")
                    #axs.plot(x0, y_pred_inf, color = "lightgreen")

                    axs[l].plot(x0, y_pred+y_conf_sup, '--', color = 'black')
                    axs[l].plot(x0, y_pred + y_conf_inf, '--', color = 'black')

                    axs[l].plot(x0, y_pred+int_prev_y, '-.', color = 'grey')
                    axs[l].plot(x0, y_pred - int_prev_y, '-.', color = 'grey')

                    axs[l].plot(x0, y_pred, color = 'navy')

                    r2_list.append(r_sq)
                    slope_list.append(slope)

                    r_sq = np.sqrt(r_sq)

                    x0, xf = xlim[k]
                    y0, yf = ylim[i]
                    # axs.text(0.8, 0.15, 'P-Value: %6.2f' % (pvalue), fontsize=8, horizontalalignment='center',
                    #             verticalalignment='center', transform=axs.transAxes)


                    #=======
                    x = xaxall
                    y = yayall
                    #p,V = np.polyfit(x, y, 1, cov=True)




                    axs[l].text(0.85, 0.05, 'r²: %5.3f' % ((r_sq)**2), fontsize=10, horizontalalignment='center',
                                verticalalignment='center', transform=axs[l].transAxes, fontweight = "bold")

                    # axs.text(0.8, 0.19, 'Durbin-Watson stat: %6.2f' % (dw), fontsize=8, horizontalalignment='center',
                    #             verticalalignment='center', transform=axs.transAxes)

                    axs[l].text(0.85, 0.15, 'Slope %5.3f ' % (slope), fontsize=10, horizontalalignment='center',
                                verticalalignment='center', transform=axs[l].transAxes, fontweight = "bold")

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





                    axs[l].set(ylim=(y0, yf))



        axs[16].set(xlabel=labsxax[k])
        axs[17].set(xlabel=labsxax[k])
        axs[18].set(xlabel=labsxax[k])
        axs[19].set(xlabel=labsxax[k])

        save_pdf(pdf3, fig, save, show)

    pdf3.close()


    if sorting == 'stellar mass':
        sorted_prop = sorted_stellar_mass
        sorting = sorting + ' (log10(M$_{\odot}$))'
        print('ok')
    if sorting == 'sfr':
        sorting = str.upper(sorting) +' log10(M$_{\odot}$/yr)'
        sorted_prop = sorted_SFR

    sorted_prop = np.array(sorted_prop)
    slope_list = np.array(slope_list)
    r2_list = np.array(r2_list)

    sorted_prop = sorted_prop[np.array([0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])]
    slope_list = slope_list[np.array([0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])]
    r2_list = r2_list[np.array([0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])]

    plt.plot( sorted_prop,r2_list, marker = '+', markersize = 10, color = 'black')

    plt.ylabel('Correlation coefficient r²', fontsize = 30)
    plt.xlabel(sorting, fontsize = 30)
    plt.grid()

    plt.tick_params(axis="x", labelsize=25)
    plt.tick_params(axis="y", labelsize=25)

    plt.show()

    plt.plot( sorted_prop,slope_list, marker = '+', markersize = 10, color = 'black')

    plt.ylabel('Slope', fontsize = 30)
    plt.xlabel(sorting, fontsize = 30)
    plt.grid()

    plt.tick_params(axis="x", labelsize=25)
    plt.tick_params(axis="y", labelsize=25)

    plt.show()


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
    FluxCOGMCnot, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII, SigmaMol, Sigmav, COTpeak = pickle.load(
        open(dir_script_data + 'Galaxies_variables_notover_GMC%s%s.pickle' % ( namegmc, name_end), "rb"))

    shortlab = ['HIIGMCdist', 'Mco', 'GMCsize', 'Smol', 'sigmav', 'avir', 'TpeakCO', 'tauff']
    MassesCO = [1e5 * i for i in MasscoGMCover]  #



    labsyay = labsyay  # removing  vel, major axis, minor axis and PA, no need to plot them
    labsxax = labsxax

    arrayyay = GMCprop
    arrayxax = HIIprop

    # Limits in the properties of HIIR and GMCs
    xlim, ylim, xx, yy = pickle.load(open(    dir_script_data + 'limits_properties.pickle', "rb"))

    return labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak

threshold_percs = [1]#,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
vels = [50,40,30,20,10]

regions = [[1,2,3],[4,5,6],[7,8,9]]

#plotallgals(muse = 'dr2', gmc_catalog = '_native_', gmc_catalog_version= 'new', matching = "overlap_1om", randomize='', outliers = True, show =False , save = True, threshold_perc=0.1, vel=10000, gmcprop = [1])
plotallgals_galprop(sorting = 'mass', muse = 'dr2', gmc_catalog = '_native_', gmc_catalog_version= 'new', matching = "overlap_1om", randomize='', outliers = True, show =True , save = True, threshold_perc=0.9, vel=10000, gmcprop = [6])

#plot_dist_prop(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1om", outliers = True, show =True , save = True, threshold_percs = [0.1,0.4,0.7,0.9])
#hist_std(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1o1", outliers = True, show =True , save = True, threshold_perc=0.4, bin = 1000)
#plot_residus(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1o1", outliers = True, show =False , save = True, threshold_percs = [0.1,0.4,0.7,0.9])
#plot_correlations_randomized(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=True, save=True, vel=10000, threshold_percs=[0.1], randomize='gmc_prop', gmcprop=[4], random=False)
#plot_correlations(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=False, save=True, vel=10000, threshold_percs=[0.9], randomize='', gmcprop=[1,3,4,6], rgal_color=False)
#plot_correlations_regions(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=False, save=True, vel=10000, threshold_percs=[0.9], randomize='', gmcprop=[4], regions = regions, rgal_color=False)
#plot_correlations_fct_rgal(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=True, save=True, vel=10000, threshold_percs=[0.1], randomize='', gmcprop=[1], rgal_color=False)

#plot_correlations_rgal(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=False, save=True, vel=10000, threshold_percs=[0.1,0.5,0.9], randomize='', gmcprop=[0], rgal_color=False)
#plot_gmcprop_regions(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=False, save=True, vel=10000, threshold_perc=0.1, randomize='', gmcprop=[1,3,4,6], regions = regions)
#hist_std_hii_props(muse = 'dr2', gmc_catalog_version = 'new', gmc_catalog='_native_', matching= 'overlap_1om', outliers=True, randomize='', vel = 10000, show = True, save = False,threshold_percs=[0.1,0.9], gmc_props=[1], bin = 1000)

#gmcprop
# 1 : MCO
# 3 : Sigma mol
# 4 : Sigma v
# 5 : alpha vir
# 6 : CO Tpeak
# 7 : Tauff

#COMMENT  1 --> center (small bulge or nucleus)
# COMMENT  2 = bar (excluding bar ends)
# COMMENT  3 = bar ends (overlap of bar and spiral)
# COMMENT  4 = interbar (R_gal < R_bar but outside bar footprint)
# COMMENT  5 = spiral arms inside interbar (R_gal < R_bar)
# COMMENT  6 = spiral arms (R_gal > R_bar)
# COMMENT  7 = interarm (only R_gal spanned by spiral arms, and R_gal > R_bar)
# COMMENT  8 = outer disc (R_gal > spiral arm ends, only for galaxies with identif
# COMMENT  9 = disc (R_gal > R_bar) where no spiral arms were identified


