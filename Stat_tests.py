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
warnings. filterwarnings("ignore")
from matplotlib import rc
import latex
import matplotlib
#rc('text', usetex = True)


np.set_printoptions(threshold=sys.maxsize)
sns.set(style="white", color_codes=False)
# ===================================================================================

dir_script_data = os.getcwd() + "/script_data/"
dirhii, dirgmc, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots, dirhiimasks = pickle.load(
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










def plot_sigma_tpeak_thresholds(new_muse, gmc_catalog, matching, outliers,  show, save, threshold_percs, vel, gmcprop):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair
    threshold_perc  = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover = get_data(new_muse, gmc_catalog, matching,
                                                                                 outliers, threshold_perc, vel)


    print("Plots of all galaxies together")


    xlimmin, xlimmax = get_min(arrayxax)
    z=0
    print("Starting loop to create figures of all galaxies together - points")
    for threshold_perc in threshold_percs:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s%s.pdf" % (dirplots, namegmc, name_end, str(z))

        pdf3 = fpdf.PdfPages( pdf_name )
        z+=1

        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover= get_data(new_muse, gmc_catalog, matching,
                                                                                     outliers, threshold_perc, vel)

        k=1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(2, 2, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
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

            #color = "black"

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


            #for l in range(len(yaytmp)):
             #   yaytmp[l] = yaytmp[l] - np.nanmean(yaytmp[l])
              #  print(np.log10(np.nanmean(yaytmp[l])))


            #========


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



                conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr,rms_tot,rms_err,dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(xaxall, yayall, alpha)

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




                # axs.text(0.8, 0.07, 'Error RMS: %5.3f' % (rms_err), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                # axs.text(0.8, 0.15, 'Standard error of estimate: %5.3f' % (stderr), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                axs[l].text(0.8, 0.1, 'R²: %5.3f' % ((r_sq)**2), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[l].transAxes)

                # axs.text(0.8, 0.19, 'Durbin-Watson stat: %6.2f' % (dw), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                axs[l].text(0.8, 0.05, 'Slope %5.3f $\pm$ %5.3f ' % (slope, conf_a), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[l].transAxes)
                #
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
                if l == 0:
                    axs[l].legend(prop={'size': 8})





                l+=1
        #axs[l].set(xlabel=labsxax[k])
        save_pdf(pdf3, fig, save, show)

        pdf3.close()




def plot_residus(new_muse, gmc_catalog, matching, outliers,  show, save, threshold_percs):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair
    threshold_perc  = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(new_muse, gmc_catalog, matching,
                                                                                 outliers, threshold_perc)

    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")


    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    for threshold_perc in threshold_percs:
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(new_muse, gmc_catalog, matching,
                                                                                     outliers, threshold_perc)
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






def hist_std(new_muse, gmc_catalog, matching, outliers,  show, save, threshold_perc, bin):
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(new_muse, gmc_catalog,
                                                                                            matching,
                                                                                            outliers, threshold_perc)
    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold_dist%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    for threshold_perc in threshold_percs:
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(new_muse, gmc_catalog,
                                                                                                matching,
                                                                                                outliers,
                                                                                                threshold_perc)
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




def plot_dist_prop(new_muse, gmc_catalog, matching, outliers,  show, save, threshold_percs):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    threshold_perc  = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(new_muse, gmc_catalog, matching,
                                                                                 outliers, threshold_perc)

    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold_dist%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")


    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    for threshold_perc in threshold_percs:
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(new_muse, gmc_catalog, matching,
                                                                                     outliers, threshold_perc)
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

def plot_abs_dist_prop(new_muse, gmc_catalog, matching, outliers,  show, save, threshold_percs):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    threshold_perc  = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(new_muse, gmc_catalog, matching,
                                                                                 outliers, threshold_perc)

    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold_dist%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")


    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    for threshold_perc in threshold_percs:
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc,LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover = get_data(new_muse, gmc_catalog, matching,
                                                                                     outliers, threshold_perc)
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




def plot_gmcprop_regions(new_muse, gmc_catalog, matching, outliers,  show, save, threshold_perc, vel, gmcprop, regions):


    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair

    region_name = ['Center','Arms','Disc']

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover = get_data(new_muse, gmc_catalog, matching,
                                                                                 outliers, threshold_perc, vel)
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

        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover= get_data(new_muse, gmc_catalog, matching,
                                                                                     outliers, threshold_perc, vel)

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


        k=1
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(1, 1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s \n %s' % (name_end, region_name[reg]), fontsize=18, va='top')
        #axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0
        for i in gmcprop:

            axs.set(ylabel=labsyay[i])
            axs.grid()
            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            #if k < 5:
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)

            axs.plot(xaxall, yayall,'o', markerfacecolor = 'None' , markersize = 2, markeredgecolor = 'black', label='threshold = %f' % threshold_perc ,linestyle = 'None')

            axs.set_xlim(xlimmin, xlimmax)

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
                print(y)
                p,V = np.polyfit(x, y, 1, cov=True)
                print(p[0])
                print(np.sqrt(V[0][0]))
                print('\n')
                print(p[1])
                print(np.sqrt(V[1][1]))



                axs.text(0.8, 0.03, 'R²: %5.3f' % ((r_sq)**2), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs.transAxes)

                # axs.text(0.8, 0.19, 'Durbin-Watson stat: %6.2f' % (dw), fontsize=8, horizontalalignment='center',
                #             verticalalignment='center', transform=axs.transAxes)

                axs.text(0.15, 0.03, 'Slope %5.3f $\pm$ %5.3f ' % (slope, conf_a), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs.transAxes)

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
                axs.legend(prop={'size': 8}, loc = 2)





                l+=1
        axs.set(xlabel=labsxax[k])
        #axs[2].set(xlabel=labsxax[k])

        save_pdf(pdf3, fig, save, show)
        reg +=1

    pdf3.close()


def plotallgals(new_muse, gmc_catalog, matching, outliers,  show, save, threshold_perc, vel, gmcprop):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
    alpha = 0.05

    # Plots of correlations with dots for each pair
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover = get_data(new_muse, gmc_catalog, matching,
                                                                                 outliers, threshold_perc, vel)
    if save == True:
        pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")


    xlimmin, xlimmax = get_min(arrayxax)

    print("Starting loop to create figures of all galaxies together - points")
    for j in range(len(galaxias)):
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot, FluxCOGMCover, regionindexGMCover= get_data(new_muse, gmc_catalog, matching,
                                                                                     outliers, threshold_perc, vel)

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


def get_data(new_muse, gmc_catalog, matching, outliers, threshold_perc, vel):
    # ==============================================================================#

    typegmc = gmc_catalog  # '_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_

    without_out = not outliers
    name_end = name(matching, without_out, new_muse, gmc_catalog, threshold_perc, vel_limit=vel)
    # ==============================================================================#

    namegmc = "_12m+7m+tp_co21%sprops" % typegmc

    # ====================================================================================================================#

    # =========================Getting all the GMC and HII properties from the pickle files===============================#


    galaxias, GMCprop1, HIIprop1, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay, idoverhii, idovergmc = pickle.load(
        open(dir_script_data + 'Galaxies_variables_GMC%s%s.pickle' % ( namegmc, name_end), "rb"))  # retrieving the regions properties

    SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, \
    velHIIover, HIIminorover, HIImajorover, HIIangleover = HIIprop1

    HIIprop = SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover

    DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, \
    tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover, FluxCOGMCover = GMCprop1

    GMCprop = DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover

    SizepcHII, LumHacorrnot, sigmavHII, metaliHII, varmetHII, numGMConHII, \
    FluxCOGMCnot, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC = pickle.load(
        open(dir_script_data + 'Galaxies_variables_notover_GMC%s%s.pickle' % ( namegmc, name_end), "rb"))

    shortlab = ['HIIGMCdist', 'Mco', 'GMCsize', 'Smol', 'sigmav', 'avir', 'TpeakCO', 'tauff']
    MassesCO = [1e5 * i for i in MasscoGMCover]  #

    labsyay = labsyay[0:len(labsyay) - 5]  # removing  vel, major axis, minor axis and PA, no need to plot them
    labsxax = labsxax[0:len(labsxax) - 4]

    arrayyay = GMCprop
    arrayxax = HIIprop

    # Limits in the properties of HIIR and GMCs
    xlim, ylim, xx, yy = pickle.load(open(    dir_script_data + 'limits_properties.pickle', "rb"))

    return labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover

threshold_percs = [1]#,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
vels = [50,40,30,20,10]

regions = [[1,2,3],[4,5,6],[7,8,9]]

#plot_all_thresholds(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1om", outliers = True, threshold_perc = 0.1, show = True, save = False, threshold_percs = threshold_percs)
#plotallgals(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1om", outliers = True, show =False , save = True, threshold_perc=0.1, vel=10000, gmcprop = [1,3,4,6])
#plot_dist_prop(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1om", outliers = True, show =True , save = True, threshold_percs = [0.1,0.4,0.7,0.9])
#plot_threshold_Sigma_Tpeak(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1om", outliers = True, show = False, save = True, threshold_percs = [0.1,0.4,0.7,0.9])
#plot_pairs(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1om", outliers = True, show = False, save = True, threshold_perc = 0.7)
#hist_std(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1o1", outliers = True, show =True , save = True, threshold_perc=0.4, bin = 1000)
#plot_sigma_tpeak_vels(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1om", outliers = True, show =False , save = True, threshold_perc=0.1, vels=vels)
#plot_residus(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1o1", outliers = True, show =False , save = True, threshold_percs = [0.1,0.4,0.7,0.9])
#plot_sigma_tpeak_thresholds(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1om", outliers = True, show =True, save = True, threshold_percs=threshold_percs, vel=10000, gmcprop = [1,3,4,6])
plot_gmcprop_regions(new_muse = True, gmc_catalog = '_150pc_homogenized_', matching = "overlap_1om", outliers = True, show =False , save = True, threshold_perc=0.9, vel=10000, gmcprop=[1], regions = regions)


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


# def plot_sigma_tpeak_vels(new_muse, gmc_catalog, matching, outliers,  show, save, threshold_perc, vels):
#     # ===============================================================
#     xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))
#     alpha = 0.05
#
#     # Plots of correlations with dots for each pair
#     vel = vels[0]
#     labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover = get_data(new_muse, gmc_catalog, matching,
#                                                                                  outliers, threshold_perc, vel)
#     if save == True:
#         pdf_name = "%sCorrelations_HA_Sigma_Tpeak_threshold%s%s.pdf" % (dirplots, namegmc, name_end)
#
#         pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
#     else:
#         pdf3 = fpdf.PdfPages("blank")
#
#     print("Plots of all galaxies together")
#
#
#     xlimmin, xlimmax = get_min(arrayxax)
#
#     print("Starting loop to create figures of all galaxies together - points")
#     for vel in vels:
#         labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover = get_data(new_muse, gmc_catalog, matching,
#                                                                                      outliers, threshold_perc, vel)
#
#         k=1
#         sns.set(style='white', color_codes=True)
#         fig, axs = plt.subplots(1, 1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
#         plt.subplots_adjust(wspace=0.3)
#         fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s' % name_end, fontsize=18, va='top')
#         #axs = axs.ravel()
#         # Galactic distance vs: Mco, avir, sigmav,Sigmamol
#         l = 0
#         for i in [4]:
#
#             axs.set(ylabel=labsyay[i])
#             axs.grid()
#             yaytmp = arrayyay[i]
#             xaxtmp = arrayxax[k]
#             xaxall = np.concatenate([f.tolist() for f in xaxtmp])
#             yayall = np.concatenate([f.tolist() for f in yaytmp])
#             #if k < 5:
#             xaxall = np.log10(xaxall)
#             yayall = np.log10(yayall)
#
#             axs.plot(xaxall, yayall,'o', markerfacecolor = 'None' , markersize = 2, markeredgecolor = 'black', label='threshold = %f' % threshold_perc ,linestyle = 'None')
#
#             axs.set_xlim(xlimmin, xlimmax)
#
#             idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
#             xaxall = xaxall[idok]
#             yayall = yayall[idok]
#             lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
#             lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
#             indlim = np.where((xaxall < lim2) & (xaxall > lim1))
#             xaxall = xaxall[indlim]
#             yayall = yayall[indlim]
#             if xaxall.any() != 0 and yayall.any != ():
#                 xmin = np.amin(xaxall)
#                 xmax = np.amax(xaxall)
#                 #x = xaxall.reshape((-1, 1))
#                 #y = yayall
#
#                 n_gmcs = num_gmcs(idovergmc)
#                 n_hiis = num_hiis(idoverhii)
#
#                 conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr,rms_tot,rms_err,dw, mean_error = conf_intervals(xaxall, yayall, alpha)
#
#                 slope, b, r_sq = linear_regression(xaxall, yayall)
#                 y_pred = b + slope * x0
#
#                 #axs.plot(x0, y_pred, '-')
#                 #sns.regplot(x,y, scatter_kws={'s':2} )
#                 y_pred_sup = b+conf_b + (slope + conf_a)*x0
#                 y_pred_inf = b-conf_b + (slope - conf_a)*x0
#
#                 #axs.plot(x0, y_pred_sup, color = "lightgreen")
#                 #axs.plot(x0, y_pred_inf, color = "lightgreen")
#
#                 axs.plot(x0, y_pred+y_conf_sup, '--', color = 'black', label='confidence interval')
#                 axs.plot(x0, y_pred + y_conf_inf, '--', color = 'black')
#
#                 axs.plot(x0, y_pred+int_prev_y, '-.', color = 'grey', label = 'prediction interval')
#                 axs.plot(x0, y_pred - int_prev_y, '-.', color = 'grey')
#
#                 axs.plot(x0, y_pred, color = 'navy', label = 'linear regression')
#
#                 x0, xf = xlim[k]
#                 y0, yf = ylim[i]
#                 axs.text(0.8, 0.15, 'P-Value: %6.2f' % (pvalue), fontsize=8, horizontalalignment='center',
#                             verticalalignment='center', transform=axs.transAxes)
#
#                 axs.text(0.8, 0.07, 'Error RMS: %6.2f' % (rms_err), fontsize=8, horizontalalignment='center',
#                             verticalalignment='center', transform=axs.transAxes)
#
#                 axs.text(0.8, 0.11, 'Standard error of estimate: %6.2f' % (stderr), fontsize=8, horizontalalignment='center',
#                             verticalalignment='center', transform=axs.transAxes)
#
#                 axs.text(0.8, 0.03, 'R^2: %6.2f' % (r_sq), fontsize=8, horizontalalignment='center',
#                             verticalalignment='center', transform=axs.transAxes)
#
#                 axs.text(0.8, 0.19, 'Durbin-Watson stat: %5.2f' % (dw), fontsize=8, horizontalalignment='center',
#                             verticalalignment='center', transform=axs.transAxes)
#
#                 axs.text(0.15, 0.05, 'Slope %5.2f $\pm$ %5.2f ' % (slope, conf_a), fontsize=8, horizontalalignment='center',
#                             verticalalignment='center', transform=axs.transAxes)
#
#                 axs.text(0.8, 0.23, 'Mean Error %5.2f ' % (mean_error), fontsize=8, horizontalalignment='center',
#                             verticalalignment='center', transform=axs.transAxes)
#
#                 axs.text(0.8, 0.27, 'paired gmcs %5.2f ' % (n_gmcs), fontsize=8, horizontalalignment='center',
#                          verticalalignment='center', transform=axs.transAxes)
#
#                 axs.text(0.8, 0.30, 'paired hii regions %5.2f ' % (n_hiis), fontsize=8, horizontalalignment='center',
#                          verticalalignment='center', transform=axs.transAxes)
#
#                 axs.set(ylim=(y0, yf))
#                 axs.legend(prop={'size': 8})
#
#
#
#
#
#                 l+=1
#         axs.set(xlabel=labsxax[k])
#         save_pdf(pdf3, fig, save, show)
#
#     pdf3.close()