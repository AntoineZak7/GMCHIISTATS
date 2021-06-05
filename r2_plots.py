import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf
import scipy.stats
import statsmodels.stats.stattools
from sklearn.linear_model import LinearRegression
import seaborn as sns
import os
import warnings
import matplotlib
warnings. filterwarnings("ignore")
plt.style.use('science')




# ===================================================================================

dir_script_data = os.getcwd() + "/script_data_dr2/"
dirhii_dr1,dirhii_dr2, dirgmc_old,dirgmc_new, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots, dirhiimasks, dirgmcmasks, sample_table_dir = pickle.load(
    open(dir_script_data + 'Directories_muse.pickle', "rb"))  # retrieving the directories paths


def checknaninf(v1, v2, lim1, lim2):
    v1n = np.array(v1)
    v2n = np.array(v2)
    indok = np.where((np.absolute(v1n) < lim1) & (np.absolute(v2n) < lim2))[0].tolist()
    # print indok
    nv1n = v1n[indok].tolist()
    nv2n = v2n[indok].tolist()
    return nv1n, nv2n

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


def name(matching, without_out, muse, gmc_catalog, gmc_catalog_version, threshold_perc, vel_limit, randomize,
         symmetrical):
    name_end = 'muse:' + muse + '_' + 'gmc:' + gmc_catalog + '(' + gmc_catalog_version + ')_' + 'vel_limit:' + str(
        vel_limit) + '_matching:' + matching + '_' + randomize + '_' + symmetrical

    if matching != "distance":
        name_end = name_end + '(' + str(threshold_perc).split(sep='.')[0] + str(threshold_perc).split(sep='.')[1] + ')'
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
    vx = np.var(x)
    vy = np.var(y)
    r = cov / (np.sqrt(vx * vy))

    slope = cov / vx
    intercept = ym - slope * xm #intercept

    Sxx = sum([(xi - xm)**2 for xi in x])
    Syy = sum([(yi - ym)**2 for yi in y])
    Err = y - (x * slope + intercept)
    SSe = sum([err**2 for err in Err]) #Sum squared errors
    SSt = Syy
    MSt = SSt/(n-2)
    SSr = sum([(xi * slope + intercept - ym) ** 2 for xi in x])
    sigma_sq = SSe / (n-2)
    MSe = sigma_sq
    MSr = SSr

    Syx2 = SSe/(n-2)

    x0 = np.sort(x)

    k= n-2
    ta = scipy.stats.t.ppf(1-alpha/2, df = k)



    Sm2 = Syx2/Sxx # Variance of slope
    Sm = np.sqrt(Sm2) # Standard deviation of slope

    Sb2 = Syx2*(1/n+(xm**2)/Sxx)
    Sb = np.sqrt(Sb2)

    #slope and intercept confidnce interval

    conf_a = Sm*ta
    conf_b = Sb*ta

    #=====intervalle confiance droite régression=====#

    y_conf_sup = ta * np.sqrt(MSe*(1/n+ np.array([(x - xm)**2 for x in x0])/Sxx))
    y_conf_inf = -ta * np.sqrt(MSe*(1/n+ [(x - xm)**2 for x in x0]/Sxx))

    #========intervalles prévision de y en x0========#

    int_prev_y = ta*np.sqrt(MSe*(1+1/n+ [(x - xm)**2 for x in x0]/Sxx   )) #vector

    #=======Tests H1================#

    dw = statsmodels.stats.stattools.durbin_watson(Err)
    rms_err = np.std(np.array(Err))
    rms_tot = np.std(y)

    #=======test chi2==============#

    mybin = 20
    xbin, ybin, ebin, nbin = bindata(x,y,mybin= mybin)
    y_prev = intercept + slope * x
    xbin_prev, ybin_prev,ebin_prev,nbin_prev = bindata(x,y_prev, mybin = mybin)

    bin_err = ybin-(xbin * slope + intercept)
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

    ebin = ebin[np.where(ebin !=0)[0]]

    chi2list = np.nan_to_num(np.array([(bin_erri**2)/sigmai**2 for bin_erri, sigmai in zip(bin_err,ebin) ]))

    slope1, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(x,y)
    mean_error = np.nanmean(abs(Err))

    return conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr,rms_tot,rms_err,dw, mean_error, SSt, SSe, MSe, MSt


def plotallgals_r2(muse, gmc_catalog, matching, outliers, threshold_percs, show, save, vel_limit, symmetrical, randomize, gmc_catalog_version):
    # ===============================================================

    r_sqs_thres = []
    slopes_thres = []
    slopes_confs = []

    print("Starting loop to create figures of all galaxies together - points")

    for threshold_perc in threshold_percs:
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii = get_data(muse, gmc_catalog,
                                                                                                matching, outliers,
                                                                                                threshold_perc,
                                                                                                vel_limit, symmetrical=symmetrical, randomize = randomize, gmc_catalog_version=gmc_catalog_version)
        r_sqs = []
        slopes = []
        slopes_conf = []
        k=1

        for i in range(len(labsyay)):
            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)
            idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
            xaxall = xaxall[idok]
            yayall = yayall[idok]

            if xaxall.any() != 0 and yayall.any != ():
                x = xaxall.reshape((-1, 1))
                y = yayall
                model = LinearRegression().fit(x, y)
                r_sq = model.score(x, y)
                slope = model.coef_

                conf_a, conf_b, int_prev_y, x0, y_conf_sup, y_conf_inf, pvalue, stderr, rms_tot, rms_err, dw, mean_error, SSt, SSe, MSe, MSt = conf_intervals(xaxall, yayall,0.05)

                slopes.append(slope)
                slopes_conf.append(conf_a)
                r_sqs.append(r_sq)

        r_sqs_thres.append(r_sqs)
        slopes_thres.append(slopes)
        slopes_confs.append(slopes_conf)

    pdf_name = "%sr2_vs_threshold.pdf" % (dirplots)
    pdf3 = fpdf.PdfPages(pdf_name)




    fig, axs = plt.subplots(2, 3,  figsize=(8.5,5))
    fig.subplots_adjust(wspace = 0.35)
    fig.subplots_adjust(hspace = 0.3)


    #fig.suptitle('correlation coefficient vs threshold \n overlap matching, all galaxies', fontsize=18, va='top')
    axs = axs.ravel()

    for i in range(len(labsyay)):

        if i == 0 or i == 2:
            color = "tab:blue"
        elif i == 4:
            color = 'black'
        else:
            color = "tab:red"
        r_sq = [x[i] for x in r_sqs_thres]

        axs[i].tick_params(axis="x", labelsize=11)
        axs[i].tick_params(axis="y", labelsize=11)
        # axs[i].xaxis.get_label().set_fontsize(20)
        # axs[i].yaxis.get_label().set_fontsize(20)

        axs[i].set(ylim = (0,0.45))
        #axs[i].set(ylabel = labsyay[i])

        axs[i].set(xlabel = 'Minimal Overlap Fraction')
        # axs[4].set(xlabel = 'Minimal Overlap Fraction')
        # axs[5].set(xlabel = 'Minimal Overlap Fraction')


        axs[i].plot( threshold_percs,r_sq, color = color, marker = "o", markersize = '4',markerfacecolor = 'None', markeredgewidth = 0.5)

    pdf3.savefig()
    pdf3.close()
    print(dirplots)
    pdf_name = "%sslopes_vs_threshold.pdf" % (dirplots)
    pdf3 = fpdf.PdfPages(pdf_name)


    fig, axs = plt.subplots(2, 3,  figsize=(8.5,5))
    fig.subplots_adjust(wspace = 0.35)
    fig.subplots_adjust(hspace = 0.3)


    #fig.suptitle('slope vs threshold \n overlap matching, all galaxies', fontsize=18, va='top')
    axs = axs.ravel()

    for i in range(len(labsyay)):

        if i == 0 or i == 2:
            color = "tab:blue"
        elif i == 4:
            color = 'black'
        else:
            color = "tab:red"
        r_sq = [x[i] for x in r_sqs_thres]

        axs[i].tick_params(axis="x", labelsize=11)
        axs[i].tick_params(axis="y", labelsize=11)
        # axs[i].xaxis.get_label().set_fontsize(9)
        # axs[i].yaxis.get_label().set_fontsize(9)

        slope_conff = [x[i] for x in slopes_confs]
        slope = [x[i] for x in slopes_thres]
        print(slope)
        axs[i].set(ylim = (np.mean(slope)-0.28,np.mean(slope) + 0.28))
        #axs[i].set(ylabel = labsyay[i])
        axs[i].set(xlabel = 'Minimal Overlap Fraction')



        axs[i].errorbar( threshold_percs,slope, slope_conff, color = color, capsize = 1.5, capthick = 0.5, elinewidth = 0.5)

    pdf3.savefig()
    pdf3.close()





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





def clean_data(xarray, yarray):
    xaxall = xarray
    yayall = yarray


    idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
    xaxall = xaxall[idok];
    yayall = yayall[idok]
    lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
    lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
    indlim = np.where((xaxall < lim2) & (xaxall > lim1))
    xaxall = xaxall[indlim]
    yayall = yayall[indlim]

    return xaxall, yayall




def get_data(muse, gmc_catalog, matching, outliers, threshold_perc, vel_limit, symmetrical, gmc_catalog_version, randomize):
    # ==============================================================================#

    typegmc = gmc_catalog  # '_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_

    without_out = not outliers
    name_end = name(matching, without_out, muse, gmc_catalog, gmc_catalog_version, threshold_perc, vel_limit, randomize,
         symmetrical)
    # ==============================================================================#

    namegmc = "_12m+7m+tp_co21%sprops" % typegmc

    # ====================================================================================================================#

    # =========================Getting all the GMC and HII properties from the pickle files===============================#


    galaxias, GMCprop1, HIIprop1, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay, idoverhii, idovergmc, labsyay1, labsxax1 = pickle.load(
        open(dir_script_data + 'Galaxies_variables_GMC%s%s.pickle' % ( namegmc, name_end),
             "rb"))  # retrieving the regions properties

    SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, \
    velHIIover, HIIminorover, HIImajorover, HIIangleover ,a= HIIprop1

    HIIprop = SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover

    DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, \
    tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover,a,b = GMCprop1

    GMCprop = MasscoGMCover,  Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover

    SizepcHII, LumHacorr, sigmavHII, metaliHII, varmetHII, numGMConHII, \
    MasscoGMC, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC,a,b,c,d,e,f,g,h= pickle.load(
        open(dir_script_data + 'Galaxies_variables_notover_GMC%s%s.pickle' % ( namegmc, name_end), "rb"))

    shortlab = ['HIIGMCdist', 'Mco', 'GMCsize', 'Smol', 'sigmav', 'avir', 'TpeakCO', 'tauff']
    MassesCO = [1e5 * i for i in MasscoGMCover]  #

    labsyay = [r'log(M$_{\rm CO}$) [10$^5$ M$_{\odot}$]',
               r'log($\Sigma_{\rm mol}$)', r'log($\sigma_{\rm v}$) [km s$^{-1}$]', r'log($\alpha_{vir}$)',
               r'log(CO $T_{\rm peak}$ [K])', r'log($\tau_{\rm ff}$) [yr]']  # removing  vel, major axis, minor axis and PA, no need to plot them
    labsxax = labsxax[0:len(labsxax) - 4]

    arrayyay = GMCprop
    arrayxax = HIIprop

    # Limits in the properties of HIIR and GMCs
    xlim, ylim, xx, yy = pickle.load(open(    dir_script_data + 'limits_properties.pickle', "rb"))

    return labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii

threshold_percs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

plotallgals_r2(muse= 'dr2', gmc_catalog="_native_", gmc_catalog_version = 'new',threshold_percs=threshold_percs, vel_limit=100,outliers=True,save=True,show=True,matching="overlap_1om", symmetrical = '', randomize = '')