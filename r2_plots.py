import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf
from sklearn.linear_model import LinearRegression
import seaborn as sns
import os
import warnings
import matplotlib
warnings. filterwarnings("ignore")


np.set_printoptions(threshold=sys.maxsize)
sns.set(style="white", color_codes=False)
# ===================================================================================

dir_script_data = os.getcwd() + "/script_data/"
dirhii, dirgmc, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots, dirhiimasks = pickle.load(
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

def plotallgals_r2(new_muse, gmc_catalog, matching, outliers, threshold_percs, show, save, vel_limit):
    # ===============================================================

    r_sqs_thres = []
    slopes_thres = []

    print("Starting loop to create figures of all galaxies together - points")

    for threshold_perc in threshold_percs:
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii = get_data(new_muse, gmc_catalog,
                                                                                                matching, outliers,
                                                                                                threshold_perc,
                                                                                                vel_limit)
        r_sqs = []
        slopes = []
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

                slopes.append(slope)
                r_sqs.append(r_sq)

        r_sqs_thres.append(r_sqs)
        slopes_thres.append(slopes)

    pdf_name = "%sr2_vs_threshold.pdf" % (dirplots)
    pdf3 = fpdf.PdfPages(pdf_name)



    sns.set(style='white', color_codes=True)

    fig, axs = plt.subplots(2, 3,  figsize=(9, 10), dpi=80)
    fig.subplots_adjust(wspace = 0.3)


    fig.suptitle('correlation coefficient vs threshold \n overlap matching, all galaxies', fontsize=18, va='top')
    axs = axs.ravel()

    for i in range(len(labsyay)):

        if i == 0 or i == 2:
            color = "tab:blue"
        else:
            color = "tab:red"
        r_sq = [x[i] for x in r_sqs_thres]

        axs[i].tick_params(axis="x", labelsize=14)
        axs[i].tick_params(axis="y", labelsize=14)
        axs[i].xaxis.get_label().set_fontsize(20)
        axs[i].yaxis.get_label().set_fontsize(20)

        axs[i].set(ylim = (0,0.45))
        axs[i].grid()
        axs[i].set(ylabel = 'r²  ' + '(' +labsyay[i]+')')

        axs[i].set(xlabel = 'Threshold')
        axs[i].plot( threshold_percs,r_sq, color = color, marker = "+", markersize = '8')

    pdf3.savefig()
    pdf3.close()

    pdf_name = "%sslopes_vs_threshold.pdf" % (dirplots)
    pdf3 = fpdf.PdfPages(pdf_name)


    fig, axs = plt.subplots(2, 3,  figsize=(9, 10), dpi=80)
    fig.subplots_adjust(wspace = 0.3)

    fig.suptitle('slope vs threshold \n overlap matching, all galaxies', fontsize=18, va='top')
    axs = axs.ravel()

    for i in range(len(labsyay)):

        if i == 0 or i == 2:
            color = "tab:blue"
        elif i == 4:
            color = 'black'
        else:
            color = "tab:red"
        r_sq = [x[i] for x in r_sqs_thres]

        axs[i].tick_params(axis="x", labelsize=14)
        axs[i].tick_params(axis="y", labelsize=14)
        axs[i].xaxis.get_label().set_fontsize(20)
        axs[i].yaxis.get_label().set_fontsize(20)

        slope = [x[i] for x in slopes_thres]
        print(slope)
        axs[i].set(ylim = (-0.11,0.45))
        axs[i].grid()
        axs[i].set(ylabel = 'Slope  ' + '(' +labsyay[i]+')')
        axs[i].set(xlabel = 'Threshold')






        axs[i].plot( threshold_percs,slope, color = color, marker = "+", markersize = '8')

    plt.show()
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
    tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover,a = GMCprop1

    GMCprop = MasscoGMCover,  Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover

    SizepcHII, LumHacorr, sigmavHII, metaliHII, varmetHII, numGMConHII, \
    MasscoGMC, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC = pickle.load(
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

plotallgals_r2(new_muse= True, gmc_catalog="_native_",threshold_percs=threshold_percs, vel_limit=10000,outliers=True,save=False,show=True,matching="overlap_1om")