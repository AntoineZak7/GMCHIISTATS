import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf
from sklearn.linear_model import LinearRegression
import seaborn as sns
import os
import warnings
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

def plotallgals(new_muse, gmc_catalog, matching, outliers, threshold_perc, show, save, vel_limit):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data+'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii = get_data(new_muse, gmc_catalog, matching, outliers, threshold_perc, vel_limit)

    if save == True:
        pdf_name = "%sCorrelations_allgals%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    marker_style = dict(markersize=4)
    # xticks1 = np.arange(8.4,9,0.2)
    xticks5 = [8.3, 8.4, 8.5, 8.6, 8.7]

    print("Starting loop to create figures of all galaxies together - points")
    for k in range(len(arrayxax)):
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(4, 2, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s' %name_end, fontsize=18, va='top')
        axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        for i in range(len(labsyay)):
            for j in range(len(galaxias)):
                xax2 = [h for h in arrayxax[k][j]]
                yay2 = [h for h in arrayyay[i][j]]
                #if k < 5:
                xax2 = np.log10(xax2)
                yay2 = np.log10(yay2)
                axs[i].plot(xax2, yay2, '8', label='%s' % galaxias[j], alpha=0.7, **marker_style)
            axs[i].set(ylabel=labsyay[i])
            axs[i].grid()
            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            #if k < 5:
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)
            idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
            xaxall = xaxall[idok];
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
                model = LinearRegression().fit(x, y)
                r_sq = model.score(x, y)
                y_pred = model.intercept_ + model.coef_ * x.ravel()
                slope = model.coef_

                slope,b,r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope*x.ravel()
                axs[i].plot(xaxall, y_pred, '-')
                x0, xf = xlim[k]
                y0, yf = ylim[i]
                axs[i].text(0.8, 0.05, 'R^2: %6.2f' % (r_sq), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[i].transAxes)

                axs[i].text(0.15, 0.05, 'Slope %5.2f' % (slope), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[i].transAxes)

                axs[i].set(ylim=(y0, yf))
                axs[0].legend(prop={'size': 8})
        axs[6].set(xlabel=labsxax[k])
        axs[7].set(xlabel=labsxax[k])
        save_pdf(pdf3, fig, save, show)

    pdf3.close()





def plot_pairs(new_muse, gmc_catalog, matching, outliers, threshold_perc, show, save):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data+'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii = get_data(new_muse, gmc_catalog, matching, outliers, threshold_perc)



    idoverhiis = [item for sublist in idoverhii for item in sublist]




    if save == True:
        pdf_name = "%splot_correlations_pairs%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    marker_style = dict(markersize=4)
    # xticks1 = np.arange(8.4,9,0.2)
    xticks5 = [8.3, 8.4, 8.5, 8.6, 8.7]

    print("Starting loop to create figures of all galaxies together - points")
    for k in range(len(arrayxax)):
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(4, 2, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s' %name_end, fontsize=18, va='top')
        axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        for i in range(len(labsyay)):
            # for j in range(len(galaxias)):
            #     xax2 = [h for h in arrayxax[k][j]]
            #     yay2 = [h for h in arrayyay[i][j]]
            #     #if k < 5:
            #     xax2 = np.log10(xax2)
            #     yay2 = np.log10(yay2)

            axs[i].set(ylabel=labsyay[i])
            axs[i].grid()
            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
            xaxall = [item for sublist in xaxtmp for item in sublist]#np.concatenate([f.tolist() for f in xaxtmp])
            yayall = [item for sublist in yaytmp for item in sublist]#np.concatenate([f.tolist() for f in yaytmp])
            #if k < 5:
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)

            for id in idoverhiis:
                duplicates, xvalues, yvalues = dup_lists(xaxall, yayall, id, idoverhiis)
                symbol = ['o','s','^', 'P','*','h','X']
                axs[i].plot(xvalues, yvalues, '8', alpha=0.7, marker = symbol[duplicates-1], color = 'tab:blue')

            idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
            xaxall = xaxall[idok];
            yayall = yayall[idok]
            lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
            lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
            indlim = np.where((xaxall < lim2) & (xaxall > lim1))
            xaxall = xaxall[indlim];
            yayall = yayall[indlim]
            if xaxall.any() != 0 and yayall.any != ():
                xmin = np.amin(xaxall)
                xmax = np.amax(xaxall)
                xprang = (xmax - xmin) * 0.1
                x = xaxall.reshape((-1, 1))
                y = yayall
                model = LinearRegression().fit(x, y)
                r_sq = model.score(x, y)
                y_pred = model.intercept_ + model.coef_ * x.ravel()
                slope = model.coef_

                slope,b,r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope*x.ravel()
                axs[i].plot(xaxall, y_pred, '-')
                x0, xf = xlim[k]
                y0, yf = ylim[i]
                axs[i].text(0.8, 0.05, 'R^2: %6.2f' % (r_sq), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[i].transAxes)

                axs[i].text(0.15, 0.05, 'Slope %5.2f' % (slope), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[i].transAxes)

                axs[i].set(ylim=(y0, yf))
                axs[0].legend(prop={'size': 8})
        axs[6].set(xlabel=labsxax[k])
        axs[7].set(xlabel=labsxax[k])
        save_pdf(pdf3, fig, save, show)

    pdf3.close()

def get_min(arrayxax):
    xaxtmp = arrayxax[1]
    yayall = np.concatenate([f.tolist() for f in xaxtmp])
    yayall = np.log10(yayall)
    xlimmin=(np.nanmin(yayall) - 0.2)
    xlimmax=(np.nanmax(yayall) + 0.2)

    return xlimmin, xlimmax


def plot_sigma_tpeak_thresholds(new_muse, gmc_catalog, matching, outliers,  show, save, threshold_percs):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    threshold_perc  = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii = get_data(new_muse, gmc_catalog, matching,
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
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii = get_data(new_muse, gmc_catalog, matching,
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
            # for j in range(len(galaxias)):
            #     xax2 = [h for h in arrayxax[k][j]]
            #     yay2 = [h for h in arrayyay[i][j]]
            #     #if k < 5:
            #     xax2 = np.log10(xax2)
            #     yay2 = np.log10(yay2)

            axs[l].set(ylabel=labsyay[i])
            axs[l].grid()
            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            #if k < 5:
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)

            # axs[l].plot(xaxall, yayall, '8', label='threshold = %f' % threshold_perc, alpha=0.7, markersize = 2)
            axs[l].set_xlim(xlimmin, xlimmax)



            for id in idoverhiis:
                duplicates, xvalues, yvalues = dup_lists(xaxall, yayall, id, idoverhiis)
                symbol = ['o','s','^', 'P','*','h','X']
                RdBu = plt.get_cmap('inferno')

                axs[l].plot(xvalues, yvalues, '8', alpha=0.7, marker = symbol[duplicates-1], markersize = 2.5) #, color = RdBu((1-(duplicates/10))**2)
            #     axs[l].set_xlim(xlimmin, xlimmax)

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
                model = LinearRegression().fit(x, y)
                r_sq = model.score(x, y)
                y_pred = model.intercept_ + model.coef_ * x.ravel()
                slope = model.coef_

                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * x.ravel()
                axs[l].plot(xaxall, y_pred, '-')
                x0, xf = xlim[k]
                y0, yf = ylim[i]
                axs[l].text(0.8, 0.05, 'R^2: %6.2f' % (r_sq), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[l].transAxes)

                axs[l].text(0.15, 0.05, 'Slope %5.2f' % (slope), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[l].transAxes)

                axs[l].set(ylim=(y0, yf))
                axs[l].legend(prop={'size': 8})
                l+=1
        axs[2].set(xlabel=labsxax[k])
        save_pdf(pdf3, fig, save, show)

    pdf3.close()


def plot_sigma_tpeak_thresholds_dist(new_muse, gmc_catalog, matching, outliers,  show, save, threshold_percs):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    threshold_perc  = threshold_percs[0]
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii = get_data(new_muse, gmc_catalog, matching,
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
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii = get_data(new_muse, gmc_catalog, matching,
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



            for id in idoverhiis:
                duplicates, xvalues, yvalues = dup_lists(xaxall, yayall, id, idoverhiis)
                symbol = ['o','s','^', 'P','*','h','X']
                RdBu = plt.get_cmap('inferno')

                axs[l].plot(xvalues, yvalues, '8', alpha=0.7,color = RdBu((1-(duplicates/10))**2), marker = symbol[duplicates-1], markersize = 2.5) #,
                axs[l].set_xlim(xlimmin, xlimmax)

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
                model = LinearRegression().fit(x, y)
                r_sq = model.score(x, y)
                y_pred = model.intercept_ + model.coef_ * x.ravel()
                slope = model.coef_

                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * x.ravel()
                axs[l].plot(xaxall, y_pred, '-')
                x0, xf = xlim[k]
                y0, yf = ylim[i]

                yayall_dist = abs(yayall - (b + slope * xaxall))
                #axs[l].plot(xaxall, yayall_dist, '8', label='threshold = %f' % threshold_perc, alpha=0.7, markersize=2)



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


def plot_threshold_Sigma_Tpeak(new_muse, gmc_catalog, matching, outliers,  show, save, threshold_percs):
    # ===============================================================
    xlim, ylim, xx, yy = pickle.load(open(dir_script_data + 'limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair
    threshold_perc  = 0.1
    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii = get_data(new_muse, gmc_catalog, matching,
                                                                                 outliers, threshold_perc)

    if save == True:
        pdf_name = "%sCorrelations_threshold_HA_Sigma_Tpeak%s%s.pdf" % (dirplots, namegmc, name_end)

        pdf3 = fpdf.PdfPages(pdf_name)  # type: PdfPages
    else:
        pdf3 = fpdf.PdfPages("blank")

    print("Plots of all galaxies together")

    marker_style = dict(markersize=4)

    print("Starting loop to create figures of all galaxies together - points")
    for i in [1,3,6]:
        k=2
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(len(threshold_percs),1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s' % name_end, fontsize=18, va='top')
        axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        l = 0


        for threshold_perc in threshold_percs:
            k=1
            labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii = get_data(new_muse, gmc_catalog,
                                                                                         matching,
                                                                                         outliers, threshold_perc)
            idoverhiis = [item for sublist in idoverhii for item in sublist]

            # for j in range(len(galaxias)):
            #     xax2 = [h for h in arrayxax[k][j]]
            #     yay2 = [h for h in arrayyay[i][j]]
            #     # if k < 5:
            #     xax2 = np.log10(xax2)
            #     yay2 = np.log10(yay2)
            #     axs[l].plot(xax2, yay2, '8',  alpha=0.7, **marker_style)

            yaytmp = arrayyay[i]
            xaxtmp = arrayxax[k]
            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            # if k < 5:
            xaxall = np.log10(xaxall)
            yayall = np.log10(yayall)
            # axs[l].plot(xaxall, yayall, '8', alpha = 0.7, label='threshold = %f' % threshold_perc, markersize = 2)
            for id in idoverhiis:
                duplicates, xvalues, yvalues = dup_lists(xaxall, yayall, id, idoverhiis)
                symbol = ['o','s','^', 'P','*','h','X']
                RdBu = plt.get_cmap('inferno')

                axs[l].plot(xvalues, yvalues, '8', alpha=0.7,color = RdBu((1-(duplicates/10))**2), marker = symbol[duplicates-1],  markersize = 2.5) #color = RdBu((1-(duplicates/10))**2),


            axs[l].set(ylabel=labsyay[i])
            axs[l].grid()
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
                model = LinearRegression().fit(x, y)
                r_sq = model.score(x, y)
                y_pred = model.intercept_ + model.coef_ * x.ravel()
                slope = model.coef_

                slope, b, r_sq = linear_regression(xaxall, yayall)
                y_pred = b + slope * x.ravel()
                axs[l].plot(xaxall, y_pred, '-', color = 'navy')
                x0, xf = xlim[k]
                y0, yf = ylim[i]
                axs[l].text(0.8, 0.05, 'R^2: %6.2f' % (r_sq), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[l].transAxes)

                axs[l].text(0.15, 0.05, 'Slope %5.2f' % (slope), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[l].transAxes)

                axs[l].set(ylim=(y0, yf))
                axs[l].legend() #fix legend label
                l+=1

        axs[len(threshold_percs)-1].set(xlabel=labsxax[k])
        save_pdf(pdf3, fig, save, show)

    pdf3.close()






def plot_all_thresholds(new_muse, gmc_catalog, matching, outliers, threshold_perc, show, save, threshold_percs):

    xlim, ylim, xx, yy = pickle.load(open('limits_properties.pickle', "rb"))

    # Plots of correlations with dots for each pair

    labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii = get_data(new_muse, gmc_catalog, matching,
                                                                                 outliers, threshold_perc)
    arrayyay_tot = [ [[]] * len(threshold_percs) for i in range(len(arrayyay)) ]
    arrayxax_tot = [ [[]] * len(threshold_percs) for i in range(len(arrayxax)) ]

    i = 0
    for threshold_perc1 in threshold_percs:
        labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii = get_data(new_muse, gmc_catalog, matching,
                                                                                     outliers, threshold_perc1)

        for k in range(len(arrayxax)):
            xaxtmp = arrayxax[k]
            xaxall = np.concatenate([f.tolist() for f in xaxtmp])
            arrayxax_tot[k][i] = xaxall
        for k in range(len(arrayyay)):
            yaytmp = arrayyay[k]
            yayall = np.concatenate([f.tolist() for f in yaytmp])
            arrayyay_tot[k][i] = yayall
        print(np.shape(arrayxax_tot))

        i += 1

    for k in range(len(arrayxax)):
        sns.set(style='white', color_codes=True)
        fig, axs = plt.subplots(4, 2, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})
        plt.subplots_adjust(wspace=0.3)
        fig.suptitle('All galaxies - Overlapping HIIregions and GMCs \n %s' %name_end, fontsize=18, va='top')
        axs = axs.ravel()
        # Galactic distance vs: Mco, avir, sigmav,Sigmamol
        for i in range(len(labsyay)):
            r_sqs = []
            slopes = []
            model_intercepts = []
            x_axis = []
            for j in range(len(threshold_percs)):
                print(len(arrayxax_tot[k][j]))
                print(len(arrayyay_tot[k][j]))

                xax2 = [h for h in arrayxax_tot[k][j]]
                yay2 = [h for h in arrayyay_tot[i][j]]
                #if k < 5:
                xax2 = np.log10(xax2)
                yay2 = np.log10(yay2)

                RdBu = plt.get_cmap('viridis') #rainbow

                axs[i].scatter(xax2, yay2, marker = '.', label='%s' % threshold_percs[j], c = RdBu((8-j)/10)) #RdBu((8-j)/10)


                axs[i].grid()
                yaytmp = arrayyay[i]
                xaxtmp = arrayxax[k]
                xaxall = np.concatenate([f.tolist() for f in xaxtmp])
                yayall = np.concatenate([f.tolist() for f in yaytmp])
                print(len(xaxall))
                print(len(yayall))
                xaxall = np.log10(xaxall)
                yayall = np.log10(yayall)

                xaxall, yayall = clean_data(xaxall, yayall)
                xaxall1, yayall1 = clean_data(xax2, yay2)
                if xaxall.any() != 0 and yayall.any != ():
                    x = xaxall1
                    y = yayall1
                    slope,b,r_sq = linear_regression(xaxall1, yayall1)
                    x_axis.append([x])
                    slopes.append(slope)
                    model_intercepts.append(b)
                    r_sqs.append(r_sq)

            slope_mean = np.mean(slopes)
            r_sq_mean = np.mean(r_sqs)
            model_intercepts_mean = np.mean(model_intercepts)

            y_pred = model_intercepts_mean + slope_mean * x.ravel()
            axs[i].plot(xaxall1, y_pred, '-')

            x0, xf = xlim[k]
            y0, yf = ylim[i]
            axs[i].text(0.8, 0.05, 'R^2: %6.2f' % (r_sq_mean), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[i].transAxes)

            axs[i].text(0.15, 0.05, 'Slope %5.2f' % (slope_mean), fontsize=8, horizontalalignment='center',
                            verticalalignment='center', transform=axs[i].transAxes)

            axs[i].set(ylim=(y0, yf))
            axs[0].legend(prop={'size': 8})

            if r_sq < 0.1:
                color = "black"
            if r_sq >= 0.1 and r_sq < 0.2:
                color = "red"
            if r_sq >=0.2 and r_sq <0.3:
                color = "orange"
            if r_sq >= 0.3:
                color = "green"
            axs[i].set(ylabel=labsyay[i])
            #axs[i].yaxis.label.set_color(color)

        axs[6].set(xlabel=labsxax[k])
        axs[7].set(xlabel=labsxax[k])
        plt.show()


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

    GMCprop = DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover

    SizepcHII, LumHacorr, sigmavHII, metaliHII, varmetHII, numGMConHII, \
    MasscoGMC, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC = pickle.load(
        open(dir_script_data + 'Galaxies_variables_notover_GMC%s%s.pickle' % ( namegmc, name_end), "rb"))

    shortlab = ['HIIGMCdist', 'Mco', 'GMCsize', 'Smol', 'sigmav', 'avir', 'TpeakCO', 'tauff']
    MassesCO = [1e5 * i for i in MasscoGMCover]  #

    labsyay = labsyay[0:len(labsyay) - 6]  # removing  vel, major axis, minor axis and PA, no need to plot them
    labsxax = labsxax[0:len(labsxax) - 4]

    arrayyay = GMCprop
    arrayxax = HIIprop

    # Limits in the properties of HIIR and GMCs
    xlim, ylim, xx, yy = pickle.load(open(    dir_script_data + 'limits_properties.pickle', "rb"))

    return labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii

threshold_percs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]




#plot_all_thresholds(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1om", outliers = True, threshold_perc = 0.1, show = True, save = False, threshold_percs = threshold_percs)
#plotallgals(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1o1", outliers = True, show = True, save = False, threshold_perc = 0.1)
#plot_sigma_tpeak_thresholds_dist(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1om", outliers = True, show =False , save = True, threshold_percs = [0.1,0.4,0.7,0.9])
#plot_threshold_Sigma_Tpeak(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1om", outliers = True, show = False, save = True, threshold_percs = [0.1,0.4,0.7,0.9])
#plot_pairs(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1om", outliers = True, show = False, save = True, threshold_perc = 0.7)

for thres in [0.8,0.9]:
    plotallgals(new_muse=True, gmc_catalog='_native_', matching="overlap_1om", outliers=True, show=True, save=True,
                threshold_perc=thres, vel_limit = 10000)



#=============================================================================================#




#scatter_plot(new_muse = True, gmc_catalog = "_native_", matching = "overlap_1o1", outliers = True, show = False, save = True, threshold_perc = 0.5, plot_all_gals = True)
# ==============================================


# # =====================================================================
# # Plotting LHa vs Mco for each galaxy, highlighting the top ten.
#
# print("Plotting Smol-Mco for each galaxy")
#
# mycol=['grey','purple','blue','green','orange','red', 'pink', 'orange']
# spring = cm.get_cmap('spring', 256)
# mspring   = ListedColormap(spring(np.linspace(0.1, 0.8, 256)))
# mycmp=['Greys','Purples','Blues','Greens','Oranges','Reds',mspring,'Wistia']
#
# xaxarray = [Sigmamoleover, aviriaGMCover]
# yayarray = [SizepcGMCover, sigmavGMCover, MasscoGMCover]
# labsxax = [r'log($\Sigma_{\rm mol}$) [M$_{\odot}$ pc$^{-2}$]', r'log($\alpha_{\rm vir}$]']
# labsyay = ['log(GMC size) [pc]', r'log($\sigma_{\rm v}$ GMC) [km/s]', r'log(Ma$_{\rm CO}$) [M$_{\odot}$]']
# pdfx = ["SigmaMol", "Virial Parameter"]
# pdfy = ["GMC Size", "Sigma V GMC", "GMC Mass"]
# pdf = fpdf.PdfPages("Smol_Mco.pdf")
# marker_style = dict(markersize=4)
#
# for i in range(len(xaxarray)):
#     for k in range(len(yayarray)):
#         fig, axs = plt.subplots(4, 2,sharex='col',figsize=(8,10),gridspec_kw={'hspace': 0})
#         plt.subplots_adjust(wspace = 0.3)
#         r, g, b = np.random.uniform(0, 1, 3)
#         fig.suptitle(r'All galaxies - $\Sigma_{mol}$  vs CO masses', fontsize=14,va='top')
#         axs = axs.ravel()
#         for j in range(len(galaxias)):
#             xaxt = np.log10(xaxarray[i][j])
#             yayt = np.log10(yayarray[k][j])
#             idok = np.where((yayt == yayt) &  (xaxt == xaxt))
#             xax = xaxt[idok] ; yay = yayt[idok]
#             xax,yay=checknaninf(xax,yay,100000,100000)
#             axs[j].plot(xax, yay, '8', alpha=0.4, color=mycol[j],label='%s' % galaxias[j],**marker_style)
#             if len(xax) > 3:
#                sns.kdeplot(xax,yay, ax=axs[j],n_levels=3,shade=True,shade_lowest=False,alpha=0.4,cmap=mycmp[j])
#                sns.kdeplot(xax,yay, ax=axs[j],linewidths=0.5,cmap=mycmp[j],n_levels=3,shade_lowest=False)
#             axs[j].grid()
#             axs[j].legend(prop={'size': 5})
#             axs[j].set(ylabel=labsyay[k])
#         axs[6].set(xlabel=labsxax[i])
#         axs[7].set(xlabel=labsxax[i])
#         pdf.savefig(fig)
#         plt.close()
#
# pdf.close()
# #exit()


# # ---------------------------------------------------
# # Plotting HII regions prop vs num of GMCs associated.
# print("Plotting HII regions prop vs num of GMCs associated.")
# marker_style = dict(markersize=2)
# # numGMConHII[i][j][k]
# # i: number of galaxies
# # j = 0 -> number of regions < size HII * 2 ;  j = 1 -> array with indices of the GMCs associated.
# # k: indices of HII regions
#
# # LumHacorr
# GMCprop = [DisHIIGMCover, SizepcGMCover, sigmavGMCover, MasscoGMCover, aviriaGMCover, Sigmamoleover, TpeakGMCover,
#            tauffGMCover]
# shortlab = ['HIIGMCdist', 'GMCsize', 'sigmav', 'Mco', 'avir', 'Smol', 'TpeakCO', 'tauff']
# arrayxax = [GaldisHII, SizepcHII, LumHacorr,sigmavHII,metaliHII,varmetHII]  # ,sigmavHII,metaliHII,varmetHII
# arrayyay = numGMConHII
# # [DisHIIGMCover,SizepcGMCover,sigmavGMCover,MasscoGMCover,aviriaGMCover,Sigmamoleover,TpeakGMCover,tauffGMCover]
# labsxax = ['Galactocentric radius [kpc]', 'log(HII region size) [pc]',
#            r'log(Luminosity H$\alpha$) [erg/s]',r'log($\sigma_{v}$ HII region) [km/s]','Metallicity','Metallicity variation']  # ,r'log($\sigma_{v}$ HII region) [km/s]','Metallicity','Metallicity variation'
# labsyay = "Number of GMCs < size HII R"
# # labsyay = ['log(Distance  HII-GMC) [pc]','log(GMC size) [pc]',r'log($\sigma_v$) [km/s]',r'log(Mass$_{CO}$) [10^5 M$_{\odot}$]',r'log($\alpha_{vir}$)',r'log($\Sigma_{mol}$)',r'log(CO $T_{peak}$ [K])',r'log($\tau_{ff}$) [yr]']
#
# minmass = np.min([np.concatenate([f.tolist() for f in MasscoGMC])][0])
# maxmass = np.max([np.concatenate([f.tolist() for f in MasscoGMC])][0])
#
# pdf3 = fpdf.PdfPages("Num_GMCs_galbygal%s.pdf" % namegmc)  # type: PdfPages
# for k in range(len(arrayxax)):
#     print('Plotting %s' % labsxax[k])
#     sns.set(style='white', color_codes=False)
#     # Galactic distance vs: Mco, avir, sigmav,Sigmamol
#
#     fig, axs = plt.subplots(8, 7, sharex='col', figsize=(8, 10), gridspec_kw={'hspace': 0})
#     plt.subplots_adjust(wspace=0.3)
#
#     fig.suptitle('All galaxies - Overlapping HIIregions and GMCs\nColor coded %s' % shortlab[k], fontsize=15,
#                  va='top')  # labprop
#     axs = axs.ravel()
#     for j in range(len(galaxias) - 1):
#         xax = [h for h in arrayxax[k][j]]
#         yay = [h for h in arrayyay[j][0]]
#         idon = np.where(arrayyay[j][0] > 1)
#         xaxon = np.array(xax)[idon];
#         yayon = np.array(yay)[idon]
#         myay = [MasscoGMC[j][f].tolist() for f in arrayyay[j][1] if len(f) > 1]
#         rmsyay = []
#         meanyay = []
#         for myv in myay:
#             rmsyay.append(np.std(myv))
#             meanyay.append(np.mean(myv))
#         if len(rmsyay) > 0:
#             rmsyay = rmsyay / np.amax(rmsyay)
#         if k < 4:
#             xax = np.log10(xax)
#             xaxon = np.log10(xaxon)
#         axs[j].plot(xax, yay, '.', alpha=0.2, **marker_style, color='cornflowerblue')  # label='%s'%galaxias[j],
#         myp = axs[j].scatter(xaxon, yayon, marker='.', s=60, alpha=1, edgecolor='black', linewidths=0.2,
#                              color='black')  # c=np.log10(rmsyay), cmap='binary'
#         # axs[j].errorbar(xaxon, yayon, rmsyay, marker='.',capsize=5,**marker_style)
#         # fig.colorbar(myp, ax=axs[j],shrink=0.9)
#         axs[j].grid()
#         axs[j].legend(prop={'size': 5})
#     axs[0].set(ylabel=labsyay)
#     axs[6].set(xlabel=labsxax[k])
#     axs[7].set(xlabel=labsxax[k])
#     pdf3.savefig(fig)
#     plt.close()
#
# pdf3.close()
#
# print("Total number of GMCs for which more than 1 are associated to the same HII region")
#
# for i in range(len(galaxias)):
#    myids = np.where(numGMConHII[i][0]>1)[0]
# # print (galaxias[i],np.sum(numGMConHII[i][0][myids]))
#
# # =========================
# # Heyerr relationship
# rootR = [np.sqrt(f) for f in SizepcGMCover]
# arrayyay = np.divide(sigmavGMCover, rootR)
# arrayxax = Sigmamoleover  #
# # arrayxax=np.multiply(Sigmamoleover,aviriaGMCover)
#
# labsyay = r'log($\sigma_v$/R$^{0.5}$) [km/s pc$^{-1/2}$]'
# labsxax = r'log($\Sigma_{mol}$[M$_{\odot}$/pc$^2$])'  #
#
# pdf6 = fpdf.PdfPages("Correlations_Heyer_allgals_GMC_%s.pdf" % namegmc)  # type: PdfPages
#
# # print "Starting loop to create figures of all galaxies together - vs avir val"
#
# sns.set(style='white', color_codes=True)
# fig, axs = plt.subplots(1, 1, sharex='col', figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0})  # ,dpi=80
# plt.subplots_adjust(wspace=0.3)
# fig.suptitle('All galaxies - Overlapping HIIregions and GMCs', fontsize=18, va='top')
# yaytmp = arrayyay;
# xaxtmp = arrayxax
# xaxall = np.concatenate([f.tolist() for f in xaxtmp])
# yayall = np.concatenate([f.tolist() for f in yaytmp])
# xaxall = np.log10(xaxall)
# yayall = np.log10(yayall)
# idok = np.where((abs(yayall) < 100000) & (abs(xaxall) < 100000))
# xaxall = xaxall[idok];
# yayall = yayall[idok]
# lim1 = np.nanmedian(xaxall) - np.nanstd(xaxall) * 4
# lim2 = np.nanmedian(xaxall) + np.nanstd(xaxall) * 4
# indlim = np.where((xaxall < lim2) & (xaxall > lim1))
# xaxall = xaxall[indlim];
# yayall = yayall[indlim]
#
# for j in range(len(galaxias)):
#     xax2 = [h for h in arrayxax[j]]
#     yay2 = [h for h in arrayyay[j]]
#     xax = np.log10(xax2)
#     yay = np.log10(yay2)
#     axs.plot(xax, yay,'8',label='%s'%galaxias[j],alpha=0.7,markersize=5)
#
# axs.set(ylabel=labsyay)
# # axs.set_yscale('log')
# # axs.set_xscale('log')
# axs.grid()
#
# ybc = np.log10(
#     math.sqrt(math.pi * ct.G.cgs.value / 5 * ct.M_sun.cgs.value / ct.pc.cgs.value * 10 ** -10)) + 0.5 * xaxall
# axs.plot(xaxall, ybc)
#
# xmin = np.amin(xaxall)
# xmax = np.amax(xaxall)
# xprang = (xmax - xmin) * 0.1
# x = xaxall.reshape((-1, 1))
# y = yayall
# model = LinearRegression().fit(x, y)
# r_sq = model.score(x, y)
# y_pred = model.intercept_ + model.coef_ * x.ravel()
#
# axs.plot(xaxall, y_pred, '-')
# # sn.regplot(x=xaxall, y=yayall, ax=axs[i])
# x0 = xmin + xprang
# x0, xf = axs.get_xlim()
# y0, yf = axs.get_ylim()
# # x0,xf = xlim[k]
# # y0,yf = ylim[i]
# axs.text(x0, y0, 'R sq: %6.4f' % (r_sq))
# #        axs[i].set(xlim=(xmin - xprang, xmax + xprang))
# axs.set(xlim=(x0, xf))
# axs.set(ylim=(y0, yf))
# axs.legend(prop={'size': 14})
# axs.set(xlabel=labsxax)
# pdf6.savefig(fig)
# plt.close()
#
# pdf6.close()

#    mybin = 20
#    xbinned,ybinned,eybinned,nybinned=bindata(xaxall,yayall,mybin)
#    # if there is any nan inside
#    ido = np.where(np.array(nybinned) != 0)
#    xbinned  = xbinned[ido]
#    ybinned  = [g for g in np.array(ybinned)[ido]]
#    eybinned = [g for g in np.array(eybinned)[ido]]
#    nybinned = [g for g in np.array(nybinned)[ido]]
#    # Plot binned data
#    mysize = np.array(nybinned).astype(float)
#    mysize = (mysize-np.min(mysize))/(np.max(mysize)-np.min(mysize))*9+3
#    mylims = [np.argmin(mysize),np.argmax(mysize)]
#    mylabs = ["Num of pairs: %s" % min(nybinned),"Num of pairs: %s"% max(nybinned)]
#    #pdb.set_trace()
#    for j in range(len(xbinned)):
#        if j == np.argmin(mysize) or j == np.argmax(mysize):
#            axs[i].plot(xbinned[j], ybinned[j],linestyle="None",alpha=0.5,marker="o", markersize=mysize[j],color="red",label ="Num of pairs: %s"%  nybinned[j])
#        else:
#            axs[i].plot(xbinned[j], ybinned[j],linestyle="None",alpha=0.5,marker="o", markersize=mysize[j],color="red")
#    axs[i].errorbar(xbinned, ybinned,eybinned, capsize=5)
#    axs[i].set(ylabel=labsyay[i])
#    axs[i].grid()
#    # Computing the linear fit to the data, using the amount of
#    xmin = np.amin(xbinned)
##    xmax = np.amax(xbinned)
#    xprang = (xmax - xmin) * 0.03
#    x = xbinned.reshape((-1, 1))
#    y = ybinned
#    model = LinearRegression().fit(x, y,nybinned)
#    r_sq = model.score(x, y)
#    y_pred = model.intercept_ + model.coef_ * x.ravel()
#    axs[i].plot(xbinned, y_pred,'-')
#    #sn.regplot(x=xaxall, y=yayall, ax=axs[i])
#    x0 = xmin+xprang
#    y0, yf = axs[i].get_ylim()
#    my0 = y0-(yf-y0)*0.13
# 3    axs[i].text(x0, my0, 'R^2: %6.4f' % (r_sq),fontsize=10)
##    axs[i].set(ylim=(y0-(yf-y0)*0.15,yf+(yf-y0)*0.15))
##    axs[i].set(xlim=(xmin - xprang*3, xmax + xprang*3))
##
# axs[0].legend(prop={'size': 9})
# axs[4].set(xlabel=labsxax)
# pdf5.savefig(fig)
# plt.close()
#
# pdf5.close()
