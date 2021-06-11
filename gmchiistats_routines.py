import pickle
import numpy as np
import matplotlib.pyplot as plt

def prepdata_single2(data1, data2):
    dataall1 = np.log10(data1)
    dataall2 = np.log10(data2)



    idok = np.where((abs(dataall1) < 100000000) & (abs(dataall2) < 100000000) )
    dataall1 = dataall1[idok]
    dataall2 = dataall2[idok]



    return dataall1, dataall2

def prepdata2(data1, data2):


    dataall1 = np.concatenate([f.tolist() for f in data1])
    dataall1 = np.log10(dataall1)

    dataall2 = np.concatenate([f.tolist() for f in data2])
    dataall2 = np.log10(dataall2)




    idok = np.where((abs(dataall1) < 100000) & (abs(dataall2) < 100000) )
    dataall1 = dataall1[idok]
    dataall2 = dataall2[idok]

    return dataall1, dataall2

def prepdata4(data1, data2, data3, data4):


    dataall1 = np.concatenate([f.tolist() for f in data1])
    dataall1 = np.log10(dataall1)

    dataall2 = np.concatenate([f.tolist() for f in data2])
    dataall2 = np.log10(dataall2)

    dataall3 = np.concatenate([f.tolist() for f in data3])
    dataall3 = np.log10(dataall3)

    dataall4 = np.concatenate([f.tolist() for f in data4])
    dataall4 = np.log10(dataall4)


    idok = np.where((abs(dataall1) < 100000) & (abs(dataall2) < 100000) )
    dataall1 = dataall1[idok]
    dataall2 = dataall2[idok]
    dataall3 = dataall3[idok]
    dataall4 = dataall4[idok]


    return dataall1, dataall2, dataall3, dataall4

def prepdata3(data1, data2, data3):


    dataall1 = np.concatenate([f.tolist() for f in data1])
    dataall1 = np.log10(dataall1)

    dataall2 = np.concatenate([f.tolist() for f in data2])
    dataall2 = np.log10(dataall2)

    dataall3 = np.concatenate([f.tolist() for f in data3])
    dataall3 = np.log10(dataall3)



    idok = np.where((abs(dataall1) < 100000) & (abs(dataall2) < 100000) )
    dataall1 = dataall1[idok]
    dataall2 = dataall2[idok]
    dataall3 = dataall3[idok]


    return dataall1, dataall2, dataall3

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

def get_data(muse, gmc_catalog, matching, outliers, gmc_catalog_version, randomize, threshold_perc, vel, symmetrical, dir_script_data):
    # ==============================================================================#

    typegmc = gmc_catalog  # '_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_

    without_out = not outliers
    name_end = name(without_out=without_out, matching=matching, muse=muse, gmc_catalog=gmc_catalog,
                    gmc_catalog_version=gmc_catalog_version, threshold_perc=threshold_perc, vel_limit=vel, randomize=randomize, symmetrical=symmetrical)
    # ==============================================================================#

    namegmc = "_12m+7m+tp_co21%sprops" % typegmc

    # ====================================================================================================================#

    # =========================Getting all the GMC and HII properties from the pickle files===============================#


    galaxias, GMCprop1, HIIprop1, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay, idoverhii, idovergmc, labsyay1, labsxax1 = pickle.load(
        open(dir_script_data + 'Galaxies_variables_GMC%s%s.pickle' % ( namegmc, name_end), "rb"))  # retrieving the regions properties

    SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, \
    velHIIover, HIIminorover, HIImajorover, HIIangleover, Rgal_hii, HII_reff_over, HII_r25_over, HIIregionindex = HIIprop1

    HIIprop = SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, Rgal_hii

    DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, \
    tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover, FluxCOGMCover, Rgal_gmc = GMCprop1

    GMCprop = DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, tauffGMCover, Rgal_gmc

    SizepcHII, LumHacorrnot, sigmavHII, metaliHII, varmetHII, numGMConHII, \
    FluxCOGMCnot, HIIminor, HIImajor, HIIangle, angleGMC, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII, SigmaMol, Sigmav, COTpeak,a,b = pickle.load(
        open(dir_script_data + 'Galaxies_variables_notover_GMC%s%s.pickle' % ( namegmc, name_end), "rb"))

    shortlab = ['HIIGMCdist', 'Mco', 'GMCsize', 'Smol', 'sigmav', 'avir', 'TpeakCO', 'tauff']
    MassesCO = [1e5 * i for i in MasscoGMCover]  #



    labsyay = labsyay  # removing  vel, major axis, minor axis and PA, no need to plot them
    labsxax = labsxax

    arrayyay = GMCprop
    arrayxax = HIIprop


    return labsxax, labsyay, arrayxax, arrayyay, name_end, namegmc, galaxias, idoverhii, idovergmc, LumHacorrnot, FluxCOGMCnot,FluxCOGMCover, regionindexGMCover, HIImajor, majorGMC, minorGMC, MassCOGMC,SizepcGMC,SizepcHII,MasscoGMCover, SizepcGMCover, SigmaMol, Sigmav, COTpeak, labsyay1, labsxax1
