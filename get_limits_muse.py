import numpy as np
import pickle
import os

def outinf(v1):
    v1n = np.array(v1)
    v1n2 = v1n[np.isfinite(v1n)]
    return v1n2


#-------------------------------------------------------------------------------------------------------------

#==============================================================================#
typegmc1 ='' # match_, match_homogenized_ (nothing for native)
typegmc = '_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_
#==============================================================================#

dir_script_data = os.getcwd() + "/script_data/"

namegmc = "_12m+7m+tp_co21%sprops"%typegmc

dirhii, dirgmc, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots, dirhiimasks = pickle.load(open(dir_script_data+'Directories_muse.pickle', "rb"))

dirmuseproperties = os.path.dirname(os.path.realpath("Extract_info_plot_per_gal_muse.py")) + "/"

galaxias, GMCprop, HIIprop, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay = pickle.load(
    open(dir_script_data+'Galaxies_variables_GMC%s.pickle' % ( namegmc), "rb"))

GaldisHIIover, SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, \
velHIIover,HIIminorover, HIImajorover, HIIangleover = HIIprop

DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, \
tauffGMCover, velGMCover,  angleGMCover, majorGMCover, minorGMCover = GMCprop


xlim = []
ylim = []
xlimt = []
ylimt = []

for k in range(len(labsxax)):
    xaxtmp = HIIprop[k]
    xaxt = np.concatenate([f.tolist() for f in xaxtmp])
    xax = outinf(xaxt)

    if k<6:
        xax = np.log10(xax)
        xax[np.isinf(xax)] = 0
    xlim1 = np.nanmedian(xax) - np.nanstd(xax)*4
    xlim2 = np.nanmedian(xax) + np.nanstd(xax)*4
    xmin = np.nanmin(xax)
    xmax = np.nanmax(xax)
    xrang = xmax-xmin
    xi = xmin - xrang*0.1
    xf = xmax + xrang*0.1

    xlim1 = np.nanmax([xlim1,xi])
    xlim2 = np.nanmin([xlim2,xf])
    xlim.append([xi,xf])
    xlimt.append([xlim1,xlim2])

    print(xlim,xlimt)




    
for k in range(len(labsyay)):
    yaytmp = GMCprop[k]
    yayt = np.concatenate([f.tolist() for f in yaytmp])
    yay = outinf(yayt)
    yay = np.log10(yay)
    ylim1 = np.nanmedian(yay) - np.nanstd(yay)*4
    ylim2 = np.nanmedian(yay) + np.nanstd(yay)*4    
    ymin = np.nanmin(yay)
    ymax = np.nanmax(yay)
    yrang = ymax-ymin
    yi = ymin - yrang*0.1
    yf = ymax + yrang*0.1

    ylim1 = np.nanmax([ylim1,yi])
    ylim2 = np.nanmin([ylim2,yf])
    ylim.append([yi,yf])
    ylimt.append([ylim1,ylim2])


    
print ("Saving variables in external file: limits_properties.pickle")
with open(dir_script_data+'limits_properties.pickle', "wb") as f:
#        pickle.dump([xlim,ylim,labsxax,labsyay], f)
        pickle.dump([xlimt,ylimt,labsxax,labsyay], f)
