'''
Script used to match HII regions and GMCs according to the spatial overlap.
Mask generated from catalog geometrical parameters (radius etc...) of HII regions and GMCs.
'''


import os
import math
import sys
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy import constants as ct
import pickle
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf
from sklearn.linear_model import LinearRegression
import seaborn as sns
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
from regions import EllipseSkyRegion, CircleSkyRegion, EllipsePixelRegion, PixCoord
from astropy import wcs
from astropy.io import fits
from itertools import chain
from reproject import reproject_interp
from decimal import Decimal
import random
from pathlib import Path
import pathlib

table = Table.read('/home/antoine/Internship/phangs_sample_table_v1p6.fits')

galnames = ['ic5332', 'Ngc1433', 'Ngc3627',
            'Ngc0628', 'Ngc1512', 'Ngc4254',
            'Ngc1087', 'Ngc1566', 'Ngc4303',
            'Ngc1300', 'Ngc1672', 'Ngc4321',
            'Ngc1365', 'Ngc2835', 'Ngc4535',
            'Ngc1385', 'Ngc3351', 'Ngc5068',
            'Ngc7496']

galnames = [str.lower(x) for x in galnames]
ids = [id for id,x in enumerate (table['name']) if x in galnames]
velocities = table['orient_vlsr'][ids]


np.set_printoptions(threshold=sys.maxsize)
sns.set(style="white", color_codes=True)
c = 299792.458
gal_radial_vels = velocities

# ===================================================================================

def extract_info(gmc_catalog, gmc_catalog_version, muse, matching, outliers, threshold_perc, vel_limit, randomize, symmetrical):
    def undo_list(list1):
        if isinstance(list1, list) and list1 != []:
            list1 = list1[0]
            return list1
        else:
            return list1

    def extract_ind_sup(list1, value):
        ind_val = [[idx, item] for [idx, item] in enumerate(list1) if item >= value]
        indexes = [item[0] for item in ind_val]
        return indexes

    def extract_ind_inf(list1, value):
        ind_val = [[idx, item] for [idx, item] in enumerate(list1) if abs(item) <= value]
        indexes = [item[0] for item in ind_val]
        return indexes

    def make_list(a):
        list1 = []
        if not isinstance(a, list):
            list1.append(a)
            return list1
        else:
            return a

    def extract_ind(list1, value):
        list1 = make_list(list1)

        ind_val = [[idx, item] for [idx, item] in enumerate(list1) if item == value]
        indexes = [item[0] for item in ind_val]
        return indexes

    def extract_ind_not_equal(list1, value):
        list1 = make_list(list1)

        ind_val = [[idx, item] for [idx, item] in enumerate(list1) if item != value]
        indexes = [item[0] for item in ind_val]
        return indexes

    def extract_values(list1, indexes):
        list1 = make_list(list1)

        values = [list1[i] for i in indexes]
        return values

    def make_hii_list_ind(hiis_matched, gmc, hiis):
        hii_list_ind = []
        i = 0
        for hii in hiis:
            if (gmc in make_list(hii["GMCS"]) and (hii["PHANGS_INDEX"] - 1 not in hiis_matched)):
                #hii_list_ind.append((hii["PHANGS_INDEX"] - 1)) bug ?
                hii_list_ind.append(i)
            i+=1
        return hii_list_ind

    def make_hii_list(hiis_matched, gmc, hiis):
        hii_list = []
        for item in hiis:
            if (gmc in make_list(item["GMCS"]) and (item["PHANGS_INDEX"] - 1 not in hiis_matched)):
                hii_list.append(item)
        return hii_list

    def do_list(list1, gmc, key):
        done_list = []
        for hii in list1:
            sublist = hii[key]
            sublist_ind = hii["GMCS"]
            indexes = extract_ind(sublist_ind, gmc)
            value_list = extract_values(sublist, indexes)
            done_list.append(value_list)
        return done_list

    def do_list_indexes(list1, indexes):
        list1 = [[idx, item] for [idx, item] in enumerate(list1) if idx in indexes]
        list1 = [item[1] for item in list1]

        return list1

    def name_unmatch(gmc_catalog,  muse, randomize, symmetrical):

        name_end = muse + gmc_catalog + gmc_catalog_version + symmetrical

        if randomize == 'pos':
            name_end = name_end + 'random'
        else:
            name_end = name_end

        return name_end

    def name(matching, without_out, muse, gmc_catalog, gmc_catalog_version, threshold_perc, vel_limit, randomize, symmetrical):

        name_end = 'muse:' + muse + '_' + 'gmc:' + gmc_catalog + '(' + gmc_catalog_version + ')_' + 'vel_limit:' + str(
            vel_limit) + '_matching:' + matching +'_'+randomize+'_'+symmetrical

        if matching != "distance":
            name_end = name_end + '(' + str(threshold_perc).split(sep='.')[0] + str(threshold_perc).split(sep='.')[1]  + ')'
            if without_out == True:
                name_end = name_end + '_' + 'without_outliers'
            else:
                name_end = name_end + '_' + 'with_outliers'

        return name_end


    def vgsr_to_vhel(ra, dec,vgsr):

        VLSR = [9., 12., 7.]  # * u.km / u.s
        vlsr = VLSR

        c = SkyCoord(ra, dec, frame=FK5, unit="deg")
        g = c.galactic
        l, b = g.l, g.b

        lsr = vgsr

        v_correct = vlsr[0] * np.cos(b) * np.cos(l) + \
                    vlsr[1] * np.cos(b) * np.sin(l) + \
                    vlsr[2] * np.sin(b)
        vhel = lsr - v_correct

        return vhel

    def without_outlier(idoverhii, idovergmc, velhii_gal, velgmc_gal):

        # =========================Getting all the GMC and HII properties from the pickle files===============================#
        namegmc = "_12m+7m+tp_co21%sprops" % typegmc

        dirmuseproperties = os.path.dirname(os.path.realpath("Extract_info_plot_per_gal_muse.py")) + "/"

        galaxias, GMCprop1, HIIprop1, RAgmc, DECgmc, RAhii, DEChii, labsxax, labsyay = pickle.load(
            open('%sGalaxies_variables_GMC%s.pickle' % (dirmuseproperties, namegmc), "rb"))

        GaldisHIIover, SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover, \
        velHIIover, HIIminorover, HIImajorover, HIIangleover = HIIprop1

        DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover, \
        tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover = GMCprop1

        # =======================================================================================================
        vel_offset_tot = []
        for j in range(len(galaxias)):
            velhii = velHIIover[j]
            velgmc = velGMCover[j]
            vel_offset = (velhii - velgmc)
            vel_offset = vel_offset[(vel_offset < 2000) & (vel_offset > -2000)]

            vel_offset_tot.append(vel_offset)

        vel_offset_tot = list(chain.from_iterable((vel_offset_tot)))

        rms = np.std(vel_offset_tot)
        print('Mean =====')
        print(np.mean(vel_offset_tot))
        print('Median =====')
        print(np.median(vel_offset_tot))
        velhii = velhii_gal
        velgmc = velgmc_gal
        vel_offset = (velhii - velgmc)
        idovergmc_original = np.array(idovergmc)

        if str(np.where(vel_offset <= 3 * rms) and vel_offset >= -3 * rms) != '(array([], dtype=int64),)':
            idoverhii = np.array(idoverhii)[np.where(abs(vel_offset) <= 3 * rms)]
            idovergmc = np.array(idovergmc)[np.where(abs(vel_offset) <= 3 * rms)]

        out_number = len(idovergmc_original) - len(idovergmc)
        return idoverhii, idovergmc, out_number, rms

    def checkdist_2D(rahii, dechii, ragmc, decgmc, sizehii_obs, radgmc, distance):

        radgmc = 0.5*radgmc
        sizehii_obs = 0.5*sizehii_obs
        dists = np.zeros((2, len(ragmc)))
        for j in range(len(ragmc)):
            distas = ((ragmc[j] - rahii) ** 2 + (decgmc[j] - dechii) ** 2) ** 0.5
            dist = distas  # distance between the GMC and all HII regions in pc
            # Save the index and value of the minimum distance for a given GMC
            dists[0, j] = int(np.argmin(dist))
            dists[1, j] = min(dist)
        mindist = dists[1,]  # degrees
        inddist = dists[0,]

        idgmc = []
        idhii = []
        distmin = []
        indall = range(len(inddist))  # it will be the index of my index position of the HIIR.

        # Removing HIIR that are doubled, i.e. the same HIIR paired with more than 1 GMC.
        for idint, it in enumerate(inddist):  # a for loop in all gmcs, reading the index of the HIIR
            indw = np.where(inddist == it)[0]  #### Looking for the same index of HIIR, it, in the entire saved index.
            # If the same GMC has being paired twice, we should have several indices in indw
            if len(indw) > 1:
                igmc = np.extract(inddist == it,
                                  indall)  # extract the index of the gmcs associated to the same HIIR, igmc.
                imin = np.argmin(np.extract(inddist == it,
                                            mindist))  # get the index of the minimum distance between all index that are it
                dmin = np.min(
                    np.extract(inddist == it, mindist))  # get the  minimum distance between all index that are it
                indgmc = igmc[
                    imin]  # Index of the GMC that is the closest to the HIIR that was associated to different GMCs. Only this one will be saved.
                if it not in idhii:
                    idhii.append(int(it))
                    distmin.append(dmin)
                    idgmc.append(indgmc)
            else:
                idhii.append(int(it))  # indice del HIIRegion
                distmin.append(np.extract(inddist == it, mindist))
                idgmc.append(idint)

        # @ Index  idhii idgmc
        sizehii = sizehii_obs / 3600  # degrees
        radgmc = np.degrees(radgmc / distance)  # degrees
        addsize = sizehii[idhii] + radgmc[idgmc]
        tmpoverid = np.argwhere(mindist[idgmc] < addsize)
        overid = [int(item) for item in tmpoverid]
        idovergmc = [idgmc[item] for item in overid]
        idoverhii = [idhii[item] for item in overid]
        return mindist, inddist, idovergmc, idoverhii

    def undo_sublist(list):
        list_undone = []
        for x in list:
            x = undo_list(x)
            list_undone.append(x)
        return list_undone

    def matching1o1(gmcs, hiis):
        print("MATCHING 1O1")

        hiis_matched = []  # List of hii regions that have a definitive matched gmc
        for j in range(len(gmcs) - 1):
            gmc = j + 1  # here gmc refers to a PHANGS INDEX, so index from the list gmcs + 1
            hii_list_ind = make_hii_list_ind(hiis_matched, gmc,hiis)  # list of hii regions indexes still not definetively matched with a gmc and potentially matched with the gmc of phangs index j+1
            hii_list = make_hii_list(hiis_matched, gmc,hiis)  # list of hii regions still not definetively matched with a gmc which overlap with the gmc of index j+1


            if np.size(hii_list) > 1:  # if more than one hii region potentially matched with gmc of phangs index j+1
                hii_gmc_overlap = do_list(hii_list, gmc, "OVERLAP_PIX")  # list of overlap in pixels for each hii region in hii_list
                hii_gmc_vel = do_list(hii_list, gmc, "DELTAV") # list of velocity offset in km/s for each hii region in hii_list

                hii_gmc_overlap = undo_sublist(hii_gmc_overlap)

                #best_gmcs_inds = np.nanargmax(hii_gmc_overlap) # list of indexes of the hii regions in hii_list that overlap the most with the gmc of phangs index j+1 (gmc of interest)

                best_gmcs_inds = np.argwhere(hii_gmc_overlap == np.nanmax(hii_gmc_overlap))
                best_gmcs_inds = best_gmcs_inds.flatten().tolist()


                if np.size(best_gmcs_inds) > 1: # if more than 1 hii region in this list
                    hii_gmc_vel = np.array(hii_gmc_vel)
                    best_gmc_ind = np.nanargmin(np.abs(hii_gmc_vel[best_gmcs_inds])) # check in this list of hii regions which minimizes the velocity offset and keep it as finak match (hii region of interest)
                    best_hii_ind = hii_list_ind[best_gmcs_inds[best_gmc_ind]] # get the true index in the hiis list of the hii region of interest

                    best_hii = hii_list[best_gmcs_inds[best_gmc_ind]] # get the hii region of interest from hii_list
                    # remove gmcs in best and in others
                    hiis[best_hii_ind]["GMCS"] = gmc # remov all the gmcs except the one of interest for the hii regions of interest
                    hiis_matched.append(best_hii)



                    if np.size(hiis[best_hii_ind]["GMCS"]) > 1:
                        print('\nbug 3\n')

                    for hii in hiis:
                        if (hii["PHANGS_INDEX"] != best_hii["PHANGS_INDEX"] and hii["GMCS"] != []): #for all the hii regions with at least one potential match and not the hii region of interest
                            make_list(hii["GMCS"]) # turn potential single float in list as float -> [float] to use indexes
                            make_list(hii["OVERLAP_PIX"])
                            make_list(hii["DELTAV"])
                            ind_gmc = extract_ind_not_equal(hii["GMCS"],gmc) # check which indexes corresponds to the potential gmcs that are not the gmc of interest and only keep them
                            hii["OVERLAP_PIX"] = extract_values(hii["OVERLAP_PIX"],ind_gmc)
                            hii["DELTAV"] = extract_values(hii["DELTAV"], ind_gmc)  # np.array(hii["DELTAV"])[ind_gmc]
                            hii["GMCS"] = extract_values(hii["GMCS"], ind_gmc)  # np.array(hii["GMCS"])[ind_gmc]

                        else:
                            if hii["PHANGS_INDEX"]== best_hii["PHANGS_INDEX"]: # if hii is the hii region of interest
                                make_list(hii["GMCS"])
                                make_list(hii["OVERLAP_PIX"])
                                make_list(hii["DELTAV"])
                                ind_gmc = extract_ind(hii["GMCS"], gmc)  #extract the index of the gmc and only keep it
                                hii["OVERLAP_PIX"] = extract_values(hii["OVERLAP_PIX"],ind_gmc)
                                hii["DELTAV"] = extract_values(hii["DELTAV"], ind_gmc)
                                hii["GMCS"] = extract_values(hii["GMCS"], ind_gmc)

                else:
                    best_gmcs_inds = undo_list(best_gmcs_inds)
                    best_hii_ind = hii_list_ind[best_gmcs_inds]
                    best_hii = hii_list[best_gmcs_inds]
                    hiis_matched.append(best_hii)
                    hiis[best_hii_ind]["GMCS"] = gmc

                    for hii in hiis:
                        if (hii["PHANGS_INDEX"]!= best_hii["PHANGS_INDEX"] and hii["GMCS"] != []):
                            hii["GMCS"] = make_list(hii["GMCS"])
                            hii["OVERLAP_PIX"] = make_list(hii["OVERLAP_PIX"])
                            hii["DELTAV"] = make_list(hii["DELTAV"])
                            ind_gmc = extract_ind_not_equal(hii["GMCS"],gmc)
                            hii["OVERLAP_PIX"] = extract_values(hii["OVERLAP_PIX"],ind_gmc)
                            hii["DELTAV"] = extract_values(hii["DELTAV"], ind_gmc)
                            hii["GMCS"] = extract_values(hii["GMCS"], ind_gmc)

                        else:
                            if hii["PHANGS_INDEX"]== best_hii["PHANGS_INDEX"]:
                                hii["GMCS"] = make_list(hii["GMCS"])
                                hii["OVERLAP_PIX"] = make_list(hii["OVERLAP_PIX"])
                                hii["DELTAV"] = make_list(hii["DELTAV"])
                                ind_gmc = extract_ind(hii["GMCS"], gmc)
                                hii["OVERLAP_PIX"] = extract_values(hii["OVERLAP_PIX"],ind_gmc)
                                hii["DELTAV"] = extract_values(hii["DELTAV"], ind_gmc)
                                hii["GMCS"] = extract_values(hii["GMCS"], ind_gmc)

            else:
                if make_list(hii_list_ind):
                    hiis[undo_list(hii_list_ind)]["GMCS"] = gmc
                    hiis_matched.append(hiis[undo_list(hii_list_ind)])



                    if np.size(hiis[undo_list(hii_list_ind)]["GMCS"]) >1:
                        print('\nbug1\n')


        hiis = [item for item in hiis if item["GMCS"] != []]
        idoverhii = [hii.get("PHANGS_INDEX") - 1 for hii in hiis]
        idovergmc = [undo_list(undo_list(hii.get("GMCS"))) - 1 for hii in hiis]



        return idoverhii, idovergmc

    def matching1om(hiis):
        print('MATCHING 1OM')

        hiireject = []

        for hii in hiis:
            if len(hii['GMCS']) > 0:
                hii_rejected = hii.copy()

                max_overlap = np.nanmax(hii['OVERLAP_PIX'])
                max_overlap_ind = extract_ind(hii['OVERLAP_PIX'], max_overlap)
                vel_list = hii["DELTAV"]

                if len((max_overlap_ind))  > 1:
                    min_vel_ind = np.argmin([vel_list[i] for i in max_overlap_ind])



                    hii['GMCS'] = hii['GMCS'][max_overlap_ind[min_vel_ind]]


                    hii_rejected['GMCS'] = [item for item in hii_rejected['GMCS'] if item != hii['GMCS']]
                    hiireject.append(hii_rejected)


                    hiireject.append(hii_rejected)

                else:
                    hii['GMCS'] = hii['GMCS'][undo_list(max_overlap_ind)]








        hiis = [hii for hii in hiis if len(make_list(hii['GMCS'])) > 0 ]
        idoverhii = [hii.get("PHANGS_INDEX") - 1 for hii in hiis]
        idovergmc = [undo_list(hii.get("GMCS")) - 1 for hii in hiis]

        idrejectedhii = [hii.get("PHANGS_INDEX") - 1 for hii in hiireject]
        idrejectedgmc = [undo_list(hii.get("GMCS")) - 1 for hii in hiireject]
        return idoverhii, idovergmc, idrejectedhii,idrejectedgmc

    def overlap_percent(rahii, dechii, major_gmc, minor_gmc, angle_gmc, decgmc, ragmc, radiushii, radgmc, header,
                        threshold_perc, gmcs, hiis, matching_1o1, matching_1om, vel_limit, randomize):


        name_unmatched = name_unmatch(gmc_catalog,  muse, randomize, symmetrical=symmetrical)
        print("raw_%s%s%s.pickle" % (name_unmatched, namegmc, galnam))

        overwrite = False

        if os.path.isfile((dir_script_data+"raw_%s%s%s.pickle" % (name_unmatched, namegmc, galnam,))) and overwrite == False:
            hiis, gmcs = pickle.load(open(dir_script_data+'raw_%s%s%s.pickle' %(name_unmatched, namegmc, galnam), "rb"))


        else:
            WC = wcs.WCS(header)
            name = "_ha.fits"  # "_Hasub_flux.fits"
            hdul = fits.open(dirmaps + str.upper(galaxias[0]) + name)
            if muse !='dr22':
                mask_suffix = '_HIIreg_mask.fits'
            else:
                mask_suffix = '_nebulae_mask_V2.fits'
            mask = fits.open(dirhiimasks + str.upper(galaxias[0]) + mask_suffix)
            mask_header = mask[0].header
            WCH = wcs.WCS(mask_header)
            image_reprojected, footprint = reproject_interp(hdul, mask_header)
            image_data = hdul[0].data

            # =============================================== Checking which GMCs have a HII region associated (distance) ===================

            ind_hii_dist = []  # indexes of hiis that potentially have gmcs
            arr_hii_dist = []
            dists = np.zeros((len(hiis)),dtype=object)  # initializing array, j : index of hii region, and for each gmc the associated hii regions, so array of not constant shape
            size_loop = len(hiis)

            ax1 = header['NAXIS1'] + 1000  # getting the sizes of the galaxy image where the regions are projected
            ax2 = header['NAXIS2'] + 1000







            for j in range(size_loop):  # This works as intended, but need to check the coordinates system of hiis and gmcs. Maybe too restrictive
                distas = np.sqrt((np.square(np.full(fill_value=rahii[j], shape=len(ragmc)) - ragmc)) + np.square((np.full(fill_value=dechii[j],shape=len(ragmc)) - decgmc)))  # array of distance between GMC(j) and all the HII regions
                dist = ((rahii[j] - ragmc) ** 2 + (dechii[j] - decgmc) ** 2) ** 0.5
                aradhii = np.full(fill_value=radiushii[j], shape=len(ragmc))  # array filled with radgmc(j) value
                deltadist = (distas - np.array(major_gmc) - 1.2* aradhii)
                dists[j] = np.where(deltadist <= 0)[0]  # indexes of where the distance is <= radius hii + radius gmc

                if np.array(dists[j]).size != 0:  # if a gmc has zero hii region where distance is <= radius hii + radius gmc, then its index is not kept
                    ind_hii_dist.append(j)
                    arr_hii_dist.append(dists[j])

            # ========================================================================================================================================

            hiis_loop = [hiis[i] for i in ind_hii_dist] # if (int(hiis[i]["BPT_NII"]) == 0 and int(hiis[i]["BPT_SII"]) == 0)]

            j = 0
            for hii in hiis:  # hiis_loop:  # range(len(ragmc)):  # for the gmc regions that passed the precedent condition, "local" masks are calculated from astropy regions using table_muse angles, radius etc...
                hii["GMCS"] = []
                hii["OVERLAP_PIX"] = []
                hii["DELTAV"] = []

                if hii in hiis_loop:
                    print("%i/%i" % (undo_list(hii["PHANGS_INDEX"]), len(hiis)))

                    rahii = hii['RA']  # Right ascension of the region in degrees
                    dechii = hii['DEC']  # declinaison of the region in degrees
                    radiushii = hii['RADIUS'] / 3600

                    center_hii = SkyCoord(rahii * u.deg, dechii * u.deg, frame=FK5, unit='deg')
                    circle_hii = CircleSkyRegion(center=center_hii, radius=radiushii * u.deg)
                    circle_pixel_hii = circle_hii.to_pixel(wcs=WC)

                    hxmin = circle_pixel_hii.bounding_box.ixmin  # getting bounding box limits to transform "local mask" to "galaxy size" masks but for the hii region this time
                    hxmax = circle_pixel_hii.bounding_box.ixmax
                    hymin = circle_pixel_hii.bounding_box.iymin
                    hymax = circle_pixel_hii.bounding_box.iymax

                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    hii_area = 0
                    mask_hii_all = np.zeros(shape=(ax2, ax1))  # initializing "galaxy size" masks

                    mask_hii = circle_pixel_hii.to_mask().data

                    for l in range(
                            hymax - hymin):  # transforming "local masks" to galaxy size masks (hii region)
                        for c in range(hxmax - hxmin):
                            mask_hii_all[l + hymin][c + hxmin] = mask_hii[l][c]

                    hii_area = np.count_nonzero(mask_hii_all)  # sum([list(i).count(2) for i in (

                    gmcs_loop = [gmcs[i] for i in arr_hii_dist[j] if (gmcs[i]['MOMMAJ_PC'] != 0 and gmcs[i]["MOMMIN_PC"] != 0)]

                    j += 1
                    for gmc in gmcs:  # gmcs_loop:#gmcs1:
                        gmc["HIIS"] = []
                        gmc["OVERLAP_PIX"] = []
                        gmc["DELTAV"] = []

                        if gmc in gmcs_loop:
                            ragmc = gmc['XCTR_DEG']
                            decgmc = gmc['YCTR_DEG']
                            major_gmc = 2*np.degrees(gmc['MOMMAJ_PC'] / gmc['DISTANCE_PC'])
                            minor_gmc = 2*np.degrees(gmc['MOMMIN_PC'] / gmc['DISTANCE_PC'])
                            angle_gmc = 90 - np.degrees(gmc['POSANG'])

                            center_gmc = SkyCoord(ragmc * u.deg, decgmc * u.deg, frame=FK5, unit='deg')
                            ellipse_gmc = EllipseSkyRegion(center=center_gmc, height=major_gmc * u.deg,width=minor_gmc * u.deg, angle=angle_gmc * u.deg)
                            ellipse_pixel_gmc = ellipse_gmc.to_pixel(wcs=WC)

                            gxmin = ellipse_pixel_gmc.bounding_box.ixmin  # getting bounding box limits to transform "local mask" to "galaxy size" masks later
                            gxmax = ellipse_pixel_gmc.bounding_box.ixmax
                            gymin = ellipse_pixel_gmc.bounding_box.iymin
                            gymax = ellipse_pixel_gmc.bounding_box.iymax

                            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                            gmc_area = 0  # initializing the areas at 0 in case no overlap  is found
                            mask_gmc = ellipse_pixel_gmc.to_mask().data


                            mask_gmc_all = np.zeros(shape=(ax2, ax1))

                            for l in range(gymax - gymin):  # transforming "local masks" to galaxy size masks (gmc)
                                for c in range(gxmax - gxmin):
                                    mask_gmc_all[l + gymin][c + gxmin] = mask_gmc[l][c]

                            gmc_area = np.count_nonzero(mask_gmc_all)#sum([list(i).count(2) for i in (

                            count1 = np.count_nonzero((mask_gmc_all+ mask_hii_all) == 2)

                            if gmc_area != 0 and hii_area != 0:

                                if symmetrical == 'sym':
                                    count1 = count1 / (0.5*(gmc_area + hii_area))

                                elif symmetrical == 'hii':
                                    count1 = count1 / hii_area

                                elif symmetrical == 'gmc':
                                    count1 = count1 / gmc_area

                                else:
                                    sys.exit('Invalid symmetrical argument')

                                if gmc["S2N"] > 5 :#and int(hii['BPT_SII']) == 0 and int(hii['BPT_NII']) == 0: #count1 >= threshold_perc and gmc["S2N"] > 5 and int(hii['BPT_SII']) == 0 and int(hii['BPT_NII']) == 0:
                                    gmc["HIIS"].append(hii['PHANGS_INDEX'])
                                    gmc["OVERLAP_PIX"].append(count1)
                                    hii["GMCS"].append(gmc['CLOUDNUM'])
                                    hii["OVERLAP_PIX"].append(count1) # when gmcs have the right size change to count2
                                    hiivel = (hii['HA_VEL'] + 1.4)  # Velocity (km/s)
                                    gmcvel = gmc['VCTR_KMS']



                                    gmcvel = vgsr_to_vhel(vgsr=gmcvel, ra=ragmc, dec=decgmc)

                                    gmcvel = gmcvel/(1-(gmcvel/299792))


                                    deltav = gmcvel - hiivel
                                    gmc["DELTAV"].append(deltav)
                                    hii["DELTAV"].append(deltav)


            with open((dir_script_data+'raw_%s%s%s.pickle' % (name_unmatched, namegmc, galnam)), "wb") as f:
                pickle.dump(
                    [hiis, gmcs], f)


        # print("====================SHUFFLE====================")
        # print("BEFORE")
        # print([gmc["CLOUDNUM"] for gmc in gmcs])
        # random.shuffle(gmcs)
        # random.shuffle(hiis)
        # print("AFTER")
        # print([gmc["CLOUDNUM"] for gmc in gmcs])

        if matching_1om == True or matching_1o1 == True:
            vel = vel_limit

            hiis, gmcs = threshold_filter(hiis,gmcs, threshold_perc)
            hiis, gmcs = vel_filter(hiis, gmcs, vel)




        if matching_1o1 == True and matching_1om == False:
            idoverhii, idovergmc = matching1o1(gmcs, hiis)

        if matching_1om == True and matching_1o1 == False:
            idoverhii, idovergmc, idrejectedhii,idrejectedgmc = matching1om(hiis)
        maxarea = 1
        ind_gmc = 1

        return maxarea, ind_gmc, idovergmc, idoverhii, idrejectedhii,idrejectedgmc

    def vel_filter(hiis, gmcs, vel):


        #gmcs = [gmc for gmc in gmcs if gmc["HIIS"]]

        # for gmc in gmcs:
        #     indexes_gmc = extract_ind_inf(gmc['DELTAV'], vel)
        #     for key, list1 in gmc.items():
        #         if key in ["DELTAV", "OVERLAP_PIX", "HIIS"]:
        #             list1 = do_list_indexes(list1, indexes_gmc)
        #             gmc[key] = list1
        # gmcs = [gmc for gmc in gmcs if gmc["HIIS"]]



        hiis = [hii for hii in hiis if hii["GMCS"]]

        for hii in hiis:
            indexes_hii = extract_ind_inf([x - gal_radial_vel for x in hii['DELTAV']], vel)
            for key, list1 in hii.items():
                if key in ["DELTAV", "OVERLAP_PIX", "GMCS"]:
                    list1 = do_list_indexes(list1, indexes_hii)
                    hii[key] = list1


        hiis = [hii for hii in hiis if hii["GMCS"]]


        return hiis, gmcs

    def threshold_filter(hiis, gmcs, threshold_perc):

        hiis = [hii for hii in hiis if hii["GMCS"]]
        #gmcs = [gmc for gmc in gmcs if gmc["HIIS"]]
        print("threshold_perc")
        print(threshold_perc)
        # for gmc in gmcs:
        #     indexes_gmc = extract_ind_sup(gmc['OVERLAP_PIX'], threshold_perc)
        #     for key, list1 in gmc.items():
        #         if key in ["DELTAV", "OVERLAP_PIX", "HIIS"]:
        #             list1 = do_list_indexes(list1, indexes_gmc)
        #             gmc[key] = list1
        # gmcs = [gmc for gmc in gmcs if gmc["HIIS"]]

        for hii in hiis:
            indexes_hii = extract_ind_sup(hii['OVERLAP_PIX'], threshold_perc)
            for key, list1 in hii.items():
                if key in ["DELTAV", "OVERLAP_PIX", "GMCS"]:
                    list1 = do_list_indexes(list1, indexes_hii)
                    hii[key] = list1


        hiis = [hii for hii in hiis if hii["GMCS"]]


        return hiis, gmcs

    def writeds9_vel(galnam, namegmc, rahii, dechii, pind, ragmc, decgmc, dist, comment, dirregions, name_end):

        f = open('%s%s%s-HIIregions-Vel%s%s.reg' % (dirregions, galnam, namegmc, comment, name_end), "w+")
        f.write("# Region file format: DS9 version 4.1\n")
        # max_dist = np.nanmax(dist)

        f.write(
            'global dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        f.write("fk5\n")
        for j in range(len(rahii)):
            d = 1 / 3600  # dist[j] / maxdist
            f.write('circle(%12.8f,%12.8f,%12.8f") # text={%i} width=3 \n' % (rahii[j], dechii[j], d, pind[j]))
        f.close()

        f = open('%s%s%s-GMCs-%s%s.reg' % (dirregions, galnam, namegmc, comment, name_end), "w+")
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(
            'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        f.write("fk5\n")
        for j in range(len(ragmc)):
            f.write('circle(%12.8f,%12.8f,1") # text={%i} width=3 color=red\n' % (ragmc[j], decgmc[j], j))
        f.close()

    def writeds9(galnam, namegmc, rahii, dechii, pind, ragmc, decgmc, comment, dirregion, name_end):

        f = open('%s%s%s-HIIregions-%s%s.reg' % (dirregion, galnam, namegmc, comment, name_end), "w+")
        f.write("# Region file format: DS9 version 4.1\n")
        if comment == 'overlapped':
            f.write(
                'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
            f.write("fk5\n")
            for j in range(len(rahii)):
                f.write('circle(%12.8f,%12.8f,0.1") # text={%i} width=3 color=green\n' % (rahii[j], dechii[j], pind[j]))
            f.close()
        else:
            f.write(
                'global color=blue dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
            f.write("fk5\n")
            for j in range(len(rahii)):
                f.write('circle(%12.8f,%12.8f,0.1") # text={%i} width=3 color=blue\n' % (rahii[j], dechii[j], pind[j]))
            f.close()

        f = open(
            '%s%s%s-GMCs-%s%s.reg' % (dirregions, galnam, namegmc, comment, name_end),
            "w+")
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(
            'global color=red dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        f.write("fk5\n")
        for j in range(len(ragmc)):
            f.write('circle(%12.8f,%12.8f,0.1") # text={%i} width=3 color=red\n' % (ragmc[j], decgmc[j], j))
        f.close()


    # ===================================================================================
    typegmc = gmc_catalog

    if not (matching == "distance" or matching == "overlap_1o1" or matching == "overlap_1om"):
        sys.exit("WRONG MATCHING INPUT")

    matching_1o1 = False
    matching_1om = False
    matching_dist = False


    if matching == "overlap_1o1":
        matching_1o1 = True
    if matching == "overlap_1om":
        matching_1om = True
    if matching == "distance":
        matching_dist = True

    without_out = not outliers
    name_end = name(matching, without_out, muse, gmc_catalog, vel_limit=vel_limit, gmc_catalog_version= gmc_catalog_version, threshold_perc=threshold_perc, randomize=randomize, symmetrical=symmetrical)
    # ---------------------------------------------


    # Defining paths to the files and files names !!! TO CHANGE !!!
    # ======================================================================================================================================================================"
    dirhii_dr1= "/home/antoine/Internship/muse_hii_new/"  # new muse tables directory
    dirhii_dr2 = "/home/antoine/Internship/muse_hii_new_dr2/"
    dirhii_dr22 = "/home/antoine/Internship/muse_hii_new_dr22/"

    dirgmc_old = '/home/antoine/Internship/gmccats_st1p5_amended/'  # gmc tables directory
    dirgmc_new = "/home/antoine/Internship/gmc_new/"
    dirgmc_4 = "/home/antoine/Internship/gmc_v4/"

    dirmaps = "/home/antoine/Internship/Galaxies/New_Muse/"  # maps directory (to plot the overlays)
    dirhiimasks_dr1 = '/home/antoine/Internship/hii_masks_dr1/'
    dirhiimasks_dr2 = '/home/antoine/Internship/hii_masks_dr2/'
    dirhiimasks_dr22 = '/home/antoine/Internship/hii_masks_dr22/'



    dirgmcmasks = '/home/antoine/Internship/gmc_masks/'

    name_muse_dr1 = 'Nebulae_Catalogue.fits'
    name_muse_dr2 = "HIIregion_cat_DR2_native.fits"
    name_muse_dr22 = "Nebulae_catalogue_v2.fits"

    sample_table_dir = "/home/antoine/Internship/PHANGS Sample (Public) - Basic Data.csv"

    namemap = '_ha.fits'

    # ========================================================================================================================================================================="


    if not os.path.isdir(os.getcwd() + "/script_data"):
        os.makedirs(os.getcwd()+"/script_data", exist_ok=True)
    if not os.path.isdir(os.getcwd() + "/ds9tables/Muse/DR1"):
        os.makedirs(os.getcwd() + "/ds9tables/Muse/DR1", exist_ok=True)
    if not os.path.isdir(os.getcwd() + "/ds9tables/Muse/DR2"):
        os.makedirs(os.getcwd() + "/ds9tables/Muse/DR2", exist_ok=True)
        if not os.path.isdir(os.getcwd() + "/ds9tables/Muse/DR22"):
            os.makedirs(os.getcwd() + "/ds9tables/Muse/DR22", exist_ok=True)
    if not os.path.isdir(os.getcwd() + "/Plots_Muse/DR1"):
        os.makedirs(os.getcwd() + "/Plots_Muse/DR1", exist_ok=True)
    if not os.path.isdir(os.getcwd() + "/Plots_Muse/DR2"):
        os.makedirs(os.getcwd() + "/Plots_Muse/DR2", exist_ok=True)
        if not os.path.isdir(os.getcwd() + "/Plots_Muse/DR22"):
            os.makedirs(os.getcwd() + "/Plots_Muse/DR22", exist_ok=True)



    dir_script_data1 = os.getcwd()+"/script_data/"
    dir_script_data2 = os.getcwd()+"/script_data_dr2/"
    dir_script_data22 = os.getcwd()+"/script_data_dr22/"

    dirregions1 = os.getcwd() + "/ds9tables/Muse/DR1/"  # Must create a directory to save the region ds9 files before running the code for the first time
    dirregions2 = os.getcwd() + "/ds9tables/Muse/DR2/"  # same but for new muse catalog
    dirregions22 = os.getcwd() + "/ds9tables/Muse/DR22/"  # same but for new muse catalog

    dirplots1 = os.getcwd() +"/Plots_Muse/DR1/"  # directories to save the plots (old muse catalog)
    dirplots2 = os.getcwd() +"/Plots_Muse/DR2/"  # directories to save the plots (new muse catalog)
    dirplots22 = os.getcwd() +"/Plots_Muse/DR22/"  # directories to save the plots (new muse catalog)



    if muse != 'dr1' and muse != 'dr2' and muse != 'dr22':
        print('wrong muse input, not dr1 or dr2')
        exit()

    # ========================================================================================================================================================================="

    # ---defining which directory to store the plots in----#
    if muse == 'dr1':
        dirplots = dirplots1
        dirregions = dirregions1
        dirhiimasks = dirhiimasks_dr1
        dir_script_data = dir_script_data1
        dirhii = dirhii_dr1
        name_muse = name_muse_dr1

    elif muse == 'dr2':
        dirplots = dirplots2
        dirregions = dirregions2
        dirhiimasks = dirhiimasks_dr2
        dir_script_data = dir_script_data2
        dirhii = dirhii_dr2
        name_muse = name_muse_dr2

    elif muse == 'dr22':
        dirplots = dirplots22
        dirregions = dirregions22
        dirhiimasks = dirhiimasks_dr22
        dir_script_data = dir_script_data22
        dirhii = dirhii_dr22
        name_muse = name_muse_dr22



    with open((dir_script_data+'Directories_muse.pickle'), "wb") as f:
        pickle.dump([dirhii_dr1,dirhii_dr2, dirgmc_old,dirgmc_new, dirregions1, dirregions2, dirregions22, dirmaps, dirplots1, dirplots2, dirplots22, dirplots, dirhiimasks, dirgmcmasks, sample_table_dir], f)

    # -----------------------------------------------------#
    if typegmc == "_native_":
        namegmc = "_12m+7m+tp_co21%sprops" % typegmc
        namegmc1 = "_12m+7m+tp_co21%sprops" % typegmc

    else:
        typegmc1 = "_" + str(typegmc.split("_")[1]) + "_"
        namegmc = "_12m+7m+tp_co21%sprops" % typegmc
        namegmc1 = "_12m+7m+tp_co21%sprops" % typegmc1

    if gmc_catalog_version == 'new':
        dirgmc = dirgmc_new
    elif gmc_catalog_version == 'old':
        dirgmc = dirgmc_old
    elif gmc_catalog_version == 'v4':
        dirgmc = dirgmc_4

    if gmc_catalog_version != 'v4':
        dirgmc = ('%scats%samended/' % (dirgmc, typegmc))
    else:
        typegmc2 = "native"
        dirgmc = ('%s%s/' % (dirgmc, typegmc2))

    # Defining empty vectors to save the variables of all galaxies
    total_outliers = []
    galaxias = []
    LumHacorr = []
    HaLumAllGal = []
    HaLumAllGal1 = []

    idoverhiis = []
    idovergmcs = []

    idrejecthiis = []
    idrejectgmcs = []
    HaLumAllGal_over = []
    HaLumAllGal_over1 = []

    COFluxAllGal = []
    COFluxAllGal1 = []

    COFluxAllGal_over = []
    COFluxAllGal_over1 = []

    gmc_vel_gal = []
    hii_vel_gal = []

    GMC_tot = []
    HII_tot = []
    GMC_tot_overlap = []
    HII_tot_overlap = []
    SizepcHII = []
    sigmavHII = []
    metaliHII = []
    varmetHII = []
    HIImajor = []
    HIIminor = []
    HIIangle = []
    idoverhii_s = []
    idovergmc_s = []

    VelGMC = []
    VelHII = []

    DisHIIGMC = []
    SizepcGMC = []
    sigmavGMC = []
    MasscoGMC = []
    aviriaGMC = []
    Sigmamole = []
    TpeakGMC = []
    tauffGMC = []
    angleGMC = []
    majorGMC = []
    minorGMC = []
    FluxCOGMC = []

    numGMConHII = []
    LumHacorrover = []



    SizepcHIIover = []
    sigmavHIIover = []
    metaliHIIover = []
    varmetHIIover = []
    velHIIover = []
    HIIangleover = []
    HIIminorover = []
    HIImajorover = []

    velGMCover = []
    DisHIIGMCover = []
    SizepcGMCover = []
    sigmavGMCover = []
    MasscoGMCover = []
    aviriaGMCover = []
    Sigmamoleover = []
    TpeakGMCover = []
    tauffGMCover = []
    angleGMCover = []
    majorGMCover = []
    minorGMCover = []
    regionindexGMCover = []
    regionindexHIIover = []
    FluxCOGMCover = []

    RAgmcover = []
    DECgmcover = []
    Rgal_HII = []
    Rgal_GMC = []
    Rgal_HII_over = []
    Rgal_GMC_over = []
    RAhiiover = []
    DEChiiover = []
    RAgmcall = []
    DECgmcall = []
    RAhiiall = []
    DEChiiall = []

    HII_reff_over = []
    HII_r25_over = []

    # ============================================================================================================
    # Limits in the properties of HIIR and GMCs
    # if limots.py has not been run, comment the following line
    # xlim, ylim, xx, yy = pickle.load(open('limits_properties.pickle', "rb"))
    # =============================================================================================================



    table_muse = Table.read("%s%s" % (dirhii, name_muse))

    w = 0
    galaxies_name = ['NGC0628']

    for i in range(len(table_muse['gal_name']) - 1):
        if galaxies_name[w] != str(table_muse['gal_name'][i]) and str(table_muse['gal_name'][i]) != 'IC5332':
            galaxies_name.append(str(table_muse['gal_name'][i]))
            w += 1

    hiicats = galaxies_name
    w = 1



    # ========================================================================================#

    #rand_ids_gmc_list = pickle.load(
    #open(dir_script_data + 'randomized_lists', "rb"))

    reff_gals = [0.41174614, 211.2647, 0.0347, 0.0303, 0.0205, 0.0002, 1.1346598, 0.0001, 0.0002, 0.258768, 0.0013,
                 165.18874, 11.32, 13.1, 0.071, 15.21, 0.0195, 0.294296, 0.0084]  # arcsec
    reff_unc = [2, 4, 11, 12, 13, 16, 17]  # FIT TO MAJOR AXIS


    # Loop in all galaxies. Do histograms and individual galaxy plots.
    print("Starting loop in all galaxies [i], do histograms and individual galaxy plots")
    for i in range(len(hiicats)):

        gal_radial_vel = gal_radial_vels[i]
        reff = reff_gals[i]

        galnam = str.lower(galaxies_name[i])

        print("%s%s%s.fits" % (dirgmc, galnam, namegmc1))
        if os.path.isfile(("%s%s%s.fits" % (dirgmc, galnam, namegmc1))):
            galaxias.append(galnam)

            print("-*" * 20)
            print("Galaxy name: %s" % galnam)
            print("-------------------------")
            #==========================HII REGIONS===========================================#

            thii = table_muse[np.where(table_muse['gal_name'] == str.upper(galnam))]
            thii = thii[np.where(thii['BPT_NII'] == 0)]
            thii = thii[np.where(thii['BPT_SII'] == 0)]

            if muse == 'dr22':
                thii = thii[np.where(thii['BPT_OI'] == 0)]

            sample_table = Table.read(sample_table_dir)
            sample_table_gal = sample_table[np.where(sample_table['Name'] == str.upper(galnam))]
            gal_ra = sample_table_gal['R.A.'][0].astype(float)
            gal_dec = sample_table_gal['Dec.'][0].astype(float)





            if randomize == 'pos':
                cen_ra = thii['cen_ra']
                thii.remove_column('cen_ra')
                cen_ra = np.array(cen_ra)
                np.random.shuffle(cen_ra)
                thii['cen_ra'] = cen_ra.astype(float)

                cen_dec = thii['cen_dec']
                thii.remove_column('cen_dec')
                cen_dec = np.array(cen_dec)
                np.random.shuffle(cen_dec)
                thii['cen_dec'] = cen_dec.astype(float)



            # Information of individual HII regions
            rahii = thii['cen_ra']  # Right ascension of the region in degrees
            dechii = thii['cen_dec']  # declinaison of the region in degrees
            hii_rgal_deg = thii['deproj_dist']/3600

            pind = np.full(shape=len(thii['cen_ra']), fill_value=int(w))
            w += 1
            hiivel = (thii['HA6562_VEL'] )  # Veclocity (km/s)
            hiivel = np.array(hiivel)

            # angle = thii['PA']  # Position angle of the HII region (deg)
            major = thii['region_circ_rad'] / 3600  # np.degrees( thii['SIZE'] / (thii['DISTMPC'][0] * 1e6))# thii['SIZE_OBS'] / 3600  # major axis of the HII region (deg)
            minor = thii['region_circ_rad'] / 3600  # minor axis of the HII region (deg)
            radiushii = thii['region_circ_rad'] / 3600

             #np.sqrt(thii['region_area']) * thii['kpc_per_pixel'] * 1000  # pc better way to go ?
            sigmahii = thii['HA6562_SIGMA']
            metalhii = thii['met_scal']

            if muse == 'dr22':
                HIIregionindex = thii['Environment']
                hii_r25 = thii['r_R25']
                hii_reff = thii['r_reff']
            else:
                HIIregionindex = thii['region_ID']
                hii_r25 = thii['region_ID']
                hii_reff = thii['region_ID']

            if muse == 'dr1':
                vamethii = thii['Delta_met']
            else :
                vamethii = thii['met_scal']



            # =============================================================
            # Corresponding CO data and GMC catalog
            # =============================================================
            print("Reading CO table_muse")

            tgmc = Table.read(("%s%s%s.fits" % (dirgmc, galnam, namegmc1)))

            # if randomize== 'pos':
            #     xctr_deg = tgmc['XCTR_DEG']
            #     tgmc.remove_column('XCTR_DEG')
            #     xctr_deg = np.array(xctr_deg)
            #     np.random.shuffle(xctr_deg)
            #     tgmc['XCTR_DEG'] = xctr_deg.astype(float)
            #
            #     yctr_deg = tgmc['YCTR_DEG']
            #     tgmc.remove_column('YCTR_DEG')
            #     yctr_deg = np.array(yctr_deg)
            #     np.random.shuffle(yctr_deg)
            #     tgmc['YCTR_DEG'] = yctr_deg.astype(float)



            gmcs = []
            hiis = []
            s2n = 5
            for i in range(len(tgmc)):
                gmc = {}
                for j in tgmc.colnames:
                    gmc[j] = tgmc[j][i]
                if gmc['S2N'] >= s2n:
                    gmcs.append(gmc)

            j1  = 1
            for gmc in gmcs:
                gmc['CLOUDNUM'] = j1
                j1 += 1

            for i in range(len(thii)):
                hii = {}
                for j in thii.colnames:
                    hii[j] = thii[j][i]
                hiis.append(hii)

            i1 = 0
            for hii in hiis:
                hii['region_ID'] = i1
                i1+=1

            for hii in hiis:
                hii["PHANGS_INDEX"] = int(hii['region_ID']) + 1
                hii['RA'] = hii['cen_ra']
                hii['DEC'] = hii['cen_dec']
                hii['RADIUS'] = hii['region_circ_rad']
                hii['HA_VEL'] = hii["HA6562_VEL"]

            s2n = tgmc['S2N']  # signal to noise ratio
            ids2n = (np.where(s2n >= 5)[0]).tolist()  # signal to noise condition

            # if randomize=='gmc_prop' :
            #
            #     print(rand_ids_gmc_list.keys())
            #
            #     rand_ids = rand_ids_gmc_list[galnam]
            #     tgmc['SIGV_KMS'] = tgmc['SIGV_KMS'][rand_ids]
            #     tgmc['FLUX_KKMS_PC2'] = tgmc['FLUX_KKMS_PC2'][rand_ids]
            #     tgmc['TMAX_K'] = tgmc['TMAX_K'][rand_ids]
            #     tgmc['MLUM_MSUN'] = tgmc['MLUM_MSUN'][rand_ids]
            #     tgmc['VIRPARAM'] = tgmc['VIRPARAM'][rand_ids]
            #     tgmc['TFF_MYR'] = tgmc['TFF_MYR'][rand_ids]
            #     tgmc['SURFDENS'] = tgmc['SURFDENS'][rand_ids]






            dist_gal_Mpc = tgmc['DISTANCE_PC'][0] / 1e6  # [ids2n][0] / 1e6

            if gmc_catalog_version != 'v4':
                region_gmc = tgmc['REGION_INDEX'][ids2n]
            else:
                region_gmc = tgmc['ENV_BAR'][ids2n]


            angle_gmc = 90 - np.degrees(tgmc['POSANG'][ids2n])  # )

            major_gmc = 2*np.degrees(tgmc['MOMMAJ_PC'][ids2n]/ tgmc['DISTANCE_PC'][ids2n])
            minor_gmc = 2*np.degrees(tgmc['MOMMIN_PC'][ids2n]/ tgmc['DISTANCE_PC'][ids2n])







            radgmc = tgmc['RAD_PC'][ids2n]  # [ids2n]
            radgmc_deg = np.degrees(radgmc / (dist_gal_Mpc * 1e6))

            radnogmc = tgmc['RAD_NODC_NOEX'][ids2n]  # [ids2n]


            ragmc = tgmc['XCTR_DEG'][ids2n] # [ids2n]
            decgmc = tgmc['YCTR_DEG'][ids2n]  # [ids2n]

            #gmc_rgal_deg = np.sqrt((ragmc-gal_ra)**2 + (decgmc-gal_dec)**2)
            #gmc_rgal_pc = np.radians(gmc_rgal_deg)*dist_gal_Mpc*1e6
            #hii_rgal_pc = np.radians(hii_rgal_deg)*dist_gal_Mpc*1e6

            hii_rgal_deg = thii['deproj_dist']/3600

            hii_rgal_pc = np.radians(hii_rgal_deg) * dist_gal_Mpc*1e6
            gmc_rgal_pc = tgmc['RGAL_KPC'] * 1000


            gmc_rgal_deg = gmc_rgal_pc / (dist_gal_Mpc*1e6)



            hii_rgal_reff =( thii['deproj_dist'])#/reff
            gmc_rgal_reff = (gmc_rgal_deg*3600)#/reff

            print("rgal test, r_norm_max = %5.2f" %np.nanmedian(hii_rgal_reff))

            sigvgmc = tgmc['SIGV_KMS'][ids2n]  # [ids2n]
            fluxco = tgmc['FLUX_KKMS_PC2'][ids2n]  # [ids2n]
            tpgmc = tgmc['TMAX_K'][ids2n] # [ids2n]

            if muse == False:
                gmcvel = tgmc['VCTR_KMS'][ids2n]  # [ids2n]
            else:
                gmcvel = tgmc['VCTR_KMS'][ids2n]  # [ids2n]
                gmcvel = np.array(gmcvel)


            gmcvel = vgsr_to_vhel(vgsr=gmcvel, ra=ragmc, dec=decgmc)
            gmcvel = gmcvel / (1 - gmcvel / c)
            gmcvel = gmcvel - gal_radial_vel


            massco = tgmc['MLUM_MSUN'][ids2n]  # [ids2n]
            avir = tgmc['VIRPARAM'][ids2n]  # [ids2n]
            tauff = tgmc['TFF_MYR'][ids2n] * 10 ** 6  # [ids2n] * 10 ** 6
            Sigmamol_co = tgmc['SURFDENS'][ids2n]  # [ids2n]
            Sigmamol_vir = tgmc['MVIR_MSUN'][ids2n] / (radgmc ** 2 * math.pi)  # [ids2n]

            # =========================================================================================
            pc2cm = 3.086e18
            dist_cm = dist_gal_Mpc * 1e6 * pc2cm
            if muse != 'dr22':
                lhahiicorr = thii['HA6562_FLUX_CORR'] * 4 * np.pi * 1.e-20 * dist_cm * dist_cm  # erg/s [ids2n]
            else:
                lhahiicorr = thii['Lum_HA6562_CORR']
            sizehii = np.radians(radiushii) * dist_gal_Mpc * 1e6


            # ==========================================================================================
            # Write to DS9 readable table_muse
            writeds9(galnam, namegmc, rahii, dechii, pind, ragmc, decgmc, "all_regions", dirregions, name_end)
            # ==========================================================================================

            data = fits.open(dirmaps + str.upper(galnam) + namemap)
            header = data[0].header

            if matching_1o1 == True or matching_1om == True:
                mindist, inddist, a, b = checkdist_2D(rahii, dechii, ragmc, decgmc, sizehii, radgmc, dist_gal_Mpc)
                maxarea, indarea, idovergmc, idoverhii, idrejectedhii,idrejectedgmc = overlap_percent(rahii, dechii, major_gmc, minor_gmc, angle_gmc,
                                                                         decgmc, ragmc, radiushii, radgmc_deg, header,
                                                                         threshold_perc, gmcs, hiis, matching_1o1,
                                                                         matching_1om, vel_limit=vel_limit, randomize = randomize)
                idovergmc_s.append(idovergmc)
                idoverhii_s.append(idoverhii)




            elif matching_dist == True:
                mindist, inddist, idovergmc, idoverhii = checkdist_2D(rahii, dechii, ragmc, decgmc, sizehii, major_gmc,dist_gal_Mpc)
                idovergmc_s.append(idovergmc)
                idoverhii_s.append(idoverhii)

            if without_out == True:
                idoverhii, idovergmc, outliers, rms = without_outlier(idoverhii, idovergmc, hiivel[idoverhii],gmcvel[idovergmc])
                total_outliers.append(outliers)


            LumHacorr_galo = lhahiicorr[idoverhii]
            SizepcHII_galo = sizehii[idoverhii]
            sigmavHII_galo = sigmahii[idoverhii]
            metaliHII_galo = metalhii[idoverhii]
            varmetHII_galo = vamethii[idoverhii]
            velHII_galo = hiivel[idoverhii]
            HIImajor_galo = major[idoverhii]
            HIIminor_galo = minor[idoverhii]
            velGMC_galo = gmcvel[idovergmc]
            distas = np.array(((ragmc[idovergmc] - rahii[idoverhii]) ** 2 + (decgmc[idovergmc] - dechii[idoverhii]) ** 2) ** 0.5)  # array of distance between GMC(j) and all the HII regions
            distas = np.array(distas) * dist_gal_Mpc * 10e6 * np.pi / 180
            DisHIIGMC_galo = mindist[idovergmc] * dist_gal_Mpc * 10e6 * np.pi / 180
            SizepcGMC_galo = radgmc[idovergmc]
            sigmavGMC_galo = sigvgmc[idovergmc]
            MasscoGMC_galo = massco[idovergmc] / 1e5
            aviriaGMC_galo = avir[idovergmc]
            Sigmamole_galo = Sigmamol_co[idovergmc]
            FluxCOGMC_galo = fluxco[idovergmc]
            TpeakGMC_galo = tpgmc[idovergmc]
            tauffGMC_galo = tauff[idovergmc]
            region_index_galo = region_gmc[idovergmc]
            region_index_hii_galo = HIIregionindex[idoverhii]
            HII_reff_galo = hii_reff[idoverhii]
            HII_r25_galo = hii_r25[idoverhii]
            majorGMC_galo = major_gmc[idovergmc]
            minorGMC_galo = minor_gmc[idovergmc]
            angleGMC_galo = angle_gmc[idovergmc]
            RAgmc = ragmc[idovergmc]
            DECgmc = decgmc[idovergmc]
            RAhii = rahii[idoverhii]
            DEChii = dechii[idoverhii]
            phii = pind[idoverhii]

            Rgal_GMC_galo = gmc_rgal_reff[idovergmc] #pc
            Rgal_HII_galo = hii_rgal_reff[idoverhii] #pc

            # print("Saving variables in external file: Clouds_HIIregions_positions_%s%s.pickle" % (galnam, namegmc))
            # with open(('Clouds_HIIregions_positions_%s%s.pickle' % (galnam, namegmc)), "wb") as f:
            #   pickle.dump([galnam, rahii, dechii, pind, idoverhii, ragmc, decgmc, idovergmc], f)

            # ==========================================================================================
            # Write to ds9 readable file
            writeds9(galnam, namegmc, RAhii, DEChii, phii, RAgmc, DECgmc, "matched", dirregions, name_end)
            writeds9_vel(galnam, namegmc, RAhii, DEChii, phii, RAgmc, DECgmc, DisHIIGMC_galo, "matched", dirregions,
                         name_end)  # write region to ds9 files with size corresponding to GMC to HII distance
            # ==========================================================================================

            # Save in a single array for all galaxies -
            mindist = np.zeros(shape=(len(LumHacorr), 1))


            LumHacorr.append(lhahiicorr)
            SizepcHII.append(sizehii)
            sigmavHII.append(sigmahii)
            metaliHII.append(metalhii)
            varmetHII.append(vamethii)
            DisHIIGMC.append(mindist)
            HIImajor.append(major)
            HIIminor.append(minor)
            SizepcGMC.append(radgmc)
            sigmavGMC.append(sigvgmc)
            MasscoGMC.append(massco / 1e5)
            aviriaGMC.append(avir)
            Sigmamole.append(Sigmamol_co)
            TpeakGMC.append(tpgmc)
            tauffGMC.append(tauff)
            angleGMC.append(angle_gmc)
            minorGMC.append(minor_gmc)
            majorGMC.append(major_gmc)
            FluxCOGMC.append(fluxco)
            Rgal_HII.append(hii_rgal_pc)
            Rgal_GMC.append(gmc_rgal_pc)
            VelGMC.append(gmcvel)
            VelHII.append(hiivel)

            Rgal_HII_over.append(Rgal_HII_galo)
            Rgal_GMC_over.append(Rgal_GMC_galo)
            LumHacorrover.append(LumHacorr_galo)
            SizepcHIIover.append(SizepcHII_galo)
            sigmavHIIover.append(sigmavHII_galo)
            metaliHIIover.append(metaliHII_galo)
            varmetHIIover.append(varmetHII_galo)
            HIImajorover.append(HIImajor_galo)
            HIIminorover.append(HIIminor_galo)
            velHIIover.append(velHII_galo)
            velGMCover.append(velGMC_galo)
            DisHIIGMCover.append(DisHIIGMC_galo)
            SizepcGMCover.append(SizepcGMC_galo)
            sigmavGMCover.append(sigmavGMC_galo)
            MasscoGMCover.append(MasscoGMC_galo)
            aviriaGMCover.append(aviriaGMC_galo)
            Sigmamoleover.append(Sigmamole_galo)
            TpeakGMCover.append(TpeakGMC_galo)
            tauffGMCover.append(tauffGMC_galo)
            angleGMCover.append(angleGMC_galo)
            majorGMCover.append(majorGMC_galo)
            minorGMCover.append(minorGMC_galo)
            regionindexGMCover.append(region_index_galo)
            FluxCOGMCover.append(FluxCOGMC_galo)
            RAgmcover.append(RAgmc)
            DECgmcover.append(DECgmc)
            RAhiiover.append(RAhii)
            DEChiiover.append(DEChii)
            RAgmcall.append(ragmc)
            DECgmcall.append(decgmc)
            RAhiiall.append(rahii)
            DEChiiall.append(dechii)
            regionindexHIIover = region_index_hii_galo
            HII_reff_over = HII_reff_galo
            HII_r25_over = HII_r25_galo

            idoverhiis.append(idoverhii)
            idovergmcs.append(idovergmc)

            idrejectgmcs.append(idrejectedgmc)
            idrejecthiis.append(idrejectedhii)

            # Quantifying
            LHa_all = np.nansum(lhahiicorr[np.isfinite(lhahiicorr)])
            LHa_galo = np.nansum(LumHacorr_galo[np.isfinite(LumHacorr_galo)])

            LHa_all1 = lhahiicorr[np.isfinite(lhahiicorr)]
            LHa_galo1 = LumHacorr_galo[np.isfinite(LumHacorr_galo)]

            Lco_all = np.nansum(fluxco)
            Lco_galo = np.nansum(FluxCOGMC_galo)

            Lco_all1 = fluxco
            Lco_galo1 = FluxCOGMC_galo

            HaLumAllGal.append(LHa_all)
            HaLumAllGal_over.append(LHa_galo)
            COFluxAllGal.append(Lco_all)
            COFluxAllGal_over.append(Lco_galo)
            GMC_tot.append(len(fluxco))
            HII_tot.append(len(lhahiicorr))
            GMC_tot_overlap.append(len(FluxCOGMC_galo))
            HII_tot_overlap.append(len(LumHacorr_galo))


            print("Total HII regions: %i, overlapping HII regions: %i  (%5.1f %%)" % (len(lhahiicorr), len(idoverhii), len(idoverhii)*100/ len(lhahiicorr)))
            print("Total GMCs %i,  overlapping clouds: %i  (%5.1f %%)" % (len(fluxco), len(set(idovergmc)), len(set(idovergmc))*100/len(fluxco) ))
            print("Total Ha lum [erg/s]: %10.2E" % LHa_all)
            print(
                "Ha lum for those with GMCs overlapped[erg/s]: %10.2E %5.1f %%" % (LHa_galo, LHa_galo * 100 / LHa_all))
            print("Total CO Flux [K km/s pc2]: %10.2E" % Lco_all)
            assert isinstance(Lco_all, object)
            print("CO Flux for those with HII regions overlapped[erg/s]: %10.2E %5.1f %%" % (
                Lco_galo, Lco_galo * 100 / Lco_all))
            print("-" * 30)

            # Writing general information (number of pairs etc...) in txt file
            if i == 0:
                file = open("Table_muse%s.txt" % typegmc, "w")
                file.write(
                    r" \multirow{2}{*}{Galaxy} & \multicolumn{2}{c}{Total}& \multicolumn{2}{c}{Overlapping} & \multicolumn{2}{c}{H$\alpha$ luminosity [erg s$^{-1}$]} & \multicolumn{2}{c}{CO flux [K km s$^{-1}$ pc$^2$]}\\" + "\n")
                file.write(r"\cline{2-9}" + "\n")
                file.write(r"\hline" + "\n")
                file.write(
                    r" & \hii\ regions & GMCs & \% \hii\ & \% GMCs & Total & Overlapping & Total & Overlapping\\" + "\n")
                file.write(r"\hline" + "\n" + r"\noalign{\smallskip}" + "\n")
            if i != 0:
                file = open("Table_muse%s.txt" % typegmc, "a")
            if len(LumHacorr_galo) != 0:
                file.write("%s & %i & %i & %i\\%% & %i\\%% & %10.2E & %i\\%% & %10.2E & %i\\%% \\\ \n" % (
                    galnam, len(lhahiicorr), len(fluxco), round(len(LumHacorr_galo) * 100.00 / len(lhahiicorr)),
                    round(len(LumHacorr_galo) * 100 / len(fluxco)), LHa_all, round(LHa_galo * 100 / LHa_all), Lco_all,
                    round(Lco_galo * 100 / Lco_all)))
            file.close()

            if len(LumHacorr_galo) == 0:
                continue





    # ===========================================================================================================================================
    # Obtaining LHa/HIIsize^2 ratio
    ratlin = [None] * len(LumHacorrover)
    for j in range(len(galaxias)):
        #if galaxias[j] != 'ic5332':
        ratlin[j] = ((LumHacorrover[j]) / (SizepcHIIover[j] ))
        #else:
            #ratlin[j] = LumHacorrover[j]

    # =============================================================================================================================================
    # Saving the parameters to be read by another procedure.

    print("Plots of all galaxies together")

    arrayxax = [SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover,
                velHIIover, HIIminorover, HIImajorover, HIIangleover, Rgal_HII_over, HII_reff_over, HII_r25_over, HIIregionindex]

    arrayyay = [DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover,
                tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover, FluxCOGMCover, Rgal_GMC_over]

    labsxax = ['log(HII region size / 1 pc)',
               r'log(Luminosity H$\alpha$ / 1 erg s$^{-1}$)',
               r'log($\sigma_{\rm v}$ HII region / 1 km s$^{-1}$)', r'log(Lum H$\alpha$/HII region size$^2$)',
               'Metallicity',
               'Metallicity variation', 'HII velocity [km s$^{-1}$]', 'HII region minor axis (deg)',
               'HII region major axis (deg)', 'HII region PA (deg)', 'R_gal HII region (pc)', 'R_eff HII region)', 'R_25 HII region', 'HII region index']

    labsyay = ['log(Dist. HII-GMC / 1 pc)', r'log(M$_{\rm CO}$ / 10$^5$ M$_{\odot}$)', 'log(GMC size / 1 pc)',
               r'log($\Sigma_{\rm mol}$ / 1 M$_{\odot}$ pc$^{-2}$)', r'log($\sigma_{\rm v}$ / 1 km s$^{-1}$)', r'log($\alpha_{vir}$)',
               r'log(CO $T_{\rm peak}$ / 1 K)', r'log($\tau_{\rm ff}$) [yr]', 'GMC velocity [km s$^{-1}$]',
               'GMC position angle (deg)',
               'GMC major axis (deg)', 'GMC minor axis (deg)', 'GMC region index', 'log(CO flux / 1 K km/s pc2) ', 'R_gal GMC (pc)']

    labsxax1 = ['HII region size [pc]',
               r'Luminosity H$\alpha$ [erg s$^{-1}$]',
               r'$\sigma_{\rm v}$ HII region [km s$^{-1}$]', r'Lum H$\alpha$/HII region size$^2$',
               'Metallicity',
               'Metallicity variation', 'HII velocity [km s$^{-1}$]', 'HII region minor axis (deg)',
               'HII region major axis (deg)', 'HII region PA (deg)', 'R_gal HII region (pc)']

    labsyay1 = ['Dist. HII-GMC [pc]', r'M$_{\rm CO}$ [10$^5$ M$_{\odot}$]', 'GMC size [pc]',
               r'$\Sigma_{\rm mol}$', r'$\sigma_{\rm v}$ [km s$^{-1}$]', r'$\alpha_{vir}$',
               r'CO $T_{\rm peak}$ [K]', r'$\tau_{\rm ff}$ [yr]', 'GMC velocity [km s$^{-1}$]',
               'GMC position angle (deg)',
               'GMC major axis (deg)', 'GMC minor axis (deg)', 'GMC region index', 'CO flux [K km/s pc2] ',
               'R_gal GMC (pc)']


    print("Saving variables in external files.")
    with open((dir_script_data+'Galaxies_variables_GMC%s%s.pickle' % (namegmc, name_end)), "wb") as f:
        pickle.dump([galaxias, arrayyay, arrayxax, RAgmcover, DECgmcover, RAhiiover, DEChiiover, labsxax, labsyay, idoverhiis, idovergmcs, labsxax1, labsyay1], f)

    with open((dir_script_data+'Galaxies_variables_notover_GMC%s%s.pickle' % (namegmc, name_end)), "wb") as f:
        pickle.dump([SizepcHII, LumHacorr, sigmavHII, metaliHII, varmetHII, numGMConHII,  FluxCOGMC, HIIminor,
                     HIImajor, HIIangle, angleGMC, majorGMC, minorGMC, MasscoGMC, SizepcGMC, SizepcHII, Sigmamole, sigmavGMC, TpeakGMC, VelHII, VelGMC], f)

    if without_out == True:
        print(total_outliers)
        print(np.sum(total_outliers))
        print(rms)

    print(name_end + "\n")

    if matching_1o1 == True:
        with open((dir_script_data+'matching_1o1%s%s.pickle' % (namegmc, name_end)), "wb") as f:
            pickle.dump([idoverhii_s, idovergmc_s], f)
    if matching_1om == True:
        with open((dir_script_data+'matching_1om%s%s.pickle' % (namegmc, name_end)), "wb") as f:
            pickle.dump([idoverhii_s, idovergmc_s], f)
    if matching_dist == True:
        with open((dir_script_data+'dist%s%s.pickle' % (namegmc, name_end)), "wb") as f:
            pickle.dump([idoverhii_s, idovergmc_s],f)



    HaLumAllGal.append(LHa_all)
    HaLumAllGal_over.append(LHa_galo)
    COFluxAllGal.append(Lco_all)
    COFluxAllGal_over.append(Lco_galo)

    HaLumAllGal1.append(LHa_all1)
    HaLumAllGal_over1.append(LHa_galo1)
    COFluxAllGal1.append(Lco_all1)
    COFluxAllGal_over1.append(Lco_galo1)

    print("Total HII regions: %i, overlapping HII regions: %i, %5.1f %% HII regions are matched/paired" % (sum(HII_tot), sum(HII_tot_overlap), sum(HII_tot_overlap)*100/sum(HII_tot)))
    print("Total GMCs %i,  overlapping GMCs: %i,  %5.1f %% GMCs are matched/paired" % (sum(GMC_tot), sum(GMC_tot_overlap), sum(GMC_tot_overlap) *100 / sum(GMC_tot)))
    print("Total Ha lum [erg/s]: %10.2E" % sum(HaLumAllGal))
    print("Ha lum for those with GMCs overlapped[erg/s]: %10.2E %5.1f %%" % (sum(HaLumAllGal_over), sum(HaLumAllGal_over) * 100 / sum(HaLumAllGal)))
    print("Total CO Flux [K km/s pc2]: %10.2E" % sum(COFluxAllGal))
    assert isinstance(Lco_all, object)
    print("CO Flux for those with HII regions overlapped[erg/s]: %10.2E %5.1f %%" %  (sum(COFluxAllGal_over), sum(COFluxAllGal_over) * 100 / sum(COFluxAllGal)))

    # print('mean HII Lum (all) = %.2E  (%.2E)' %(Decimal(np.nanmean(HaLumAllGal1)), Decimal((np.nanmean(np.log10(LHa_all1))))))
    # print('mean HII Lum (paired) = %.2E  (%.2E)' %(Decimal(np.nanmean(HaLumAllGal_over1)), Decimal((np.nanmean(np.log10(LHa_galo1))))))
    #
    # print('mean CO flux (all) = %.2E  (%.2E)' %(Decimal(np.nanmean(COFluxAllGal1)), Decimal(np.log10(np.nanmean(COFluxAllGal1)))))
    # print('mean CO flux (paired) = %.2E  (%.2E)' %(Decimal(np.nanmean(COFluxAllGal_over1)), Decimal(np.log10(np.nanmean(COFluxAllGal_over1)))))




#extract_info(gmc_catalog="_native_", gmc_catalog_version='new', muse='dr2', matching="overlap_1om", outliers=True,threshold_perc=0.5 , vel_limit= 10000, randomize = True)
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.1 , vel_limit= 10000, randomize = '', symmetrical='gmc')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.2 , vel_limit= 10000, randomize = '', symmetrical='gmc')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.3 , vel_limit= 10000, randomize = '', symmetrical='gmc')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.4 , vel_limit= 10000, randomize = '', symmetrical='gmc')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.5 , vel_limit= 10000, randomize = '', symmetrical='gmc')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.6 , vel_limit= 10000, randomize = '', symmetrical='gmc')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.7 , vel_limit= 10000, randomize = '', symmetrical='gmc')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.8 , vel_limit= 10000, randomize = '', symmetrical='gmc')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.9 , vel_limit= 10000, randomize = '', symmetrical='gmc')

extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.1 , vel_limit= 10000, randomize = '', symmetrical='hii')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.2 , vel_limit= 10000, randomize = '', symmetrical='hii')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.3 , vel_limit= 10000, randomize = '', symmetrical='hii')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.4 , vel_limit= 10000, randomize = '', symmetrical='hii')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.5 , vel_limit= 10000, randomize = '', symmetrical='hii')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.6 , vel_limit= 10000, randomize = '', symmetrical='hii')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.7 , vel_limit= 10000, randomize = '', symmetrical='hii')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.8 , vel_limit= 10000, randomize = '', symmetrical='hii')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.9 , vel_limit= 10000, randomize = '', symmetrical='hii')

extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.1 , vel_limit= 10000, randomize = '', symmetrical='sym')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.2 , vel_limit= 10000, randomize = '', symmetrical='sym')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.3 , vel_limit= 10000, randomize = '', symmetrical='sym')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.4 , vel_limit= 10000, randomize = '', symmetrical='sym')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.5 , vel_limit= 10000, randomize = '', symmetrical='sym')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.6 , vel_limit= 10000, randomize = '', symmetrical='sym')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.7 , vel_limit= 10000, randomize = '', symmetrical='sym')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.8 , vel_limit= 10000, randomize = '', symmetrical='sym')
extract_info(gmc_catalog="_native_", gmc_catalog_version='v4', muse='dr22', matching="overlap_1om", outliers=True,threshold_perc=0.9 , vel_limit= 10000, randomize = '', symmetrical='sym')



