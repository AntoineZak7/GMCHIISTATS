import os
import math
import sys
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import constants as ct
import pickle
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf
from sklearn.linear_model import LinearRegression
import seaborn as sns
from astropy.coordinates import FK5
from regions import EllipseSkyRegion, CircleSkyRegion
from astropy import wcs
from astropy.io import fits
from itertools import chain


np.set_printoptions(threshold=sys.maxsize)
sns.set(style="white", color_codes=True)
c = 299792.458
# ===================================================================================

def extract_info(gmc_catalog, new_muse, overlap_matching, outliers,threshold_perc):

    def name(overperc, without_out, new_muse):
        name_append = ['perc_matching_', 'with_outliers', 'without_outliers', 'new_muse_', 'old_muse_', str(threshold_perc)]

        if new_muse == True:
            name_end = name_append[3]
            if overperc == True:
                name_end = name_end + name_append[0] +name_append[5]
                if without_out == True:
                    name_end = name_end + name_append[2]
                else:
                    name_end = name_end + name_append[1]

        else:
            name_end = name_append[4]
            if overperc == True:
                name_end = name_end + name_append[0] + name_append[5]
                if without_out == True:
                    name_end = name_end + name_append[2]
                else:
                    name_end = name_end + name_append[1]
        return name_end

    def vgsr_to_vhel(ra, dec, vgsr):  # This is a copy paste of a gala v0.1.0 routine https://gala-astro.readthedocs.io/en/v0.1.0/_modules/gala/coordinates/core.html#vhel_to_vgsr
        """
        Convert a radial velocity in the Galactic standard of rest (GSR) to
        a barycentric radial velocity.

        Parameters
        ----------
        coordinate : :class:`~astropy.coordinates.SkyCoord`
            An Astropy SkyCoord object or anything object that can be passed
            to the SkyCoord initializer.
        vgsr : :class:`~astropy.units.Quantity`
            GSR line-of-sight velocity.
        vcirc : :class:`~astropy.units.Quantity`
            Circular velocity of the Sun.
        vlsr : :class:`~astropy.units.Quantity`
            Velocity of the Sun relative to the local standard
            of rest (LSR).

        Returns
        -------
        vhel : :class:`~astropy.units.Quantity`
            Radial velocity in a barycentric rest frame.

        """
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
        # tmpoverid = np.argwhere(dists[1,idgmc] < (sizehii[idhii]*2)) # HII with GMCs < 2 sizeHII type: List[int]
        tmpoverid = np.argwhere(mindist[idgmc] < addsize)
        overid = [int(item) for item in tmpoverid]
        idovergmc = [idgmc[item] for item in overid]
        idoverhii = [idhii[item] for item in overid]
        return mindist, inddist, idovergmc, idoverhii

    def findist(rahii, dechii, ragmc, decgmc,  distance):
        distas = np.array(((ragmc[idovergmc] - rahii[idoverhii]) ** 2 + (decgmc[idovergmc] - dechii[idoverhii]) ** 2) ** 0.5)  # array of distance between GMC(j) and all the HII regions
        distas = np.array(distas) * distance[0]
        print(distas)
        return distas          # indexes of where the distance is <= radius hii + radius gmc


    def overlap_percent(rahii, dechii, major_gmc, minor_gmc, angle_gmc, decgmc, ragmc, radiushii, radgmc, header,
                        threshold_perc):
        WC = wcs.WCS(header)
        # =============================================== Checking which GMCs have a HII region associated in terms of distance ===================
        ind_gmc_dist = [] # indexes of gmcs that respect the distance condition
        dists = np.zeros((len(ragmc)),
                         dtype=object)  # initializing array, j : index of gmc, and for each gmc the associated hii regions, so array of not constant shape
        for j in range(len(ragmc)):
            distas = ((np.full(fill_value=ragmc[j], shape=len(radiushii)) - rahii) ** 2 + (np.full(fill_value=decgmc[j],
                                                                                                   shape=len(
                                                                                                       radiushii)) - dechii) ** 2) ** 0.5  # array of distance between GMC(j) and all the HII regions
            distas = np.array(distas)
            aradgmc = np.full(fill_value=radgmc[j], shape=len(radiushii))  # array filled with radgmc(j) value
            dists[j] = np.where(
                distas <= np.array(radiushii) + aradgmc)  # indexes of where the distance is <= radius hii + radius gmc

            if str(dists[
                       j]) != '(array([], dtype=int64),)':  # if a gmc has zero hii region where distance is <= radius hii + radius gmc, then its index is not kept
                ind_gmc_dist.append(j)
        # ========================================================================================================================================

        ind_hii = []  # indexes of the hii regions that pass the threshold condition
        ind_gmc = []  # indexes of the gmcs that pass the threshold condition
        maxarea = []  # maximum overlapping area for the gmc in %
        maxarea_hii = []  # maximum overlapping area for the hii region in %
        area_gmc_pixel = [] # area of the gmc in pixels


        name = "_ha.fits"  # "_Hasub_flux.fits"
        hdul = fits.open(dirmaps + str.upper(galaxias[0]) + name)
        image_data = hdul[0].data
        fig = plt.figure()
        fig.add_subplot(111, projection=WC)
        plt.imshow(image_data, vmax=6000, cmap='Greys')

        for j in ind_gmc_dist:  # range(len(ragmc)):  # for the gmc regions that passed the precedent condition, "local" masks are calculated from astropy regions using table angles, radius etc...

            center_gmc = SkyCoord(ragmc[j] * u.deg, decgmc[j] * u.deg, frame=FK5, unit='deg')
            ellipse_gmc = EllipseSkyRegion(center=center_gmc, height=major_gmc[j] * u.deg, width=minor_gmc[j] * u.deg,
                                           angle=angle_gmc[j] * u.deg)
            ellipse_pixel_gmc = ellipse_gmc.to_pixel(wcs=WC)

            gxmin = ellipse_pixel_gmc.bounding_box.ixmin  # getting bounding box limits to transform "local mask" to "galaxy size" masks later
            gxmax = ellipse_pixel_gmc.bounding_box.ixmax
            gymin = ellipse_pixel_gmc.bounding_box.iymin
            gymax = ellipse_pixel_gmc.bounding_box.iymax

            print('%s/%s' % (j, len(ragmc)))  # just to check the loop

            ind = []  # indexes of hii regions associated to the GMC of index j
            for k in dists[
                j]:  # dist[j] is an array of no constant shape (dtype object), so cannot be manipulated with numpy but as a string. An ind list is filled with its values
                l = 0
                for i in str(k).replace('[', '').replace(']', '').split()[:]:
                    # print(str(k).replace('[','').replace(']','').split()[l])
                    ind.append(int(str(k).replace('[', '').replace(']', '').split()[l]))
                    l += 1

            intersec = []  # size in pixels of the overlapping region between gmcs and hii regions
            intersec_area_gmc = [] # size in pixels of the gmc
            intersec_id = []  # id of the gmc region overlapping
            intersec_hii = []  # id of the hii region overlapping


            for i in ind:
                center_hii = SkyCoord(rahii[i] * u.deg, dechii[i] * u.deg, frame=FK5, unit='deg')

                circle_hii = CircleSkyRegion(center=center_hii, radius=radiushii[i] * u.deg)
                circle_pixel_hii = circle_hii.to_pixel(wcs=WC)

                hxmin = circle_pixel_hii.bounding_box.ixmin  # getting bounding box limits to transform "local mask" to "galaxy size" masks but for the hii region this time
                hxmax = circle_pixel_hii.bounding_box.ixmax
                hymin = circle_pixel_hii.bounding_box.iymin
                hymax = circle_pixel_hii.bounding_box.iymax

                gmc_area = 0  # initializing the areas at 0 in case no overlap  is found
                hii_area = 0
                mask_gmc = ellipse_pixel_gmc.to_mask().data
                mask_hii = circle_pixel_hii.to_mask().data

                ax1 = header['NAXIS1'] +100 # getting the sizes of the galaxy image where the regions are projected
                ax2 = header['NAXIS2']+100
                mask_hii_all = np.zeros(shape=(ax2, ax1))  # initializing "galaxy size" masks
                mask_gmc_all = np.zeros(shape=(ax2, ax1))

                for l in range(gymax - gymin):  # transforming "local masks" to galaxy size masks (gmc)
                    for c in range(gxmax - gxmin):
                        mask_gmc_all[l + gymin][c + gxmin] = mask_gmc[l][c]

                for l in range(hymax - hymin):  # transforming "local masks" to galaxy size masks (hii region)
                    for c in range(hxmax - hxmin):
                        mask_hii_all[l + hymin][c + hxmin] = mask_hii[l][c]

                for l in range(
                        gymax - gymin):  # calculating gmc area in pixels (probably more straightforward way to do it ?)
                    for c in range(gxmax - gxmin):
                        if mask_gmc[l][c] == 1:
                            gmc_area += 1

                for l in range(
                        hymax - hymin):  # calculating hii region area in pixels (probably more straightforward way to do it )
                    for c in range(hxmax - hxmin):
                        if mask_hii[l][c] == 1:
                            hii_area += 1

                count1 = sum([list(i).count(2) for i in (
                            mask_gmc_all + mask_hii_all)])  # adding the global hii mask to the global gmc masks and counting the number of values = 2, i.e overlapping area = (number of pixels which values are 2)

                if count1 != 0:
                    count = count1 / gmc_area  # overlapping area in terms of percentage of the gmc
                    count_hii = count1 / hii_area  # overlapping area in terms of percentage of the hii region

                    if count >= threshold_perc:
                        intersec.append(
                            count)  # add the overlapping percentage of each hii region that overlap with the gmc of index j
                        intersec_id.append(i)
                        intersec_hii.append(count_hii)
                        intersec_area_gmc.append(count1)

            if intersec != []:  # keeping only one hii region, the one with the most overlapping area
                maxarea.append(np.nanmax(intersec))
                ind_hii.append(int(
                    intersec_id[np.nanargmax(intersec)]))  # keeping only the hii region with the most common surface
                maxarea_hii.append(intersec_hii[np.nanargmax(intersec)])
                ind_gmc.append(j)
                area_gmc_pixel.append(intersec_area_gmc[np.nanargmax(intersec)])

        # plt.show()
        ind_hii = np.array(ind_hii)
        ind_gmc = np.array(ind_gmc)
        maxarea_hii = np.array(maxarea_hii)
        maxarea = np.array(maxarea)
        area_gmc_pixel = np.array(area_gmc_pixel)

        hii_ind = []  # hii indexes to keep
        gmc_ind = []  # gmc indexes to keep

        for j in range(len(ind_hii)):
            if len(np.where(ind_hii[j] == ind_hii)) != 0:  # checking which hii indexes appear more than once
                #indd = ind_hii[np.where(ind_hii == ind_hii[j])]
                area_ind = maxarea[np.where(ind_hii == ind_hii[j])] # overlapping areas of the different gmcs associated to the hii region
                area_max = maxarea[j] # overlapping area of the gmc[j] with its hii region
                if area_max == np.nanmax(area_ind): # check if the overlapping area is greater than for the others hii regions (first condition)
                    if len(ind_hii[np.where(maxarea == area_max)]) > 1: # a lot of gmcs are located inside hii regions, so the are in % = 100% and so a condition on the gmc area in pixel (or maybe another) is required
                        area_ind_gmc = area_gmc_pixel[np.where(ind_hii == ind_hii[j])]
                        area_j_gmc = area_gmc_pixel[j]
                        if area_j_gmc == np.nanmax(area_ind_gmc): # from the gmcs with a 100% overlap area, only the largest one is kept. Maybe use a velocity condition instead ? (second condition)
                            hii_ind.append(int(ind_hii[j]))
                            gmc_ind.append(int(ind_gmc[j]))
                    else: # if first condition is enough
                        hii_ind.append(int(ind_hii[j]))
                        gmc_ind.append(int(ind_gmc[j]))  # ((ind_gmc[np.where(maxarea_hii == area_max_hii)]))
            else: # if the hii region is only associated to one gmc
                hii_ind.append(int(ind_hii[j]))
                gmc_ind.append((ind_gmc[np.where(ind_hii == ind_hii[j])]))

        idovergmc = np.array(gmc_ind)
        idoverhii = np.array(hii_ind)

        return maxarea, ind_gmc, idovergmc, idoverhii

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

    def outnan(var1):  # Taking out the NaN values
        nonan = var1[~np.isnan(var1)]
        return nonan


    # ===================================================================================
    typegmc = gmc_catalog#'_native_'  # native, _150pc_, _120pc_, _90pc_, _60pc_
    overperc = overlap_matching
    without_out = not outliers
    name_end = name(overperc, without_out, new_muse)
    # ---------------------------------------------

    # Defining paths to the files and files names !!! TO CHANGE !!!
 #======================================================================================================================================================================"
    dirhii = "/home/antoine/Internship/muse_hii/"  # old muse tables directory
    dirhii_new = "/home/antoine/Internship/muse_hii_new/"  # new muse tables directory
    dirgmc = '/home/antoine/Internship/gmccats_st1p5_amended/'  # gmc tables directory

    dirregions1 = "/home/antoine/Internship/ds9tables/Muse/Old_Muse/"  # Must create a directory to save the region ds9 files before running the code for the first time
    dirregions2 = "/home/antoine/Internship/ds9tables/Muse/New_Muse/" # same but for new muse catalog

    dirmaps = "/home/antoine/Internship/Galaxies/New_Muse/"  # maps directory (to plot the overlays)

    dirplots1 = "/home/antoine/Internship/Plots_Muse/Old_Muse/"  # directories to save the plots (old muse catalog)
    dirplots2 = "/home/antoine/Internship/Plots_Muse/New_Muse/" # directories to save the plots (new muse catalog)
#========================================================================================================================================================================="


    # ---defining which directory to store the plots in----#
    if new_muse == False:
        dirplots = dirplots1
        dirregions = dirregions1
    else:
        dirplots = dirplots2
        dirregions = dirregions2

    with open(('Directories_muse.pickle'), "wb") as f:
        pickle.dump([dirhii, dirgmc, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots], f)

    # -----------------------------------------------------#
    if typegmc == "_native_":
        namegmc = "_12m+7m+tp_co21%sprops" % typegmc
        namegmc1 = "_12m+7m+tp_co21%sprops" % typegmc

    else:
        typegmc1 = "_" + str(typegmc.split("_")[1]) + "_"
        print(typegmc1)
        namegmc = "_12m+7m+tp_co21%sprops" % typegmc
        namegmc1 = "_12m+7m+tp_co21%sprops" % typegmc1
    #typegmc = typegmc + typegmc1
    dirgmc = ('%scats%samended/' % (dirgmc, typegmc))
    namemap = '_ha.fits'
    name_new_muse = 'Nebulae_Catalogue.fits'

    # -----------------------------------------

    pdf1 = fpdf.PdfPages("%sHistograms_all_GMC_muse%s%s.pdf" % (dirplots, typegmc, typegmc))
    pdf2 = fpdf.PdfPages("%sCorrelations_galbygal_GMC_muse%s%s.pdf" % (dirplots, typegmc, typegmc))
    pdf3 = fpdf.PdfPages("%sCorrelations_allgals_GMC_muse%s%s.pdf" % (dirplots, typegmc, typegmc))

    # =======================================================================================

    # Defining empty vectors to save the variables of all galaxies
    total_outliers = []
    galaxias = []
    LumHacorr = []
    if new_muse == False:  # Remove galactocentric distance dist to compare old and new muse catalogs more easily (no galactocentric distance for new muse ?)
        GaldisHII = []
    SizepcHII = []
    sigmavHII = []
    metaliHII = []
    varmetHII = []
    HIIaxisratio = []
    HIImajor = []
    HIIminor = []
    HIIangle = []

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

    numGMConHII = []
    LumHacorrover = []

    if new_muse == False:
        GaldisHIIover = []

    SizepcHIIover = []
    sigmavHIIover = []
    metaliHIIover = []
    varmetHIIover = []
    velHIIover = []
    HIIaxisratioover = []
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

    RAgmcover = []
    DECgmcover = []
    RAhiiover = []
    DEChiiover = []
    RAgmcall = []
    DECgmcall = []
    RAhiiall = []
    DEChiiall = []

    # ============================================================================================================
    # Limits in the properties of HIIR and GMCs
    # if limots.py has not been run, comment the following line
    # xlim, ylim, xx, yy = pickle.load(open('limits_properties.pickle', "rb"))
    # =============================================================================================================
    hiicats = [f for f in os.listdir(dirhii)]

    if new_muse == True:
        table_new_muse = Table.read("%s%s" % (dirhii_new, name_new_muse))
        w = 0
        galaxies_name = ['IC5332']

        for i in range(len(table_new_muse['gal_name']) - 1):
            if galaxies_name[w] != str(table_new_muse['gal_name'][i]):
                galaxies_name.append(str(table_new_muse['gal_name'][i]))
                w += 1

        hiicats = galaxies_name
        w = 1

    # Loop in all galaxies. Do histograms and individual galaxy plots.
    print("Starting loop in all galaxies [i], do histograms and individual galaxy plots")
    for i in range(len(hiicats)):
        if new_muse == True:
            galnam = str.lower(galaxies_name[i])
            table = table_new_muse
        else:
            galnam = hiicats[i].split("_")[0]

        # print("%s%s%s.fits" % (dirgmc, galnam, namegmc))
        # print(galnam)
        if os.path.isfile(("%s%s%s.fits" % (dirgmc, galnam, namegmc1))):
            galaxias.append(galnam)

            print("-*" * 20)
            print("Galaxy name: %s" % galnam)
            print("-------------------------")

            if new_muse == False:
                thii = Table.read(dirhii + hiicats[i])
                # Information of the galaxy
                PAgal = thii['PA'][0]  # PA of galaxy (deg)
                inclgal = thii['INCL'][0]  # incl inclination of galaxy (deg)
                racen = thii['RA_CENTER'][0]  # Right ascension of the galaxy center (deg)
                deccen = thii['DEC_CENTER'][0]  # Declinaison of the galaxy center (deg)

                # Information of individual HII regions
                rahii = thii['RA']  # Right ascension of the region in degrees
                dechii = thii['DEC']  # declinaison of the region in degrees
                pind = thii['PHANGS_INDEX']
                hiivel = thii['HA_VEL'] + 1.4  # Veclocity (km/s)

                angle = 90 - thii['PA']  # Position angle of the HII region (deg)
                major = thii[
                            'SIZE_OBS'] / 3600  # np.degrees( thii['SIZE'] / (thii['DISTMPC'][0] * 1e6))# thii['SIZE_OBS'] / 3600  # major axis of the HII region (deg)
                minor = thii['SIZE_OBS'] / 3600  # minor axis of the HII region (deg)
                radiushii = thii['SIZE_OBS'] / 3600

                # lhahiicorr = thii['LHA']*10.**(0.4*np.interp(6563, [5530,6700],[1.,0.74])*3.1*thii['EBV']) # Correcting
                # lhahiicorr = thii['CLHA']  # erg/s
                # elhahiicorr = thii['CLHA_ERR'] # Error in Lhalpha extinction corrected
                sizehii = thii['SIZE']  # pc
                sigmahii = thii['HA_SIG']
                metalhii = thii['METAL_SCAL']
                vamethii = thii['OFF_SCAL']
                disthii = thii['DISTMPC'][0]

            else:
                thii = table[np.where(table['gal_name'] == str.upper(galnam))]
                thii = thii[np.where(thii['BPT_NII'] == 0)]
                thii = thii[np.where(thii['BPT_SII'] == 0)]

                # Information of the galaxy
                # PAgal = thii['PA'][0]  # PA of galaxy (deg)
                # inclgal = thii['INCL'][0]  # incl inclination of galaxy (deg)
                # racen = thii['RA_CENTER'][0]  # Right ascension of the galaxy center (deg)
                # deccen = thii['DEC_CENTER'][0]  # Declinaison of the galaxy center (deg)

                # Information of individual HII regions
                rahii = thii['cen_ra']  # Right ascension of the region in degrees
                dechii = thii['cen_dec']  # declinaison of the region in degrees
                pind = np.full(shape=len(thii['cen_ra']), fill_value=int(w))
                w += 1
                hiivel = (thii['HA6562_VEL'] + 1.4)  # Veclocity (km/s)
                hiivel = np.array(hiivel)

                # angle = thii['PA']  # Position angle of the HII region (deg)
                major = thii[
                            'region_circ_rad'] / 3600  # np.degrees( thii['SIZE'] / (thii['DISTMPC'][0] * 1e6))# thii['SIZE_OBS'] / 3600  # major axis of the HII region (deg)
                minor = thii['region_circ_rad'] / 3600  # minor axis of the HII region (deg)
                radiushii = thii['region_circ_rad'] / 3600

                sizehii = np.sqrt(thii['region_area']) * thii['kpc_per_pixel'] * 1000  # pc
                sigmahii = thii['HA6562_SIGMA']
                metalhii = thii['met_scal']
                vamethii = thii['Delta_met']

            # =============================================================
            # Corresponding CO data and GMC catalog
            # --------------------------------------------

            print("Reading table")

            tgmc = Table.read(("%s%s%s.fits" % (dirgmc, galnam, namegmc1)))

            s2n = tgmc['S2N']  # signal to noise ratio
            ids2n = (np.where(s2n > 5)[0]).tolist()  # signal to noise condition

            dist_gal_Mpc = tgmc['DISTANCE_PC'][ids2n][0] / 1e6

            region_gmc = tgmc['REGION_INDEX']

            angle_gmc = 90 - np.degrees(tgmc['POSANG'][ids2n])
            major_gmc = np.degrees(tgmc['MOMMAJ_PC'][ids2n] / tgmc['DISTANCE_PC'][ids2n])
            minor_gmc = np.degrees(tgmc['MOMMIN_PC'][ids2n] / tgmc['DISTANCE_PC'][ids2n])

            sigvgmc = tgmc['SIGV_KMS'][ids2n]
            ragmc = tgmc['XCTR_DEG'][ids2n]
            decgmc = tgmc['YCTR_DEG'][ids2n]
            fluxco = tgmc['FLUX_KKMS_PC2'][ids2n]
            radgmc = tgmc['RAD_PC'][ids2n]
            radgmc_deg = np.degrees(radgmc / (dist_gal_Mpc * 1e6))
            radnogmc = tgmc['RAD_NODC_NOEX'][ids2n]
            tpgmc = tgmc['TMAX_K'][ids2n]
            if new_muse == False:
                gmcvel = tgmc['VCTR_KMS'][ids2n]
            else:
                gmcvel = tgmc['VCTR_KMS'][ids2n]
                gmcvel = np.array(gmcvel)

            gmcvel = vgsr_to_vhel(vgsr=gmcvel, ra=ragmc, dec=decgmc)
            gmcvel = gmcvel / (1 - gmcvel / c)

            if galnam != 'ngc1672' and galnam != 'ic5332':
                massco = tgmc['MLUM_MSUN'][ids2n]
                avir = tgmc['VIRPARAM'][ids2n]
                tauff = tgmc['TFF_MYR'][ids2n] * 10 ** 6
                Sigmamol_co = tgmc['SURFDENS'][ids2n]
            else:
                massco = tgmc['FLUX_KKMS_PC2'][ids2n] * 4.3 / 0.69
                avir = 5. * (
                            sigvgmc * 1e5) ** 2 * radgmc * ct.pc.cgs.value / massco / ct.M_sun.cgs.value / ct.G.cgs.value
                Sigmamol_co = massco / (radgmc ** 2 * math.pi)
                RAD3 = (radgmc ** 2 * 50) ** 0.33
                rhogmc = 1.26 * massco / (4 / 3 * math.pi * RAD3 ** 3) * ct.M_sun.cgs.value / ct.pc.cgs.value ** 3
                arg = 3. * math.pi / (32 * ct.G.cgs.value * rhogmc)
                tauff = [math.sqrt(f) / 365 / 24 / 3600 for f in arg]
                tauff = np.array(tauff)

            Sigmamol_vir = tgmc['MVIR_MSUN'][ids2n] / (radgmc ** 2 * math.pi)

            # =========================================================================================
            # Correct LHa by the new distance measurement.
            if new_muse == False:
                lhahiicorr = thii['CLHA'] * (dist_gal_Mpc / disthii) ** 2  # erg/s
            else:
                pc2cm = 3.086e18
                dist_cm = dist_gal_Mpc * 1e6 * pc2cm
                lhahiicorr = thii['HA6562_FLUX_CORR'] * 4 * np.pi * 1.e-20 * dist_cm * dist_cm  # erg/s

            # ==========================================================================================
            # Write to DS9 readable table
            writeds9(galnam, namegmc, rahii, dechii, pind, ragmc, decgmc, "all_regions", dirregions, name_end)
            # ==========================================================================================

            if without_out == True:
                idoverhii, idovergmc, outliers, rms = without_outlier(idoverhii, idovergmc, hiivel[idoverhii],
                                                                      gmcvel[idovergmc])
                total_outliers.append(outliers)

            data = fits.open(dirmaps + str.upper(galnam) + namemap)
            header = data[0].header
            #threshold = 50
            #threshold_perc = 0.5

            if overperc == True:
                mindist, inddist, a, b = checkdist_2D(rahii, dechii, ragmc, decgmc, sizehii, radgmc,
                                                      dist_gal_Mpc)
                maxarea, indarea, idovergmc, idoverhii = overlap_percent(rahii, dechii, major_gmc, minor_gmc, angle_gmc,
                                                                         decgmc, ragmc, radiushii, radgmc_deg, header,
                                                                         threshold_perc)
            else:
                mindist, inddist, idovergmc, idoverhii = checkdist_2D(rahii, dechii, ragmc, decgmc, sizehii, radgmc,
                                                                      dist_gal_Mpc)

            LumHacorr_galo = lhahiicorr[idoverhii]
            # if new_muse == False:
            # GaldisHII_galo = rgalhii[idoverhii]
            SizepcHII_galo = sizehii[idoverhii]
            sigmavHII_galo = sigmahii[idoverhii]
            metaliHII_galo = metalhii[idoverhii]
            varmetHII_galo = vamethii[idoverhii]

            velHII_galo = hiivel[idoverhii]
            # HIIangle_galo = angle[idoverhii]
            HIImajor_galo = major[idoverhii]
            HIIminor_galo = minor[idoverhii]
            velGMC_galo = gmcvel[idovergmc]
            distas = np.array(((ragmc[idovergmc] - rahii[idoverhii]) ** 2 + (
                        decgmc[idovergmc] - dechii[idoverhii]) ** 2) ** 0.5)  # array of distance between GMC(j) and all the HII regions
            #print(dist_gal_Mpc*10e6 * np.pi/180)

            distas = np.array(distas) * dist_gal_Mpc*10e6 * np.pi/180
            #print(distas)
            #test = findist(rahii[idoverhii], dechii[idoverhii], ragmc[idovergmc], decgmc[idovergmc],dist_gal_Mpc[idovergmc])
            #print(test)
            DisHIIGMC_galo = mindist[idovergmc] * dist_gal_Mpc*10e6 * np.pi/180
            SizepcGMC_galo = radgmc[idovergmc]
            sigmavGMC_galo = sigvgmc[idovergmc]
            MasscoGMC_galo = massco[idovergmc] / 1e5
            aviriaGMC_galo = avir[idovergmc]
            Sigmamole_galo = Sigmamol_co[idovergmc]
            FluxCOGMC_galo = fluxco[idovergmc]
            TpeakGMC_galo = tpgmc[idovergmc]
            tauffGMC_galo = tauff[idovergmc]
            region_index_galo = region_gmc[idovergmc]

            veloffset_galo = velHII_galo - velGMC_galo

            majorGMC_galo = major_gmc[idovergmc]
            minorGMC_galo = minor_gmc[idovergmc]
            angleGMC_galo = angle_gmc[idovergmc]
            # HIIaxisratio_galo = axis_ratio[idoverhii]

            RAgmc = ragmc[idovergmc]
            DECgmc = decgmc[idovergmc]
            RAhii = rahii[idoverhii]
            DEChii = dechii[idoverhii]
            phii = pind[idoverhii]

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
            LumHacorr.append(lhahiicorr)
            mindist = np.zeros(shape=(len(LumHacorr), 1))

            # if new_muse == False:
            # GaldisHII.append(rgalhii)
            SizepcHII.append(sizehii)
            sigmavHII.append(sigmahii)
            metaliHII.append(metalhii)
            varmetHII.append(vamethii)
            DisHIIGMC.append(mindist)
            HIImajor.append(major)
            HIIminor.append(minor)
            # HIIangle.append(angle)

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

            LumHacorrover.append(LumHacorr_galo)
            # if new_muse == False:
            # GaldisHIIover.append(GaldisHII_galo)
            SizepcHIIover.append(SizepcHII_galo)
            sigmavHIIover.append(sigmavHII_galo)
            metaliHIIover.append(metaliHII_galo)
            varmetHIIover.append(varmetHII_galo)
            # HIIangleover.append(HIIangle_galo)
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

            RAgmcover.append(RAgmc)
            DECgmcover.append(DECgmc)
            RAhiiover.append(RAhii)
            DEChiiover.append(DEChii)

            RAgmcall.append(ragmc)
            DECgmcall.append(decgmc)
            RAhiiall.append(rahii)
            DEChiiall.append(dechii)

            # Quantifying
            LHa_all = np.nansum(lhahiicorr[np.isfinite(lhahiicorr)])
            LHa_galo = np.nansum(LumHacorr_galo[np.isfinite(LumHacorr_galo)])

            Lco_all = np.nansum(fluxco)
            Lco_galo = np.nansum(FluxCOGMC_galo)

            print("Total HII regions: %i, overlapping clouds: %i" % (len(lhahiicorr), len(LumHacorr_galo)))
            print("Total GMCs %i,  overlapping HII regions: %i" % (len(fluxco), len(FluxCOGMC_galo)))
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
            # ==========================================================================================
            ## PLOTS
            # ==========================================================================================
            print("Starting plots ")
            title_font = {'fontname': 'Arial', 'size': '18', 'color': 'black', 'weight': 'normal',
                          'verticalalignment': 'bottom'}
            marker_style = dict(markersize=3)

            # ------------------------------------------------------------------------------------------
            # Histograms
            # Taking out the NaNs values
            Sigmamol_co_n = outnan(Sigmamol_co)
            avir_n = outnan(avir)
            massco_n = outnan(massco)
            sigv_kms_n = outnan(sigvgmc)
            rad_nodc_noex_n = outnan(radnogmc)
            dists_n = outnan(mindist)
            sigmahii_n = outnan(sigmahii)
            lhacorrall_n = outnan(lhahiicorr[(np.array(lhahiicorr) < 1e50)])
            sizehii_n = outnan(sizehii)
            # if new_muse == False:
            #   rgalhii_n = outnan(rgalhii)
            metalhii_n = outnan(metalhii[(np.abs(metalhii) < 30)])
            vamethii_n = outnan(vamethii[(np.abs(vamethii) < 30)])
            sigmahii_cl_n = outnan(sigmavHII_galo)
            lhacorrall_cl_n = outnan(LumHacorr_galo)
            sizehii_cl_n = outnan(SizepcHII_galo)
            # if new_muse == False:
            #   rgalhii_cl_n = outnan(GaldisHII_galo)
            metalhii_cl_n = outnan(metaliHII_galo)
            vamethii_cl_n = outnan(varmetHII_galo)

            arrays = [sizehii_n, lhacorrall_n, sigmahii_n, metalhii_n, vamethii_n, dists_n, rad_nodc_noex_n,
                      sigv_kms_n, massco_n, avir_n, Sigmamol_co_n]  # rgalhii_n,
            labsname = ['Galactocentric radius [kpc]', 'HII region size [pc]', r'Luminosity H$\alpha$ [erg/s]',
                        r'$\sigma_{v}$ HII region [km/s]', 'Metallicity', 'Variation metallicity',
                        'Distance  HII-GMC [pc]',
                        'GMC size [pc]', r'$\sigma_v$ [km/s]', r'Mass$_{CO}$ [M$_{\odot}$]', r'$\alpha_{vir}$',
                        r'$\Sigma_{mol}$']
            arraycl = [sizehii_cl_n, lhacorrall_cl_n, sigmahii_cl_n, metalhii_cl_n, vamethii_cl_n]  # rgalhii_cl_n,

            fig, axs = plt.subplots(6, 2, figsize=(8, 12))
            plt.subplots_adjust(hspace=0.4)
            fig.suptitle('Galaxy %s - Histograms' % galnam.upper(), fontsize=15, va='top')
            axs = axs.ravel()  # to put in 1D the axs

            print(" Histograms saved in: Histograms_all_GMC%s.pdf" % namegmc)

            for z in range(len(arrays)):
                if len(arrays[z]) == 0:
                    minv = 0
                    maxv = 0
                else:
                    if len(arrays[z]) == 1:
                        minv = arrays[z][0]
                        maxv = arrays[z][0] + 1
                    else:

                        minv = np.min(arrays[z])
                        maxv = np.max(arrays[z])

                if z < 5 and z != 2:
                    axs[z].hist(arrays[z], alpha=0.5, bins=np.arange(minv, maxv, (maxv - minv) / 20),
                                label=['All HII regions'])
                    axs[z].hist(arraycl[z], alpha=0.5, bins=np.arange(minv, maxv, (maxv - minv) / 20),
                                label=['HII regions with overlapping GMCs'])
                if (z > 5) and (z != 9) and (z != 11):
                    axs[z].hist(arrays[z], alpha=0.5, bins=20)
                if (z == 2) or (z == 9) or (z == 11):
                    if z == 2:
                        axs[z].hist(arrays[z], alpha=0.5, bins=np.logspace(np.log10(minv), np.log10(maxv), 20))
                        axs[z].hist(arraycl[z], alpha=0.5, bins=np.logspace(np.log10(minv), np.log10(maxv), 20))
                    else:
                        axs[z].hist(arrays[z], alpha=0.5, bins=np.logspace(np.log10(minv), np.log10(maxv), 20))
                    axs[z].set_xscale("log")
                axs[z].title.set_text(labsname[z])
            axs[1].legend(prop={'size': 6})

            pdf1.savefig(fig)
            plt.close()

            # Plot HII parameters vs GMC parameters====================================================================
            title_font = {'fontname': 'Arial', 'size': '18', 'color': 'black', 'weight': 'normal',
                          'verticalalignment': 'bottom'}
            marker_style = dict(markersize=3)

            # --------------------------------------------------------------------------------------------------------
            arrayxax = [SizepcHII_galo, LumHacorr_galo, sigmavHII_galo, metaliHII_galo, varmetHII_galo]
            arrayyay = [DisHIIGMC_galo, MasscoGMC_galo, SizepcGMC_galo, Sigmamole_galo, sigmavGMC_galo, aviriaGMC_galo,
                        TpeakGMC_galo, tauffGMC_galo]  # GaldisHII_galo,

            labsxax = ['HII region size [pc]', r'Luminosity H$\alpha$ [erg/s]',
                       r'$\sigma_{v}$ HII region [km/s]', 'Metallicity',
                       'Metallicity variation']  # 'Galactocentric radius [kpc]',
            labsyay = ['Dist. HII-GMC [pc]', r'M$_{\rm CO}$ [10^5 M$_{\odot}$]', 'GMC size [pc]', r'$\Sigma_{\rm mol}$',
                       r'$\sigma_{\rm v}$ [km/s]', r'$\alpha_{\rm vir}$', r'CO $T_{\rm peak}$', r'$\tau_{\rm ff}$']

            print("Plotting HII region vs GMC parameters for individual galaxies")

            for k in range(len(arrayxax)):
                sns.set(style="white", color_codes=True)
                fig, axs = plt.subplots(4, 2, sharex='col', figsize=(8, 10), gridspec_kw={'hspace': 0})
                plt.subplots_adjust(wspace=0.3)
                fig.suptitle('Galaxy %s' % galnam.upper(), fontsize=18, va='top')
                axs = axs.ravel()
                for j in range(len(labsyay)):
                    xax = arrayxax[k]
                    yay = arrayyay[j]
                    xaxt = xax
                    if "Metallicity" not in labsxax[k]:
                        xaxt = np.log10(xax)
                    yayt = np.log10(yay)
                    idok = np.where((abs(yayt) < 100000) & (abs(xaxt) < 100000))
                    xax = xaxt[idok]
                    yay = yayt[idok]
                    lim1 = np.nanmedian(xax) - np.nanstd(xax) * 4
                    lim2 = np.nanmedian(xax) + np.nanstd(xax) * 4
                    indlim = np.where((xax < lim2) & (xax > lim1))
                    xax = xax[indlim]
                    yay = yay[indlim]
                    sns.set(color_codes=True)
                    if len(xax) > 2:
                        xmin = np.amin(xax)
                        xmax = np.amax(xax)
                        xprang = (xmax - xmin) * 0.1
                        x = xax.reshape((-1, 1))
                        y = yay
                        model = LinearRegression().fit(x, y)
                        r_sq = model.score(x, y)
                        y_pred = model.intercept_ + model.coef_ * x.ravel()
                        sns.regplot(x=xax, y=yay, ax=axs[j])
                        # x0, xf = xlim[k]
                        # y0, yf = ylim[i]
                        # xprang = (xf - x0) * 0.05
                        # yprang = (yf - y0) * 0.05
                        # axs[i].text(x0 + xprang / 2, y0 + yprang, 'R sq: %6.4f' % (r_sq))
                        # axs[i].set(xlim=(x0, xf))
                        # axs[i].set(ylim=(y0, yf))
                        # Uncomment above line if limits file has been generated by limits python program
                # axs[i].set(ylabel=labsyay[j])
                # axs[i].grid()
                axs[6].set(xlabel=labsxax[k])
                axs[7].set(xlabel=labsxax[k])
                # axs[7].set(ylim=(5.2, yf))

                pdf2.savefig(fig)
                plt.close()

    pdf1.close()
    pdf2.close()

    # ===========================================================================================================================================
    # Obtaining LHa/HIIsize^2 ratio
    ratlin = [None] * len(LumHacorrover)
    for j in range(len(galaxias)):
        if galaxias[j] != 'ic5332':
            ratlin[j] = ((LumHacorrover[j]) / (SizepcHIIover[j] ** 2))
        else:
            ratlin[j] = LumHacorrover[j]

    # =============================================================================================================================================
    # Saving the parameters to be read by another procedure.


    print("Plots of all galaxies together")

    arrayxax = [SizepcHIIover, LumHacorrover, sigmavHIIover, ratlin, metaliHIIover, varmetHIIover,
                velHIIover, HIIminorover, HIImajorover, HIIangleover]

    arrayyay = [DisHIIGMCover, MasscoGMCover, SizepcGMCover, Sigmamoleover, sigmavGMCover, aviriaGMCover, TpeakGMCover,
                tauffGMCover, velGMCover, angleGMCover, majorGMCover, minorGMCover, regionindexGMCover]

    labsxax = ['log(HII region size) [pc]',
               r'log(Luminosity H$\alpha$) [erg s$^{-1}$]',
               r'log($\sigma_{\rm v}$ HII region) [km s$^{-1}$]', r'log(Lum H$\alpha$/HII region size$^2$)',
               'Metallicity',
               'Metallicity variation', 'HII velocity [km s$^{-1}$]', 'HII region minor axis (deg)',
               'HII region major axis (deg)', 'HII region PA (deg)']

    labsyay = ['log(Dist. HII-GMC) [pc]', r'log(M$_{\rm CO}$) [10$^5$ M$_{\odot}$]', 'log(GMC size) [pc]',
               r'log($\Sigma_{\rm mol}$)', r'log($\sigma_{\rm v}$) [km s$^{-1}$]', r'log($\alpha_{vir}$)',
               r'log(CO $T_{\rm peak}$ [K])', r'log($\tau_{\rm ff}$) [yr]', 'GMC velocity [km s$^{-1}$]',
               'GMC position angle (deg)',
               'GMC major axis (deg)', 'GMC minor axis (deg)', 'GMC region index']

    print("Saving variables in external files.")
    with open(('Galaxies_variables_GMC%s%s.pickle' % (namegmc, name_end)), "wb") as f:
        pickle.dump([galaxias, arrayyay, arrayxax, RAgmcover, DECgmcover, RAhiiover, DEChiiover, labsxax, labsyay], f)

    with open(('Galaxies_variables_notover_GMC%s%s.pickle' % (namegmc, name_end)), "wb") as f:
        pickle.dump([SizepcHII, LumHacorr, sigmavHII, metaliHII, varmetHII, numGMConHII, MasscoGMC, HIIminor,
                     HIImajor, HIIangle, angleGMC, majorGMC, minorGMC], f)

    if without_out == True:
        print(total_outliers)
        print(np.sum(total_outliers))
        print(rms)

    print(name_end)
