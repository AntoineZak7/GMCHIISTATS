import os
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
from astropy.io import fits
from astropy import wcs
from astropy.table import Column


def all_catalogs(dirallcats, dirmask, overwrite):
    gmccats = True


    dirmasks = dirmask
    dirmasks = dirmasks + '/'
    dircats = dirallcats
    dircats = dircats + '/'
    all_cats = [f for f in os.listdir(dircats)]
    maskname = "_simple.fits"

    if gmccats == True:
        del_index = []
        for k in range(len(all_cats) - 1):
            if all_cats[k].split("_")[0] != 'cats' or all_cats[k].split("_")[len(all_cats[k].split("_")) - 1] != 'amended':
                del_index.append(k)
        all_cats = [i for j, i in enumerate(all_cats) if j not in del_index]


    for k1 in range(len(all_cats)):
        gmc_cat = dircats + all_cats[k1]
        single_catalog(dircat = gmc_cat, dirmask = dirmasks, overwrite= overwrite)


dircat = '/home/antoine/Internship/gmccats_st1p5_amended/cats_120pc_homogenized_amended'
dirmask = '/home/antoine/Internship/masks_v5p3_simple'

def single_catalog(dircat, dirmask, overwrite):

    dirmasks = dirmask
    dirmasks = dirmasks + '/'
    gmccats = True
    dirtable = dircat + '/'
    maskname = "_simple.fits"
    dirtable_name = dircat.split('/')[len(dircat.split('/')) - 1]
    gmc_cat = [f for f in os.listdir(dirtable)]

    if gmccats == True:
        del_index = []
        for k in range(len(gmc_cat) - 1):
            if gmc_cat[k].split("_")[1] != '12m+7m+tp' or gmc_cat[k].split("_")[len(gmc_cat[k].split("_")) - 1] != 'props.fits':
                del_index.append(k)
        gmc_cat = [i for j, i in enumerate(gmc_cat) if j not in del_index]

    for k in range(len(gmc_cat)):
        galnam = gmc_cat[k].split("_")[0]
        typegmc = "_" + dirtable_name.split("_")[1] + "_"

        if galnam == "eso097-013":
            namegmc = "_7m+tp_co21%sprops" % typegmc
        else:
            namegmc = "_12m+7m+tp_co21%sprops" % typegmc


        if os.path.isfile(dirmasks + str.upper(galnam) + maskname):

            tgmc = Table.read(dirtable + galnam + namegmc + '.fits')
            masks = dirmasks + str.upper(galnam) + maskname
            ragmc = tgmc['XCTR_DEG']
            decgmc = tgmc['YCTR_DEG']
            index = tgmc['CLOUDNUM']
            mask = fits.open(masks)
            mask_data = mask[0].data
            mask_hdr = mask[0].header
            WC = wcs.WCS(mask_hdr)

            gmc_coords = SkyCoord(ragmc, decgmc, frame=FK5, unit=(u.deg))
            gmc_coords_pix = gmc_coords.to_pixel(wcs=WC, mode='all')
            indexes = []

            for j in range(len(gmc_coords_pix[0])):
                x = int(round(gmc_coords_pix[0][j], 0))
                y = int(round(gmc_coords_pix[1][j], 0))
                indexes.append(int(mask_data[y][x]))

            region_index = Column(indexes, name='REGION_INDEX')

            colname = tgmc.colnames

            if ('REGION_INDEX' in colname) == True:
                tgmc['REGION_INDEX'] = indexes
            if ('REGION_INDEX' in colname) == False:
                tgmc.add_column(region_index)

            if overwrite == True:
                table_name = dirtable + galnam + namegmc + '.fits'
            if overwrite == False:
                table_name = dirtable + galnam + namegmc + '_add' + '.fits'

            tgmc.write(table_name, format='fits', overwrite=True)



def value_at_position(ra, dec, dirmask, galnam):

    dirmasks = dirmask
    dirmasks = dirmasks + '/'
    maskname = "_simple.fits"
    ragmc = ra
    decgmc = dec

    if os.path.isfile(dirmasks + str.upper(galnam) + maskname):

        masks = dirmasks + str.upper(galnam) + maskname
        mask = fits.open(masks)
        mask_data = mask[0].data
        mask_hdr = mask[0].header
        WC = wcs.WCS(mask_hdr)

        gmc_coords = SkyCoord(ragmc, decgmc, frame=FK5, unit=(u.deg))
        gmc_coords_pix = gmc_coords.to_pixel(wcs=WC, mode='all')
        indexes = []

        for j in range(len(gmc_coords_pix[0])):
            x = int(round(gmc_coords_pix[0][j], 0))
            y = int(round(gmc_coords_pix[1][j], 0))
            indexes.append(int(mask_data[y][x]))
            print('ra: %f \ dec:%f \ region index: %i' % (ragmc[j], decgmc[j], indexes[j]))

    return indexes



all_catalogs(dirallcats = '/home/antoine/Internship/gmc_new/', dirmask = '/home/antoine/Internship/masks_v5p3_simple' , overwrite = True)


