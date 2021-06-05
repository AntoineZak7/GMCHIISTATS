import numpy as np
from astropy.table import Table
import pickle
import os

dir_script_data = os.getcwd() + "/script_data_dr2/"
dirhii_dr1, dirhii_dr2, dirgmc_old, dirgmc_new, dirregions1, dirregions2, dirmaps, dirplots1, dirplots2, dirplots, dirhiimasks, dirgmcmasks = pickle.load(
    open(dir_script_data + 'Directories_muse.pickle', "rb"))  # retrieving the directories paths

name_muse_dr2 = "HIIregion_cat_DR2_native.fits"
name_muse = name_muse_dr2
typegmc = "_native_"
namegmc1 = "_12m+7m+tp_co21%sprops" % typegmc
dirgmc_new = dirgmc_new + 'cats_native_amended/'
# =======randomizing gmc&hii rpops====================#

table_muse = Table.read("%s%s" % (dirhii_dr2, name_muse))

w = 0
galaxies_name = ['IC5332']

for i in range(len(table_muse['gal_name']) - 1):
    if galaxies_name[w] != str(table_muse['gal_name'][i]):
        galaxies_name.append(str(table_muse['gal_name'][i]))
        w += 1

hiicats = galaxies_name
w = 1

rand_ids_gmc_list = {}
for galnam in galaxies_name:
    galnam = str.lower(galnam)
    thii = Table.read(("%s%s%s.fits" % (dirgmc_new, galnam, namegmc1)))

    ids = np.linspace(0, len(thii['XCTR_DEG']) - 1, len(thii['XCTR_DEG']), dtype=int)
    rand_ids = ids
    np.random.shuffle(rand_ids)
    rand_ids_gmc_list[galnam] = rand_ids

with open(dir_script_data + 'randomized_lists', "wb") as f:
    pickle.dump(rand_ids_gmc_list, f)