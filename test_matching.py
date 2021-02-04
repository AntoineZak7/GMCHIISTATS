import numpy as np

def make_list(a):
    if not isinstance(a, list):
        a = [a]
    return a

def extract_ind(list1,  value):
    ind_val = [[idx,item] for [idx, item] in enumerate(list1) if item == value]
    indexes = [item[0] for item in ind_val]
    return indexes

def extract_ind_sup(list1,  value):
    list1 = make_list(list1)
    ind_val = [[idx,item] for [idx, item] in enumerate(list1) if item >= value]
    indexes = [item[0] for item in ind_val]
    return indexes


def extract_ind_not_equal(list1,  value):
    ind_val = [[idx,item] for [idx, item] in enumerate(list1) if item != value]
    print(ind_val)
    indexes = [item[0] for item in ind_val]
    return indexes

def extract_values(list1, indexes):
    values = [list1[i] for i in indexes]
    return values

def make_hii_list_ind(hiis_matched, gmc, hiis):
    hii_list_ind = []
    for item in hiis:
        if (gmc in make_list(item["GMCS"]) and (item["PHANGS_INDEX"] - 1 not in hiis_matched)):
            hii_list_ind.append((item["PHANGS_INDEX"] - 1))
    return hii_list_ind
    #[(item['CLOUDNUM'] - 1) for item in hiis if (gmc in item["GMCS"] and (item["CLOUDNUM"] - 1 not in hiis_matched))]

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
    list1 = make_list(list1)
    list1 = [[idx,item] for [idx, item] in enumerate(list1) if idx in indexes]
    list1 = [item[1] for item in list1]

    return list1

def undo_list(list1):
    if isinstance(list1, list):
        list1 = list1[0]
        return list1
    else:
        return list1

def undo_sublist(list):
    list_undone = []
    for x in list:
        x = undo_list(x)
        list_undone.append(x)
    return list_undone


list_test = [[1.0], [1.0]]
print(undo_sublist(list_test))

hiivel = np.array([2,48,24,31,7,11,0,0])
gmcvel = np.array([15,22,33,12,5,3,17,-6])


hiis = []
hii1 = {"PHANGS_INDEX":1, "GMCS":[1,2], "OVERLAP_PIX": [0.1,1], "HA_VEL": 2, "DELTAV": [-13, 9]}
hii2 = {"PHANGS_INDEX":2, "GMCS":[3], "OVERLAP_PIX": 0.5, "HA_VEL": 48, "DELTAV": [25]}
hii3 = {"PHANGS_INDEX":3, "GMCS":[], "OVERLAP_PIX": [], "HA_VEL": 24, "DELTAV": []}
hii4 = {"PHANGS_INDEX":4, "GMCS":[1,2,3,4], "OVERLAP_PIX": [0.1,1,0.5,0.6], "HA_VEL": 31, "DELTAV": [16,9,-2,19]}
hii5 = {"PHANGS_INDEX":5, "GMCS":[5,6], "OVERLAP_PIX": [0.8,0.9], "HA_VEL": 7, "DELTAV": [2,4]}
hii6 = {"PHANGS_INDEX":6, "GMCS":[7], "OVERLAP_PIX": [1], "HA_VEL": 11, "DELTAV": [-6]}
hii7 = {"PHANGS_INDEX":7, "GMCS":[8], "OVERLAP_PIX": [0.8], "HA_VEL": 11, "DELTAV": [-6]}
hii8 = {"PHANGS_INDEX":8, "GMCS":[8], "OVERLAP_PIX": [0.8], "HA_VEL": 11, "DELTAV": [-6]}

hiis.append(hii1)
hiis.append(hii2)
hiis.append(hii3)
hiis.append(hii4)
hiis.append(hii5)
hiis.append(hii6)
hiis.append(hii7)
hiis.append(hii8)


gmcs = []
gmc1 = {"CLOUDNUM":1, "HIIS":[1,4], "OVERLAP_PIX":[0.1,0.1], "GMC_VEL":15}
gmc2 = {"CLOUDNUM":2, "HIIS":[1,4], "OVERLAP_PIX":[0.1,1], "GMC_VEL":22}
gmc3 = {"CLOUDNUM":3, "HIIS":[2,4], "OVERLAP_PIX":[0.5,0.5], "GMC_VEL":33}
gmc4 = {"CLOUDNUM":4, "HIIS":[4], "OVERLAP_PIX":[0.6], "GMC_VEL":12}
gmc5 = {"CLOUDNUM":5, "HIIS":[5], "OVERLAP_PIX":[0.8], "GMC_VEL":5}
gmc6 = {"CLOUDNUM":6, "HIIS":[5], "OVERLAP_PIX":[0.9], "GMC_VEL":3}
gmc7 = {"CLOUDNUM":7, "HIIS":[6], "OVERLAP_PIX":[1], "GMC_VEL":17}
gmc8 = {"CLOUDNUM":8, "HIIS":[7,8], "OVERLAP_PIX":[0.8,0.8], "GMC_VEL":19}
gmcs.append(gmc1)
gmcs.append(gmc2)
gmcs.append(gmc3)
gmcs.append(gmc4)
gmcs.append(gmc5)
gmcs.append(gmc6)
gmcs.append(gmc7)
gmcs.append(gmc8)



threshold_perc = 0.5




hiis_matched = []
# for hii in hiis:
#     if len(make_list(hii['GMCS'])) > 0:
#         max_overlap_ind = np.nanargmax(make_list(hii["OVERLAP_PIX"]))
#         print(np.size(max_overlap_ind))
#         vel_list = hii["DELTAV"]
#         if np.size(max_overlap_ind) > 1:
#             min_vel_ind = np.argmin([vel_list[i] for i in max_overlap_ind])
#             hii['GMCS'] = hii['GMCS'][max_overlap_ind][min_vel_ind]
#             hiis_matched.append(hii)
#         else:
#             hii['GMCS'] = hii['GMCS'][max_overlap_ind]
#             hiis_matched.append(hii)












#print([item["CLOUDNUM"] -1 for item in hiis if gmc in item["GMCS"]])

# hiis_matched = []
#
for j in range(len(gmcs) ):

    print("=="*10 + "PASS" + "="*10)
    print(hiis)
    print("hiis_matched =")
    print(hiis_matched)
    print("\n")

    gmc = j + 1

    print("gmc =")
    print(gmc)
    print("\n")

    print(hiis)

    hii_list_ind = make_hii_list_ind(hiis_matched, gmc, hiis)#[(item['CLOUDNUM'] - 1) for item in hiis if (gmc in item["GMCS"]  and (item["CLOUDNUM"] -1 not in hiis_matched))]
    hii_list = make_hii_list(hiis_matched, gmc, hiis)#[item for item in hiis if gmc in item["GMCS"]]



    print("hii_list_ind =" )
    print(hii_list_ind )
    print("\n")


    if np.size(hii_list) > 1:


        #hii_gmc_overlap = [   np.array(sublist["OVERLAP_PIX"])[np.where(np.array(sublist["GMCS"]) == gmc)[0]] for sublist in hii_list   ]
        hii_gmc_overlap = do_list(hii_list, gmc, "OVERLAP_PIX")
        print("hii_gmc_overlap =")
        print(hii_gmc_overlap)
        print("\n")

        #hii_gmc_vel = [   np.array(sublist["DELTAV"])[np.where(np.array(sublist["GMCS"]) == gmc)[0]] for sublist in hii_list]
        hii_gmc_vel = do_list(hii_list, gmc, "DELTAV")

        print("hii_gmc_vel =")
        print(hii_gmc_vel)
        print("\n")

        best_gmcs_inds = np.nanargmax(hii_gmc_overlap)

        print("best_gmcs_ind =")
        print(best_gmcs_inds)
        print("\n")

        if  np.size(best_gmcs_inds) > 1:
            best_gmc_ind = np.nanargmin(hii_gmc_vel[best_gmcs_inds])
            best_hii_ind = hii_list_ind[best_gmcs_inds[best_gmc_ind]]
            best_hii = hii_list[best_gmcs_inds[best_gmc_ind]]
            # remove gmcs in best and in others
            hiis[best_hii_ind]["GMCS"] = gmc
            hiis_matched.append(best_hii)
            for hii in hiis:
                if hii["PHANGS_INDEX"] - 1 != best_hii_ind and hii["GMCS"] != []:
                    make_list(hii["GMCS"])
                    make_list(hii["OVERLAP_PIX"])
                    make_list(hii["DELTAV"])
                    #hii["GMCS"] = [item for item in hii["GMCS"] if item != gmc]
                    ind_gmc = extract_ind_not_equal(hii["GMCS"], gmc)#np.where(np.array(hii["GMCS"] != gmc))[0]

                    hii["OVERLAP_PIX"] = extract_values(hii["OVERLAP_PIX"], ind_gmc)#np.array(hii["OVERLAP_PIX"])[ind_gmc]
                    hii["DELTAV"] = extract_values(hii["DELTAV"], ind_gmc) #np.array(hii["DELTAV"])[ind_gmc]
                    hii["GMCS"] = extract_values(hii["GMCS"], ind_gmc)#np.array(hii["GMCS"])[ind_gmc]

                else:
                    make_list(hii["GMCS"])
                    make_list(hii["OVERLAP_PIX"])
                    make_list(hii["DELTAV"])
                    ind_gmc = extract_ind(hii["GMCS"], gmc)#np.where(np.array(hii["GMCS"] != gmc))[0]
                    hii["OVERLAP_PIX"] = extract_values(hii["OVERLAP_PIX"], ind_gmc)#np.array(hii["OVERLAP_PIX"])[ind_gmc]
                    hii["DELTAV"] = extract_values(hii["DELTAV"], ind_gmc) #np.array(hii["DELTAV"])[ind_gmc]
                    hii["GMCS"] = extract_values(hii["GMCS"], ind_gmc)#np.array(hii["GMCS"])[ind_gmc]





        else:
            best_hii_ind = hii_list_ind[best_gmcs_inds]
            best_hii = hii_list[best_gmcs_inds]
            hiis_matched.append(best_hii)

            # remove gmcs in best and in others
            #list1 = [[idx, item] for [idx, item] in enumerate(hii) if item != gmc]
            #print(list1)

            hiis[best_hii_ind]["GMCS"] = gmc
            for hii in hiis:
                if hii["PHANGS_INDEX"] - 1 != best_hii_ind and hii["GMCS"] != []:
                    hii["GMCS"] =make_list(hii["GMCS"])
                    hii["OVERLAP_PIX"] = make_list(hii["OVERLAP_PIX"])
                    hii["DELTAV"] = make_list(hii["DELTAV"])
                    #hii["GMCS"] = [item for item in hii["GMCS"] if item != gmc]
                    ind_gmc = extract_ind_not_equal(hii["GMCS"], gmc)#np.where(np.array(hii["GMCS"] != gmc))[0]
                    hii["OVERLAP_PIX"] = extract_values(hii["OVERLAP_PIX"], ind_gmc)#np.array(hii["OVERLAP_PIX"])[ind_gmc]
                    hii["DELTAV"] = extract_values(hii["DELTAV"], ind_gmc) #np.array(hii["DELTAV"])[ind_gmc]
                    hii["GMCS"] = extract_values(hii["GMCS"], ind_gmc)#np.array(hii["GMCS"])[ind_gmc]
                else:

                    hii["GMCS"] = make_list(hii["GMCS"])
                    hii["OVERLAP_PIX"] = make_list(hii["OVERLAP_PIX"])
                    hii["DELTAV"] = make_list(hii["DELTAV"])
                    ind_gmc = extract_ind(hii["GMCS"], gmc)#np.where(np.array(hii["GMCS"] != gmc))[0]
                    hii["OVERLAP_PIX"] = extract_values(hii["OVERLAP_PIX"], ind_gmc)#np.array(hii["OVERLAP_PIX"])[ind_gmc]
                    hii["DELTAV"] = extract_values(hii["DELTAV"], ind_gmc) #np.array(hii["DELTAV"])[ind_gmc]
                    hii["GMCS"] = extract_values(hii["GMCS"], ind_gmc)#np.array(hii["GMCS"])[ind_gmc]

    else:

        if make_list(hii_list_ind) :

            hiis[undo_list(hii_list_ind)]["GMCS"] = gmc
            hiis_matched.append(hiis[undo_list(hii_list_ind)])

            print(gmc)
            print(hiis[undo_list(hii_list_ind)])

#====================================================#

hiis = [item for item in hiis if item["GMCS"] != []]

idoverhii = [ undo_list(hii.get("PHANGS_INDEX"))-1 for hii in hiis]
idovergmc = [ undo_list(hii.get("GMCS"))-1 for hii in hiis]

print('hii')
print(idoverhii)
print(hiivel[idoverhii])
print('\n')
print('gmc')
print(idovergmc)
print(gmcvel[idovergmc])
print([hii['PHANGS_INDEX'] for hii in hiis_matched])