import pickle
import numpy as np

a,b,c,d,e,f,g,h,i= pickle.load(open("/home/antoine/PycharmProjects/Intership/MUSE/script_data/Directories_muse.pickle", "rb"))
print(e)
hiis, gmcs = pickle.load(open("/home/antoine/PycharmProjects/Intership/MUSE/script_data/raw_new_muse__native_with_outliers_12m+7m+tp_co21_native_propsngc1087.pickle", "rb"))
hii = hiis[100]
gmc = gmcs[75]
vel  = 50

print(hii["DELTAV"])
print(gmc["DELTAV"])
# print(np.array(hii["DELTAV"])[np.where(np.abs(np.array(hii["DELTAV"])) <= vel)[0]])
# print(np.array(gmc["DELTAV"])[np.where(np.abs(np.array(gmc["DELTAV"])) <= vel)[0]])

hii["DELTAV"] = [x for x in hii["DELTAV"] if np.abs(x) <= vel]
print(hii["DELTAV"])
gmc["DELTAV"] = [x for x in gmc["DELTAV"] if np.abs(x) <= vel]
print(gmc["DELTAV"])
