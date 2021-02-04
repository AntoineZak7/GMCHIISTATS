import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as fpdf
from sklearn.linear_model import LinearRegression
import seaborn as sns
import os
import warnings
import statistics
import scipy.stats
import statsmodels.stats.stattools

r2_center1 = [0.24,0.28,0.26,0.42]
r2_center2 = [0.17]
r2_center3 = [0.0042,0.044]
r2_arms1 = [0.054,0.13,0.22,0.30]
r2_arms2 = [0.23]
r2_arms3 = [0.028,0.14]
r2_disk = [0.05,0.09,0.1,0.21,0.11,0.0148,0.0416]
r2_global = [0.09,0.19,0.23,0.33,0.21]



pdf3 = fpdf.PdfPages('/home/antoine/PycharmProjects/Intership/MUSE/Plots_Muse/New_Muse/r2_graph.pdf')
x = np.linspace(1,len(r2_disk), len(r2_disk))
x1 = [1,2,3,4]
x2 = [5]
x3 = [6,7]
print(x)
print(x1)
print(x2)

labsyay = [ 'GMC size',r'$\sigma_{\rm v}$', r'CO $T_{\rm peak}$',r'M$_{\rm CO}$',  r'$\Sigma_{\rm mol}$', r'$\alpha_{\rm vir}$', r'$\tau_{\rm ff}$']



plt.bar(x, height=r2_disk, color = 'tab:red', tick_label = labsyay )
plt.bar(x1, height=r2_center1,color = 'tab:green',label = 'Galaxy center')
plt.bar(x1, height=r2_arms1, color = 'tab:blue',label = 'Galaxy arms')
plt.bar(x2, height=r2_arms2, color = 'tab:blue')
plt.bar(x2, height=r2_center2, color ='tab:green' )
plt.bar(x3, height=r2_arms3, color = 'tab:blue')

plt.bar(x3, height=r2_center3, color ='tab:green' )

plt.bar(x, height=r2_disk, color ='tab:red',label = 'Galaxy disc' )

plt.ylabel('Correlation Coefficients')
plt.legend()

pdf3.savefig()
pdf3.close()


#plt.show()