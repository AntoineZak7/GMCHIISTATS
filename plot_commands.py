import Stats_test_2_22 as stattests2
import Stats_test_2_21 as stattest



'''
######################################################
#              Stat_tests v2.22                      #
#             Matching using Masks                   #
######################################################
'''


threshold_percs = [1]  # ,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
vels = [50, 40, 30, 20, 10]

regions = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]




# plot_correlations(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=False, save=True, vel=10000, threshold_percs=[0.1,0.5,0.9], randomize='', gmcprop=[1,4], rgal_color=False, symmetrical='sym')

#
# stattests2.plotallgals_galprop(sorting='mass', muse='dr2', gmc_catalog='_native_', gmc_catalog_version='new',
#                     matching="overlap_1om", randomize='', outliers=True, show=False, save=True, threshold_perc=0.5,
#                     vel=10000, gmcprop=[6], symmetrical='gmc')
# plotallgals(muse = 'dr2', gmc_catalog = '_native_', gmc_catalog_version= 'new', matching = "overlap_1om", randomize='', outliers = True, show =False , save = True, threshold_perc=0.1, vel=10000, gmcprop = [1])

# hist_std(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1o1", outliers = True, show =True , save = True, threshold_perc=0.4, bin = 1000)

# plot_correlations_randomized(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=True, save=True, vel=10000, threshold_percs=[0.1], randomize='gmc_prop', gmcprop=[4], random=False)
# plot_correlations_regions(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=False, save=True, vel=100, threshold_percs=[0.9], randomize='', gmcprop=[1], regions = regions, rgal_color=False, symmetrical='')
# plot_single_correlations(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=False, save=True, vel=10000, threshold_percs=[0.1,0.5,0.9], randomize='', gmcprop=[1], rgal_color=False, symmetrical='')
# plot_correlations_fct_rgal(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=True, save=True, vel=10000, threshold_percs=[0.1], randomize='', gmcprop=[1], rgal_color=False)
# plot_correlations_rgal(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=False, save=True, vel=10000, threshold_percs=[0.1,0.5,0.9], randomize='', gmcprop=[0], rgal_color=False)

# plot_gmcprop_regions(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=False, save=True, vel=10000, threshold_perc=0.1, randomize='', gmcprop=[1,3,4,6], regions = regions)

# plotallgals_covariance(muse = 'dr2', gmc_catalog_version = 'new', gmc_catalog='_native_', matching= 'overlap_1om', outliers=True, randomize='', vel = 10000, show = False, save = True,threshold_perc=0.9, gmcprop=[1,4,6],  symmetrical = '')

# hist_all(muse = 'dr2', gmc_catalog_version = 'new', gmc_catalog='_native_', matching= 'overlap_1om', outliers=True, randomize='', vel = 10000, show = False, save = True,threshold_percs=[0.1,0.5,0.9], gmc_props=[1], bin = 100, symmetrical = '')
# hist_std_props(muse = 'dr2', gmc_catalog_version = 'new', gmc_catalog='_native_', matching= 'overlap_1om', outliers=True, randomize='', vel = 100, show = False, save = True,threshold_percs=[0.1,0.5,0.9], gmc_props=[1], bin = 100, symmetrical = '')

# cdf_std_hii_props(muse = 'dr2', gmc_catalog_version = 'new', gmc_catalog='_native_', matching= 'overlap_1om', outliers=True, randomize='', vel = 10000, show = False, save = True,threshold_percs=[0.1,0.5,0.9], gmc_props=[1], bin = 100, symmetrical = '')
# cdf_std_props(muse = 'dr2', gmc_catalog_version = 'new', gmc_catalog='_native_', matching= 'overlap_1om', outliers=True, randomize='', vel = 100, show = False, save = True,threshold_percs=[0.1,0.5,0.9], gmc_props=[1], bin = 100, symmetrical = '')


'''
######################################################
#              Stat_tests v2.21                      #
#             Matching using Catalogs                #
######################################################
'''

threshold_percs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#threshold_percs = [0.1,0.5,0.9]
vels = [50,40,30,20,10]

regions = [[1,2,3],[4,5,6],[7,8,9]]

#stattest.plot_correlations(muse='dr22', gmc_catalog="_native_", gmc_catalog_version='v4', matching="overlap_1om", symmetrical='gmc', outliers=True, show=False, save=True, vel=10000, threshold_percs=threshold_percs, randomize='', gmcprop=[1,4], rgal_color=False)
# stattest.plot_correlations(muse='dr22', gmc_catalog="_native_", gmc_catalog_version='v4', matching="overlap_1om", symmetrical='sym', outliers=True, show=False, save=True, vel=10000, threshold_percs=threshold_percs, randomize='', gmcprop=[1,4], rgal_color=False)
# stattest.plot_correlations(muse='dr22', gmc_catalog="_native_", gmc_catalog_version='v4', matching="overlap_1om", symmetrical='hii', outliers=True, show=False, save=True, vel=10000, threshold_percs=threshold_percs, randomize='', gmcprop=[1,4], rgal_color=False)

#stattest.plotallgals_covariance(muse = 'dr22', gmc_catalog_version = 'v4', gmc_catalog='_native_', matching= 'overlap_1om', outliers=True, randomize='', vel = 10000, show = False, save = True,threshold_perc=0.9, gmcprop=[1,4,6],  symmetrical = 'gmc')


stattest.plotallgals_galprop(sorting = 'sfr', muse = 'dr22', gmc_catalog = '_native_', gmc_catalog_version= 'v4', matching = "overlap_1om", randomize='', outliers = True, show =False , save = True, threshold_perc=0.9, vel=10000, gmcprop = [1], symmetrical = 'gmc')
#plotallgals(muse = 'dr2', gmc_catalog = '_native_', gmc_catalog_version= 'new', matching = "overlap_1om", randomize='', outliers = True, show =False , save = True, threshold_perc=0.1, vel=10000, gmcprop = [1]4

#hist_std(new_muse = True, gmc_catalog = '_native_', matching = "overlap_1o1", outliers = True, show =True , save = True, threshold_perc=0.4, bin = 1000)

#plot_correlations_randomized(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=True, save=True, vel=10000, threshold_percs=[0.1], randomize='gmc_prop', gmcprop=[4], random=False)
#plot_correlations_regions(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=False, save=True, vel=100, threshold_percs=[0.9], randomize='', gmcprop=[1], regions = regions, rgal_color=False, symmetrical='')
#plot_single_correlations(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=False, save=True, vel=10000, threshold_percs=[0.1,0.5,0.9], randomize='', gmcprop=[1], rgal_color=False, symmetrical='')
#plot_correlations_fct_rgal(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=True, save=True, vel=10000, threshold_percs=[0.1], randomize='', gmcprop=[1], rgal_color=False)
#plot_correlations_rgal(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=False, save=True, vel=10000, threshold_percs=[0.1,0.5,0.9], randomize='', gmcprop=[0], rgal_color=False)

#plot_gmcprop_regions(muse='dr2', gmc_catalog="_native_", gmc_catalog_version='new', matching="overlap_1om", outliers=True, show=False, save=True, vel=10000, threshold_perc=0.1, randomize='', gmcprop=[1,3,4,6], regions = regions)

#plotallgals_covariance(muse = 'dr2', gmc_catalog_version = 'new', gmc_catalog='_native_', matching= 'overlap_1om', outliers=True, randomize='', vel = 10000, show = False, save = True,threshold_perc=0.9, gmcprop=[1,4,6],  symmetrical = '')

#hist_all(muse = 'dr2', gmc_catalog_version = 'new', gmc_catalog='_native_', matching= 'overlap_1om', outliers=True, randomize='', vel = 10000, show = False, save = True,threshold_percs=[0.1,0.5,0.9], gmc_props=[1], bin = 100, symmetrical = '')
#hist_std_props(muse = 'dr2', gmc_catalog_version = 'new', gmc_catalog='_native_', matching= 'overlap_1om', outliers=True, randomize='', vel = 100, show = False, save = True,threshold_percs=[0.1,0.5,0.9], gmc_props=[1], bin = 100, symmetrical = '')

#cdf_std_hii_props(muse = 'dr2', gmc_catalog_version = 'new', gmc_catalog='_native_', matching= 'overlap_1om', outliers=True, randomize='', vel = 10000, show = False, save = True,threshold_percs=[0.1,0.5,0.9], gmc_props=[1], bin = 100, symmetrical = '')
#cdf_std_props(muse = 'dr2', gmc_catalog_version = 'new', gmc_catalog='_native_', matching= 'overlap_1om', outliers=True, randomize='', vel = 100, show = False, save = True,threshold_percs=[0.1,0.5,0.9], gmc_props=[1], bin = 100, symmetrical = '')

