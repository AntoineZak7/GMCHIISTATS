import numpy as np
import matplotlib.pyplot as plt


zetha_e = 0.97 #perte charge entrée air
etha_c = 0.9 #rendeent polytropique compresseur
etha_f = 0.92 #rendement polytropique fan
etha_comb = 0.99 # rendement polytropique de combustion
zetha_cc = 0.95 #perte charge chambre combustion
etha_m = 0.98 # rendement mécanique sur l'arbre
etha_tHP = 0.89 # rendement polytropique turbine HP
etha_tBP = 0.90 # rendement polytropique turbine BP
zetha_tuy = 0.97 #pertes de charge tuyere ejection
lambdaa = 11. #taux de dilution mspoint/mppoint (BPR)
r = 287.1 #J/kg/K GP
r_etoile = 291.6 #apres combustion
gamma = 1.4 #GP
gamma_etoile = 1.33 # après combustion
Pk = 42800000. #J/kg, pouvoir calo inf kérosène





do_plots = 1


def turbofan(M0, P0, T0, Tt4, pi_c, lambdaa, m_point):
    m_pointp = m_point / (1 + lambdaa)
    m_points = m_pointp * lambdaa

    # =============================================================TURBINE MONO CORPS MONO FLUX (verif gamma point et cp point)===========================================================#

    # ======Conditions amont====="
    Pt0 = P0 * np.power((1 + (gamma - 1) / 2 * M0 ** 2), gamma / (gamma - 1))
    Tt0 = T0 * (1 + (gamma - 1) / 2 * M0 ** 2)
    a0 = np.sqrt(gamma * r * T0)

    # =========Conditions amont et aval compresseur======# COMPLETELY OKKKK
    Tt2 = Tt0
    Pt2 = zetha_e * Pt0

    Pt3 = pi_c * Pt2  # pression tot sortie compresseur
    Tt3 = np.power(pi_c, (gamma - 1) / (gamma * etha_c)) * Tt2  # temp totale sortie compresseur

    Pt4 = zetha_cc * Pt3  # température sortie de chambre

    cp = r * gamma / (gamma - 1)
    cp_etoile = r_etoile * gamma_etoile / (gamma_etoile - 1)

    mk_point = m_pointp * (cp_etoile * Tt4 - cp * Tt3) / (etha_comb * Pk - cp_etoile * Tt4)

    Pceff = m_pointp / mk_point * cp * (Tt4 - Tt3)
    alpha = cp * (Tt4 - Tt3) / Pceff

    # =======turbine HP=======# OKKKKK except t4_5
    w_comp = m_pointp * cp * (Tt3 - Tt2)
    w_turb_hp = -w_comp / etha_m

    Tt4_5 = Tt4 + w_turb_hp/ (m_pointp * (1 + alpha) * cp_etoile)  # Tt4_5 ??
    pi_t = np.power((Tt4_5 / Tt4),gamma_etoile/ ((gamma_etoile - 1)* etha_tHP ) )  # taux de détente, formule correcte
    Pt4_5 = Pt4 * np.power((Tt4_5 / Tt4),gamma_etoile/ ((gamma_etoile - 1)* etha_tHP ) )  # formule correcte


    # ===Turbine BP========#
    w_cycle_mono_sp,a,b,c = turbo_mono(M0,P0,T0,Tt4,pi_c, m_pointp)
    w_cycle_mono = w_cycle_mono_sp * m_pointp #puissance utile cycle coeur
    w_turb_bp = (w_cycle_mono - (2/(lambdaa+2))*w_cycle_mono)

    Tt5 = Tt4_5 - w_turb_bp / (m_pointp * (1 + alpha) * cp_etoile)
    Pt5 = Pt4_5 * np.power((Tt5 / Tt4_5),gamma_etoile/ ((gamma_etoile - 1)* etha_tHP ) )  # formule correcte


    # ======Tuyère sortie coeur=========#
    Pt9 = zetha_tuy * Pt5
    Tt9 = Tt5

    V0 = M0 * np.sqrt(gamma * r * T0) #OKKKKKKK
    M9 = np.sqrt(2 / (gamma_etoile - 1) * (np.power(Pt9 / P0, (gamma_etoile - 1) / gamma_etoile) - 1))
    T9 = Tt9 / (1 + (gamma_etoile - 1) / 2 * M9 ** 2)
    a9 = np.sqrt(gamma_etoile * r_etoile * T9)
    V9 = M9 * a9 #OKKKKKKKK
    P9 = P0

    F_sp_c = (1 + alpha) * V9 - V0
    F_c = m_pointp * F_sp_c #OKKKKKKKKKKK


    #============ Fan ============================# OKKKKK
    w_fan = (lambdaa / (lambdaa + 2)) * w_cycle_mono * etha_m
    Tt_17 = Tt0 + w_fan / (m_points * cp)
    Pt_17 = Pt2 * np.power((Tt_17 / Tt2), gamma * etha_f / ((gamma - 1)))


    #======== Tuyère sortie fan =================# COMPLETELY OKKKKK
    P19 = P0
    Pt_19 = zetha_tuy*Pt_17
    Tt_19 = Tt_17
    M19 = np.sqrt(2 / (gamma - 1) * (np.power(Pt_19 / P19, (gamma - 1) / gamma) - 1))
    T_19 = Tt_19 / (1 + (gamma - 1) / 2 * M19 ** 2)
    a_19 = np.sqrt(gamma*r*T_19)
    V19 = M19 * a_19
    F_fan_net = 1 * V19 - V0
    F_fan = m_points * F_fan_net # poussée fan


    # ========Rendements et puissances=========#
    F_tot = F_fan + F_c
    F_tot_spe = F_tot/(m_point)

    w_cycle = m_pointp * (0.5 * V9 ** 2 - 0.5 * V0 ** 2) + m_point * (0.5 * V19 ** 2 - 0.5 * V0 ** 2)
    w_cycle_sp = w_cycle / (m_point)

    w_pr_sp = V0 * F_tot_spe
    w_pr = V0 * F_tot

    w_chim = Pk * mk_point
    w_chim_sp = w_chim / m_pointp

    rend_th = w_cycle / w_chim
    rend_prop = w_pr_sp / w_cycle_sp
    rend_thermoprop = w_pr / w_chim


    # print("Pt0 = %8.0f Pa" % Pt0)
    # print("Tt0 = %8.0f K" % Tt0)
    # print("a0 = %8.0f m/s" % a0)
    #
    # print("Pt2 = %8.0f Pa" % Pt2)
    # print("Tt2 = %8.0f K" % Tt2)
    #
    # print("Pt3 = %8.0f Pa" % Pt3)
    # print("Tt3/Tt2 = %8.2f " % (Tt3 / Tt2))
    # print("Tt3 = %8.0f K" % Tt3)
    #
    # print("w_uc= %8.0f" % w_comp)
    # print("m_pointp = %5.0f" % m_pointp)
    # print("m_k = %5.3f" % mk_point)
    # print("alpha = %8.3f %%" % (alpha * 100))
    # print("Pt4 = %8.0f Pa" % Pt4)
    #
    # print("w_ut= %8.0f" % w_turb_hp)
    #
    # print("Tt4_5 = %8.0f K" % Tt4_5)
    # print("Tt4_5/Tt4 = %8.2f " % (Tt4_5 / Tt4))
    # print("pi_t = %8.2f" % pi_t)
    # print("Pt4_5 = %8.0f Pa" % Pt4_5)
    #
    # print("puissance utile cycle coeur= %8.0f" % w_cycle_mono)
    # print("puissance vers fan = %5.3f" %((lambdaa/(lambdaa+2))*w_cycle_mono))
    # print("puissance fan = %8.3f" %((lambdaa/(lambdaa+2))*w_cycle_mono*etha_m))
    #
    # print("Tt5 = %8.0f K" % Tt5)
    # print("Pt5 = %8.0f Pa" % Pt5)
    #
    # print("Pt9 = %8.0f Pa" % Pt9)
    # print("T9 = %5.3f" %T9)
    # print("a9 = %5.3f" %a9)
    # print("V9 = %5.3f" %V9)
    # print('poussée coeur = %5.3f' % F_c)
    #
    # print("T_t17 = %5.3f" % Tt_17)
    # print("Pt_17 = %8.0f Pa" % Pt_17)
    #
    # print("Pt_19 = %8.0f Pa" % Pt_19)
    # print("M19 = %5.3f" %M19)
    # print("T19 = %5.3f" %T_19)
    # print("a_19 = %5.3f" %a_19)
    # print("V19 = %5.3f" % V19)
    #
    # print("F_F = %5.3f" %F_fan)
    # print("Poussée totale = %5.3F" %F_tot)
    # print("Poussée spécifique = %5.3f" %F_tot_spe)
    #
    # print("rendement thermique = %5.3f %%" % (rend_th * 100))
    # print("rendement propulsif = %5.3f %%" % (rend_prop * 100))
    # print("rendement thermopropulsif = %5.3f %%" % (rend_thermoprop * 100))


    return w_cycle, w_chim, w_pr, F_sp_c


def turbo_mono(M0, P0, T0, Tt4, pi_c, m_point): #100% correcte

    #=============================================================TURBINE MONO CORPS MONO FLUX===========================================================#

    #======Tt et Pt, temp et pression totale en aval du compresseur====="

    Pt0 = P0*np.power((1+(gamma - 1)/2 * M0**2),gamma/(gamma-1))
    Tt0 = T0 * (1 + (gamma - 1)/2 * M0**2)
    a0 = np.sqrt(gamma*r*T0)

    #=========Temp et Pression tot en aval compr, chambre et turbine======#

    Tt2 = Tt0
    Pt2 = zetha_e * Pt0

    Pt3 = pi_c*Pt2
    Tt3 = np.power(pi_c,(gamma-1)/(gamma*etha_c))*Tt2

    Pt4 = zetha_cc * Pt3

    cp = r*gamma/(gamma-1)
    cp_etoile = r_etoile*gamma_etoile/(gamma_etoile-1)

    mk_point = m_point * (cp_etoile * Tt4 - cp * Tt3) / (etha_comb * Pk - cp_etoile * Tt4)

    Pceff = m_point / mk_point * cp * (Tt4 - Tt3)
    alpha = cp*(Tt4-Tt3)/Pceff


    #=======travail turbine=======#

    w_comp = m_point * cp * (Tt3 - Tt2)

    exp = gamma_etoile/((gamma_etoile-1)*etha_tHP)

    Tt5 = Tt4-w_comp/(m_point * (1 + alpha) * cp_etoile)
    Pt5 = Pt4*np.power((Tt5/Tt4),exp)
    Tt9 = Tt5
    Pt9 = zetha_tuy*Pt5

    #=====Poussée spécifique======#

    V0 = M0*np.sqrt(gamma*r*T0)
    M9 = np.sqrt(2/(gamma_etoile-1)*(np.power(Pt9/P0,(gamma_etoile-1)/gamma_etoile)-1))
    T9 = Tt9/(1+(gamma_etoile-1)/2 * M9**2)
    a9 = np.sqrt(gamma_etoile*r_etoile*T9)
    V9 = M9 * a9
    P9 = P0

    F_sp_net = (1+alpha)*V9 - V0
    F_net = m_point * F_sp_net

    #========Rendements et puissances=========#

    w_cycle = m_point * ((1 + alpha) * 0.5 * V9 ** 2 - 0.5 * V0 ** 2)
    w_sp_cycle = w_cycle / m_point

    w_pr_sp = V0 * F_sp_net
    w_pr = V0*F_net

    w_chim_sp = alpha*Pceff
    w_chim = w_chim_sp * m_point

    rend_th = w_sp_cycle/w_chim_sp
    rend_prop = w_pr_sp/w_sp_cycle
    rend_thermoprop = w_pr_sp/w_chim_sp

    return w_sp_cycle, w_chim_sp, w_pr_sp, F_sp_net



M0 = 0.8
alt = 35000. #feet
P0 = 22700. #Pa
T0 = 217. #K



if do_plots == 1:

    #======================================================================================================================#
    #                                                   PLOTS                                                              #
    #======================================================================================================================#

    #============= variation rendements et poussée en fct OPR et Tt4=====================================#

    pi_c_list = np.arange(5,200,1)
    Tt4 = 1600.
    m_point = 215
    lambdaa = 1
    
    F_mono_sp_nets = []
    w_mono_chim_sps = []
    w_mono_pr_sps = []
    w_mono_sp_cycles = []

    F_tbfan_sp_nets = []
    w_tbfan_chim_sps = []
    w_tbfan_pr_sps = []
    w_tbfan_sp_cycles = []

    for pi_c in pi_c_list:
        lambdaa = 11
        w_mono_sp_cycle, w_mono_chim_sp, w_mono_pr_sp, F_mono_sp_net = turbofan(M0, P0, T0, Tt4, pi_c, lambdaa, m_point)
        lambdaa = 1
        w_tbfan_sp_cycle, w_tbfan_chim_sp, w_tbfan_pr_sp, F_tbfan_sp_net = turbofan(M0, P0, T0, Tt4, pi_c, lambdaa, m_point)


        w_mono_chim_sps.append(w_mono_chim_sp)
        w_mono_pr_sps.append(w_mono_pr_sp)
        F_mono_sp_nets.append(F_mono_sp_net)
        w_mono_sp_cycles.append(w_mono_sp_cycle)

        w_tbfan_chim_sps.append(w_tbfan_chim_sp)
        w_tbfan_pr_sps.append(w_tbfan_pr_sp)
        F_tbfan_sp_nets.append(F_tbfan_sp_net)
        w_tbfan_sp_cycles.append(w_tbfan_sp_cycle)


    w_mono_sp_cycles = np.array(w_mono_sp_cycles)
    F_mono_sp_nets = np.array(F_mono_sp_nets)
    w_mono_chim_sps = np.array(w_mono_chim_sps)
    w_mono_pr_sps = np.array(w_mono_pr_sps)

    w_tbfan_chim_sps = np.array(w_tbfan_chim_sps)
    w_tbfan_pr_sps = np.array(w_tbfan_pr_sps)
    F_tbfan_sp_nets = np.array(F_tbfan_sp_nets)
    w_tbfan_sp_cycles = np.array(w_tbfan_sp_cycles)



    rend_ths_mono = w_mono_sp_cycles / w_mono_chim_sps
    rend_prs_mono = w_mono_pr_sps / w_mono_sp_cycles
    rend_thermoprops_mono = w_mono_pr_sps / w_mono_chim_sps

    rend_ths_tbfan = w_tbfan_sp_cycles / w_tbfan_chim_sps
    rend_prs_tbfan = w_tbfan_pr_sps / w_tbfan_sp_cycles
    rend_thermoprops_tbfan = w_tbfan_pr_sps / w_tbfan_chim_sps



    fig, axs = plt.subplots(2, 2, figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0.2})
    plt.subplots_adjust(wspace=0.3)
    fig.suptitle('Rendements et poussée spécifique Turboréacteur mono-corps mono-flux et Turbofan (BPR = 1 et Tt4 = 1600 K)')


    axs[0,0].plot(pi_c_list, rend_ths_mono, label = "BPR = 11")
    axs[0,0].plot(pi_c_list, rend_ths_tbfan, label = "BPR = 1")
    axs[0,0].set(ylabel = "Rendement thermique")
    axs[0,0].set(xlabel = "Taux de compression")
    axs[0,0].grid()
    axs[0,0].legend()
    rend_ths_mono_max = np.nanmax(rend_ths_mono)
    rend_ths_mono_argmax = np.nanargmax(rend_ths_mono)
    rend_ths_tbfan_max = np.nanmax(rend_ths_tbfan)
    rend_ths_tbfan_argmax = np.argmax(rend_ths_tbfan)
    #axs[0,0].text(rend_ths_mono_argmax, rend_ths_mono_max, s = "Rendement thermique max = %2.2f à pi_c = %4.2f" %(rend_ths_mono_max, rend_ths_mono_argmax))
    axs[0, 0].text(pi_c_list[rend_ths_tbfan_argmax], rend_ths_tbfan_max+0.01, s="Rendement thermique max = %2.2f \n pour OPR = %4.0f" % ( rend_ths_tbfan_max,pi_c_list[rend_ths_tbfan_argmax]))
    axs[0,0].plot(pi_c_list[rend_ths_tbfan_argmax],rend_ths_tbfan_max, color = "black", marker = "+")


    axs[0,1].plot(pi_c_list, rend_prs_mono, label = "mono-corps mono-flux")
    axs[0,1].plot(pi_c_list, rend_prs_tbfan, label = "turbofan")
    axs[0,1].set(ylabel = "Rendement propulsif")
    axs[0,1].set(xlabel = "Taux de compression")
    axs[0,1].grid()
    rend_prs_mono_max = np.nanmin(rend_prs_mono)
    rend_prs_mono_argmax = np.nanargmin(rend_prs_mono)
    rend_prs_tbfan_max = np.nanmin(rend_prs_tbfan)
    rend_prs_tbfan_argmax = np.argmin(rend_prs_tbfan)
    axs[0,1].text(rend_prs_mono_argmax, rend_prs_mono_max+0.01, s = "Rendement propulsif min = %2.2f \n pour OPR = %4.0f" %(rend_prs_mono_max, pi_c_list[rend_prs_mono_argmax]))
    axs[0,1].text(rend_prs_tbfan_argmax, rend_prs_tbfan_max+0.01, s="Rendement propulsif min = %2.2f \n pour OPR = %4.0f" % ( rend_prs_tbfan_max,pi_c_list[rend_prs_tbfan_argmax]))
    axs[0,1].plot(pi_c_list[rend_prs_tbfan_argmax],rend_prs_tbfan_max, color = "black", marker = "+")
    axs[0,1].plot(pi_c_list[rend_prs_mono_argmax],rend_prs_mono_max, color = "black", marker = "+")


    axs[1,0].plot(pi_c_list, rend_thermoprops_mono, label = "mono-corps mono-flux")
    axs[1,0].plot(pi_c_list, rend_thermoprops_tbfan, label = "turbofan")
    axs[1,0].set(ylabel = "Rendement thermopropulsif")
    axs[1,0].set(xlabel = "Taux de compression")
    axs[1,0].grid()
    rend_thermoprops_tbfan_max = np.nanmax(rend_thermoprops_tbfan)
    rend_thermoprops_tbfan_argmax = np.nanargmax(rend_thermoprops_tbfan)
    axs[1, 0].text(rend_thermoprops_tbfan_argmax, rend_thermoprops_tbfan_max+0.01, s="Rendement thermopropulsif max = %2.2f \n pour OPR = %4.0f" % ( rend_thermoprops_tbfan_max,pi_c_list[rend_thermoprops_tbfan_argmax]))
    axs[1,0].plot(pi_c_list[rend_thermoprops_tbfan_argmax],rend_thermoprops_tbfan_max, color = "black", marker = "+")


    axs[1,1].plot(pi_c_list, F_mono_sp_nets, label = "mono-corps mono-flux")
    axs[1,1].plot(pi_c_list, F_tbfan_sp_nets, label = "turbofan")
    axs[1,1].set(ylabel = "Poussée spécifique (m/s)")
    axs[1,1].set(xlabel = "Taux de compression")
    axs[1,1].grid()
    F_mono_max = np.nanmax(F_mono_sp_nets)
    F_mono_argmax = np.nanargmax(F_mono_sp_nets)
    F_tbfan_max = np.nanmax(F_tbfan_sp_nets)
    F_tbfan_argmax = np.argmax(F_tbfan_sp_nets)
    axs[1,1].text(F_mono_argmax, F_mono_max+20, s = "Poussée spécifique max = %2.2f m/s \n pour OPR = %4.0f" %(F_mono_max, pi_c_list[F_mono_argmax]))
    axs[1,1].text(F_tbfan_argmax, F_tbfan_max+20, s="Poussée spécifique max = %2.2f m/s \n pour OPR = %4.0f" % ( F_tbfan_max,pi_c_list[F_tbfan_argmax]))
    axs[1,1].plot(pi_c_list[F_tbfan_argmax],F_tbfan_max, color = "black", marker = "+")
    axs[1,1].plot(pi_c_list[F_mono_argmax],F_mono_max, color = "black", marker = "+")

    #plt.show()








    F_mono_sp_nets = []
    w_mono_chim_sps = []
    w_mono_pr_sps = []
    w_mono_sp_cycles = []

    F_tbfan_sp_nets = []
    w_tbfan_chim_sps = []
    w_tbfan_pr_sps = []
    w_tbfan_sp_cycles = []

    pi_c = 40
    Tt4_list = np.arange(840,1400,1)
    m_point = 215

    for Tt4 in Tt4_list:
        lambdaa = 11
        w_mono_sp_cycle, w_mono_chim_sp, w_mono_pr_sp, F_mono_sp_net = turbofan(M0, P0, T0, Tt4, pi_c, lambdaa, m_point)
        lambdaa = 1
        w_tbfan_sp_cycle, w_tbfan_chim_sp, w_tbfan_pr_sp, F_tbfan_sp_net = turbofan(M0, P0, T0, Tt4, pi_c, lambdaa, m_point)

        w_mono_chim_sps.append(w_mono_chim_sp)
        w_mono_pr_sps.append(w_mono_pr_sp)
        F_mono_sp_nets.append(F_mono_sp_net)
        w_mono_sp_cycles.append(w_mono_sp_cycle)

        w_tbfan_chim_sps.append(w_tbfan_chim_sp)
        w_tbfan_pr_sps.append(w_tbfan_pr_sp)
        F_tbfan_sp_nets.append(F_tbfan_sp_net)
        w_tbfan_sp_cycles.append(w_tbfan_sp_cycle)

    w_mono_sp_cycles = np.array(w_mono_sp_cycles)
    F_mono_sp_nets = np.array(F_mono_sp_nets)
    w_mono_chim_sps = np.array(w_mono_chim_sps)
    w_mono_pr_sps = np.array(w_mono_pr_sps)

    w_tbfan_chim_sps = np.array(w_tbfan_chim_sps)
    w_tbfan_pr_sps = np.array(w_tbfan_pr_sps)
    F_tbfan_sp_nets = np.array(F_tbfan_sp_nets)
    w_tbfan_sp_cycles = np.array(w_tbfan_sp_cycles)

    rend_ths_mono = w_mono_sp_cycles / w_mono_chim_sps
    rend_prs_mono = w_mono_pr_sps / w_mono_sp_cycles
    rend_thermoprops_mono = w_mono_pr_sps / w_mono_chim_sps

    rend_ths_tbfan = w_tbfan_sp_cycles / w_tbfan_chim_sps
    rend_prs_tbfan = w_tbfan_pr_sps / w_tbfan_sp_cycles
    rend_thermoprops_tbfan = w_tbfan_pr_sps / w_tbfan_chim_sps

    fig, axs = plt.subplots(2, 2, figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0.2})
    plt.subplots_adjust(wspace=0.3)
    fig.suptitle('Rendements et poussée spécifique Turboréacteur mono-corps mono-flux et Turbofan (BPR = 1 et OPR = 40)')

    axs[0, 0].plot(Tt4_list, rend_ths_mono, label="BPR = 11")
    axs[0, 0].plot(Tt4_list, rend_ths_tbfan, label="BPR = 1")
    axs[0, 0].set(ylabel="Rendement thermique")
    axs[0, 0].set(xlabel="Température totale fin combustion (K)")
    axs[0, 0].grid()
    axs[0, 0].legend()


    axs[0, 1].plot(Tt4_list, rend_prs_mono, label="mono-corps mono-flux")
    axs[0, 1].plot(Tt4_list, rend_prs_tbfan, label="turbofan")
    axs[0, 1].set(ylabel="Rendement propulsif")
    axs[0, 1].set(xlabel="Température totale fin combustion (K)")
    axs[0, 1].grid()
    axs[0,1].set(ylim = (0,1))
    axs[0,1].set(xlim = (915,1400))
    rend_prs_tbfan_max = np.nanmax(rend_prs_tbfan)
    rend_prs_tbfan_argmax = np.argmax(rend_prs_tbfan)
    axs[0, 1].text(rend_prs_tbfan_argmax, rend_prs_tbfan_max + 0.01,s="Rendement propulsif max = %2.2f \n pour OPR = %4.0f" % (rend_prs_tbfan_max, Tt4_list[rend_prs_tbfan_argmax]))
    axs[0, 1].plot(Tt4_list[rend_prs_tbfan_argmax], rend_prs_tbfan_max, color="black", marker="+")


    axs[1, 0].plot(Tt4_list, rend_thermoprops_mono, label="mono-corps mono-flux")
    axs[1, 0].plot(Tt4_list, rend_thermoprops_tbfan, label="turbofan")
    axs[1, 0].set(ylabel="Rendement thermopropulsif")
    axs[1, 0].set(xlabel="Température totale fin combustion (K)")
    axs[1, 0].grid()


    axs[1, 1].plot(Tt4_list, F_mono_sp_nets, label="mono-corps mono-flux")
    axs[1, 1].plot(Tt4_list, F_tbfan_sp_nets, label="turbofan")
    axs[1, 1].set(ylabel="Poussée spécifique (m/s)")
    axs[1, 1].set(xlabel="Température totale fin combustion (K)")
    axs[1, 1].grid()








    F_tbfan_sp_nets = []
    w_tbfan_chim_sps = []
    w_tbfan_pr_sps = []
    w_tbfan_sp_cycles = []

    pi_c = 40
    Tt4 = 1400
    lambdaa_list = np.arange(0.2,15,0.1)
    m_point = 215


    for lambdaa in lambdaa_list:
        w_mono_sp_cycle, w_mono_chim_sp, w_mono_pr_sp, F_mono_sp_net = turbofan(M0, P0, T0, Tt4, pi_c, lambdaa, m_point)
        w_tbfan_sp_cycle, w_tbfan_chim_sp, w_tbfan_pr_sp, F_tbfan_sp_net = turbofan(M0, P0, T0, Tt4, pi_c, lambdaa, m_point)


        w_tbfan_chim_sps.append(w_tbfan_chim_sp)
        w_tbfan_pr_sps.append(w_tbfan_pr_sp)
        F_tbfan_sp_nets.append(F_tbfan_sp_net)
        w_tbfan_sp_cycles.append(w_tbfan_sp_cycle)



    w_tbfan_chim_sps = np.array(w_tbfan_chim_sps)
    w_tbfan_pr_sps = np.array(w_tbfan_pr_sps)
    F_tbfan_sp_nets = np.array(F_tbfan_sp_nets)
    w_tbfan_sp_cycles = np.array(w_tbfan_sp_cycles)



    rend_ths_tbfan = w_tbfan_sp_cycles / w_tbfan_chim_sps
    rend_prs_tbfan = w_tbfan_pr_sps / w_tbfan_sp_cycles
    rend_thermoprops_tbfan = w_tbfan_pr_sps / w_tbfan_chim_sps

    fig, axs = plt.subplots(2, 2, figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0.2})
    plt.subplots_adjust(wspace=0.3)
    fig.suptitle('Rendements et poussée spécifique Turboréacteur mono-corps mono-flux et Turbofan (pi_c = 40 et Tt4 = 1600K')

    axs[0, 0].plot(lambdaa_list, rend_ths_tbfan, label="turbofan")
    axs[0, 0].set(ylabel="Rendement thermique")
    axs[0, 0].set(xlabel="Taux de dilution (BPR)")
    axs[0, 0].grid()
    axs[0, 0].legend()


    axs[0, 1].plot(lambdaa_list, rend_prs_tbfan, label="turbofan")
    axs[0, 1].set(ylabel="Rendement propulsif")
    axs[0, 1].set(xlabel="Taux de dilution (BPR)")
    axs[0, 1].grid()


    axs[1, 0].plot(lambdaa_list, rend_thermoprops_tbfan, label="turbofan")
    axs[1, 0].set(ylabel="Rendement thermopropulsif")
    axs[1, 0].set(xlabel="Taux de dilution (BPR)")
    axs[1, 0].grid()


    axs[1, 1].plot(lambdaa_list, F_tbfan_sp_nets, label="turbofan")
    axs[1, 1].set(ylabel="Poussée spécifique (m/s)")
    axs[1, 1].set(xlabel="Taux de dilution (BPR)")
    axs[1, 1].grid()


    F_tbfan_sp_nets = []
    w_tbfan_chim_sps = []
    w_tbfan_pr_sps = []
    w_tbfan_sp_cycles = []

    F_mono_sp_nets = []
    w_mono_chim_sps = []
    w_mono_pr_sps = []
    w_mono_sp_cycles = []

    pi_c = 40
    Tt4 = 1400
    lambdaa = 1
    m_point = 215
    M0_list = np.arange(0.3,2,0.05)

    for M0 in M0_list:
        lambdaa = 11
        w_mono_sp_cycle, w_mono_chim_sp, w_mono_pr_sp, F_mono_sp_net = turbofan(M0, P0, T0, Tt4, pi_c, lambdaa, m_point)
        lambdaa = 1
        w_tbfan_sp_cycle, w_tbfan_chim_sp, w_tbfan_pr_sp, F_tbfan_sp_net = turbofan(M0, P0, T0, Tt4, pi_c, lambdaa, m_point)

        w_mono_chim_sps.append(w_mono_chim_sp)
        w_mono_pr_sps.append(w_mono_pr_sp)
        F_mono_sp_nets.append(F_mono_sp_net)
        w_mono_sp_cycles.append(w_mono_sp_cycle)

        w_tbfan_chim_sps.append(w_tbfan_chim_sp)
        w_tbfan_pr_sps.append(w_tbfan_pr_sp)
        F_tbfan_sp_nets.append(F_tbfan_sp_net)
        w_tbfan_sp_cycles.append(w_tbfan_sp_cycle)

    w_mono_sp_cycles = np.array(w_mono_sp_cycles)
    F_mono_sp_nets = np.array(F_mono_sp_nets)
    w_mono_chim_sps = np.array(w_mono_chim_sps)
    w_mono_pr_sps = np.array(w_mono_pr_sps)

    w_tbfan_chim_sps = np.array(w_tbfan_chim_sps)
    w_tbfan_pr_sps = np.array(w_tbfan_pr_sps)
    F_tbfan_sp_nets = np.array(F_tbfan_sp_nets)
    w_tbfan_sp_cycles = np.array(w_tbfan_sp_cycles)

    rend_ths_mono = w_mono_sp_cycles / w_mono_chim_sps
    rend_prs_mono = w_mono_pr_sps / w_mono_sp_cycles
    rend_thermoprops_mono = w_mono_pr_sps / w_mono_chim_sps

    rend_ths_tbfan = w_tbfan_sp_cycles / w_tbfan_chim_sps
    rend_prs_tbfan = w_tbfan_pr_sps / w_tbfan_sp_cycles
    rend_thermoprops_tbfan = w_tbfan_pr_sps / w_tbfan_chim_sps

    fig, axs = plt.subplots(2, 2, figsize=(9, 10), dpi=80, gridspec_kw={'hspace': 0.2})
    plt.subplots_adjust(wspace=0.3)
    fig.suptitle(
        'Rendements et poussée spécifique Turboréacteur mono-corps mono-flux et Turbofan (BPR = 1, OPR = 40 et P0 = 227 hPa)')

    axs[0, 0].plot(M0_list, rend_ths_mono, label="BPR = 11")
    axs[0, 0].plot(M0_list, rend_ths_tbfan, label="BPR = 1")
    axs[0, 0].set(ylabel="Rendement thermique")
    axs[0, 0].set(xlabel="Nb Mach")
    axs[0, 0].grid()
    axs[0, 0].legend()

    axs[0, 1].plot(M0_list, rend_prs_mono, label="mono-corps mono-flux")
    axs[0, 1].plot(M0_list, rend_prs_tbfan, label="turbofan")
    axs[0, 1].set(ylabel="Rendement propulsif")
    axs[0, 1].set(xlabel="Nb Mach")
    axs[0, 1].grid()



    axs[1, 0].plot(M0_list, rend_thermoprops_mono, label="mono-corps mono-flux")
    axs[1, 0].plot(M0_list, rend_thermoprops_tbfan, label="turbofan")
    axs[1, 0].set(ylabel="Rendement thermopropulsif")
    axs[1, 0].set(xlabel="Nb Mach")
    axs[1, 0].grid()

    axs[1, 1].plot(M0_list, F_mono_sp_nets, label="mono-corps mono-flux")
    axs[1, 1].plot(M0_list, F_tbfan_sp_nets, label="turbofan")
    axs[1, 1].set(ylabel="Poussée spécifique (m/s)")
    axs[1, 1].set(xlabel="Nb Mach")
    axs[1, 1].grid()



    plt.show()