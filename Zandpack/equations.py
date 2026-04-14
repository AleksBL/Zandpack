#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:11:47 2026

@author: aleks
"""
                                                                                                                                                                                                                                                      
Eqs = {
        #"liouville_von_neumann":
        #"i * h_bar * d/dt sigma_F(t) = [H_F(t), sigma_F(t)]",

        #"hamiltonian_matrix":
        #"H_F = [ [H_L, H_LD, 0], [H_DL, H_D, H_DR], [0, H_RD, H_R] ]",

        #"density_matrix":
        #"sigma_F = [ [sigma_L, sigma_LD, sigma_LR], [sigma_DL, sigma_D, sigma_DR], [sigma_RL, sigma_RD, sigma_R] ]",

        "device_density_matrix_eom":
        "i * h_bar * d/dt sigma_D(t) = [H_D(t), sigma_D(t)] + i * sum_alpha [Pi_alpha(t) + Pi_alpha_dagger(t)]",

        "current_matrix":
        "Pi_alpha(t) = integral from t0 to t dt' [G_D^>(t, t') * Sigma_alpha^<(t', t) - G_D^<(t, t') * Sigma_alpha^>(t', t)]",

        "lesser_self_energy":
        "Sigma_alpha^<(t2, t) = (i / (2 * pi * h_bar)) * exp[(i / h_bar) * integral from 0 to t dt1 Delta_alpha(t1)] * "
        "integral from -infinity to +infinity d_epsilon f_alpha(epsilon) * Gamma_alpha(epsilon) * exp[-(i / h_bar) * epsilon * (t2 - t)]",

        "greater_self_energy":
        "Sigma_alpha^>(t2, t) = -(i / (2 * pi * h_bar)) * exp[(i / h_bar) * integral from 0 to t dt1 Delta_alpha(t1)] * "
        "integral from -infinity to +infinity d_epsilon (1 - f_alpha(epsilon)) * Gamma_alpha(epsilon) * exp[-(i / h_bar) * epsilon * (t2 - t)]",

        "level_width_function":
        "Gamma_alpha(epsilon) = sum_l^Nl L_al(epsilon) * W_al",

        "lorentzian_function":
        "L_c(epsilon) = gamma_v^2 / [(epsilon - epsilon_v)^2 + gamma_v^2]",

        "eigendecomposition":
        "W_al = sum_c lambda_alc * xi_alc ⊗ xi_alc_dagger",

        "fermi_function_expansion":
        "F(s) = 1 / (1 + exp(s)) = 1/2 - sum_p R_p / (s - z_p^+ + R_p / (s - z_p^-))",

        "self_energy_pole_expansion":
        "Sigma_alpha^</>(t2, t) = (1 / h_bar) * sum_{c, ±} Lambda_axc^</>± * xi_axc ⊗ xi_axc_dagger * exp[-(i / h_bar) * integral from 0 to t dt' chi_ax^±(t')]",

        "auxiliary_mode_eom":
        "i * h_bar * d/dt Psi_axc = [H(t) - I * chi_ax^+(t)] * Psi_axc + h_bar * Lambda_axc^<,+ * xi_axc + "
        "h_bar * (Lambda_axc^>,+ - Lambda_axc^<,+) * sigma * xi_axc + (1 / h_bar) * sum_{a' x' c'} Omega_axc a' x' c' * xi_a' x' c'",

        "omega_eom":
        "i * h_bar * d/dt Omega_axc a' x' c' = (chi_a' x'^- (t) - chi_ax^+ (t)) * Omega_axc a' x' c' + "
        "i * h_bar * (Lambda_axc^>, - - Lambda_axc^<, -) * xi_a' x' c'_dagger * Psi_axc + "
        "i * h_bar * (Lambda_axc^>, + - Lambda_axc^<, +) * Psi_a' x' c'_dagger * xi_axc",

        "steady_state_psi":
        "Psi_axc^0 = -[H^0 - I * chi_asc^+]^(-1) * [h_bar * Lambda_asc^×+ * xi_asc + "
        "h_bar * (Lambda_asc^×+ - Lambda_asc^×+) * sigma^0 * xi_asc + (1 / h_bar) * sum_{a' s' c'} Omega_asc a' s' c'^0 * xi_a' s' c']",

        "steady_state_density_matrix":
        "sigma^0 = (i / (2 * pi)) * integral from -infinity to +infinity G^r(epsilon) * Sigma^<(epsilon) * G^a(epsilon) d_epsilon",

        #"mutual_information":
        #"MI_AB = S([sigma]_A) + S([sigma]_B) - S([sigma]_{A ∪ B})",

        #"entropy":
        #"S(sigma) = -Tr[sigma * log(sigma)] - Tr[(1 - sigma) * log(1 - sigma)]",

        #"dipole_hamiltonian":
        #"H_dip = V_dip(t) * sum_{<i,j>, s} (x_ij / |r_ij|) * c_is_dagger * c_js",

        "continuity_equation":
        "dN/dt = sum_a J_a(t)",

        "lowdin_transformation":
        "M_D' = S^(-1/2) * M_D * S^(-1/2)",

        "mulliken_charge_matrix":
        "Q_0 = (1/2) * [sigma^NO * S + S * sigma^NO]",

        "linearized_hamiltonian":
        "H(sigma(t)) = H^0 + sum_i dH/dq_i * [Q(sigma(t)) - Q_0]_ii",

        #"gold_junction_potential":
        #"V_ii(t) = (z_i - z_center) / (z_max - z_min) * V_bias(t)",

        "bcs_hamiltonian":
        "H_D^BCS = sum_{ij ∈ i'} h_{ij ∈ i'} c_iλ_dagger c_jλ' + sum_{ij}^+ℓ' [Δ_ij^+ℓ c_iλ_dagger c_jλ'_dagger + Δ_ij^+ℓ'+ c_iλ c_jλ']",

        #"bogoliubov_de_gennes_matrix":
        #"H_ij^BdG = [ [h_ij^↑↑/2, h_ij^↑↓/2, Δ_ij^↑↑, Δ_ij^↑↓], "
        #"[h_ij^↓↑/2, h_ij^↓↓/2, Δ_ij^↓↑, Δ_ij^↓↓], "
        #"[Δ_ij^↑↑*, Δ_ij^↓↑*, -h_ij^↑↑/2, -h_ij^↑↓/2], "
        #"[Δ_ij^↑↓*, Δ_ij^↓↓*, -h_ij^↓↑/2, -h_ij^↓↓/2] ]",

        #"bogoliubov_level_width":
        #"Gamma_alpha^BdG(epsilon) = (1/2) * [ [Gamma_alpha(epsilon), 0, 0, 0], "
        #"[0, Gamma_alpha(epsilon), 0, 0], "
        #"[0, 0, Gamma_alpha(-epsilon), 0], "
        #"[0, 0, 0, Gamma_alpha(-epsilon)] ]",

        #"cost_function":
        #"E = integral from -infinity to +infinity (f(x) - L(x))^2 dx",

        #"repulsive_term":
        #"R = alpha_PO * sum_{l ≠ l'} tan(pi/2 * O_ll') with O_ll' = M_ll' / sqrt(M_ll * M_l'l')",

        #"gradient_cost_function":
        #"∂E/∂x_i = Gamma_i^2 * ∂/∂x_i M_ii + sum_{l ≠ l'} Gamma_l * Gamma_l' * ∂/∂x_i M_il + 2 * sum_j f_j * Gamma_j * ∂/∂x_i K_ji",

        "Hamiltonian_elec_dev_overlap_correction":
        "Sigma_a^0(t) = -Delta_a(t) * S_Da * S_a^(-1) * S_aD",

        "Overlap_elec_dev_overlap Correction":
        "Sigma_a^1 = S_Da * S_a^(-1) * S_aD",
    }
_Eqs = {
        "liouville_von_neumann": r"""
        $$
        i \hbar \frac{\mathrm{d}}{\mathrm{d} t} \boldsymbol{\sigma}_F(t) = \left[ \mathbf{H}_F(t), \boldsymbol{\sigma}_F(t) \right],
        $$
        """,

        "hamiltonian_matrix": r"""
        $$
        \mathbf{H}_F = \begin{bmatrix}
        \mathbf{H}_L & \mathbf{H}_{L,D} & 0 \\
        \mathbf{H}_{D,L} & \mathbf{H}_D & \mathbf{H}_{D,R} \\
        0 & \mathbf{H}_{R,D} & \mathbf{H}_R
        \end{bmatrix}
        $$
        """,

        "density_matrix": r"""
        $$
        \boldsymbol{\sigma}_F = \begin{bmatrix}
        \boldsymbol{\sigma}_L & \boldsymbol{\sigma}_{L,D} & \boldsymbol{\sigma}_{L,R} \\
        \boldsymbol{\sigma}_{D,L} & \boldsymbol{\sigma}_D & \boldsymbol{\sigma}_{D,R} \\
        \boldsymbol{\sigma}_{R,L} & \boldsymbol{\sigma}_{R,D} & \boldsymbol{\sigma}_R
        \end{bmatrix}
        $$
        """,

        "device_density_matrix_eom": r"""
        $$
        i \hbar \frac{\mathrm{d}}{\mathrm{d} t} \sigma_D(t) = \left[ \mathbf{H}_D(t), \sigma_D(t) \right] + i \sum_{\alpha} \left[ \boldsymbol{\Pi}_{\alpha}(t) + \boldsymbol{\Pi}_{\alpha}^{\dagger}(t) \right]
        $$
        """,

        "current_matrix": r"""
        $$
        \boldsymbol{\Pi}_{\alpha}(t) = \int_{t_0}^{t} \mathrm{d} t' \left[ \mathbf{G}_D^{>}(t, t') \boldsymbol{\Sigma}_{\alpha}^{<}(t', t) - \mathbf{G}_D^{<}(t, t') \boldsymbol{\Sigma}_{\alpha}^{>}(t', t) \right]
        $$
        """,

        "lesser_self_energy": r"""
        $$
        \boldsymbol{\Sigma}_{\alpha}^{<}(t_2, t) = \frac{i}{2 \pi \hbar} \mathrm{e}^{\frac{i}{\hbar} \int_0^t \mathrm{d} t_1 \Delta_{\alpha}(t_1)} \int_{-\infty}^{\infty} \mathrm{d} \epsilon f_{\alpha}(\epsilon) \boldsymbol{\Gamma}_{\alpha}(\epsilon) \mathrm{e}^{-\frac{i}{\hbar} \epsilon (t_2 - t)}
        $$
        """,

        "greater_self_energy": r"""
        $$
        \boldsymbol{\Sigma}_{\alpha}^{>}(t_2, t) = -\frac{i}{2 \pi \hbar} \mathrm{e}^{\frac{i}{\hbar} \int_0^t \mathrm{d} t_1 \Delta_{\alpha}(t_1)} \int_{-\infty}^{\infty} \mathrm{d} \epsilon (1 - f_{\alpha}(\epsilon)) \boldsymbol{\Gamma}_{\alpha}(\epsilon) \mathrm{e}^{-\frac{i}{\hbar} \epsilon (t_2 - t)}
        $$
        """,

        "level_width_function": r"""
        $$
        \boldsymbol{\Gamma}_{\alpha}(\epsilon) = \sum_{l}^{N_l} L_{al}(\epsilon) \mathbf{W}_{al}
        $$
        """,

        "lorentzian_function": r"""
        $$
        L_{c}(\epsilon) = \frac{\gamma_{v}^{2}}{(\epsilon - \epsilon_{v})^{2} + \gamma_{v}^{2}} = \frac{i \gamma_{v}}{2} \left[ \frac{1}{\epsilon - z_{v}^{+} - \frac{1}{\epsilon - z_{v}^{+}}} \right]
        $$
        """,

        "eigendecomposition": r"""
        $$
        \mathbf{W}_{al} = \sum_{c} \lambda_{alc} \vec{\xi}_{alc} \otimes \vec{\xi}_{alc}^{\dagger}
        $$
        """,

        "fermi_function_expansion": r"""
        $$
        F(s) = \frac{1}{1 + \mathrm{e}^{s}} = \frac{1}{2} - \sum_{p} \frac{R_{p}}{s - z_{p}^{+} + \frac{R_{p}}{s - z_{p}^{-}}}
        $$
        """,

        "self_energy_pole_expansion": r"""
        $$
        \mathbf{\Sigma}_{\alpha}^{</>}(t_2, t) = \frac{1}{\hbar} \sum_{c, \pm} \Lambda_{axc}^{</>\pm} \vec{\xi}_{axc} \otimes \vec{\xi}_{axc}^{\dagger} \mathrm{e}^{-\frac{i}{\hbar} \int_{0}^{t} \mathrm{d} t' \chi^{\pm}_{ax}(t')}
        $$
        """,

        "auxiliary_mode_eom": r"""
        $$
        i \hbar \frac{\mathrm{d}}{\mathrm{d} t} \vec{\Psi}_{axc} = \left[ \mathbf{H}(t) - \mathbf{1} \chi^{+}_{ax}(t) \right] \vec{\Psi}_{axc} + \hbar \Lambda_{axc}^{<,+} \vec{\xi}_{axc} + \hbar (\Lambda_{axc}^{>,+} - \Lambda_{axc}^{<,+}) \sigma \vec{\xi}_{axc} + \frac{1}{\hbar} \sum_{a' x' c'} \Omega_{axc a' x' c'} \vec{\xi}_{a' x' c'}
        $$
        """,

        "omega_eom": r"""
        $$
        i \hbar \frac{\mathrm{d}}{\mathrm{d} t} \Omega_{axc a' x' c'} = (\chi^{-}_{a' x'}(t) - \chi^{+}_{ax}(t)) \Omega_{axc a' x' c'} + i \hbar (\Lambda_{axc}^{>, -} - \Lambda_{axc}^{<, -}) \vec{\xi}^{\dagger}_{a' x' c'} \vec{\Psi}_{axc} + i \hbar (\Lambda_{axc}^{>, +} - \Lambda_{axc}^{<, +}) \vec{\Psi}^{\dagger}_{a' x' c'} \vec{\xi}_{axc}
        $$
        """,

        "steady_state_psi": r"""
        $$
        \vec{\Psi}^{0}_{asc} = -\left[ \mathbf{H}^{0} - \mathbf{1} \chi^{+}_{asc} \right]^{-1} \left( \hbar \Lambda^{\times+}_{asc} \vec{\xi}_{asc} + \hbar (\Lambda^{\times+}_{asc} - \Lambda^{\times+}_{asc}) \bm{\sigma}^{0} \vec{\xi}_{asc} + \frac{1}{\hbar} \sum_{a' s' c'} \Omega^{0}_{asca' s' c'} \vec{\xi}_{a' s' c'} \right)
        $$
        """,

        "steady_state_density_matrix": r"""
        $$
        \bm{\sigma}^{0} = \frac{i}{2 \pi} \int_{-\infty}^{+\infty} \mathbf{G}^{r}(\epsilon) \bm{\Sigma}^{<}(\epsilon) \mathbf{G}^{a}(\epsilon) \mathrm{d} \epsilon
        $$
        """,

        "mutual_information": r"""
        $$
        \mathrm{MI}_{AB} = S\left([\boldsymbol{\sigma}]_{A}\right) + S\left([\boldsymbol{\sigma}]_{B}\right) - S\left([\boldsymbol{\sigma}]_{A \cup B}\right)
        $$
        """,

        "entropy": r"""
        $$
        S(\sigma) = -\mathrm{Tr}[\sigma \log \sigma] - \mathrm{Tr}\left[(1 - \sigma) \log (1 - \sigma)\right]
        $$
        """,

        "dipole_hamiltonian": r"""
        $$
        H_{dip} = V_{dip}(t) \sum_{<i,j>, s} \frac{x_{ij}}{|\vec{r}_{ij}|} c_{is}^{\dagger} c_{js}
        $$
        """,

        "continuity_equation": r"""
        $$
        \frac{\mathrm{d} N}{\mathrm{d} t} = \sum_{a} J_{a}(t)
        $$
        """,

        "lowdin_transformation": r"""
        $$
        \mathbf{M}_{D}^{\prime} = \tilde{\mathbf{S}}^{-\frac{1}{2}} \mathbf{M}_{D} \tilde{\mathbf{S}}^{-\frac{1}{2}}
        $$
        """,

        "mulliken_charge_matrix": r"""
        $$
        \mathbf{Q}_{0} = \frac{1}{2} \left[ \boldsymbol{\sigma}^{NO} \mathbf{S} + \mathbf{S} \boldsymbol{\sigma}^{NO} \right]
        $$
        """,

        "linearized_hamiltonian": r"""
        $$
        \mathbf{H}(\boldsymbol{\sigma}(t)) = \mathbf{H}^{0} + \sum_{i} \frac{\mathrm{d} \mathbf{H}}{\mathrm{d} q_{i}} \left[ \mathbf{Q}(\boldsymbol{\sigma}(t)) - \mathbf{Q}_{0} \right]_{ii}
        $$
        """,

        "gold_junction_potential": r"""
        $$
        V_{ii}(t) = \frac{z_{i} - z_{\text{center}}}{z_{\text{max}} - z_{\text{min}}} V_{\text{bias}}(t)
        $$
        """,

        "bcs_hamiltonian": r"""
        $$
        H_{D}^{BCS} = \sum_{ij \in i'} h_{ij \in i'} c^{\dagger}_{i \lambda} c^{\phantom{\dagger}}_{j \lambda'} + \sum_{ij}^{+\ell'} \left[ \Delta^{+\ell}_{ij} c^{\dagger}_{i \lambda} c^{\dagger}_{j \lambda'} + \Delta^{+\ell' +}_{ij} c_{i \lambda} c_{j \lambda'} \right]
        $$
        """,

        "bogoliubov_de_gennes_matrix": r"""
        $$
        \mathbf{H}_{ij}^{BdG} = \begin{bmatrix}
        h_{ij}^{\uparrow \uparrow}/2 & h_{ij}^{\uparrow \downarrow}/2 & \Delta_{ij}^{\uparrow \uparrow} & \Delta_{ij}^{\uparrow \downarrow} \\
        h_{ij}^{\downarrow \uparrow}/2 & h_{ij}^{\downarrow \downarrow}/2 & \Delta_{ij}^{\downarrow \uparrow} & \Delta_{ij}^{\downarrow \downarrow} \\
        \Delta_{ij}^{\uparrow \uparrow*} & \Delta_{ij}^{\downarrow \uparrow*} & -h_{ij}^{\uparrow \uparrow}/2 & -h_{ij}^{\uparrow \downarrow}/2 \\
        \Delta_{ij}^{\uparrow \downarrow*} & \Delta_{ij}^{\downarrow \downarrow*} & -h_{ij}^{\downarrow \uparrow}/2 & -h_{ij}^{\downarrow \downarrow}/2
        \end{bmatrix}
        $$
        """,

        "bogoliubov_level_width": r"""
        $$
        \mathbf{\Gamma}_{\alpha}^{BdG}(\epsilon) = \frac{1}{2} \begin{bmatrix}
        \mathbf{\Gamma}_{\alpha}(\epsilon) & 0 & 0 & 0 \\
        0 & \mathbf{\Gamma}_{\alpha}(\epsilon) & 0 & 0 \\
        0 & 0 & \mathbf{\Gamma}_{\alpha}(-\epsilon) & 0 \\
        0 & 0 & 0 & \mathbf{\Gamma}_{\alpha}(-\epsilon)
        \end{bmatrix}
        $$
        """,

        "cost_function": r"""
        $$
        E = \int_{-\infty}^{\infty} (f(x) - \mathcal{L}(x))^{2} \mathrm{d} x
        $$
        """,

        "repulsive_term": r"""
        $$
        R = \alpha_{PO} \sum_{l \neq l'} \tan \left( \frac{\pi}{2} O_{ll'} \right) \quad \text{with} \quad O_{ll'} = \frac{M_{ll'}}{\sqrt{M_{ll} M_{l' l'}}}
        $$
        """,

        "gradient_cost_function": r"""
        $$
        \frac{\partial E}{\partial x_i} = \Gamma_i^2 \frac{\partial}{\partial x_i} M_{ii} + \sum_{l \neq l'} \Gamma_l \Gamma_{l'} \frac{\partial}{\partial x_i} M_{il} + 2 \sum_{j} f_j \Gamma_j \frac{\partial}{\partial x_i} K_{ji}
        $$
        """,

        "sigma_0": r"""
        $$
        \bm{\Sigma}_{a}^{0}(t) = -\Delta_{a}(t) \mathbf{S}_{Da} \mathbf{S}_{a}^{-1} \mathbf{S}_{aD}
        $$
        """,

        "sigma_1": r"""
        $$
        \bm{\Sigma}_{a}^{1} = \mathbf{S}_{Da} \mathbf{S}_{a}^{-1} \mathbf{S}_{aD}
        $$
        """,
    }
     
