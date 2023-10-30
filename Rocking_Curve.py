import numpy as np
import Crystals


def RockingCurve(central_angle,angular_range, step_size, Lattice_planes_miscut, lam, lattice_indices):
    
    
    DD=Crystals.CrystalData(Lattice_planes_miscut, crystal='Si', lattice_indices=lattice_indices)#'Ge')
    C_a,(Bragg_angle, Max_Darvin_curve, Polarizability, Polarizability_minus, Beta, assymetry_ratio, Lattice_plane_trace_and_surface,struct_factor) = DD.core_angle(lam,Lattice_planes_miscut)
    ####print('This Bragg angle: ', Bragg_angle)
    #central_angle= Bragg_angle
    
    #central_angle=DD.calc_theta_B
    #struct_factor=DD.get_Fc(12.3984193E-10/lam)
    Polarizability_0=DD.get_chi_0(12.3984193E-10/lam)#Get_Polarizability_0(lam)
    Reflectivity=np.zeros(int(((2*angular_range))//(step_size)))
    Angle_deviation=np.zeros(int(((2*angular_range))//(step_size)))
    ii=0
    #Angle_array= np.linspace((central_angle-angular_range),(central_angle+angular_range),int(((2*angular_range))//(step_size)))
    R_e=2.8179403262e-09
    cell_dim=0.000543095#0.00056575
    for i in range(int(((2*angular_range))//(step_size))):
        Bragg_angle_deviation=central_angle-angular_range+i*step_size
        incident_angle=Bragg_angle-Lattice_planes_miscut-Bragg_angle_deviation
        if incident_angle<=0:
            incident_angle=-incident_angle
        
        R_e=2.8179403262e-09
        cell_dim=0.000543095
        Bragg_angle=np.arcsin(lam*1e6 / (2 * cell_dim / np.sqrt(lattice_indices[0] ** 2 + lattice_indices[1] ** 2 + lattice_indices[2] ** 2)))
        
        
        gamma_o = np.sin(Bragg_angle - Lattice_planes_miscut)
        gamma_h = np.sin(- Bragg_angle - Lattice_planes_miscut)
        gamma = gamma_h / gamma_o
        
        
        
        bragg_shift_oc=( -gamma_o + np.sqrt(gamma_o ** 2 - (gamma_o- gamma_h) * np.sqrt(1 - gamma_o ** 2) * Polarizability_0/ np.sin(2 * Bragg_angle))) / np.sqrt(1 - gamma_o ** 2)
        bragg_shift_os=-1*Polarizability_0*(1-gamma)/(2*np.sin(2*Bragg_angle))
        delta_os= R_e * (lam*1e6)**2 / (np.pi * (cell_dim**3) * np.sin(2 * Bragg_angle)) * np.sqrt(struct_factor) * np.sqrt(np.abs(gamma))
        delta_oc=delta_os * (Bragg_angle - Lattice_planes_miscut) / (bragg_shift_oc + Bragg_angle - Lattice_planes_miscut)

        Deviation_parameter_complex=(Bragg_angle_deviation - bragg_shift_oc) / delta_oc
        ##Deviation_parameter_complex=(Bragg_angle_deviation - bragg_shift_os) / delta_os
        
        ############################Reflectivity[ii] =np.abs(Polarizability/Polarizability_minus)*np.abs(Deviation_parameter_complex-np.sign(Deviation_parameter_complex.real)*np.sqrt(Deviation_parameter_complex*Deviation_parameter_complex-1))**2
        ###Reflectivity[ii]=-1*np.sqrt(Polarizability*Polarizability_minus/np.abs(assymetry_ratio))*(Deviation_parameter_complex-np.sign(Deviation_parameter_complex.real)*np.sqrt(Deviation_parameter_complex*Deviation_parameter_complex-1))/Polarizability_minus
        Reflectivity[ii]=np.abs(Polarizability/Polarizability_minus)*abs(Deviation_parameter_complex-np.sign(Deviation_parameter_complex.real)*np.sqrt(Deviation_parameter_complex*Deviation_parameter_complex-1))**2
        
        
        Angle_deviation[i] = Bragg_angle_deviation
        ii+=1
        '''
        bragg_shift_oc=( -gamma_o + np.sqrt(gamma_o ** 2 - (gamma_o- gamma_h) * np.sqrt(1 - gamma_o ** 2) * Polarizability_0/ np.sin(2 * Bragg_angle))) / np.sqrt(1 - gamma_o ** 2)
        delta_os= R_e * (lam*1e6)**2 / (np.pi * (cell_dim**3) * np.sin(2 * Bragg_angle)) * np.sqrt(struct_factor ** 2) * np.sqrt(np.abs(gamma))
        delta_oc=delta_os * (Bragg_angle - Lattice_planes_miscut) / (bragg_shift_oc + Bragg_angle - Lattice_planes_miscut)

        #Deviation_parameter_complex=(Bragg_angle_deviation - bragg_shift_oc) / delta_oc
        Deviation_parameter_complex=DD.calc_deviation_par_h(12.3984193E-10/lam,Lattice_planes_miscut,Bragg_angle_deviation)
        
        Polarizability=DD.calc_chih(12.3984193E-10/lam)
        Polarizability_minus=DD.calc_chih(12.3984193E-10/lam)
        Reflectivity[ii] =np.abs(Polarizability/Polarizability_minus)*np.abs(Deviation_parameter_complex-np.sign(Deviation_parameter_complex.real)*np.sqrt(Deviation_parameter_complex*Deviation_parameter_complex-1))**2
        
        Angle_deviation[i] = Bragg_angle_deviation
        ii+=1
        '''
        
    return Angle_deviation,  abs(Reflectivity), Bragg_angle, C_a



