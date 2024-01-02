import numpy as np
import pathlib
import matplotlib.pyplot as plt

class Slits:
    def __init__(self, _sizes):
        self.hor_size1=_sizes[0]
        self.vert_size1=_sizes[2]
        self.hor_size2=_sizes[1]
        self.vert_size2=_sizes[3]

class CrystalData:
    def __init__(self, Lattice_planes_miscut, crystal='Si', lattice_indices=[2,2,0]):
        path = pathlib.Path(__file__).parent.absolute()
        self.Lattice_planes_miscut = Lattice_planes_miscut  # miscut crystal
        #self.miller = lattice_indices[0]
        self.lattice_indices = lattice_indices
        self.R_str_fac_to_polarizability=0.00000000000000281794 #in meters
        
        if crystal == 'Si':
            ##self.a = 0.54309e-3  # lattice constant of unit cell [um]
            self.unit_cell_dimension= 0.000000000543095 #in meters
            '''
            if self.miller == 2:
                self.form_factor_real = 8.66557#8.722
            elif self.miller == 1:
                self.form_factor_real = 10.3811#10.525
            elif self.miller == 4:
                self.form_factor_real = 6.1045#6.240+5.769
            else:
                print('Crystal latice isn\'t set')
                self.form_factor_real = 8.66557
            '''
            #self.form_factor_real = np.loadtxt(str(path) + "/crystal_constants/f1_Si_50keV.txt")
            self.form_factor_real = np.loadtxt(str(path) + "/crystal_constants/f1_Si_AllAngles.txt")
            self.form_factor_imag = np.loadtxt(str(path) + "/crystal_constants/f2_Si_50keV.txt")
            self.chio_real = np.loadtxt(str(path) + "/crystal_constants/chio_Si_real.txt")
            self.chio_imag = np.loadtxt(str(path) + "/crystal_constants/chio_Si_imag.txt")

        if crystal == 'Ge':
            ##self.a = 0.56578e-3  # lattice constant of unit cell [um]
            self.unit_cell_dimension= 0.00000000056578 #in meters
            '''
            if self.miller == 2:
                self.form_factor_real = 23.7972
            elif self.miller == 1:
                self.form_factor_real = 27.3664
            else:
                print('Crystal latice isn\'t set')
                self.form_factor_real = 23.7972
            '''
            self.form_factor_imag = np.loadtxt(str(path) + "/crystal_constants/f2_Ge_50keV.txt")
            self.chio_real = np.loadtxt(str(path) + "/crystal_constants/chio_Ge_real.txt")
            self.chio_imag = np.loadtxt(str(path) + "/crystal_constants/chio_Ge_imag.txt")
    def get_chi_0(self, E):
        chio_real = np.interp(E, self.chio_real[:, 0], self.chio_real[:, 1])
        chio_imag = np.interp(E, self.chio_imag[:, 0], self.chio_imag[:, 1])
        return chio_real + 1j * chio_imag
    def get_form_factor(self, E):
        lam=(12.3984193E-10/E)
        Bragg_angle=np.arcsin(1/(2*self.unit_cell_dimension/(np.sqrt(self.lattice_indices[0]**2+self.lattice_indices[1]**2+self.lattice_indices[2]**2)*lam)))
        #####print('sinus Bragg:   ___ ',(np.sin(Bragg_angle)/(lam*1e10)))
        form_factor_imag = np.interp(E, self.form_factor_imag[:, 0], self.form_factor_imag[:, 1])
        #form_factor_real = np.interp(E, self.form_factor_real[:, 0], self.form_factor_real[:, 1])
        form_factor_real = np.interp((np.sin(Bragg_angle)/(lam*1e10)), self.form_factor_real[:, 0], self.form_factor_real[:, 1])
        form_factor = form_factor_real + 1j * form_factor_imag
        ######print('The form factor real:   ___ ',form_factor_real)
        return form_factor
    '''
    def get_Fc(self, E):
        form_factor_imag = np.interp(E, self.form_factor_imag[:, 0], self.form_factor_imag[:, 1])
        form_factor = self.form_factor_real + 1j * form_factor_imag
        if self.miller == 1:
            structure_factor = 4 * (1 + 1j)
        else:
            structure_factor = 8  # default is 220

        Fc = form_factor * structure_factor  # structure factor complex
        return Fc
    '''



    def core_angle(self,lam,planes_miscut):
        #miscut_sign=np.sign(planes_miscut)
        #planes_miscut=np.abs(planes_miscut)
        
        #DD=CrystalData(self.Lattice_planes_miscut, crystal='Si', lattice_indices=lattice_indices)
        k = np.pi*2/lam
        scatter_fac=self.get_form_factor(12.3984193E-10/lam)
        struct_factor=scatter_fac*(1+(-1)**(self.lattice_indices[0]+self.lattice_indices[1])+(-1)**(self.lattice_indices[0]+self.lattice_indices[2])+(-1)**(self.lattice_indices[1]+self.lattice_indices[2]))*(1+pow(-1*1j,(self.lattice_indices[0]+self.lattice_indices[1]+self.lattice_indices[2])))
        Polarizability=-1*self.R_str_fac_to_polarizability*lam*lam*struct_factor/(np.pi*self.unit_cell_dimension**3)
        #Polarizability_0=self.get_chi_0(12.3984193E-10/lam)
        struct_factor_minus=scatter_fac*(1+(-1)**(-self.lattice_indices[0]-self.lattice_indices[1])+(-1)**(-self.lattice_indices[0]-self.lattice_indices[2])+(-1)**(-self.lattice_indices[1] -self.lattice_indices[2]))*(1+pow(-1*1j,(-self.lattice_indices[0]-self.lattice_indices[1]-self.lattice_indices[2])))
        
        Polarizability_minus=-1*self.R_str_fac_to_polarizability*lam*lam*struct_factor_minus/(np.pi*self.unit_cell_dimension**3)
        
        Beta=np.angle((Polarizability_minus*Polarizability),deg=False)/2
        
        Bragg_angle=np.arcsin(1/(2*self.unit_cell_dimension/(np.sqrt(self.lattice_indices[0]**2+self.lattice_indices[1]**2+self.lattice_indices[2]**2)*lam)))
        Lattice_plane_trace_and_surface=-np.pi/2-planes_miscut #angle between surface normal and trace of lattice planes
        assymetry_ratio = np.cos(Lattice_plane_trace_and_surface-Bragg_angle)/np.cos(Lattice_plane_trace_and_surface+Bragg_angle)
        Central_incid_angle=Bragg_angle - planes_miscut# +(1-assymetry_ratio)*(np.real(Polarizability)*np.cos(Beta)+np.imag(Polarizability)*np.sin(Beta))/(2*np.sin(2*Bragg_angle))
        
        #DD=CrystalData(self.Lattice_planes_miscut, crystal='Si', lattice_indices=lattice_indices)
        R_e=2.8179403262e-09
        cell_dim=0.000543095
        #if isinstance(lam, (list, tuple)):
        #    lam=lam[0]
        Bragg_angle_darwin=np.arcsin(lam*1e6 / (2 * cell_dim / np.sqrt(self.lattice_indices[0] ** 2 + self.lattice_indices[1] ** 2 + self.lattice_indices[2] ** 2)))
           
        gamma_o = np.sin(Bragg_angle_darwin - planes_miscut)
        gamma_h = np.sin(- Bragg_angle_darwin - planes_miscut)
        gamma = gamma_h / gamma_o
        Max_Darvin_curve=(( -gamma_o + np.sqrt(gamma_o ** 2 - (gamma_o- gamma_h) * np.sqrt(1 - gamma_o ** 2) * self.get_chi_0(12.3984193E-10/lam)/ np.sin(2 * Bragg_angle_darwin))) / np.sqrt(1 - gamma_o ** 2)).real
        ####print(' Here I think I have maximum!  ',Max_Darvin_curve)
        
        return Central_incid_angle+Max_Darvin_curve, (Bragg_angle, Max_Darvin_curve, Polarizability, Polarizability_minus, (2*np.pi*np.sqrt(self.lattice_indices[0]**2+self.lattice_indices[1]**2+self.lattice_indices[2]**2))/(self.unit_cell_dimension), assymetry_ratio, Lattice_plane_trace_and_surface,(struct_factor*struct_factor_minus))#scatter_fac)

    
    def crystal_reflection(self, incident_angle, Lattice_planes_miscut, lam):
        #l_p_m=self.Lattice_planes_miscut
        #miscut_sign=np.sign(Lattice_planes_miscut)
        
        #Lattice_planes_miscut=np.abs(Lattice_planes_miscut)
        
        if isinstance(incident_angle, np.ndarray):
            #print('INCID ANGLE was::: ',incident_angle)
            #incident_angle[(incident_angle>np.pi/2)]-=np.pi/2
            #incident_angle[(incident_angle<-np.pi/2)]+=np.pi/2
            
            #print('do it!  ',np.size(incident_angle[(incident_angle<0)]))#[(incident_angle<0)]
            incident_angle[(incident_angle<0)]*=-1#+=np.pi/2#*=-1
            #print('INCID ANGLE is now::: ',incident_angle)
        
        Central_incid_angle,(Bragg_angle, Max_Darvin_curve, Polarizability, Polarizability_minus, h_vector, assymetry_ratio, Lattice_plane_trace_and_surface,struct_factor_sqear)=self.core_angle(lam,Lattice_planes_miscut)
        Bragg_angle_deviation=-Bragg_angle+incident_angle+Lattice_planes_miscut
        incident_angle_deviation=incident_angle-Central_incid_angle#incident_angle+2*lattice_miscut_angle
        if isinstance(Central_incid_angle,np.ndarray):
            print('   Central_incid_angle  ::  ',Central_incid_angle[3])
        
        R_e=2.8179403262e-09
        cell_dim=0.000543095

        ###outcoming_angle=Central_incid_angle+2*Lattice_planes_miscut-(np.arccos(np.cos(Central_incid_angle+incident_angle_deviation)-np.cos(Central_incid_angle)+np.cos(Central_incid_angle+2*Lattice_planes_miscut)))
        
        gamma_o = np.sin(Bragg_angle - Lattice_planes_miscut)
        gamma_h = np.sin(- Bragg_angle - Lattice_planes_miscut)
        gamma = gamma_h / gamma_o
        
        
        ##struct_factor=self.get_Fc(12.3984193E-10/lam)
        bragg_shift_oc=( -gamma_o + np.sqrt(gamma_o ** 2 - (gamma_o- gamma_h) * np.sqrt(1 - gamma_o ** 2) * self.get_chi_0(12.3984193E-10/lam)/ np.sin(2 * Bragg_angle))) / np.sqrt(1 - gamma_o ** 2)
        
        bragg_shift_os= -1* (1-gamma)*self.get_chi_0(12.3984193E-10/lam)/ (2*np.sin( 2 * Bragg_angle))
        
        delta_os= R_e * (lam*1e6)**2 / (np.pi * (cell_dim**3) * np.sin(2 * Bragg_angle)) * np.sqrt(struct_factor_sqear) * np.sqrt(np.abs(gamma))
        delta_oc = delta_os * (Bragg_angle - Lattice_planes_miscut) / (bragg_shift_oc + Bragg_angle - Lattice_planes_miscut)

        #Deviation_parameter_complex=(Bragg_angle_deviation - bragg_shift_os) / delta_os   
        Deviation_parameter_complex=(Bragg_angle_deviation - bragg_shift_oc) / delta_oc   
            
        #############Amplitude_ratio_=-1*np.sqrt(Polarizability*Polarizability_minus/np.abs(assymetry_ratio))*(Deviation_parameter_complex-np.sign(Deviation_parameter_complex.real)*np.sqrt(Deviation_parameter_complex*Deviation_parameter_complex-1))/Polarizability_minus
        Amplitude_ratio_=np.abs(Polarizability/Polarizability_minus)*abs(Deviation_parameter_complex-np.sign(Deviation_parameter_complex.real)*np.sqrt(Deviation_parameter_complex*Deviation_parameter_complex-1))**2
        
        #Amplitude_ratio_[((np.cos(Central_incid_angle-incident_angle_deviation)-np.cos(Central_incid_angle)+np.cos(Central_incid_angle+2*Lattice_planes_miscut))>1)]=0
        #Amplitude_ratio_[((np.cos(Central_incid_angle-incident_angle_deviation)-np.cos(Central_incid_angle)+np.cos(Central_incid_angle+2*Lattice_planes_miscut))<-1)]=0
        #if miscut_sign>0:
       
        return [Amplitude_ratio_, 0, h_vector], Central_incid_angle+2*self.Lattice_planes_miscut, 0 # np.mean(i_a)+2*l_p_m#np.mean(Central_incid_angle+2*Lattice_planes_miscut)
        
        #return [Amplitude_ratio_, outcoming_angle, K_vector_range], Centr_angle+2*self.Lattice_planes_miscut, (-Bragg_angle_deviation+bragg_shift_oc+delta_oc+1j*(Bragg_angle_deviation-bragg_shift_oc+delta_oc)) # np.mean(i_a)+2*l_p_m#np.mean(Central_incid_angle+2*Lattice_planes_miscut)
        
        ############################################################################# 
    
