#######################################################################################################
#here the main classes for creating geometry and ray tracing are described
####################################################################################################
import numbers
import numpy as np
import copy
import Ray_plane_interaction_ev as RPL
import matplotlib.pyplot as plt
from Crystals import CrystalData, Slits

from numpy.random import default_rng

class Geometry: 
    ##this class contains the data about geometry of the system, namely: positions of every optical element (for now - crystal) in 3D space, 
    ##and normals for crystal surface plane and crystal lattice plane.
    def __init__(self, cryst, object_coords, object_normals, object_latice_normals, object_vertical_or, cr_frame_1, cr_frame_2, sys_plane_x, sys_plane_y):
        self.object_coords = object_coords
        self.object_normals = object_normals
        self.object_latice_normals = object_latice_normals
        self.object_vertical_or = object_vertical_or # if it has 'true', then this crystal reflects rays in vertical plane (like for example, monochromator crystals do), if 'false'- in horisontalplane  

        self.cryst = cryst# a list of crystal objects, in the sequence as they go from source to detector. If the object is jaus a (perpendicular to the system axis) plane, then the object is int type: '0'
        self.frame = [cr_frame_1, cr_frame_2]
        self.sys_plane_x = sys_plane_x
        self.sys_plane_y = sys_plane_y

class LightRay:  
    ##generator class for the arrays of optical rays
    def __init__(self,
                #max_iter, # int, how many times this generator will return the sets of the light beams (max. iterations)
                buffer, # int, size of the matrix, which represents the light source
                small_buffer, # float, rays per point
                lam, # float, the 'central' wavelength of the generated light
                lam_step, # float, wavelength step
                iter,
                rand_seed=1):  # int, random seed for repeatability

        self.buffer = buffer
        print('buffer buffer: :: :::: ', buffer)
        self.small_buffer = small_buffer
        self.lam = lam 

        self.delta_lam = lam*lam_step #the width of the wavelength distribution of the generated light
        self.iter = iter
        self.rand_seed = rand_seed # some seed number - is then multiplied on iteration number and thereby changed
        self.hor_pos_=np.zeros(self.buffer)#*0
        self.vert_pos_=np.zeros(self.buffer)#*0
        source_size2=0.14#0.04
        self.source_size2=source_size2
        self.source_size1=source_size2
        
        if self.buffer>1 :
            source_vert_bins=10#[300, 10]
            source_size1=source_size2*(self.buffer/(source_vert_bins*source_vert_bins))#0.165#*5
            self.source_size1=source_size1
            print('Source size:: ',source_size1)
            bb=np.linspace(-source_size2/2,source_size2/2,int(self.buffer/(source_vert_bins*source_size1/source_size2)))
            aa=np.linspace(-source_size1/2,source_size1/2,int(self.buffer/source_vert_bins))

            for i in range(0,int(source_vert_bins*source_size1/source_size2)):
                self.hor_pos_[i*int(self.buffer/(source_vert_bins*source_size1/source_size2)):i*int(self.buffer/(source_vert_bins*source_size1/source_size2))+int(self.buffer/(source_vert_bins*source_size1/source_size2))]=bb
            jj=-1
            for i in range(0,self.buffer):
                if(i%source_vert_bins==0):
                    jj+=1
                self.vert_pos_[i]=aa[jj]
        
        

    def generate_beams(self):

        ##'Random' objects generation
        # just for an example, I initialize 2 random number 
        # generators with individual seeds; 
        
        rng2 = default_rng(self.iter + self.rand_seed + 10)
        rng3 = default_rng(self.iter + self.rand_seed)
        angles_number = np.int32(100)# np.int32(np.round(1e-8 / self.small_buffer**2))

        print('Angles number:: ', angles_number)

        ### generate 1d arrays (angles_number * self.buffer) to exploit 
        # numpy vectorization

        hor_angle = np.zeros(angles_number*self.buffer, dtype=np.float32) 
        roll_angle = np.zeros(angles_number*self.buffer, dtype=np.float32) 

        #hor_angle = np.zeros(angles_number*self.buffer, dtype=np.float32) 
        roll_angle_sin = np.zeros(angles_number*self.buffer, dtype=np.float32)
        roll_angle_cos = np.zeros(angles_number*self.buffer, dtype=np.float32)

        lambdas = np.zeros(angles_number*self.buffer, dtype=np.float32) 
        init_coord = np.zeros((angles_number*self.buffer, 3), dtype=np.float32)

        for i in range(self.buffer):

            h_p = self.hor_pos_[i] #horizontal position of the rays origin point
            v_p = self.vert_pos_[i] #vertical position of the rays origin point
            
            #proportion=0.000687/0.00006175
            
            roll_angle[(i*angles_number):((i+1)*angles_number)] = rng2.uniform(0, 2*np.pi, angles_number)#rng2.uniform(0, 0.000687, angles_number)#*self.max_iter)#rng2.uniform(-LightRay.source_divergence[1], LightRay.source_divergence[1], angles_number)#rng2.uniform(0, np.pi, angles_number)#(rng2.normal(0, np.pi, angles_number))
            #hor_angle[(i*angles_number):((i+1)*angles_number)] = rng2.triangular(0, 0.00006175, 0.00006175, angles_number)#rng2.uniform(0, 0.00006175, angles_number)
        
            hor_angle[(i*angles_number):((i+1)*angles_number)] = rng2.triangular(0, 0.00006175, 0.00006175, angles_number)##0, 0.000006175, 0.000006175#rng2.uniform(0, 0.00006175, angles_number)
            
            lambdas[(i*angles_number):((i+1)*angles_number)] = np.ones(angles_number, dtype=np.float32) * self.lam - rng3.uniform(-1, 1, angles_number)*self.delta_lam 

            # here 0 coordinate is horizontal position of the rays origin point and 
            # 2 coordinate is vertical position of the rays origin point
           
            init_coord[(i*angles_number):((i+1)*angles_number), 1] = h_p
            init_coord[(i*angles_number):((i+1)*angles_number), 0] = v_p

        beams = RayVisualizer([0,0,0], # system_trace_begin
                            init_coord, # init_coords
                            [hor_angle, roll_angle], # init_angles
                            lambdas,
                            [self.source_size1,self.source_size2])

        return beams
        

class RayVisualizer: 
    
    def __init__(self, 
                system_trace_begin, 
                init_coords, 
                init_angles, 
                lambdas,
                source_sizes): 
        
        self.n_rays = init_coords.shape[0]
        self.can = 0
        
        self.system_tilt_vert = False
        self.system_tilt_hor = False

        ### self.sys_x, self.sys_y, self.sys_z is wrapped as self.sys
        self.sys = np.array([0, 0, 1], dtype=np.float32)
        self.coef = [1, 1]
        self.moving_coord_sys = np.eye(3, dtype=np.float32)
        
        self.system_pos = np.zeros(3, dtype=np.float32)
        self.system_trace = system_trace_begin
        self.crystal_normals = system_trace_begin[:]
        self.crystal_edges = []
        
        ### self.N_x, self.N_y, self.N_z and self.N_a_x, self.N_a_y, self.N_a_z 
        # are wrapped as self.N and self.N_a
        self.N = self.sys.copy()
        self.N_a = self.sys.copy()
        
        self.crystal_inline = np.array([0, 1, 0], dtype=np.float32)#self.N_y
        
        self.crystal_frame = [self.crystal_inline, self.moving_coord_sys[0]]
        
        self.lambdas = lambdas[:]
        self.wavelength = lambdas[0]
        
        ############################################
        ### self.rays_pos_x, self.rays_pos_y, self.rays_pos_z are wrapped 
        # as self.rays_pos
        self.rays_pos = init_coords.copy()


        
        self.amplitudes = np.ones(self.n_rays, dtype=np.float32)
        
        #Here I cut all the points of the extended light source, which don't fit into an ellipse - after this the are of the source should have elliptic shape
        '''
        for n_ray in range(len(self.amplitudes)):
            if ((2*self.rays_pos[n_ray,1]/source_sizes[1] )**2 + (2*self.rays_pos[n_ray,0]/source_sizes[0])**2 > 1):
                #print('AAATT!')
                self.amplitudes[n_ray]=0
        '''
        self.rays_directions = RPL.get_cartesian_direction(init_angles[0], init_angles[1])
        #self.rays_directions = RPL.get_cartesian_direction_rectang(init_angles[0], init_angles[1])
        '''
        #self.rays_directions = init_angles[0], init_angles[1]
        '''
        self.cr_edge_x = []
        self.cr_edge_y = []
        self.cr_edge_z = []
        
        self.rays_traces_x = []
        self.rays_traces_y = []
        self.rays_traces_z = []

        self.rays_traces_x.extend(self.rays_pos[:, 0])#[:] )
        self.rays_traces_y.extend(self.rays_pos[:, 1])#[:] )
        self.rays_traces_z.extend(self.rays_pos[:, 2])#[:] )
        
        self.rays_traces_lam = []
        
        energies = 12.3984193E-10/self.lambdas
        self.rays_traces_lam.extend(energies)

        self.size_of_cr_plane=1#*100

        ### instead of 
        # self.rays_plane_x=np.zeros((8,np.shape(self.rays_directions)[0]), dtype=np.float32)
        # I would suggest to do
        # np.zeros((8,self.rays_directions.shape[0]), dtype=np.float32)
        # In the first case you first get 1D array, and then get it's hape, in the second case, you get the shape directly
        # I have also introduced another variable self.n_rays not to get shape all the time
        self.rays_plane_x=np.zeros((11,self.n_rays), dtype=np.float32)
        self.rays_plane_y=np.zeros((11,self.n_rays), dtype=np.float32)
        self.lambdas_plane=np.zeros((11,self.n_rays), dtype=np.float32)
        self.amplitudes_plane = np.zeros((11,self.n_rays), dtype=np.float32)

    
    def adjust_surface_normals(self, crystal_angles):

        ### here I shortened and simplified the code by making operations inline 
        # instead of introducing extra variables. Plus, I wrapped N_x, N_y, N_z and
        # N_a_x, N_a_y, N_a_z as N and N_a, respectively

        pitch, roll, yaw = crystal_angles

        self.N = np.matmul(RPL.rotate_around_axis(self.crystal_frame[0], roll), normal=self.N)##np.matmul(np.matmul(np.matmul(yaw_matr,pitch_matr),roll_matr), [self.sys_x, self.sys_y, self.sys_z])### this will be gone
        self.N_a= np.matmul(RPL.rotate_around_axis(self.crystal_frame[0], roll), self.N_a)

        self.N  = np.matmul(RPL.rotate_around_axis(self.crystal_frame[1], pitch), self.N)##np.matmul(np.matmul(np.matmul(yaw_matr,pitch_matr),roll_matr), [self.sys_x, self.sys_y, self.sys_z])### this will be gone
        self.N_a = np.matmul(RPL.rotate_around_axis(self.crystal_frame[1], pitch), self.N_a)


    def generate_surface_normals(self, crystal_angles, misscut, Vertical_or):

        pitch, roll, yaw = crystal_angles

        ### Note, I wrapped N_x, N_y, N_z and N_a_x, N_a_y, N_a_z as N and N_a, respectively
        
        if Vertical_or:
            inline_vert_cryst = np.matmul(RPL.rotate_around_axis(self.moving_coord_sys[0], pitch), self.moving_coord_sys[2])
            

            axis_vert_cryst = np.matmul(RPL.rotate_around_axis(self.moving_coord_sys[0], pitch+np.pi/2), self.moving_coord_sys[2])
            axis_vert_cryst = np.matmul(RPL.rotate_around_axis(inline_vert_cryst, roll), axis_vert_cryst)

            axis_crystal_planes_vert_cryst = np.matmul(RPL.rotate_around_axis(self.moving_coord_sys[0], misscut+pitch+np.pi/2), self.moving_coord_sys[2])
            axis_crystal_planes_vert_cryst = np.matmul(RPL.rotate_around_axis(inline_vert_cryst, roll), axis_crystal_planes_vert_cryst)
            
            #inline_vert_cryst = np.matmul(RPL.rotate_around_axis(self.moving_coord_sys[1], yaw), inline_vert_cryst) #???
        
            
            axis_crystal_planes_vert_cryst = np.matmul(RPL.rotate_around_axis(axis_vert_cryst, yaw), axis_crystal_planes_vert_cryst)

            self.crystal_inline = inline_vert_cryst / np.linalg.norm(inline_vert_cryst)
            self.crystal_frame = [self.crystal_inline, self.moving_coord_sys[0]]
            self.N = -axis_vert_cryst
            self.N_a = -axis_crystal_planes_vert_cryst

            self.coef[0] = (-1) ** int(self.system_tilt_vert)
            self.system_tilt_vert = not self.system_tilt_vert
                
        else:

            inline_hor_cryst = np.matmul(RPL.rotate_around_axis(self.moving_coord_sys[1], pitch), self.moving_coord_sys[2])
            #inline_hor_cryst = np.matmul(RPL.rotate_around_axis(self.moving_coord_sys[0], yaw), inline_hor_cryst) #???

            axis_hor_cryst = np.matmul(RPL.rotate_around_axis(self.moving_coord_sys[1], pitch + np.pi / 2), self.moving_coord_sys[2])# instead - will be this!
            axis_hor_cryst = np.matmul(RPL.rotate_around_axis(inline_hor_cryst, roll), axis_hor_cryst)
        
            axis_crystal_planes_hor_cryst = np.matmul(RPL.rotate_around_axis(self.moving_coord_sys[1], misscut+pitch+np.pi/2),self.moving_coord_sys[2])
            axis_crystal_planes_hor_cryst = np.matmul(RPL.rotate_around_axis(inline_hor_cryst, roll), axis_crystal_planes_hor_cryst)

            axis_crystal_planes_hor_cryst = np.matmul(RPL.rotate_around_axis(axis_hor_cryst, yaw), axis_crystal_planes_hor_cryst)

            self.crystal_inline = inline_hor_cryst / np.linalg.norm(inline_hor_cryst)
            self.crystal_frame = [self.crystal_inline, self.moving_coord_sys[1]]
            self.N = -axis_hor_cryst#_inv
            self.N_a = -axis_crystal_planes_hor_cryst
            
            self.coef[1] = (-1) ** int(self.system_tilt_hor)
            self.system_tilt_hor = not self.system_tilt_hor    
        
        
    def define_crystal_corners(self,moving_coord_sys_vector,size_of_cr_plane, cr_proportion):

        crystal_corner1 = size_of_cr_plane * moving_coord_sys_vector + size_of_cr_plane * cr_proportion * self.crystal_inline#+self.system_pos
        crystal_corner2 = size_of_cr_plane * moving_coord_sys_vector - size_of_cr_plane * cr_proportion * self.crystal_inline
        crystal_corner3 = -size_of_cr_plane * moving_coord_sys_vector - size_of_cr_plane * cr_proportion * self.crystal_inline
        crystal_corner4 = -size_of_cr_plane * moving_coord_sys_vector + size_of_cr_plane * cr_proportion * self.crystal_inline

        return crystal_corner1, crystal_corner2, crystal_corner3, crystal_corner4
        
    
    def new_moving_axis(self, new_sys_direction):
        ### self.sys_x, self.sys_y, self.sys_z is wrapped as self.sys
        new_moving_x_axis = np.cross(new_sys_direction, self.sys)
        new_moving_x_axis = new_moving_x_axis / np.linalg.norm(new_moving_x_axis)
        new_moving_y_axis = np.cross(new_moving_x_axis, new_sys_direction)
        new_moving_y_axis = new_moving_y_axis / np.linalg.norm(new_moving_y_axis)
        return new_moving_x_axis, new_moving_y_axis

    def make_plane_frame(self, system_pos, Vertical_or, size_of_cr_plane, cr_proportion, new_sys_direction):

        if Vertical_or: 
            crystal_corner1, crystal_corner2, crystal_corner3, crystal_corner4 = self.define_crystal_corners(self.moving_coord_sys[0], size_of_cr_plane, cr_proportion)
            new_moving_x_axis, new_moving_y_axis = self.new_moving_axis(new_sys_direction)
            self.moving_coord_sys = np.array([self.coef[0] * new_moving_x_axis, \
                                            self.coef[0] * self.coef[1] * new_moving_y_axis, \
                                            new_sys_direction], \
                                            dtype=np.float32)           

        else:
            crystal_corner1, crystal_corner2, crystal_corner3, crystal_corner4 = self.define_crystal_corners(self.moving_coord_sys[1], size_of_cr_plane, cr_proportion)
            new_moving_y_axis, new_moving_x_axis = self.new_moving_axis(new_sys_direction)
            self.moving_coord_sys = np.array([self.coef[0] * self.coef[1] * new_moving_x_axis, \
                                            self.coef[1] * new_moving_y_axis, \
                                            new_sys_direction], \
                                            dtype=np.float32)
        
        self.cr_edge_x = [crystal_corner1[0], crystal_corner2[0], crystal_corner3[0], crystal_corner4[0]]
        self.cr_edge_y = [crystal_corner1[1], crystal_corner2[1], crystal_corner3[1], crystal_corner4[1]]#
        self.cr_edge_z = [crystal_corner1[2], crystal_corner2[2], crystal_corner3[2], crystal_corner4[2]]#
        ##print('Sys_pos:  ', system_pos)
        self.cr_edge_x += system_pos[0]
        self.cr_edge_y += system_pos[1]
        self.cr_edge_z += system_pos[2]

        self.crystal_edges.extend([self.cr_edge_x[0], self.cr_edge_y[0], self.cr_edge_z[0]])
        self.crystal_edges.extend([self.cr_edge_x[1], self.cr_edge_y[1], self.cr_edge_z[1]])
        self.crystal_edges.extend([self.cr_edge_x[2], self.cr_edge_y[2], self.cr_edge_z[2]])
        self.crystal_edges.extend([self.cr_edge_x[3], self.cr_edge_y[3], self.cr_edge_z[3]])
        #print('new_sys_direction', new_sys_direction     )
        ### self.sys_x, self.sys_y, self.sys_z is wrapped as self.sys
        self.sys = new_sys_direction#[0]   
        #print('new_sys_direction', self.sys_x, self.sys_y, self.sys_z )


    def reflect(self,central_wavelength, Vertical_or,crystal,crystal_angles, syst_pos, size_of_cr_plane, cr_proportion):

        #this function is only used during geometry generation
        #reflect - only turning (reflecting) the system axis
        self.generate_surface_normals(crystal_angles, crystal.Lattice_planes_miscut, Vertical_or)
        
        incid_angle = RPL.get_angle_ray_plane( np.array(self.moving_coord_sys[2]), self.N)
        crystal_plane_angle = RPL.get_angle_ray_plane( np.array(self.moving_coord_sys[2]), self.N_a)
        
        [Amplitude_ratio_, outcoming_angle, K_vector_range], angle_out_main, elem2 = crystal.crystal_reflection(incid_angle, (-incid_angle+crystal_plane_angle), central_wavelength)#self.lambdas[0]###self.lambdas[0]##(incid_angle-crystal_plane_angle)##(crystal.Lattice_planes_miscut)#(incid_angle-crystal_plane_angle), central_wavelength, central_wavelength)
        out_angle = angle_out_main - outcoming_angle
        
        #new_sys_direction = RPL.get_refl_vector(np.array([self.sys_x, self.sys_y, self.sys_z]),np.array([N_x, N_y, N_z]), out_angle)
        new_sys_direction = RPL.get_refl_vector(np.array(self.moving_coord_sys[2]), self.N, out_angle)
        new_sys_direction = new_sys_direction/np.linalg.norm(new_sys_direction)
        
        self.make_plane_frame(syst_pos, Vertical_or, size_of_cr_plane, cr_proportion, new_sys_direction)
        ### Note, I wrapped N_x, N_y, N_z and N_a_x, N_a_y, N_a_z as N and N_a, respectively
        return self.N.copy(), self.N_a.copy(), self.crystal_frame[0].copy(), self.crystal_frame[1].copy(),out_angle

    def Build_geometry(self, central_wavelength,all_crystals, distances, cross_distances, crystal_angles, crystal_orientation, size_of_cr_plane=10, cr_proportion=5):
 
        self.sys_pos = np.zeros((len(distances)+1,3), dtype=np.float32)
        self.sys_plane_x = np.zeros(len(distances)+1, dtype=np.float32)
        self.sys_plane_y = np.zeros(len(distances)+1, dtype=np.float32)
        self.cr_surface = np.zeros((len(all_crystals),3), dtype=np.float32)
        self.cr_lattice = np.zeros((len(all_crystals),3), dtype=np.float32)
        self.cr_frame_1 = np.zeros((len(all_crystals),3), dtype=np.float32)
        self.cr_frame_2 = np.zeros((len(all_crystals),3), dtype=np.float32)
        for dd in range(len(all_crystals)):
            self.sys_pos[dd+1] = self.sys_pos[dd]+distances[dd] * self.sys
            #print('system trace! ! !  system trace  ! :: :: ', self.sys_pos)
            #print('system direction! ! !:: :: ', np.array([self.sys_x, self.sys_y, self.sys_z]))
            if isinstance(all_crystals[dd], CrystalData):
                self.cr_surface[dd][:], self.cr_lattice[dd][:], self.cr_frame_1[dd][:], self.cr_frame_2[dd][:],out_angle = self.reflect(central_wavelength, crystal_orientation[dd], all_crystals[dd], crystal_angles[dd], self.sys_pos[dd+1], size_of_cr_plane,cr_proportion)
                if cross_distances[dd+1] > 0:
                    distances[dd+1]=cross_distances[dd+1]/np.sin(out_angle)
            else:
                self.cr_surface[dd][:] = -self.sys
                self.cr_lattice[dd][:] = -self.sys
                self.cr_frame_1[dd][:] = self.moving_coord_sys[0].copy()
                self.cr_frame_2[dd][:] = self.moving_coord_sys[1].copy()
            sys_in_plane = RPL.get_project_ray_plane(self.sys_pos[dd+1],np.array(self.cr_surface[dd][:]))
            self.sys_plane_x[dd]=np.transpose(np.dot(sys_in_plane,self.cr_frame_1[dd][:]))
            self.sys_plane_y[dd]=np.transpose(np.dot(sys_in_plane,self.cr_frame_2[dd][:]))
            print('system plane: :: :  ', self.sys_plane_x[dd],' and ',self.sys_plane_y[dd])  
            #self.system_trace.extend(self.system_pos)
        #self.sys_pos[-1] = self.sys_pos[-2] + distances[-1]*np.array([self.sys_x, self.sys_y, self.sys_z])
        self.sys = np.array([0, 0, 1], dtype=np.float32)
        geom_ = Geometry(all_crystals, self.sys_pos, self.cr_surface, self.cr_lattice, crystal_orientation, self.cr_frame_1, self.cr_frame_2, self.sys_plane_x, self.sys_plane_y)    
        #return geom_, np.transpose(np.reshape(np.array(self.crystal_edges),(int(len(self.crystal_edges)/3),3)))##[lin_x_b,lin_y_b,lin_z_b]
        
        ### note, that I changed output to return not only crystal edges but also system_trace and crystal normals
        return geom_, \
            np.transpose(np.reshape(np.array(self.crystal_edges),(len(self.crystal_edges) // 3, 3))),\
            self.system_trace,\
            self.crystal_normals    


    def Intensity_control(self): 

        ## Delete all the rays, that have become to weak after reflections..
        self.lambdas = self.lambdas[self.amplitudes>0.05]

        ### self.rays_pos_x, self.rays_pos_y, self.rays_pos_z are wrapped 
        # as self.rays_pos
        self.rays_pos = self.rays_pos[self.amplitudes>0.05, :]
        
        self.rays_directions = self.rays_directions[self.amplitudes>0.05, :]
        self.amplitudes = self.amplitudes[self.amplitudes>0.05]
    
    def Slits_cut(self, Slits, can, len_l):
        
        #[rays_pos_x, rays_pos_y, rays_pos_z] = np.transpose(self.rays_pos)
        rays_pos_x=self.rays_plane_x[can][:len_l]
        rays_pos_y=self.rays_plane_y[can][:len_l]
        
        self.lambdas = self.lambdas[rays_pos_x<Slits.hor_size2 ]
        self.rays_directions = self.rays_directions[rays_pos_x<Slits.hor_size2,: ]
        self.amplitudes = self.amplitudes[rays_pos_x<Slits.hor_size2 ]
        self.rays_pos = self.rays_pos[rays_pos_x<Slits.hor_size2 ,:]
        copy_pos = copy.deepcopy(rays_pos_x)
        rays_pos_x=rays_pos_x[copy_pos<Slits.hor_size2 ]
        rays_pos_y=rays_pos_y[copy_pos<Slits.hor_size2 ]
        #[rays_pos_x, rays_pos_y, rays_pos_z] = np.transpose(self.rays_pos)

        self.lambdas = self.lambdas[rays_pos_x>-Slits.hor_size1 ]
        self.rays_directions = self.rays_directions[rays_pos_x>-Slits.hor_size1,: ]
        self.amplitudes = self.amplitudes[rays_pos_x>-Slits.hor_size1 ]
        self.rays_pos = self.rays_pos[rays_pos_x>-Slits.hor_size1 ,:]
        copy_pos = copy.deepcopy(rays_pos_x)
        rays_pos_x=rays_pos_x[copy_pos>-Slits.hor_size1 ]
        rays_pos_y=rays_pos_y[copy_pos>-Slits.hor_size1 ]
        #[rays_pos_x, rays_pos_y, rays_pos_z] = np.transpose(self.rays_pos)

        self.lambdas = self.lambdas[rays_pos_y>-Slits.vert_size1 ]
        self.rays_directions = self.rays_directions[rays_pos_y>-Slits.vert_size1,: ]
        self.amplitudes = self.amplitudes[rays_pos_y>-Slits.vert_size1 ]
        self.rays_pos = self.rays_pos[rays_pos_y>-Slits.vert_size1 ,:]
        copy_pos = copy.deepcopy(rays_pos_y)
        rays_pos_x=rays_pos_x[copy_pos>-Slits.vert_size1 ]
        rays_pos_y=rays_pos_y[copy_pos>-Slits.vert_size1 ]
        #[rays_pos_x, rays_pos_y, rays_pos_z] = np.transpose(self.rays_pos)

        self.lambdas = self.lambdas[rays_pos_y<Slits.vert_size2 ]
        self.rays_directions = self.rays_directions[rays_pos_y<Slits.vert_size2,: ]
        self.amplitudes = self.amplitudes[rays_pos_y<Slits.vert_size2 ]
        self.rays_pos = self.rays_pos[rays_pos_y<Slits.vert_size2 ,:]
        copy_pos = copy.deepcopy(rays_pos_y)
        rays_pos_x=rays_pos_x[copy_pos<Slits.vert_size2]
        rays_pos_y=rays_pos_y[copy_pos<Slits.vert_size2]

    def rays_directions_change(self,crystal,object_normals, object_latice_normals):

        if isinstance(crystal, CrystalData):

            rays_incid_angle = RPL.get_angle_ray_plane(self.rays_directions, np.array(object_normals))
            rays_crystal_plane_angle = RPL.get_angle_ray_plane(self.rays_directions, np.array(object_latice_normals))

            [Amplitude_ratio_, out_angle_ray, K_vector_range], angle_out_main, elem2 = crystal.crystal_reflection(rays_incid_angle, (-rays_incid_angle+rays_crystal_plane_angle), self.lambdas)##(crystal.Lattice_planes_miscut)

            self.amplitudes *= Amplitude_ratio_
            
            out_angle_ray=-out_angle_ray+angle_out_main

            self.rays_directions = RPL.get_refl_vector(self.rays_directions,np.array(object_normals), out_angle_ray)#out_angle_rays[ray])   
            self.rays_directions = self.rays_directions / np.linalg.norm(self.rays_directions)
        
    ##########################
    #Olny this function ( propagation )- might be replaced by GPU code with similar performance - that would make the whole program faster..
    #########################
    def propagation(self, Geom, screen_length):

        for itr in range(0,len(Geom.object_normals)):


            self.Intensity_control()

            self.rays_pos = RPL.get_ray_plane_crossing(self.rays_pos, self.rays_directions, Geom.object_coords[itr+1], Geom.object_normals[itr])## np.array([0,0,20000]), np.array([0, 0, 1])
            ###
            ##print( 'rays positions : . .:  ', np.transpose(self.rays_pos))
            [rays_pos_x, rays_pos_y, rays_pos_z] = np.transpose(self.rays_pos)

            rays_in_plane = RPL.get_project_ray_plane(self.rays_pos,np.array(Geom.object_normals[itr]))
            
            normilizer=[(np.linalg.norm(Geom.frame[0][itr])**2), (np.linalg.norm(Geom.frame[1][itr])**2)]
            #self.rays_plane_x[self.can][:len(self.lambdas)]=np.transpose(np.dot(rays_in_plane[:, np.newaxis],self.crystal_frame[0]))/normilizer[0]
            #self.rays_plane_y[self.can][:len(self.lambdas)]=np.transpose(np.dot(rays_in_plane[:, np.newaxis],self.crystal_frame[1]))/normilizer[1]
            self.rays_plane_x[self.can][:len(self.lambdas)]=np.transpose(np.dot(rays_in_plane[:, np.newaxis],Geom.frame[0][itr]))
            self.rays_plane_y[self.can][:len(self.lambdas)]=np.transpose(np.dot(rays_in_plane[:, np.newaxis],Geom.frame[1][itr]))
                     
            self.lambdas_plane[self.can][:len(self.lambdas)]= 12.3984193E-10 / self.lambdas
            self.amplitudes_plane[self.can][:len(self.lambdas)]=self.amplitudes
            ###
            if isinstance(Geom.cryst[itr], Slits):
               self.Slits_cut(Geom.cryst[itr], self.can, len(self.lambdas))

            self.rays_directions_change(Geom.cryst[itr], Geom.object_normals[itr], Geom.object_latice_normals[itr])
            
            self.Intensity_control()

            self.rays_traces_x.extend(self.rays_pos[:, 0])#[:] )
            self.rays_traces_y.extend(self.rays_pos[:, 1])#[:] )
            self.rays_traces_z.extend(self.rays_pos[:, 2])#[:] )

            energies = 12.3984193E-10/self.lambdas
            self.rays_traces_lam.extend(energies)
            
            self.crystal_sizes=[screen_length[0]/1, screen_length[1]/1]##screen_length##2/240

            self.can+=1
