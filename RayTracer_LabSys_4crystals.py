###############################################################
##  Main Body of the program
############################################################
#%%
# external imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import time
import json
import os
import datetime
import copy
from scipy import stats

from matplotlib.colorbar import Colorbar
from tif_image_analysis import open_tif_image,show_image_contours
from matplotlib.colors import LogNorm
from mpl_toolkits import mplot3d
from math import isnan, floor
from Rocking_Curve import RockingCurve

# internal imports
from Crystals import CrystalData, Slits
#from Open_trajectories import show_trajectories, show_contours
#from Lab_system_ev import Propagation as Propagation_ev
from Ray_Visualizer_ev import LightRay as LightRay_ev
# fitting the gaussian
from astropy import modeling
fitter = modeling.fitting.LevMarLSQFitter()
model = modeling.models.Gaussian1D()#+modeling.models.Gaussian1D()#modeling.models.Linear1D()

#%%
# parse files with set-up configuration
os.chdir('C:\\Users\\gx4419\\Documents\\Python_Scripts\\backup\\FromGH')##os.chdir('C:\\Users\\gx4419\\Documents\\Python_Scripts\\Evelinas_histograms_created')

with open('Parameters_KARA.txt') as f:#('/ccpi/data/RayTracing_BraggMagnifier/Parameters_KARA.txt') 
    data_with_params = f.read()
'''
with open('Parameters_P23.txt') as f:
    data_with_params = f.read()
'''
js = json.loads(data_with_params)
print(js)

geom_chromatic = 7
geometries_section=3#3# which optical elements will be in the graphic window? (0 - beginning of the system)

Graphics = js["Graphics"]#True
Beam_path = js["Beam_path"]#True

Miscut_220_small = js["Miscut_220_small"]#0.01640609
Miscut_220 = js["Miscut_220"]#0.1033235#for our real setup###0.1033235,
Miscut_111 = js["Miscut_111"]#0.11992157# for the 15keV setup, 20 magnification 111 orientation

energy = js["energy"]# energy = 29 # in keV    #29 #30.5
lam = 12.3984193E-10 / energy #in Angström
k = np.pi*2/lam

#Slits of the beamline
Primary_sl=Slits([100000,100000,1000000,1000000])#Slits([0.52,0.73,0.3,1.43])#Primary_sl= Slits([0.3,0.3,0.3,0.3])#Primary_sl=Slits([0.0,0.13,1000000,1000000])#Primary_sl= ([100000,100000,1000000,1000000])#

object_sl= Slits([1000000,1000000,10000000,10000000])

#############detector_sl= Slits([-150.5,178.5,150,-122])#Slits([-399,427,150,-122]) #{hor,hor,vert,vert}
####detector_sl= Slits([380,-352,380,-352])
#detector_sl= Slits([376,-348,380,-352])
#detector_sl= Slits([133,-105,128,-100])
#detector_sl= Slits([228,0,228,0])#Slits([128,-100,128,-100])
#detector_sl= Slits([123,-95,145,-117])# for roll mistune 0.005
#detector_sl= Slits([128,-100,138,-110])## for roll mistune 0.008
detector_sl= Slits([128,-100,137,-109])## for vert cr mistune 0.0001##vert,hor(left->right)
# Crystal Lattices
Lattice1=[1,1,1]
Lattice2=[2,2,0]
missc1=0.03
missc2=0.165
'''
Lattice1=[1,0,0]
Lattice2=[2,2,0]
missc1=0.06
missc2=0.165
'''
# Create crystal objects

Si_111 = CrystalData(0, crystal='Si', lattice_indices=Lattice1)
#Si_111 = CrystalData(0, crystal='Si', lattice_indices=[2,2,0])
Si_220_miscut = CrystalData(Miscut_220, crystal='Si', lattice_indices=[2,2,0])
Si_minus220_miscut = CrystalData(-Miscut_220, crystal='Si', lattice_indices=[2,2,0])
Si_111_miscut = CrystalData(Miscut_111, crystal='Si', lattice_indices=[1,1,1])
Si_220_small_miscut = CrystalData(Miscut_220_small, crystal='Si', lattice_indices=[2,2,0])

Angle_for_220 = Si_220_miscut.core_angle(lam, Miscut_220)[1][0] + Si_220_miscut.core_angle(lam, Miscut_220)[1][1] - Miscut_220##+0.0008#############
Angle_for_minus220 = Si_minus220_miscut.core_angle(lam, -Miscut_220)[1][0] + Si_minus220_miscut.core_angle(lam, -Miscut_220)[1][1] + Miscut_220
##print('Angle for BM1: ::',Angle_for_220)
Angle_for_111_miscut = Si_111_miscut.core_angle(lam, Miscut_111)[1][0] + Si_111_miscut.core_angle(lam, Miscut_111)[1][1] - Miscut_111
#print('111 miscut angle: ', Angle_for_111_miscut)
##########print('!!!!', core_angle( Miscut_220, lam, [2,2,0])[1][1])
Angle_for_220_small_misscut = Si_220_small_miscut.core_angle(lam, Miscut_220_small)[1][0] + Si_220_small_miscut.core_angle(lam, Miscut_220_small)[1][1] - Miscut_220_small
Angle_for_111 = Si_111.core_angle(lam, 0)[1][0] + Si_111.core_angle(lam, 0)[1][1]

Backpropagation_path = []
backpropagation_flag = 0

source_array_size = js["source_bins"][0]#2000
image_resolution = js["image_resolution"]#600

### when you allocate any array with numpy, you always need to 
# specify data type as default data type (float32 or float64) 
# depends on actual python installation
# normally you want to perform calculations in float32, 
# this is sufficient unless you really want to trace a ray 
# from the sun to the earth and calculate a tiny
# deflection over a huge distance or something like this. 
# float64 will double the memory footprint and also might 
# not be the best option for GPUs
### note, the five variables below are allocated but unused

xedge_tot_ = np.zeros(image_resolution+1, dtype=np.float32)
yedge_tot_ = np.zeros(image_resolution+1, dtype=np.float32)
xedge_marked_tot_= np.zeros(image_resolution+1, dtype=np.float32)
yedge_marked_tot_ = np.zeros(image_resolution+1, dtype=np.float32)
xedge_lin_tot_ = np.zeros(image_resolution+1, dtype=np.float32)

repeat = js["repeat"]#1#6
#### for test
#repeat = 6000
delta_lam_ = js["delta_lam"]#0.05#0.0002#0.05#0.001
lambda_step = delta_lam_#0.0001#0.005#0.01#delta_lam_

lambda_step_tot = js["lambda_step_tot"]#1#4
rocking_step_tot = js["rocking_step_tot"]#1#6#6
#### for test
#rocking_step_tot = 2
rocking_step = js["rocking_step"]#0.0000035/2#0.00002#0.00003#0##0.00000175
#rocking_current=Angle_for_111+js["rocking_init_pos"]*0.000007#*rocking_step#Angle_for_111-5*rocking_step#Angle_for_111-6*rocking_step#Angle_for_111+rocking_step#Angle_for_111-rocking_step#+rocking_step*(rocking_step_tot//2+1)#Angle_for_220-rocking_step*(rocking_step_tot//2)#Angle_for_220-rocking_step*(rocking_step_tot//2)
rocking_current = js["rocking_init_pos"]###*0.000007


### lambda_sequence is to be used for the simulation of imaging
#lambda_sequence = []

rocking_current_2 = Angle_for_111# + rocking_step * (rocking_step_tot // 2)

lambda_current = lam * (1 - lambda_step * lambda_step_tot / 2)

cycle_now = 0
cycle_now_tot = 1
if lambda_step_tot > 1:
    cycle_now_tot = lambda_step_tot
elif rocking_step_tot > 1:
    cycle_now_tot = rocking_step_tot


cycle_flag = 0

x_plot_tot_old = []#[[],[]]
y_plot_tot_old = []#[[],[]]
z_plot_tot_old = []#[[],[]]
c_tot_old = []#[[],[]]

x_plot_tot = []#[[],[]]
y_plot_tot = []#[[],[]]
z_plot_tot = []#[[],[]]
c_tot = []#[[],[]]

all_crystals = [Primary_sl, Si_111, Si_111, object_sl, Si_220_miscut,Si_220_small_miscut, Si_220_miscut,Si_220_small_miscut,detector_sl,0,0]#Primary_sl,##good one
## list of the positions of the crystal that is being detuned: starting from the position of the best alignment
rocking_positions_mono2 = np.array([rocking_current_2,rocking_current_2-3*rocking_step,rocking_current_2-2*rocking_step,rocking_current_2-1*rocking_step, rocking_current_2+1*rocking_step, rocking_current_2+2*rocking_step])#np.linspace(rocking_current_2, rocking_current_2+rocking_step_tot*rocking_step, rocking_step_tot)
#rocking_positions_bm1 = np.array([0,-2*rocking_step,-1*rocking_step,1*rocking_step, 2*rocking_step, 3*rocking_step])#np.linspace(rocking_current_2, rocking_current_2+rocking_step_tot*rocking_step, rocking_step_tot)


plane_image_x_tot=[[] for i in range(len(all_crystals))]#
plane_image_y_tot=[[] for i in range(len(all_crystals))]#
plane_image_lam_tot=[[] for i in range(len(all_crystals))]#
plane_image_intens_tot=[[] for i in range(len(all_crystals))]#
plane_image_steps_tot=[[] for i in range(len(all_crystals))]#

distances = np.array([13816, 9184,185,1000,300,600,300,600,295,2,1], dtype=np.float32)#13816
cross_distances = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)#
shift_to_add= 0#0.00000175
print('Initially the angle is this::  ', Angle_for_220_small_misscut)

#init crystal normals with theit angles to the Central Axis in this order : pitch, roll yaw.
crystals_normals=[0,]*rocking_step_tot
for pos in range(rocking_step_tot):
    crystals_normals[pos] = [None,\
                        #None,\
                        [rocking_current_2,0,0], \
                        #[rocking_current_2,0,0], \
                        [rocking_positions_mono2[pos],0,0], \
                        #[rocking_positions_mono2[pos],0,0]
                        None,\
                        [Angle_for_220-0.00001,0,0], \
                        #-0.0001
                        #-0.00001
                        #-0.00003
                        #-0.005,0], \
                        #0.02??#-0.01#0.004
                        #[Angle_for_minus220,0,0], \
                        [Angle_for_220_small_misscut,0,0], \
                        #+0.000063#+0.01-0.00056#+0.01-0.000558
                        #None,\
                        [Angle_for_220,0.005,0], \
                        #-0.00003
                        #+2*0.00000175,0.005
                        #0.01
                        #[Angle_for_220+2*rocking_step,0.005,0], \
                        #[rocking_positions_bmcr1[pos],0,0], \
                        [Angle_for_220_small_misscut,0,0], \
                        #+0.000063#+0.01-0.000564
                        None, None ]###None,#rocking_positions[pos]##[rocking_current_2,rocking_positions[pos],0]

#%%

start = time.time()

### Trick is: to use a single ray (Central Axis) to initialize geometry

# the same applies to lin_n (crystal_normals)
#####init_ray = LightRay_ev(1, js["rays_per_point"], lam, delta_lam_, 1).generate_beams()

Geometry, edges, lin, lin_n = [0,]*rocking_step_tot,[0,]*rocking_step_tot,[0,]*rocking_step_tot,[0,]*rocking_step_tot
print('Distances::  ',distances)

for pos in range(rocking_step_tot):
    init_ray = LightRay_ev(1, js["rays_per_point"], lam, delta_lam_, 1).generate_beams()
    Geo, lin,edges, lin_n = init_ray.Build_geometry(lam, \
                                                    all_crystals, \
                                                    distances, \
                                                    cross_distances, \
                                                    crystals_normals[pos],\
                                                    #[None, False, False,None,False,False,True,None],\#for Ady
                                                    #here I swapped BM1 and BM2
                                                    #[None, False, False,None,True,False,None],\
                                                    
                                                    [None, False, False,None, True,True, False,False,None],\
                                                    size_of_cr_plane=8, \
                                                    cr_proportion=3)#2.6)
    if pos>0:
        Geometry[pos]=copy.deepcopy(Geometry[0])
        Geometry[pos].object_normals[2]=copy.deepcopy(Geo.object_normals[2])
        Geometry[pos].object_latice_normals[2]=copy.deepcopy(Geo.object_latice_normals[2])
    elif pos==0:
        Geometry[pos]=copy.deepcopy(Geo)

print('Distances became::  ',distances)

# number of bins in the generated histograms
n_bins = image_resolution#50

# nested lists to keep bin edges and histograms
# fixed for all cycles 
x_edge = [[] for i in range(len(all_crystals))]
y_edge = [[] for i in range(len(all_crystals))]

# create a folder to store the histograms - 

root_folder=os.getcwd()
date_object = datetime.date.today()
time_object = datetime.datetime.now().time()
name_for_new_dir = str('__'+str(js["energy"])+'keV_'+str(date_object)+'__'+str(time_object.hour)+'__'+str(time_object.minute)+'__'+str(time_object.second))
os.mkdir(name_for_new_dir)
os.chdir(name_for_new_dir)

#%%
hists=[np.zeros((n_bins,n_bins), dtype=np.float32) for i in range(rocking_step_tot)]
edges_x =np.zeros((2,int(rocking_step_tot)))
edges_y =np.zeros((2,int(rocking_step_tot)))

histograms_for_comparison = [[] for i in range(rocking_step_tot)]
xlim_for_comparison = [[] for i in range(rocking_step_tot)]
ylim_for_comparison = [[] for i in range(rocking_step_tot)]
x_scale_vector = np.zeros(rocking_step_tot)
y_scale_vector = np.zeros(rocking_step_tot)

for cycle in range(rocking_step_tot):

    hist_intensity = [np.zeros((n_bins,n_bins), dtype=np.float32) for i in range(len(all_crystals))]
    hist_wavelength = [np.zeros((n_bins,n_bins), dtype=np.float32) for i in range(len(all_crystals))]
    hist_denom = [np.zeros((n_bins,n_bins), dtype=np.float32) for i in range(len(all_crystals))]
    for iter in range(repeat // rocking_step_tot):   

        # In general if you initialize random number generator seed once and run with the same settingds it will be repeatble
        current_iter = iter + cycle * repeat // rocking_step_tot
        RaySet = LightRay_ev(source_array_size, js["rays_per_point"], lam, delta_lam_, current_iter).generate_beams()
        RaySet.propagation(Geometry[cycle], screen_length=[90,20])##rocking_adjust=[0,0,0]
       
        x_plot = RaySet.rays_traces_x
        y_plot = RaySet.rays_traces_y
        z_plot = RaySet.rays_traces_z
        print(print(len(x_plot)))

        c = RaySet.rays_traces_lam

        # as we calculate a new histogram for every cycle, 
        # I set it to zero here, in the beginning of the cycle
        geometry_el_num=len(Geometry[cycle].object_vertical_or)+2###########################################################
        plane_image_x = [[] for i in range(geometry_el_num)]
        plane_image_y = [[] for i in range(geometry_el_num)]
        plane_image_lam = [[] for i in range(geometry_el_num)]
        ampl = [[] for i in range(geometry_el_num)]
        for geom_id in range(len(all_crystals)):
            ### try to get rid of 'Propagation' file####
            #print('AAAAAAAAAAAAAAAAAAAAAAAA!!!')
            plane_image_x[geom_id].extend(RaySet.rays_plane_x[geom_id][(0.05<RaySet.amplitudes_plane[geom_id][:])])
            plane_image_y[geom_id].extend(RaySet.rays_plane_y[geom_id][(0.05<RaySet.amplitudes_plane[geom_id][:])])
            plane_image_lam[geom_id].extend(RaySet.lambdas_plane[geom_id][(0.05<RaySet.amplitudes_plane[geom_id][:])])
            ampl[geom_id].extend(RaySet.amplitudes_plane[geom_id][(0.05<RaySet.amplitudes_plane[geom_id][:])])
            #print('Geom id ',geom_id, 'length of array  ',len(plane_image_y[geom_id]))
            ####
        
        print(len(plane_image_y))
        for geom_id in range(len(all_crystals)):
            # edges depend on geometry only
            if iter == 0 and cycle == 0:
                # get range along each axis and extent it by 20%
                
                range_x_min = np.min(plane_image_x[geom_id])
                range_x_max = np.max(plane_image_x[geom_id])
                range_y_min = np.min(plane_image_y[geom_id])
                range_y_max = np.max(plane_image_y[geom_id])
                range_x_gap = np.abs(range_x_max - range_x_min)
                range_y_gap = np.abs(range_y_max - range_y_min)
                range_y_max = range_y_min + max(range_x_gap,range_y_gap)##range_y_gap
                range_x_max = range_x_min + max(range_x_gap,range_y_gap)##range_x_gap##Here you switch: normal scale or small scale 2d histograms
                
                range_x_l = range_x_min# - 0.2 * np.abs(range_x_max - range_x_min)#*0.2
                range_x_r = range_x_max# + 0.2 * np.abs(range_x_max - range_x_min)#*0.2
                range_y_l = range_y_min# - 0.2 * np.abs(range_y_max - range_y_min)#*0.2
                range_y_r = range_y_max# + 0.2 * np.abs(range_y_max - range_y_min)#*0.2
                print('current geometry = {}'.format(geom_id))
                print('min_x = {}, max_x = {}, av_x = {}, sys_x = {}, diff = {}, \nmin_y = {}, max_y = {}, av_y = {}, sys_y = {}, diff = {}'.format(range_x_min, \
                                                                                                                                                range_x_max, \
                                                                                                                                                0.5*(range_x_max + range_x_min), \
                                                                                                                                                Geometry[cycle].sys_plane_x[geom_id], \
                                                                                                                                                np.abs(0.5*(range_x_max + range_x_min) - Geometry[cycle].sys_plane_x[geom_id]), \
                                                                                                                                                range_y_min, \
                                                                                                                                                range_y_max, \
                                                                                                                                                0.5*(range_y_max+range_y_min), \
                                                                                                                                                Geometry[cycle].sys_plane_y[geom_id], \
                                                                                                                                                np.abs(0.5*(range_y_max + range_y_min) - Geometry[cycle].sys_plane_y[geom_id])))

                # calculate histogram edges
                x_edge[geom_id] = np.linspace(range_x_l, range_x_r, num=n_bins+1, endpoint=True)
                y_edge[geom_id] = np.linspace(range_y_l, range_y_r, num=n_bins+1, endpoint=True)
                if geom_id==(len(all_crystals)-1):
                    #print('AAAAAAA',edges_x[0][3])
                    edges_x[0][int(cycle)]=np.min(x_edge[len(all_crystals)-1])
                    edges_x[1][int(cycle)]=np.max(x_edge[len(all_crystals)-1])
                    edges_y[0][int(cycle)]=np.min(y_edge[len(all_crystals)-1])
                    edges_y[1][int(cycle)]=np.max(y_edge[len(all_crystals)-1])
            
            # calculate current histogram 
            # for intensity
            [h_, tmp1, tmp2] = np.histogram2d(plane_image_y[geom_id],\
                                            plane_image_x[geom_id], \
                                            [y_edge[geom_id], x_edge[geom_id]],
                                            normed = False, \
                                            weights= ampl[geom_id])
            # and add it to existing
            hist_intensity[geom_id] += np.rot90(h_,axes=(1, 0))#np.rot90(np.fliplr(h_))##make it += h_ for real picture

            # for wavelength
            [h_, tmp1, tmp2] = np.histogram2d(plane_image_y[geom_id],\
                                            plane_image_x[geom_id], \
                                            [y_edge[geom_id], x_edge[geom_id]],
                                            normed = False, \
                                            weights= plane_image_lam[geom_id])
            hist_wavelength[geom_id] += np.rot90(h_,axes=(1, 0))#np.rot90(np.fliplr(h_))##make it += h_ for real picture

            [h_, tmp1, tmp2] = np.histogram2d(plane_image_y[geom_id],\
                                            plane_image_x[geom_id], \
                                            [y_edge[geom_id], x_edge[geom_id]],
                                            normed = False)
            hist_denom[geom_id] += np.rot90(h_,axes=(1, 0))#np.rot90(np.fliplr(h_))##make it += h_# for real picture
            
            if geom_id == geom_chromatic:
                if iter == 0:
                    previous_min = stats.binned_statistic_2d(plane_image_y[geom_id], plane_image_x[geom_id], plane_image_lam[geom_id], statistic='min', bins=[y_edge[geom_id], x_edge[geom_id]])
                    previous_max = stats.binned_statistic_2d(plane_image_y[geom_id], plane_image_x[geom_id], plane_image_lam[geom_id], statistic='max', bins=[y_edge[geom_id], x_edge[geom_id]])
                    previous_max = previous_max[0]
                    previous_min = previous_min[0]
                    previous_min[np.isnan(previous_min)] = 0
                    previous_max[np.isnan(previous_max)] = 0
                current_min = stats.binned_statistic_2d(plane_image_y[geom_id], plane_image_x[geom_id], plane_image_lam[geom_id], statistic='min', bins=[y_edge[geom_id], x_edge[geom_id]])
                current_max = stats.binned_statistic_2d(plane_image_y[geom_id], plane_image_x[geom_id], plane_image_lam[geom_id], statistic='max', bins=[y_edge[geom_id], x_edge[geom_id]])
                current_min = current_min[0]
                current_max = current_max[0]
                current_min[np.isnan(current_min)] = 0
                current_max[np.isnan(current_max)] = 0
                
                previous_min[np.logical_and(previous_min < 1, current_min > 1)] = current_min[np.logical_and(previous_min < 1, current_min > 1)]
                cond = np.logical_and(current_min > 1, previous_min > 1)
                previous_min[cond] = np.minimum(previous_min[cond], current_min[cond])
                previous_max = np.maximum(previous_max, current_max)
    if cycle == 0: 
        x_edge_translated = [[] for i in range(len(all_crystals))]
        y_edge_translated = [[] for i in range(len(all_crystals))]
        for geom_id in range(len(all_crystals)):
            x_edge_translated[geom_id] = x_edge[geom_id]# - np.min(x_edge[geom_id]) - np.abs(np.min(x_edge[geom_id]) - np.max(x_edge[geom_id])) / 2
            y_edge_translated[geom_id] = y_edge[geom_id]# - np.min(y_edge[geom_id]) - np.abs(np.min(y_edge[geom_id]) - np.max(y_edge[geom_id])) / 2
   
    cmap_ = copy.copy(mpl.cm.get_cmap("brg"))#seismic
    cmap_.set_under(color='black')
    # normalize histograms - calculate average in every bin
    for geom_id in range(len(all_crystals)):
        tmp1 = hist_wavelength[geom_id].copy()
        tmp2 = hist_denom[geom_id].copy()
        hist_wavelength[geom_id][tmp2 > 0] = tmp1[tmp2 > 0] / tmp2[tmp2 > 0]
    
    # calculate color (wavelength) range for all geometries
    for geom_id in range(8):
        tmp = hist_wavelength[geom_id].copy()
        tmp = tmp[tmp > 0]
        if geom_id == 0:
            vmin = np.amin(tmp)
            vmax = np.amax(tmp)
        else:
            vmin = min([np.amin(tmp), vmin])
            vmax = max([np.amax(tmp), vmax])

    fig, (axs) = plt.subplots(2, 4, figsize=(10,5))
    fig.suptitle('Wavelength, cycle{}'.format(cycle))

    for fig_i in range(2):
        for fig_j in range(4):
            extent=[np.min(x_edge_translated[fig_i*4+fig_j+geometries_section]), \
                    np.max(x_edge_translated[fig_i*4+fig_j+geometries_section]), \
                    np.max(y_edge_translated[fig_i*4+fig_j+geometries_section]), \
                    np.min(y_edge_translated[fig_i*4+fig_j+geometries_section])]
            
            # calcualte alpha channel
            alpha_ = hist_intensity[fig_i*4+fig_j+geometries_section].copy()
            # normalize alpha
            alpha_ /= np.amax(alpha_)
            # set background (alpha=0) to be opaque
            alpha_[hist_intensity[fig_i*4+fig_j+geometries_section] < 1e-16] = 1
            # to adjust alpha range
            alpha_lim = 0.7
            alpha_ *= (1 - alpha_lim)
            alpha_ += alpha_lim
            # here we normalize hist_wavelength between 0  and 1. this is needed
            # to properly map values with the colormap
            hist_normalized = (hist_wavelength[fig_i*4+fig_j+geometries_section] - vmin) / (vmax - vmin)
            # and we rely on matplotlab to map values to colours
            img_array = cmap_(hist_normalized)
            # now img_array has 3 channels (RGB)
            # we add alpha channel (the fourth)
            img_array[..., 3] = alpha_
            # finally we need to set all background (out of range) values to black
            img_array[hist_wavelength[fig_i*4+fig_j+geometries_section] < vmin, :-1] = 0
            # so, we generated n_bins x n_bins x 4 RGBA array which we simply 
            # plot using the standrad imshow. Note, I use interpolation='none'
            # to disable 'smoothing', I guess this is preferable for histograms
            im = axs[fig_i, fig_j].imshow(img_array, cmap=cmap_, interpolation='none', aspect='auto', extent=extent)
            axs[fig_i, fig_j].set_title('geom {}'.format(fig_i*4+fig_j), fontsize=14)
            if fig_i == 1:
                axs[fig_i, fig_j].set_xlabel('x, mm', fontsize=13)
            if fig_j == 0:
                axs[fig_i, fig_j].set_ylabel('y, mm', fontsize=13)
            axs[fig_i, fig_j].xaxis.set_tick_params(labelsize=11)
            axs[fig_i, fig_j].yaxis.set_tick_params(labelsize=11)
    # some whitespace adjustements
    fig.subplots_adjust(right=0.825, hspace=0.5, wspace=0.5)
    # set colorbar limits
    im.set_clim([vmin, vmax])
    # and add a common colorbar
    cax = fig.add_axes([0.83, 0.025, 0.02, 0.915])
    cbar = fig.colorbar(im, cax=cax)
    # colorbar title
    cbar.ax.set_ylabel('Energy, eV', fontsize=14)
    
    #plt.savefig('test_cycle{}.png'.format(cycle), dpi=400, format='png')
    #plt.show()
    

    for geom_id in range(len(all_crystals)-1,len(all_crystals)):    
        hist_intensity_filename = name_for_new_dir + '_hist_intensity_cycle_{}_geom_{}.npy'.format(cycle, geom_id)
        hist_wavelength_filename = name_for_new_dir + '_hist_wavelength_cycle_{}_geom_{}.npy'.format(cycle, geom_id)
        np.save(hist_intensity_filename, hist_intensity[geom_id])
        np.save(hist_wavelength_filename, hist_wavelength[geom_id])
        # edges depend on geometry only
        # here I save translated edges
        if cycle == 0:
            edge_x_filename = name_for_new_dir + '_edge_x_geom_{}.npy'.format(geom_id)
            edge_y_filename = name_for_new_dir + '_edge_y_geom_{}.npy'.format(geom_id)
            np.save(edge_x_filename, x_edge_translated[geom_id])
            np.save(edge_y_filename, y_edge_translated[geom_id])
    hists[cycle]=hist_intensity[len(all_crystals)-1].copy() 
    
    histograms_for_comparison[cycle] = hist_intensity[len(all_crystals)-1]#[len(all_crystals)-1]
    xlim_for_comparison[cycle] = x_edge_translated[len(all_crystals)-1]#[len(all_crystals)-1]
    ylim_for_comparison[cycle] = y_edge_translated[len(all_crystals)-1]#[len(all_crystals)-1]
    x_scale_vector[cycle]=np.max(xlim_for_comparison[cycle])-np.min(xlim_for_comparison[cycle])
    y_scale_vector[cycle]=np.max(ylim_for_comparison[cycle])-np.min(ylim_for_comparison[cycle])
