###############################################################
##  Main Body of the program
############################################################
#%%
# external imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import Colorbar
import matplotlib as mpl
import copy
import json
import os
import datetime
import copy
import scipy

from tif_image_analysis import open_tif_image, open_exper_data,show_image_contours

from matplotlib.colors import LogNorm
from mpl_toolkits import mplot3d
from math import isnan, floor
from Rocking_Curve import RockingCurve

#from fast_histogram import histogram2d

#import pickle

import time

# internal imports

from Crystals import CrystalData, Slits
from Open_trajectories import show_trajectories, show_contours
#from Lab_system_ev import Propagation as Propagation_ev
from Ray_Visualizer_ev import LightRay as LightRay_ev
from astropy import modeling

fitter = modeling.fitting.LevMarLSQFitter()
model = modeling.models.Gaussian1D()#+modeling.models.Gaussian1D()#modeling.models.Linear1D()


#%%
# parse files with set-up configuration
#os.chdir('C:\\Users\\gx4419\\Documents\\Python_Scripts\\Evelinas_histograms_created')
os.chdir('C:\\Users\\gx4419\\Documents\\Python_Scripts\\backup\\try_to_finalize_new_reflection')

with open('Parameters_KARA.txt') as f:#('/ccpi/data/RayTracing_BraggMagnifier/Parameters_KARA.txt') 
    data_with_params = f.read()
'''
with open('Parameters_KARA.txt') as f:
    data_with_params = f.read()
'''
'''
with open('Parameters_P23.txt') as f:
    data_with_params = f.read()
'''
geom_chromatic = 9

geometries_section=3#3#0# which optical elements will be in the graphic window? (0 - beginning of the system)


js = json.loads(data_with_params)
print("Data type after reconstruction : ", type(js))
print(js)

Graphics = js["Graphics"]#True
Beam_path = js["Beam_path"]#True

energy = js["energy"]# energy = 29 # in keV    #29 #30.5


lam = 12.3984193E-10 / energy #in Angström

k = np.pi*2/lam

Primary_sl= Slits([0.32,0.53,0.099,1.23])#Slits([0.62,0.73,0.095,1.19]) #Slits(js["mono_slits"])##[0.12,0.33,0.095,1.19]#[0.72,0.73,0.595,1.19]####[-0.05,0.25,0.05,0.05]
##Primary_sl= Slits([0.52,0.73,0.3,1.43])#Slits([0.62,0.73,0.095,1.19]) #Slits(js["mono_slits"])##[0.12,0.33,0.095,1.19]#[0.72,0.73,0.595,1.19]####[-0.05,0.25,0.05,0.05]
#Primary_sl= Slits([0.42,0.63,0.3,1.43])
#Primary_sl= Slits([100000,100000,1000000,1000000])
#Primary_sl= Slits([0.92,1.13,1.8,2.93])
Primary_sl=Slits([0.52,0.73,0.3,1.43])#Primary_sl=Slits([0.0,0.13,1000000,1000000])#Primary_sl= ([100000,100000,1000000,1000000])#
Primary_sl= Slits([100000,100000,1000000,1000000])
#Primary_sl= Slits([0.3,0.3,0.3,0.3])
#object_sl= Slits([-272.095,272.102,0.001,0.001])#Slits([-272.107,272.1095,0.001,0.001])#Slits([-272.11,272.1125,0.001,0.001])#(js["mono_slits"])
#object_sl= Slits([-272.095,272.102,0.001,0.001])#
object_sl= Slits([100000,100000,1000000,1000000])
#detector_sl= Slits([-325,353,-110,138])#([-30,300,-10,90])#([-272.095,272.102,0.001,0.001])
'''
detector_sl= Slits([-397,425,153,-125])#Slits([-5470,5498,-5817,5845])#Slits([-5482,5504,-5843,5871])#Best variant:##Slits([-5470,5498,-5813,5841])#Slits([-5462,5590,-5810,5843])#Slits([-5472,5500,-5820,5848])#Slits([100000,100000,1000000,1000000])#Slits([-5475,5485,-5820,5840])
'''
#detector_sl= Slits([-399,427,148,-120])
#detector_sl= Slits([-395,423,144,-116])
'''
detector_sl= Slits([-150.5,178.5,150,-122])#Slits([-399,427,150,-122]) #{hor,hor,vert,vert}
'''
detector_sl= Slits([100000,100000,1000000,1000000])
'''
#THese slits - for system with 220 monochromator
detector_sl= Slits([-150.5,178.5,140,-112])
'''
Miscut_220_small = js["Miscut_220_small"]#0.01640609
Miscut_220 = js["Miscut_220"]#0.1033235#for our real setup###0.1033235,

Miscut_111 = js["Miscut_111"]#0.11992157# for the 15keV setup, 20 magnification 111 orientation

Lattice1=[1,1,1]#[1,1,1]#[2,2,0]
Lattice2=[2,2,0]
missc1=0.03
missc2=0.165
'''
Lattice1=[1,0,0]
Lattice2=[2,2,0]
missc1=0.06
missc2=0.165
'''
Si_111 = CrystalData(0, crystal='Si', lattice_indices=Lattice1)


#Si_111 = CrystalData(0, crystal='Si', lattice_indices=[2,2,0])
Si_220_miscut = CrystalData(Miscut_220, crystal='Si', lattice_indices=[2,2,0])
Si_minus220_miscut = CrystalData(-Miscut_220, crystal='Si', lattice_indices=[2,2,0])
Si_111_miscut = CrystalData(Miscut_111, crystal='Si', lattice_indices=[1,1,1])
Si_220_small_miscut = CrystalData(Miscut_220_small, crystal='Si', lattice_indices=[2,2,0])

##Angle_for_220 = core_angle( Miscut_220, lam, [2,2,0])[1][0]-0.000069-Miscut_220

Angle_for_220 = Si_220_miscut.core_angle(lam, Miscut_220)[1][0] + Si_220_miscut.core_angle(lam, Miscut_220)[1][1] - Miscut_220##+0.0008#############
Angle_for_minus220 = Si_minus220_miscut.core_angle(lam, -Miscut_220)[1][0] + Si_minus220_miscut.core_angle(lam, -Miscut_220)[1][1] + Miscut_220
##print('Angle for BM1: ::',Angle_for_220)
Angle_for_111_miscut = Si_111_miscut.core_angle(lam, Miscut_111)[1][0] + Si_111_miscut.core_angle(lam, Miscut_111)[1][1] - Miscut_111
#print('111 miscut angle: ', Angle_for_111_miscut)
##########print('!!!!', core_angle( Miscut_220, lam, [2,2,0])[1][1])
Angle_for_220_small_misscut = Si_220_small_miscut.core_angle(lam, Miscut_220_small)[1][0] + Si_220_small_miscut.core_angle(lam, Miscut_220_small)[1][1] - Miscut_220_small
Angle_for_111 = Si_111.core_angle(lam, 0)[1][0] + Si_111.core_angle(lam, 0)[1][1]
'''
rocking_x10, rocking_y10, ang10, dif10=RockingCurve(0, 0.002, 0.0000001, 0, lam*0.99995, Lattice1)
rocking_x20, rocking_y20, ang20, dif20=RockingCurve(0, 0.002, 0.0000001, 0.1033235, lam*0.99995, Lattice2)#0.1033235
rocking_x30, rocking_y30, ang30, dif30=RockingCurve(0, 0.002, 0.0000001, 0, lam*1.00005, Lattice1)
rocking_x40, rocking_y40, ang40, dif40=RockingCurve(0, 0.002, 0.0000001, 0.1033235, lam*1.00005, Lattice2)#0.1033235
rocking_x50, rocking_y50, ang50, dif50=RockingCurve(0, 0.002, 0.0000001, 0, lam, Lattice1)
rocking_x60, rocking_y60, ang60, dif60=RockingCurve(0, 0.002, 0.0000001, 0.1033235, lam, Lattice2)#0.1033235
print('Your bragg angle: ',ang50, ' and ', ang60)

plt.plot(rocking_x10+ang10, rocking_y10,'g', rocking_x20+ang20, rocking_y20,'r', rocking_x30+ang30, rocking_y30,'c', rocking_x40+ang40, rocking_y40,'y',rocking_x50+ang50, rocking_y50,'b', rocking_x60+ang60, rocking_y60,'k')
'''
#plt.axvline(x=Angle_for_111, linestyle='dashed')
'''
rocking_x_1, rocking_y_1, ang_1, dif_1=RockingCurve(0, 0.2, 0.00001, 0, lam, Lattice1)
rocking_x_2, rocking_y_2, ang_2, dif_2=RockingCurve(0, 0.002, 0.0000001, 0, lam, Lattice2)#0.1033235
print('Your bragg angle: ',ang_1, ' and ', ang_2)
plt.plot(rocking_x_1+ang_1, rocking_y_1,'g', rocking_x_2+ang_2, rocking_y_2,'r')

plt.show()

print(str(Lattice1)+' Bragg angle: ', str(ang10*57.296), ', inc. angle,', str((ang10-missc1)*57.296), ', dep.angle :' + str((ang10+2*missc1)*57.296), ', magnification: ', str(np.sin(ang10+missc1)/np.sin(ang10-missc1)))#', the angle that fits us is:  ', str(np.arctan(np.tan(ang10)*(np.sin(2*ang10)-1)/(np.sin(2*ang10)+1))*57.296))
print(str(Lattice2)+' Bragg angle: ', str(ang20*57.296), ', inc. angle,', str((ang20-missc2)*57.296),  ', dep.angle :' + str((ang20+2*missc2)*57.296), ', magnification: ', str(np.sin(ang20+missc2)/np.sin(ang20-missc2)))#', the angle that fits us is:  ', str(np.arctan(np.tan(ang20)*(np.sin(2*ang20)-1)/(np.sin(2*ang20)+1))*57.296))
'''

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

### the two below are unused
fin_hist = np.zeros(100, dtype=np.float32)
fin_ax = np.zeros(101, dtype=np.float32)


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
rocking_positions = []
### lambda_sequence is unused
lambda_sequence = []
rocking_positions.append(rocking_current)

#rocking_current=0
rocking_current_2 = Angle_for_111# + rocking_step * (rocking_step_tot // 2)


lambda_current = lam * (1 - lambda_step * lambda_step_tot / 2)

cycle_now = 0
cycle_now_tot = 1
if lambda_step_tot > 1:
    cycle_now_tot = lambda_step_tot
elif rocking_step_tot > 1:
    cycle_now_tot = rocking_step_tot

### the five arrays below are unused
h_tot_= np.zeros((image_resolution, image_resolution, cycle_now_tot), dtype=np.float32)#h_tot_=np.zeros((image_resolution,image_resolution,lambda_step_tot))
h_tot_r = np.zeros((image_resolution, image_resolution, cycle_now_tot), dtype=np.float32)
h_tot_i = np.zeros((image_resolution, image_resolution, cycle_now_tot), dtype=np.float32)
h_tot_marked_ = np.zeros((image_resolution, image_resolution, cycle_now_tot), dtype=np.float32)#h_tot_marked_=np.zeros((image_resolution,image_resolution,lambda_step_tot))
h_lin_tot_ = np.zeros((image_resolution, cycle_now_tot), dtype=np.float32)#,lambda_step_tot))

### the three arrays below are unused
h_ = np.zeros((image_resolution,image_resolution), dtype=np.float32)
h_lin = np.zeros((image_resolution), dtype=np.float32)
h_marked = np.zeros((image_resolution,image_resolution), dtype=np.float32)

cycle_flag = 0

### the five parameters below are unused
portions_num = 0
max_hor = 0
min_hor = 0
max_vert = 0
min_vert = 0


x_plot_tot_old = []#[[],[]]
y_plot_tot_old = []#[[],[]]
z_plot_tot_old = []#[[],[]]
c_tot_old = []#[[],[]]

x_plot_tot = []#[[],[]]
y_plot_tot = []#[[],[]]
z_plot_tot = []#[[],[]]
c_tot = []#[[],[]]

### there is no need to init the nine lists below, as they wll be returned by 
# Build_geometry
lin_x_tot = []
lin_y_tot = []
lin_z_tot = []

lin_x_tot_b = []
lin_y_tot_b = []
lin_z_tot_b = []

lin_x_tot_n = []
lin_y_tot_n = []
lin_z_tot_n = []
 
### just to let you know, a more convenient form of initializing nested lists like this
# [[],[],[],[],[],[],[],[]]
# is
# [[] for i in range(8)]
# this will save you from counting brackets 


#all_crystals = [Primary_sl, Si_111, Si_111, Si_220_miscut, Si_220_miscut,0,0]#Primary_sl,
all_crystals = [Primary_sl, Si_111, Si_111, object_sl, Si_220_miscut,Si_220_small_miscut, Si_220_miscut,Si_220_small_miscut,detector_sl,0,0]#Primary_sl,##good one
#all_crystals = [Primary_sl, Si_111, Si_111, object_sl, Si_220_miscut,Si_minus220_miscut, Si_220_miscut,detector_sl,0,0]##for Ady####

#all_crystals = [Primary_sl, Si_111, Si_111,object_sl, Si_220_miscut,Si_220_small_miscut, Si_220_miscut,Si_220_small_miscut,0,0]#Primary_sl,

plane_image_x_tot=[[] for i in range(len(all_crystals))]#[[],[],[],[],[],[],[],[]]
plane_image_y_tot=[[] for i in range(len(all_crystals))]#[[],[],[],[],[],[],[],[]]
plane_image_lam_tot=[[] for i in range(len(all_crystals))]#[[],[],[],[],[],[],[],[]]
plane_image_intens_tot=[[] for i in range(len(all_crystals))]#[[],[],[],[],[],[],[],[]]
plane_image_steps_tot=[[] for i in range(len(all_crystals))]#[[],[],[],[],[],[],[],[]]


### it's better to keep data types consistent throughout the code. 
# In your version, you convert distances from a list to a np.array in Geometry.
# I initialize them as np.array instead. Another suggestion for future: 
# keep naming consistent. If you call a variable 'distances' in the main body,
# keep the same name inside classes, or vice versa. For instance, in Geometry the same variable 
# is called 'object_coords'. This will drastically improve readability of your code.
#distances = np.array([9184,4600,185,1000,300,600,300,600,295,2,1], dtype=np.float32)#13816#np.array([13816, 9184, 2000,1000,300,677,300,450, 148,140], dtype=np.float32)
distances = np.array([13816, 9184,185,1000,300,600,300,600,295,2,1], dtype=np.float32)#13816
#[9184,4600,2000,1000,300,600,300,600,295,2,1]
#9184,4600,##0,13816,
#####distances = np.array([57000, 27000, 2000,1000,300,300,295,2,1], dtype=np.float32)#for P23
#####distances = np.array([57000, 27000, 2000,1000,300,300,300,295,2,1], dtype=np.float32)#for P23, for Ady

##cross_distances = np.array([0, 0, 0,0,0,35,0,0,0,0], dtype=np.float32)#for P23, for Ady
cross_distances = np.array([0, 0, 0,0,0,0,0,0,0,0,0,0], dtype=np.float32)#

shift_to_add= 0#0.00000175

#rocking_positions = [rocking_current_2,rocking_current_2-1.305*rocking_step, rocking_current_2+rocking_step]#np.linspace(rocking_current_2, rocking_current_2+rocking_step_tot*rocking_step, rocking_step_tot)
#rocking_positions = np.array([rocking_current_2,rocking_current_2-5*rocking_step,rocking_current_2-3*rocking_step,rocking_current_2+rocking_step, rocking_current_2+3*rocking_step, rocking_current_2+5*rocking_step])#np.linspace(rocking_current_2, rocking_current_2+rocking_step_tot*rocking_step, rocking_step_tot)
#rocking_positions = np.array([rocking_current_2,rocking_current_2-4*rocking_step,rocking_current_2-2*rocking_step,rocking_current_2+2*rocking_step, rocking_current_2+3*rocking_step, rocking_current_2+4*rocking_step])#np.linspace(rocking_current_2, rocking_current_2+rocking_step_tot*rocking_step, rocking_step_tot)
#rocking_current_2 -= rocking_step/2
rocking_step*=1#1.7
rocking_positions_mono2 = np.array([rocking_current_2,rocking_current_2-2*rocking_step,rocking_current_2-1*rocking_step,rocking_current_2+1*rocking_step, rocking_current_2+2*rocking_step, rocking_current_2+3*rocking_step])#np.linspace(rocking_current_2, rocking_current_2+rocking_step_tot*rocking_step, rocking_step_tot)
#rocking_positions_mono2 = np.array([rocking_current_2,rocking_current_2-3*rocking_step,rocking_current_2-2*rocking_step,rocking_current_2-1*rocking_step, rocking_current_2+1*rocking_step, rocking_current_2+2*rocking_step])#np.linspace(rocking_current_2, rocking_current_2+rocking_step_tot*rocking_step, rocking_step_tot)

#rocking_step=0.000001#0.005
#rocking_positions_bm1 = np.array([0,-2*rocking_step,-1*rocking_step,1*rocking_step, 2*rocking_step, 3*rocking_step])#np.linspace(rocking_current_2, rocking_current_2+rocking_step_tot*rocking_step, rocking_step_tot)

#rocking_positions_mono2+=rocking_step/2#it was always there: rocking_step/2!
##rocking_positions_bmcr1 = np.array([Angle_for_220,Angle_for_220-2*rocking_step,Angle_for_220-1*rocking_step,Angle_for_220+1*rocking_step, Angle_for_220+2*rocking_step, Angle_for_220+3*rocking_step])#np.linspace(rocking_current_2, rocking_current_2+rocking_step_tot*rocking_step, rocking_step_tot)
######################################rocking step = 0.00000575
#rocking_positions+=rocking_step
#rocking_positions = [0,-rocking_step, rocking_step]#np.linspace(rocking_current_2, rocking_current_2+rocking_step_tot*rocking_step, rocking_step_tot)
####rocking_positions = [rocking_current_2,rocking_current_2-rocking_step, rocking_current_2+rocking_step]#np.linspace(rocking_current_2, rocking_current_2+rocking_step_tot*rocking_step, rocking_step_tot)


crystals_normals=[0,]*rocking_step_tot
for pos in range(rocking_step_tot):
    crystals_normals[pos] = [None,\
                        #None,\
                        [rocking_current_2,0,0], \
                        #[rocking_current_2,0,0], \
                        [rocking_positions_mono2[pos],0,0], \
                        #[rocking_positions_mono2[pos],0,0]
                        None,\
                        [Angle_for_220,-0.005,0], \
                        #-0.005,0],#[Angle_for_220,-0.005,0] \
                        #0.02??#-0.01#0.004
                        #[Angle_for_minus220,0,0], \
                        [Angle_for_220_small_misscut,0,0], \
                        #None,\
                        [Angle_for_220+3*0.00000175,0.008,0], \
                        #0.01#[Angle_for_220+2*0.00000175,0.005,0],\
                        #[Angle_for_220+2*rocking_step,0.005,0], \
                        #[rocking_positions_bmcr1[pos],0,0], \
                        [Angle_for_220_small_misscut,0,0], \
                        None, None ]###None,#rocking_positions[pos]##[rocking_current_2,rocking_positions[pos],0]

#%%

start = time.time()

### just a note aside: this is very smart to use a ray to initialize geometry
# I will use this trick for future reference. I would try to do this analytically 
# instead (definitely more complicated than your solution)

### a single ray to initialize geometry
### note, parameter lin (system_trace) seems to be initialized to 0 and never used/ updated
# the same applies to lin_n (crystal_normals)
#####init_ray = LightRay_ev(1, js["rays_per_point"], lam, delta_lam_, 1).generate_beams()

Geometry, edges, lin, lin_n = [0,]*rocking_step_tot,[0,]*rocking_step_tot,[0,]*rocking_step_tot,[0,]*rocking_step_tot
print('Distances were::  ',distances)
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
#print('Geom::: ::: ', Geometry[2])#==Geometry[0])
print('Distances became::  ',distances)
'''
Geometry, edges, lin, lin_n = init_ray.Build_geometry(lam, \
                                                        all_crystals, \
                                                        distances, \
                                                        crystals_normals,\
                                                        [None, False, False, True, False,None],\
                                                        size_of_cr_plane=8, \
                                                        cr_proportion=2.6)
'''
#print('Edges! :: : ', edges)
#print('Geometry of the optical system is created: :  ', Geometry.object_coords, ' .. ', Geometry.object_normals,' .. ', Geometry.object_latice_normals,' .. ', Geometry.object_vertical_or)


# number of bins in the generated histograms
n_bins = image_resolution#50

# nested lists to keep bin edges and histograms
# fixed for all cycles 
x_edge = [[] for i in range(len(all_crystals))]
y_edge = [[] for i in range(len(all_crystals))]

#x_edge = [[[],[],[],[],[],[]] for i in range(8)]
#y_edge = [[[],[],[],[],[],[]] for i in range(8)]

# create a folder to store the histograms - 
# I kind of reiedto follow your naming convention
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

        ### I introduced current_iter counter to update random number generator seed. I am not sure how critical it is. 
        # In general if you initialize it once and run with the same settingds it will be repeatble
        current_iter = iter + cycle * repeat // rocking_step_tot
        RaySet = LightRay_ev(source_array_size, js["rays_per_point"], lam, delta_lam_, current_iter).generate_beams()
        RaySet.propagation(Geometry[cycle], screen_length=[90,20])##rocking_adjust=[0,0,0]
        #Res = Propagation_ev(RaySet, Geometry[cycle])# Ray tracing is performed here!
        #x_plot, y_plot, z_plot, c, plane_image_x, plane_image_y, plane_image_lam, ampl = Res
        x_plot = RaySet.rays_traces_x
        y_plot = RaySet.rays_traces_y
        z_plot = RaySet.rays_traces_z
        print(len(x_plot))
        #print(x_plot[0].shape())
        c = RaySet.rays_traces_lam

        #plane_image_x = RaySet.rays_plane_x
        #plane_image_y = RaySet.rays_plane_y
        #plane_image_lam = RaySet.lambdas_plane
        #ampl = RaySet.amplitudes_plane
        '''
        for flag in range(geometry_el_num):
            plane_image_x[flag].extend(RaySet.rays_plane_x[flag][(0.5<RaySet.amplitudes_plane[flag][:])])
            plane_image_y[flag].extend(RaySet.rays_plane_y[flag][(0.5<RaySet.amplitudes_plane[flag][:])])
            plane_image_lam[flag].extend(RaySet.lambdas_plane[flag][(0.5<RaySet.amplitudes_plane[flag][:])])
            ampl[flag].extend(RaySet.amplitudes_plane[flag][(0.5<RaySet.amplitudes_plane[flag][:])])
        '''

        #print('plane_image_x   ', len(plane_image_x))
        #x_plot_tot.extend(x_plot)
        #z_plot_tot.extend(z_plot)
        #y_plot_tot.extend(y_plot)
        '''
        lin_x_tot.extend(lin[0])
        lin_y_tot.extend(lin[1])
        lin_z_tot.extend(lin[2])
        
        lin_x_tot_b.extend(edges[0])
        lin_y_tot_b.extend(edges[1])
        lin_z_tot_b.extend(edges[2])
        
        lin_x_tot_n.extend(lin_n[0])
        lin_y_tot_n.extend(lin_n[1])
        lin_z_tot_n.extend(lin_n[2])

        for flag in range(10):
            #print( 'rays positions : . .:  ', np.array(plane_image_x[flag]))
            plane_image_x_tot[flag].extend(np.array(plane_image_x[flag]))#[(0.5<ampl[flag])])
            plane_image_y_tot[flag].extend(np.array(plane_image_y[flag]))#[(0.5<ampl[flag])])
            plane_image_lam_tot[flag].extend(np.array(plane_image_lam[flag]))#[(0.5<ampl[flag])])
            plane_image_intens_tot[flag].extend(np.array(ampl[flag]))
            plane_image_steps_tot[flag].extend(np.ones(len(plane_image_lam[flag]))*cycle)#rocking_current)#[(0.5<ampl[flag])]))*rocking_current)
            
        
            plane_image_x_tot=np.insert(plane_image_x_tot,flag,np.append(plane_image_x_tot[flag], np.array(plane_image_x[flag])[(0.5<ampl[flag])]))
            plane_image_y_tot=np.insert(plane_image_y_tot,flag,np.append(plane_image_y_tot[flag], np.array(plane_image_y[flag])[(0.5<ampl[flag])]))
            plane_image_lam_tot=np.insert(plane_image_lam_tot,flag,np.append(plane_image_lam_tot[flag], np.array(plane_image_lam[flag])[(0.5<ampl[flag])]))
            plane_image_steps_tot=np.insert(plane_image_steps_tot,flag,np.append(plane_image_steps_tot[flag], np.ones(len(np.array(plane_image_lam[flag])[(0.5<ampl[flag])]))*rocking_current))
        '''
        # as we calculate a new histogram for every cycle, 
        # I set it to zero here, in the beginning of the cycle
        geometry_el_num=len(Geometry[cycle].object_vertical_or)+2###########################################################
        plane_image_x = [[] for i in range(geometry_el_num)]
        plane_image_y = [[] for i in range(geometry_el_num)]
        plane_image_lam = [[] for i in range(geometry_el_num)]
        ampl = [[] for i in range(geometry_el_num)]
        for geom_id in range(len(all_crystals)):
            ### try to get rid of 'Propagation' file####
            plane_image_x[geom_id].extend(RaySet.rays_plane_x[geom_id][(0.1<RaySet.amplitudes_plane[geom_id][:])])
            plane_image_y[geom_id].extend(RaySet.rays_plane_y[geom_id][(0.1<RaySet.amplitudes_plane[geom_id][:])])
            plane_image_lam[geom_id].extend(RaySet.lambdas_plane[geom_id][(0.1<RaySet.amplitudes_plane[geom_id][:])])
            ampl[geom_id].extend(RaySet.amplitudes_plane[geom_id][(0.1<RaySet.amplitudes_plane[geom_id][:])])
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
                range_y_max = range_y_min + range_y_gap#max(range_x_gap,range_y_gap)##range_y_gap
                range_x_max = range_x_min + range_x_gap#max(range_x_gap,range_y_gap)##range_x_gap##Here you switch: normal scale or small scale 2d histograms
                '''
                range_x_min = min([np.min(plane_image_y[geom_id]),np.min(plane_image_x[geom_id])])
                range_x_max = max([np.max(plane_image_y[geom_id]),np.max(plane_image_x[geom_id])])
                range_y_min = min([np.min(plane_image_y[geom_id]),np.min(plane_image_x[geom_id])])
                range_y_max = max([np.max(plane_image_y[geom_id]),np.max(plane_image_x[geom_id])])
                '''
                '''
                if range_x_min<range_y_min:
                    range_y_min=range_x_min
                else:
                    range_x_min=range_y_min
                if range_x_max>range_y_max:
                    range_y_max=range_x_max
                else:
                    range_x_max=range_y_max
                '''
                range_x_l = range_x_min #- 0.2 * np.abs(range_x_max - range_x_min)#*0.2
                range_x_r = range_x_max #+ 0.2 * np.abs(range_x_max - range_x_min)#*0.2
                range_y_l = range_y_min #- 0.2 * np.abs(range_y_max - range_y_min)#*0.2
                range_y_r = range_y_max #+ 0.2 * np.abs(range_y_max - range_y_min)#*0.2
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
                    previous_min = scipy.stats.binned_statistic_2d(plane_image_y[geom_id], plane_image_x[geom_id], plane_image_lam[geom_id], statistic='min', bins=[y_edge[geom_id], x_edge[geom_id]])
                    previous_max = scipy.stats.binned_statistic_2d(plane_image_y[geom_id], plane_image_x[geom_id], plane_image_lam[geom_id], statistic='max', bins=[y_edge[geom_id], x_edge[geom_id]])
                    previous_max = previous_max[0]
                    previous_min = previous_min[0]
                    previous_min[np.isnan(previous_min)] = 0
                    previous_max[np.isnan(previous_max)] = 0
                current_min = scipy.stats.binned_statistic_2d(plane_image_y[geom_id], plane_image_x[geom_id], plane_image_lam[geom_id], statistic='min', bins=[y_edge[geom_id], x_edge[geom_id]])
                current_max = scipy.stats.binned_statistic_2d(plane_image_y[geom_id], plane_image_x[geom_id], plane_image_lam[geom_id], statistic='max', bins=[y_edge[geom_id], x_edge[geom_id]])
                current_min = current_min[0]
                current_max = current_max[0]
                current_min[np.isnan(current_min)] = 0
                current_max[np.isnan(current_max)] = 0
                
                previous_min[np.logical_and(previous_min < 1, current_min > 1)] = current_min[np.logical_and(previous_min < 1, current_min > 1)]
                cond = np.logical_and(current_min > 1, previous_min > 1)
                previous_min[cond] = np.minimum(previous_min[cond], current_min[cond])
                previous_max = np.maximum(previous_max, current_max)
            
    # normalize histograms
    
    #if rocking_step_tot>1:
    #    c_tot.extend(np.ones(len(c))*rocking_current)
    #else:
    #    c_tot.extend(c)

    # if you want to, you can translate all edges in such a way that you have zero
    # in the centre
    
    if cycle == 0: 
        x_edge_translated = [[] for i in range(len(all_crystals))]
        y_edge_translated = [[] for i in range(len(all_crystals))]
        for geom_id in range(len(all_crystals)):
            x_edge_translated[geom_id] = x_edge[geom_id] - np.min(x_edge[geom_id]) - np.abs(np.min(x_edge[geom_id]) - np.max(x_edge[geom_id])) / 2
            y_edge_translated[geom_id] = y_edge[geom_id] - np.min(y_edge[geom_id]) - np.abs(np.min(y_edge[geom_id]) - np.max(y_edge[geom_id])) / 2
            #x_edge_translated[geom_id] = x_edge[geom_id]# 
            #y_edge_translated[geom_id] = y_edge[geom_id]# 
            
            # or if you want 0 in the top left corner
            # x_edge_translated[geom_id] = x_edge[geom_id] - np.min(x_edge[geom_id])
            # y_edge_translated[geom_id] = y_edge[geom_id] - np.min(y_edge[geom_id])
    
    # just for testing/ visualizing results
    '''
    fig, axs = plt.subplots(2, 4, figsize=(10,6.5), dpi=200)
    fig.suptitle('Intensity, cycle{}'.format(cycle))
    
    for fig_i in range(2):
        for fig_j in range(4):
            extent=[np.min(x_edge_translated[fig_i*4+fig_j]), \
                    np.max(x_edge_translated[fig_i*4+fig_j]), \
                    np.max(y_edge_translated[fig_i*4+fig_j]), \
                    np.min(y_edge_translated[fig_i*4+fig_j])]
            axs[fig_i, fig_j].imshow(hist_intensity[fig_i*4+fig_j], aspect='auto', extent=extent)
            axs[fig_i, fig_j].set_title('geom {}'.format(fig_i*4+fig_j))
    '''
    cmap_ = copy.copy(mpl.cm.get_cmap("brg"))#seismic
    cmap_.set_under(color='black')
    # normalize histograms - calculate average in every bin
    for geom_id in range(len(all_crystals)):
        tmp1 = hist_wavelength[geom_id].copy()
        tmp2 = hist_denom[geom_id].copy()
        hist_wavelength[geom_id][tmp2 > 0] = tmp1[tmp2 > 0] / tmp2[tmp2 > 0]
    
    # calculate color range for all geometries
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
                axs[fig_i, fig_j].set_xlabel('x, mm', fontsize=23)
            if fig_j == 0:
                axs[fig_i, fig_j].set_ylabel('y, mm', fontsize=23)
            axs[fig_i, fig_j].xaxis.set_tick_params(labelsize=20)
            axs[fig_i, fig_j].yaxis.set_tick_params(labelsize=20)
    # some whitespace adjustements
    fig.subplots_adjust(right=0.825, hspace=0.5, wspace=0.5)
    # set colorbar limits
    im.set_clim([vmin, vmax])
    # and add a common colorbar
    cax = fig.add_axes([0.83, 0.025, 0.02, 0.915])
    cbar = fig.colorbar(im, cax=cax)
    # colorbar title
    cbar.ax.set_ylabel('Energy, eV', fontsize=16)
    # so a bit of pain and magic, and we can finally save your beautiful plots!
    # plt.savefig('/path/test_cycle{}.png'.format(cycle), dpi=400, format='png')
    ##plt.show()

    #plt.savefig(path, dpi=200, format='png')
    #plt.show()
    
    

    fig, (axs) = plt.subplots(1,1, figsize=(5,5))
    #fig.suptitle('Wavelength, cycle{}'.format(cycle))
    for fig_i in range(1):
        extent=[np.min(x_edge_translated[geom_chromatic]), \
                np.max(x_edge_translated[geom_chromatic]), \
                np.max(y_edge_translated[geom_chromatic]), \
                np.min(y_edge_translated[geom_chromatic])]
        # calcualte alpha channel
        alpha_ = hist_intensity[geom_chromatic].copy()
        # normalize alpha
        alpha_ /= np.amax(alpha_)
        # set background (alpha=0) to be opaque
        alpha_[hist_intensity[geom_chromatic] < 1e-16] = 1
        # to adjust alpha range
        alpha_lim = 0.1
        alpha_ *= (1 - alpha_lim)
        alpha_ += alpha_lim
        # here we normalize hist_wavelength between 0  and 1. this is needed
        # to properly map values with the colormap
        hist_normalized = (hist_wavelength[geom_chromatic] - vmin) / (vmax - vmin)
        
        # and we rely on matplotlab to map values to colours
        img_array = cmap_(hist_normalized)
        # now img_array has 3 channels (RGB)
        # we add alpha channel (the fourth)
        #img_array[..., 3] = alpha_
        # finally we need to set all background (out of range) values to black
        img_array[hist_wavelength[geom_chromatic] < vmin, :-1] = 0
        # so, we generated n_bins x n_bins x 4 RGBA array which we simply 
        # plot using the standrad imshow. Note, I use interpolation='none'
        # to disable 'smoothing', I guess this is preferable for histograms
        im = axs.imshow(np.flipud(img_array), cmap=cmap_, interpolation='none', aspect='auto', extent=extent)
        
        #axs[0].set_title('old', fontsize=16)
        
        
        '''
        hist_average = 0.5 * (previous_max + previous_min)
        hist_normalized = (np.rot90(np.fliplr(hist_average)) - vmin) / (vmax - vmin)
        # and we rely on matplotlab to map values to colours
        img_array = cmap_(hist_normalized)
        # now img_array has 3 channels (RGB)
        # we add alpha channel (the fourth)
        img_array[..., 3] = alpha_
        # finally we need to set all background (out of range) values to black
        img_array[hist_average < vmin, :-1] = 0
        im = axs[1].imshow(img_array, cmap=cmap_, interpolation='none', aspect='auto', extent=extent)
        '''
        
       
        #axs[1].set_title('averaged', fontsize=16)
        axs.xaxis.set_tick_params(labelsize=16)
        axs.yaxis.set_tick_params(labelsize=16)
        axs.set_xlabel('x, mm', fontsize=16)
        axs.set_ylabel('y, mm', fontsize=16)
        #axs[1].xaxis.set_tick_params(labelsize=16)
        #axs[1].yaxis.set_tick_params(labelsize=16)
    
    im.set_clim([vmin, vmax])
        # and add a common colorbar
        #cax = fig.add_axes([0.005, 0.005, 0.04, 0.8])
    cbar = fig.colorbar(im)
        # colorbar title
    cbar.ax.set_ylabel('Energy, keV', fontsize=16)
    cbar.ax.yaxis.set_tick_params(labelsize=16)

    fig, (axs) = plt.subplots(1,1, figsize=(6,6))
    tmp = previous_max - previous_min
    #tmp[tmp > 1e-30] = 1
    im = axs.imshow(np.flipud(np.fliplr(tmp)), cmap='jet', interpolation='none', aspect='auto', extent=extent)
    #ax=plt.figure(figsize=(10,10))
    axs.yaxis.set_tick_params(labelsize=16)
    axs.xaxis.set_tick_params(labelsize=16)
    #plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    axs.set_xlabel('x, mm', fontsize=16)
    axs.set_ylabel('y, mm', fontsize=16)
    #im.set_clim([vmin, vmax])
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('Energy spectral width, keV', fontsize=16)
    cbar.ax.yaxis.set_tick_params(labelsize=16)


    fig, (axs) = plt.subplots(1,1, figsize=(6,6))
    #tmp[tmp > 1e-30] = 1
    im = axs.imshow(np.flipud(hist_intensity[geom_chromatic]), cmap='Greys', interpolation='none', aspect='auto', extent=extent)
    #ax=plt.figure(figsize=(10,10))
    axs.yaxis.set_tick_params(labelsize=16)
    axs.xaxis.set_tick_params(labelsize=16)
    #plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    axs.set_xlabel('x, mm', fontsize=16)
    axs.set_ylabel('y, mm', fontsize=16)
    #im.set_clim([vmin, vmax])
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('Intensity, a.u.', fontsize=16)
    cbar.ax.yaxis.set_tick_params(labelsize=16)
    
    #plt.imshow(tmp,extent=extent)
    #plt.title('diff')
    #cbar = plt.colorbar()

        # colorbar title
    #cbar.ax.set_ylabel('Spectral width, eV', fontsize=16)
    
    '''
    plt.figure(figsize=(10,10))
    plt.imshow(previous_max, vmin=28)
    plt.title('max')
    plt.colorbar()

    plt.figure(figsize=(10,10))
    plt.imshow(previous_min, vmin=28)
    plt.title('min')
    plt.colorbar()
    '''

    ##rocking_current += rocking_step
    ### not sure why you update these two below but I did the same for consistency
    #rocking_positions.append(rocking_current)
    ###rocking_current_2+=rocking_step

    # save histograms - I normally use np.save but you might prefer other way
    # to load them you can simply use
    # variable_name = np.load(filename)
    
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
    #edges_x=x_edge   
    #edges_x =np.max(x_edge_translated[7])-np.min(x_edge_translated[7])#[np.min(x_edge_translated[7]),np.max(x_edge_translated[7])]
    #edges_y =np.max(y_edge_translated[7])-np.min(y_edge_translated[7])#[np.min(y_edge_translated[7]),np.max(y_edge_translated[7])]
    histograms_for_comparison[cycle] = hist_intensity[len(all_crystals)-1]#[len(all_crystals)-1]
    xlim_for_comparison[cycle] = x_edge_translated[len(all_crystals)-1]#[len(all_crystals)-1]
    ylim_for_comparison[cycle] = y_edge_translated[len(all_crystals)-1]#[len(all_crystals)-1]
    x_scale_vector[cycle]=np.max(xlim_for_comparison[cycle])-np.min(xlim_for_comparison[cycle])
    y_scale_vector[cycle]=np.max(ylim_for_comparison[cycle])-np.min(ylim_for_comparison[cycle])

fig=plt.figure()  
gs = plt.GridSpec(2, 2, height_ratios=[2, 2])
gs.update(left=0.08, right=0.925,top=0.95, bottom=0.05,hspace=0.3, wspace=0.1)
      
ax0 = plt.subplot(gs[0, 0])
ax1 = plt.subplot(gs[0, 1]) 
ax3 = plt.subplot(gs[1, 0]) 
ax4 = plt.subplot(gs[1, 1])

hh_, xx = open_tif_image('C:\\Users\\gx4419\\Documents\\Python_Scripts\\4crystal\\','1')

print(hh_)
hh_1, xx = open_tif_image('C:\\Users\\gx4419\\Documents\\Python_Scripts\\4crystal\\','2')
print(hh_1)
hh_2, xx = open_tif_image('C:\\Users\\gx4419\\Documents\\Python_Scripts\\4crystal\\','3')
hh_3, xx = open_tif_image('C:\\Users\\gx4419\\Documents\\Python_Scripts\\4crystal\\','4')
hh_4, xx = open_tif_image('C:\\Users\\gx4419\\Documents\\Python_Scripts\\4crystal\\','5')
print(hh_4)
hh_5, xx = open_tif_image('C:\\Users\\gx4419\\Documents\\Python_Scripts\\4crystal\\','6')
#show_image_contours(histograms_for_comparison,[hh_5,hh_4, hh_3,hh_2,hh_1,hh_])
steps= rocking_positions_mono2-rocking_current_2
histogram_temp=histograms_for_comparison[0]
histograms_for_comparison[0]=histograms_for_comparison[1]
histograms_for_comparison[1]=histograms_for_comparison[2]
histograms_for_comparison[2]=histogram_temp
st_temp=steps[0]
steps[0]=steps[1]
steps[1]=steps[2]
steps[2]=st_temp
show_image_contours(histograms_for_comparison,histograms_for_comparison,steps*57.296,0,1)
show_image_contours(histograms_for_comparison,histograms_for_comparison,steps*57.296,1,1)
'''
show_image_contours(histograms_for_comparison,[hh_5,hh_4,hh_3,hh_2,hh_1,hh_],steps*57.296,0,1)
show_image_contours(histograms_for_comparison,[hh_5,hh_4,hh_3,hh_2,hh_1,hh_],steps*57.296,1,1)
'''
'''
norm_coefs1=open_exper_data([ax0,ax1,ax3,ax4],[hh_5,hh_4, hh_3,hh_2,hh_1,hh_],[28,28,28,28,28,28],[28,28,28,28,28,28],0,0,[0,0,0,0],True,'-',20)#norm_coefs)#[2.8,2.8,2.8,2.8,2.8,2.8],[2.8,2.8,2.8,2.8,2.8,2.8])
print('Now normalize on these: :  ',norm_coefs1)
norm_coefs=open_exper_data([ax0,ax1,ax3,ax4],[histograms_for_comparison[1], histograms_for_comparison[2],histograms_for_comparison[0],histograms_for_comparison[3],histograms_for_comparison[4],histograms_for_comparison[5]], x_scale_vector,y_scale_vector,-29,-29,[0,0,0,0],False,'*',1)#-30,-40,[0,0,0,0],False,'*',1
#print('Limits would be: x :  ',np.max(xlim_for_comparison[0])-np.min(xlim_for_comparison[0]), 'Limits would be: y :  ', np.max(xlim_for_comparison[0])-np.min(ylim_for_comparison[0]))
print('Now normalize on these: :  ',norm_coefs)
plt.show()


print(time.time() - start)
print('Length of elements array: :: ',len(plane_image_y_tot[geom_chromatic]))

plt.show()
'''