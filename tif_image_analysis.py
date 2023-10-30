from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker  
from scipy.signal import lfilter
from scipy.ndimage.filters import gaussian_filter

def format_func(value, tick_number):
    return value*0.055


def open_tif_image(image_address, image_name):
    #im = Image.open('C:\\Users\\gx4419\Documents\\Python Scripts\\Magnifier_simulation\\experimental_data\\dcm2pitch_0p1860_0.tif')
    im = Image.open(image_address + image_name + '.tif')
    
    imarray = np.array(im)
    imarray=np.transpose(imarray)
    
    return imarray, int(np.floor(imarray.shape[0]*np.sqrt(2))+1)

def show_image_contours(imarray_,imarray2_, rocking_angle, first_or_second_half, swap_anything):
    fig, (axs) = plt.subplots(1, int(len(imarray2_)/2), figsize=(5,5))#plt.subplots(2
    mask = np.zeros(np.shape(imarray2_[0]))
    for i_num in range(first_or_second_half,first_or_second_half+1):
        for j_num in range(int(len(imarray2_)/2)):
            ii=j_num +3*i_num 
            imarray2=imarray2_[ii]
            '''
            for jj in range(np.shape(imarray2)[0]):
                for kk in range(np.shape(imarray2)[1]):
                    if imarray2[jj][kk]>=0.3*np.max(imarray2):
                        imarray2[jj][kk]=np.nan
            '''
            imarray=imarray_[ii]
            
            array_max=np.max(imarray)
            array_min=np.min(imarray[(imarray>0)])
            array2_max=100#np.max(imarray)
            array2_min=1#np.min(imarray[(imarray>0)])
            print(array_min)
            
            
            
            
            if swap_anything:
                mask[(np.where(imarray2<1))]=1
                mask[(np.where(imarray2>150))]=1
                #mask[(np.where(np.isnan(imarray2)))]=1
                imarray2[(np.where(imarray2>150))]=0
            masked_imarray2 = np.ma.MaskedArray(imarray2, mask = mask)
            
            ##imarray[(np.where(gradients[0]>50))]=0
            #imarray[(np.where(gradients[1]<30))]=0
            
            blurred2 = gaussian_filter(masked_imarray2, sigma=11)#12#3
            blurred = gaussian_filter(imarray, sigma=11)#20
            #gradients = np.gradient(blurred2)
            #mask[(np.where(gradients[0]>0.1))]=1
            #mask[(np.where(gradients[1]>0.1))]=1
            ######
            array2_max=np.max(blurred2)
            array2_min=1
            #mask[(np.where(blurred2>0.6*array2_max))]=1
            
            blurred2 = np.ma.MaskedArray(blurred2, mask = mask)
            print(np.shape(mask))
            #########axs[0].imshow(masked_imarray2,vmin=array_min,vmax=array_max, interpolation='none', aspect='auto')
            #axs[1].imshow(blurred,vmin=1,vmax=100, interpolation='none', aspect='auto')
            #axs[1].contour(blurred, colors='black',aspect='auto')
            #axs[i_num][j_num].imshow(np.rot90(np.fliplr(masked_imarray2)),vmin=np.min(masked_imarray2),vmax=np.max(masked_imarray2), interpolation='none',cmap='jet', aspect='auto')
            if swap_anything:
                imarray2=np.rot90(np.fliplr(imarray2))
                blurred=np.fliplr(blurred)
                blurred2=np.rot90(np.fliplr(blurred2))
            axs[j_num].imshow(imarray2,vmin=np.min(imarray2),vmax=np.max(imarray2), interpolation='none',cmap='binary', aspect='auto')#vmin=np.min(imarray),vmax=np.max(imarray)##cmap='jet'
            
            axs[j_num].xaxis.set_major_formatter(plt.FuncFormatter(format_func))#set_major_locator(plt.MultipleLocator(250))
            axs[j_num].yaxis.set_major_formatter(plt.FuncFormatter(format_func))
            #axs[i_num][j_num].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f mm'))
            step_2= (np.max(blurred2)-np.min(blurred2))/5
            step_2_small=(np.max(blurred2)-np.min(blurred2)-3*step_2)/4
            levels_step=np.linspace(np.min(blurred2),np.max(blurred2),4)#12#levels_step=np.linspace(np.min(blurred2),np.max(blurred2)-3*step_2,4)#12
            #np.rot90(np.fliplr(blurred2))
            contur1=axs[j_num].contour(blurred2, levels_step,cmap='autumn',aspect='auto',linewidths=2.5,linestyles='dashdot')#colors='black',aspect='auto')
            #plt.clabel(contur1, fmt = '%1.1d', colors = 'k', fontsize=14) #contour line labels
            #axs[i_num][j_num].contour(np.flip(blurred2,axis=0), 10,cmap='autumn',aspect='auto')#colors='black',aspect='auto')
            step_= (np.max(blurred)-np.min(blurred))/5
            step_small= (np.max(blurred)-np.min(blurred)-4*step_)/4
            
            levels_step=np.linspace(1*np.min(blurred),1*np.max(blurred),4)#levels_step=np.linspace(1*np.min(blurred),1*np.max(blurred)-3*step_,4)#0.5*np.max(blurred)#np.max(blurred)-4.5*step_
            
            contur2=axs[j_num].contour(blurred,levels_step,cmap='cool',aspect='auto',linewidths=3.5)# np.flipud(np.rot90(np.fliplr(blurred))#colors='red',aspect='auto')#np.rot90(blurred,k=3)
            #plt.clabel(contur2, fmt = '%1.1d', colors = 'k', fontsize=14) #contour line labels
            
            #axs[i_num][j_num].set_title('DCM2 pitch rocking, step:  {}'.format(int(ii+1-len(imarray2_)/2)), fontsize=14)
            axs[j_num].set_title('DCM2_pitch =  DCM2_pitch$_0$ {0:+.1e} $^\circ$'.format( rocking_angle[ii]), fontsize=16)
            #axs[i_num][j_num]
            #axs[i_num][j_num].imshow(np.zeros([np.size(np.fliplr(imarray),0),np.size(np.fliplr(imarray),1)]),vmin=0,vmax=0, interpolation='none',cmap='jet', aspect='auto')
            axs[j_num].xaxis.set_tick_params(labelsize=14)
            axs[j_num].yaxis.set_tick_params(labelsize=14)
            plt.xlabel('mm', fontsize=14)#if int(ii+1-len(imarray2_)/2)>0:
            #axs[1].imshow(gradients[0],vmin=1,vmax=50, interpolation='none', aspect='auto')
            #im = plt.imshow(gradients[0],vmin=1,vmax=10, interpolation='none', aspect='auto')#, extent=extent)
            #im = plt.imshow(gradients[1],vmin=1,vmax=10, interpolation='none', aspect='auto')#, extent=extent)
            ##im = plt.imshow(imarray,vmin=1,vmax=100, interpolation='none', aspect='auto')#, extent=extent)
            ##print('Blurred',max(blurred), min(blurred))
            
            ##im = plt.imshow(blurred,vmin=1,vmax=100, interpolation='none', aspect='auto')#, extent=extent)
            #axs[1].contourf(masked_imarray, levels=[15, 25, 45],colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
    plt.show()

def image_line_profile(imarray, shift_up=0):
    #imarray = open_tif_image(image_address, image_name)
    
    image_resolution= imarray.shape
    size_sim=int(np.floor(imarray.shape[0]*np.sqrt(2))+1)
    lin_coord=np.zeros(size_sim)
    lin_scale=np.linspace(0,size_sim, size_sim)
    print('IMAGE RESOLUTION', image_resolution)
    for i in range(image_resolution[0]):
        for j in range(image_resolution[1]):
            
            if ((i>248) and (i<265)) or ((j>248) and (j<265)):
                continue
            
            if (imarray[i][j]>1000) or (imarray[i][j]<1):
                imarray[i][j]=0
            if abs(i-(j-shift_up))<2:
            #if abs(i-(image_resolution[1]-j-shift_up))<1:
                
                ####lin_coord[(int(np.floor(np.sqrt((i+shift_up)*(i+shift_up)+(image_resolution[1]-j-shift_up)*(image_resolution[1]-j-shift_up)))))]+=imarray[i][j]
                lin_coord[(int(np.floor(np.sqrt((i+shift_up/2)*(i+shift_up/2)+(j-shift_up/2)*(j-shift_up/2)))))]+=imarray[i][j]
                #imarray[i][j]+=1000
    
    lin_scale = np.delete(lin_scale, np.where((lin_coord<2) | (lin_coord>1000)))
    lin_coord = np.delete(lin_coord, np.where((lin_coord<2) | (lin_coord>1000)))
    
    '''
    lin_coord = np.delete(lin_coord, np.where((lin_scale<265) & (lin_scale>248)))
    lin_scale = np.delete(lin_scale, np.where((lin_scale<265) & (lin_scale>248)))
    '''
    '''
    fig=plt.figure()
    plt.plot(lin_scale, lin_coord, '*')
    plt.show()
    '''
    '''
    fig=plt.figure()
    im = plt.imshow(imarray,cmap='seismic',extent=[0,image_resolution[0],0,image_resolution[1]])
    plt.clim(25, 60)
    cb = plt.colorbar(im)
    plt.show()
    '''
    
    return lin_coord, lin_scale

def open_exper_data(diagrams,hist_array,side_1_length,side_2_length,side_1_shift,side_2_shift,normalization_coefficients,preparator,markers,finness_degree): #(fitter, model):#hist_array
    '''
    fig=plt.figure()  
    gs = plt.GridSpec(2, 2, height_ratios=[2, 2])

    gs.update(left=0.08, right=0.925,top=0.95, bottom=0.05,hspace=0.3, wspace=0.1)
        
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1]) 
    ax3 = plt.subplot(gs[1, 0]) 
    ax4 = plt.subplot(gs[1, 1])
    '''

    ax0 = diagrams[0]
    ax1 = diagrams[1]
    ax3 = diagrams[2]
    ax4 = diagrams[3]
    hh_,hh_1, hh_2,hh_3,hh_4,hh_5 = hist_array/np.max(np.max(np.max(hist_array)))
    '''
    if preparator:
        hh_=np.fliplr(hh_)
        hh_1=np.fliplr(hh_1)
        hh_2=np.fliplr(hh_2)
        hh_3=np.fliplr(hh_3)
        hh_4=np.fliplr(hh_4)
        hh_5=np.fliplr(hh_5)
    '''
    print(np.diagonal(hh_,offset=(-50)).shape)
    if preparator: 
        prep=1
    else:
        prep=-1
    #y_edge_ = np.ones(len(side_1_length))
    y_edge_=[np.linspace(np.sqrt(side_1_shift**2+side_2_shift**2),prep*np.sqrt(side_1_length[ii]**2+side_2_length[ii]**2)+np.sqrt(side_1_shift**2+side_2_shift**2),len(hh_)) for ii in range(0,len(side_1_length))]#range(len(hh_))
    
    #y_edge_special=[np.linspace(-np.sqrt(side_1_shift**2+side_2_shift**2),np.sqrt(side_1_length[ii]**2+side_2_length[ii]**2)-np.sqrt(side_1_shift**2+side_2_shift**2),len(hh_)) for ii in range(0,len(side_1_length))]#range(len(hh_))
    #y_edge_special=y_edge_
    y_edge_special=[np.linspace((prep+1)*np.sqrt(side_1_shift**2+side_2_shift**2),np.sqrt(side_1_length[ii]**2+side_2_length[ii]**2)+(prep+1)*np.sqrt(side_1_shift**2+side_2_shift**2),len(hh_)) for ii in range(0,len(side_1_length))]#range(len(hh_))
    
    ##y_edge_special=[np.linspace(-1*prep*np.sqrt(side_1_shift**2+side_2_shift**2),prep*np.sqrt(side_1_length[ii]**2+side_2_length[ii]**2)-1*prep*np.sqrt(side_1_shift**2+side_2_shift**2),len(hh_)) for ii in range(0,len(side_1_length))]#range(len(hh_))
    #y_edge_special=[np.linspace(2*prep*side_1_shift*side_2_shift/np.sqrt(side_1_shift**2+side_2_shift**2),np.sqrt(side_1_length[ii]**2+side_2_length[ii]**2)+2*prep*side_1_shift*side_2_shift/np.sqrt(side_1_shift**2+side_2_shift**2),len(hh_)) for ii in range(0,len(side_1_length))]#range(len(hh_))

    y_edge_hor=[np.linspace(-side_1_shift,prep*side_1_length[ii]-side_1_shift,len(hh_)) for ii in range(0,len(side_1_length))]
    y_edge_vert=[np.linspace(-side_2_shift,prep*side_2_length[ii]-side_2_shift,len(hh_)) for ii in range(0,len(side_2_length))]
    
    print(len(y_edge_))
    diag_max1=len(np.diagonal(hh_, axis1=0))

    #fitted_model = fitter(model, y_edge, lin_coord[0])
    hh_y=sum([np.pad(np.diagonal(hh_, offset=(-int(np.floor(diag_max1/2))+x),axis1=0), ((diag_max1-len(np.diagonal(hh_, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2, diag_max1-len(np.diagonal(hh_, offset=(-int(np.floor(diag_max1/2))+x),axis1=0))-(diag_max1-len(np.diagonal(hh_, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2),'constant', constant_values=(0,0)) for x in range(0, int(diag_max1))])
    hh_x=y_edge_special[0]
    hh_1y=sum([np.pad(np.diagonal(hh_1, offset=(-int(np.floor(diag_max1/2))+x),axis1=0), ((diag_max1-len(np.diagonal(hh_1, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2, diag_max1-len(np.diagonal(hh_1, offset=(-int(np.floor(diag_max1/2))+x),axis1=0))-(diag_max1-len(np.diagonal(hh_1, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2),'constant', constant_values=(0,0)) for x in range(0, int(diag_max1))])
    hh_1x=y_edge_special[1]
    hh_2y=sum([np.pad(np.diagonal(hh_2, offset=(-int(np.floor(diag_max1/2))+x),axis1=0), ((diag_max1-len(np.diagonal(hh_2, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2, diag_max1-len(np.diagonal(hh_2, offset=(-int(np.floor(diag_max1/2))+x),axis1=0))-(diag_max1-len(np.diagonal(hh_2, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2),'constant', constant_values=(0,0)) for x in range(0, int(diag_max1))])                
    hh_2x=y_edge_special[2]
    hh_3y=sum([np.pad(np.diagonal(hh_3, offset=(-int(np.floor(diag_max1/2))+x),axis1=0), ((diag_max1-len(np.diagonal(hh_3, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2, diag_max1-len(np.diagonal(hh_3, offset=(-int(np.floor(diag_max1/2))+x),axis1=0))-(diag_max1-len(np.diagonal(hh_3, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2),'constant', constant_values=(0,0)) for x in range(0, int(diag_max1))])
    hh_3x=y_edge_special[3]
    hh_4y=sum([np.pad(np.diagonal(hh_4, offset=(-int(np.floor(diag_max1/2))+x),axis1=0), ((diag_max1-len(np.diagonal(hh_4, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2, diag_max1-len(np.diagonal(hh_4, offset=(-int(np.floor(diag_max1/2))+x),axis1=0))-(diag_max1-len(np.diagonal(hh_4, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2),'constant', constant_values=(0,0)) for x in range(0, int(diag_max1))])              
    hh_4x=y_edge_special[4]
    hh_5y=sum([np.pad(np.diagonal(hh_5, offset=(-int(np.floor(diag_max1/2))+x),axis1=0), ((diag_max1-len(np.diagonal(hh_5, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2, diag_max1-len(np.diagonal(hh_5, offset=(-int(np.floor(diag_max1/2))+x),axis1=0))-(diag_max1-len(np.diagonal(hh_5, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2),'constant', constant_values=(0,0)) for x in range(0, int(diag_max1))])
    hh_5x=y_edge_special[5]

    #hh_y=hh_y((hh_y[i]-hh_y[i-1]<1000)&(hh_y[i+1]-hh_y[i]<1000) for i in range(1, len(hh_y)))
    hh_y_new, hh_x_new=denoizer(hh_x,hh_y,finness_degree)
    hh_1y_new, hh_1x_new=denoizer(hh_1x,hh_1y,finness_degree)
    hh_2y_new, hh_2x_new=denoizer(hh_2x,hh_2y,finness_degree)
    hh_3y_new, hh_3x_new=denoizer(hh_3x,hh_3y,finness_degree)
    hh_4y_new, hh_4x_new=denoizer(hh_4x,hh_4y,finness_degree)
    hh_5y_new, hh_5x_new=denoizer(hh_5x,hh_5y,finness_degree)

    maximum_d_1= np.max([np.max(hh_x_new),np.max(hh_1x_new),np.max(hh_2x_new),np.max(hh_3x_new),np.max(hh_4x_new),np.max(hh_5x_new)])
    maximum_d_1= np.max(hh_3x_new)
    if normalization_coefficients[0]==0: 
        normalization_coefficients[0]=maximum_d_1
    im3 = ax0.plot(hh_y_new,\
                    hh_x_new/normalization_coefficients[0],'b'+markers,\
                    hh_1y_new,\
                    hh_1x_new/normalization_coefficients[0],'c'+markers,\
                    hh_2y_new,\
                    hh_2x_new/normalization_coefficients[0],'g'+markers,\
                    hh_3y_new,\
                    hh_3x_new/normalization_coefficients[0],'k'+markers,\
                    hh_4y_new,\
                    hh_4x_new/normalization_coefficients[0],'m'+markers,\
                    hh_5y_new,\
                    hh_5x_new/normalization_coefficients[0],'r'+markers)

    hh_=np.rot90(hh_,axes=(0, 1))
    hh_1=np.rot90(hh_1,axes=(0, 1))
    hh_2=np.rot90(hh_2,axes=(0, 1))
    hh_3=np.rot90(hh_3,axes=(0, 1))
    hh_4=np.rot90(hh_4,axes=(0, 1))
    hh_5=np.rot90(hh_5,axes=(0, 1))

    hh_y=sum([np.pad(np.diagonal(hh_, offset=(-int(np.floor(diag_max1/2))+x),axis1=0), ((diag_max1-len(np.diagonal(hh_, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2, diag_max1-len(np.diagonal(hh_, offset=(-int(np.floor(diag_max1/2))+x),axis1=0))-(diag_max1-len(np.diagonal(hh_, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2),'constant', constant_values=(0,0)) for x in range(0, int(diag_max1))])
    hh_x=y_edge_[0]
    hh_1y=sum([np.pad(np.diagonal(hh_1, offset=(-int(np.floor(diag_max1/2))+x),axis1=0), ((diag_max1-len(np.diagonal(hh_1, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2, diag_max1-len(np.diagonal(hh_1, offset=(-int(np.floor(diag_max1/2))+x),axis1=0))-(diag_max1-len(np.diagonal(hh_1, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2),'constant', constant_values=(0,0)) for x in range(0, int(diag_max1))])
    hh_1x=y_edge_[1]
    hh_2y=sum([np.pad(np.diagonal(hh_2, offset=(-int(np.floor(diag_max1/2))+x),axis1=0), ((diag_max1-len(np.diagonal(hh_2, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2, diag_max1-len(np.diagonal(hh_2, offset=(-int(np.floor(diag_max1/2))+x),axis1=0))-(diag_max1-len(np.diagonal(hh_2, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2),'constant', constant_values=(0,0)) for x in range(0, int(diag_max1))])                
    hh_2x=y_edge_[2]
    hh_3y=sum([np.pad(np.diagonal(hh_3, offset=(-int(np.floor(diag_max1/2))+x),axis1=0), ((diag_max1-len(np.diagonal(hh_3, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2, diag_max1-len(np.diagonal(hh_3, offset=(-int(np.floor(diag_max1/2))+x),axis1=0))-(diag_max1-len(np.diagonal(hh_3, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2),'constant', constant_values=(0,0)) for x in range(0, int(diag_max1))])
    hh_3x=y_edge_[3]
    hh_4y=sum([np.pad(np.diagonal(hh_4, offset=(-int(np.floor(diag_max1/2))+x),axis1=0), ((diag_max1-len(np.diagonal(hh_4, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2, diag_max1-len(np.diagonal(hh_4, offset=(-int(np.floor(diag_max1/2))+x),axis1=0))-(diag_max1-len(np.diagonal(hh_4, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2),'constant', constant_values=(0,0)) for x in range(0, int(diag_max1))])              
    hh_4x=y_edge_[4]
    hh_5y=sum([np.pad(np.diagonal(hh_5, offset=(-int(np.floor(diag_max1/2))+x),axis1=0), ((diag_max1-len(np.diagonal(hh_5, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2, diag_max1-len(np.diagonal(hh_5, offset=(-int(np.floor(diag_max1/2))+x),axis1=0))-(diag_max1-len(np.diagonal(hh_5, offset=(-int(np.floor(diag_max1/2))+x),axis1=0)))//2),'constant', constant_values=(0,0)) for x in range(0, int(diag_max1))])
    hh_5x=y_edge_[5]
    #hh_y=hh_y((hh_y[i]-hh_y[i-1]<1000)&(hh_y[i+1]-hh_y[i]<1000) for i in range(1, len(hh_y)))
    hh_y_new, hh_x_new=denoizer(hh_x,hh_y,finness_degree)
    hh_1y_new, hh_1x_new=denoizer(hh_1x,hh_1y,finness_degree)
    hh_2y_new, hh_2x_new=denoizer(hh_2x,hh_2y,finness_degree)
    hh_3y_new, hh_3x_new=denoizer(hh_3x,hh_3y,finness_degree)
    hh_4y_new, hh_4x_new=denoizer(hh_4x,hh_4y,finness_degree)
    hh_5y_new, hh_5x_new=denoizer(hh_5x,hh_5y,finness_degree)
    maximum_d_2= np.max([np.max(hh_x_new),np.max(hh_1x_new),np.max(hh_2x_new),np.max(hh_3x_new),np.max(hh_4x_new),np.max(hh_5x_new)])
    maximum_d_2= np.max(hh_3x_new)
    if normalization_coefficients[1]==0: 
        normalization_coefficients[1]=maximum_d_2
    im4 = ax1.plot(hh_y_new,\
                    hh_x_new/normalization_coefficients[1],'b'+markers,\
                    hh_1y_new,\
                    hh_1x_new/normalization_coefficients[1],'c'+markers,\
                    hh_2y_new,\
                    hh_2x_new/normalization_coefficients[1],'g'+markers,\
                    hh_3y_new,\
                    hh_3x_new/normalization_coefficients[1],'k'+markers,\
                    hh_4y_new,\
                    hh_4x_new/normalization_coefficients[1],'m'+markers,\
                    hh_5y_new,\
                    hh_5x_new/normalization_coefficients[1],'r'+markers)
    if prep>0:
        axe=[0,1]
    else:
        axe=[1,0]
    hh_y_new, hh_x_new=denoizer(y_edge_vert[0],np.sum(hh_,axis=axe[1]),finness_degree)
    hh_1y_new, hh_1x_new=denoizer(y_edge_vert[1],np.sum(hh_1,axis=axe[1]),finness_degree)
    hh_2y_new, hh_2x_new=denoizer(y_edge_vert[2],np.sum(hh_2,axis=axe[1]),finness_degree)
    hh_3y_new, hh_3x_new=denoizer(y_edge_vert[3],np.sum(hh_3,axis=axe[1]),finness_degree)
    hh_4y_new, hh_4x_new=denoizer(y_edge_vert[4],np.sum(hh_4,axis=axe[1]),finness_degree)
    hh_5y_new, hh_5x_new=denoizer(y_edge_vert[5],np.sum(hh_5,axis=axe[1]),finness_degree)
    maximum_s_1= np.max([np.max(hh_x_new),np.max(hh_1x_new),np.max(hh_2x_new),np.max(hh_3x_new),np.max(hh_4x_new),np.max(hh_5x_new)])
    maximum_s_1= np.max(hh_3x_new)
    if normalization_coefficients[2]==0: 
        normalization_coefficients[2]=maximum_s_1
    im5 = ax3.plot(hh_y_new,hh_x_new/normalization_coefficients[2],'b'+markers,\
                    hh_1y_new,hh_1x_new/normalization_coefficients[2],'c'+markers,\
                    hh_2y_new,hh_2x_new/normalization_coefficients[2],'g'+markers,\
                    hh_3y_new,hh_3x_new/normalization_coefficients[2],'k'+markers,\
                    hh_4y_new,hh_4x_new/normalization_coefficients[2],'m'+markers,\
                    hh_5y_new,hh_5x_new/normalization_coefficients[2],'r'+markers)

    hh_y_new, hh_x_new=denoizer(y_edge_hor[0],np.sum(hh_,axis=axe[0]),finness_degree)
    hh_1y_new, hh_1x_new=denoizer(y_edge_hor[1],np.sum(hh_1,axis=axe[0]),finness_degree)
    hh_2y_new, hh_2x_new=denoizer(y_edge_hor[2],np.sum(hh_2,axis=axe[0]),finness_degree)
    hh_3y_new, hh_3x_new=denoizer(y_edge_hor[3],np.sum(hh_3,axis=axe[0]),finness_degree)
    hh_4y_new, hh_4x_new=denoizer(y_edge_hor[4],np.sum(hh_4,axis=axe[0]),finness_degree)
    hh_5y_new, hh_5x_new=denoizer(y_edge_hor[5],np.sum(hh_5,axis=axe[0]),finness_degree)
    maximum_s_2= np.max([np.max(hh_x_new),np.max(hh_1x_new),np.max(hh_2x_new),np.max(hh_3x_new),np.max(hh_4x_new),np.max(hh_5x_new)])
    maximum_s_2= np.max(hh_3x_new)
    if normalization_coefficients[3]==0: 
        normalization_coefficients[3]=maximum_s_2
    im5 = ax4.plot(hh_y_new,hh_x_new/normalization_coefficients[3],'b'+markers,\
                    hh_1y_new,hh_1x_new/normalization_coefficients[3],'c'+markers,\
                    hh_2y_new,hh_2x_new/normalization_coefficients[3],'g'+markers,\
                    hh_3y_new,hh_3x_new/normalization_coefficients[3],'k'+markers,\
                    hh_4y_new,hh_4x_new/normalization_coefficients[3],'m'+markers,\
                    hh_5y_new,hh_5x_new/normalization_coefficients[3],'r'+markers)#range(len(h_2)),
    ax4.legend(['rocking pos 1','rocking pos 2','rocking pos 3','rocking pos 4','rocking pos 5','rocking pos 6'])
    return [maximum_d_1,maximum_d_2,maximum_s_1,maximum_s_2]
    #plt.show()

if __name__ == '__main__':
    im = Image.open('C:\\Users\\gx4419\Documents\\Python Scripts\\Magnifier_simulation\\experimental_data\\dcm2pitch_0p1860_0.tif')
    im.show()


def denoizer(hh_x,hh_y,finness_degree):
    norm_c=np.max(hh_y)/finness_degree
    hh_y_new=np.zeros(len(hh_y))
    hh_y_new[0]=hh_y[0]
    hh_x_new=np.zeros(len(hh_x))
    hh_x_new[0]=hh_x[0]
    i_new=0
    for i in range(2,len(hh_y)-2):
        if ((hh_y[i]-hh_y[i-1]<norm_c)&(hh_y[i+1]-hh_y[i]<norm_c)&(hh_y[i]-hh_y[i-1]>-norm_c)&(hh_y[i+1]-hh_y[i]>-norm_c)&(hh_y[i]-hh_y[i-2]>-norm_c)&(hh_y[i+2]-hh_y[i]>-norm_c)&(hh_y[i]-hh_y[i-2]<norm_c)&(hh_y[i+2]-hh_y[i]<norm_c)):
            hh_y_new[i_new]=hh_y[i]
            hh_x_new[i_new]=hh_x[i]
            i_new+=1
    hh_y_new=hh_y_new[:i_new]
    hh_x_new=hh_x_new[:i_new]
    return hh_x_new, hh_y_new
