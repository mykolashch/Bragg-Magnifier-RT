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
            
            blurred2 = gaussian_filter(masked_imarray2, sigma=7)#12#3
            blurred = gaussian_filter(imarray, sigma=7)#20
            array2_max=np.max(blurred2)
            array2_min=1
            #mask[(np.where(blurred2>0.6*array2_max))]=1
            
            blurred2 = np.ma.MaskedArray(blurred2, mask = mask)
            print(np.shape(mask))
            if swap_anything:
                imarray2=np.rot90(np.fliplr(imarray2))
                blurred=np.flipud(blurred)
                blurred2=np.rot90(np.fliplr(blurred2))
            axs[j_num].imshow(imarray2,vmin=np.min(imarray2),vmax=np.max(imarray2), interpolation='none',cmap='binary', aspect='auto')#vmin=np.min(imarray),vmax=np.max(imarray)##cmap='jet'
            
            axs[j_num].xaxis.set_major_formatter(plt.FuncFormatter(format_func))#set_major_locator(plt.MultipleLocator(250))
            axs[j_num].yaxis.set_major_formatter(plt.FuncFormatter(format_func))
            #axs[i_num][j_num].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f mm'))
            step_2= (np.max(blurred2)-np.min(blurred2))/5
            step_2_small=(np.max(blurred2)-np.min(blurred2)-3*step_2)/4
            levels_step=np.linspace(np.min(blurred2),np.max(blurred2),4)#12#levels_step=np.linspace(np.min(blurred2),np.max(blurred2)-3*step_2,4)#12
            contur1=axs[j_num].contour(blurred2, levels_step,cmap='autumn',aspect='auto',linewidths=1.5,linestyles='dashdot')#colors='black',aspect='auto')
            step_= (np.max(blurred)-np.min(blurred))/5
            step_small= (np.max(blurred)-np.min(blurred)-4*step_)/4
            
            levels_step=np.linspace(1*np.min(blurred),1*np.max(blurred),4)#levels_step=np.linspace(1*np.min(blurred),1*np.max(blurred)-3*step_,4)#0.5*np.max(blurred)#np.max(blurred)-4.5*step_
            
            contur2=axs[j_num].contour(blurred,levels_step,cmap='cool',aspect='auto',linewidths=2.5)# np.flipud(np.rot90(np.fliplr(blurred))#colors='red',aspect='auto')#np.rot90(blurred,k=3)
            axs[j_num].set_title('DCM2_pitch =  DCM2_pitch$_0$ {0:+.1e} $^\circ$'.format( rocking_angle[ii]), fontsize=14)
            if int(ii+1-len(imarray2_)/2)>0:
                plt.xlabel('mm', fontsize=14)
           
    plt.show()
