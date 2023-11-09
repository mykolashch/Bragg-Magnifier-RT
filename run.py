from RayTracer_LabSys_4crystals import open_tif_image, show_image_contours, rocking_positions_mono2, rocking_current_2, histograms_for_comparison

if __name__ == "__main__":
    
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
    show_image_contours(histograms_for_comparison,[hh_5,hh_4,hh_3,hh_2,hh_1,hh_],steps*57.296,0,1)
    show_image_contours(histograms_for_comparison,[hh_5,hh_4,hh_3,hh_2,hh_1,hh_],steps*57.296,1,1)
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