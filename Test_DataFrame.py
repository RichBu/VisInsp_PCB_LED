
import pandas as pd
import numpy as np
import jinja2
import math
import re
import cv2
import imutils
from imutils.video import count_frames
import time


class Config_Data:
    disp_result_on_image = True
    disp_detect_win =  True
    disp_defect_win = True
    disp_movie_subwin = True
    disp_movie_eval = True         #evaluate the movie 
    eval_use_edge_detect = True    #should use edge detection instead of fixed box, not working yet
    store_movie_frame = True
    movie_frame_num = 0 
    is_fast_movie_split = False      #do we want to fast step thru movie and store

    #for the product detection
    prod_bound_lwr = [45, 92, 115]  # in B, G, R format
    prod_bound_upr = [105, 132, 165]

    mach_bound_lwr = [45, 92, 115]  # in B, G, R format
    mach_bound_upr = [105, 132, 165]

    filepath = "Z:\Shared Folders\Data_WCO\BHPB VisInsp OnSite\Samp Data"
    filename1 = "NoDefect_01.jpg"
    filename2 = "Defect_01.jpg"
    filename_still = "Still_"       #name to pre-append to movie stills
    filepath_still = "stills"


class Report_Data:
    job_name = "Test of Rotary Machine"
    time_start = 0.0
    time_count_frame = 0.0
    time_end = 0.0
    time_eval = 0.0
    time_elap_count_frame = 0.0
    time_elap_eval = 0.0
    time_elap_tot = 0.0
    time_frame_ms = 0.0
    frame_tot_est = 0
    frame_tot_act = 0
    frame_good = 0
    frame_bad = 0
    report_by = "Rich Budek"
    movie_time = 0.0



class Search_Box:
    orig_x1 = 16
    orig_y1 = 161
    orig_x2 = 735
    orig_y2 = 185

    sb_ht = 126       # was 71
    sb_ul_x = 11      # was 17
    sb_ul_y = 505     # was 550
    sb_ur_x = 735     # was 847
    sb_ur_y = 454     # was 504

const_lim_defect = 760    #



class Motion_Box:
    x1 = 30
    y1 = 307
    x2 = 817
    y2 = 504
    contin = True
    skip_frame = 23  #num of frames to skip after motion stop

  

class Sample_Box:
    X1_L = 0
    Y1_L = 0
    X2_L = 0
    Y2_L = 0
    X1_R = 0
    Y1_R = 0
    X2_R = 0
    Y2_R = 0



#funtions for color styling
def color_negative_red(val):
    #color = 'red' if val < 0 else 'black'
    color = 'black'
    return f'color: {color}'


def color_spec_dif_01(val):
    color = 'red' if float(val) > 0.001 else 'black'
    return f'color: {color}'

def color_spec_dif_02(val):    #warning if above 0.002
    color = 'blue' if float(val) > 0.002 or float(val) < -0.002 else 'black'
    return f'color: {color}'

def color_spec_dif_03(val):    #red if above 0.002
    color = 'red' if float(val) > 0.002 or float(val) < -0.002 else 'black'
    return f'color: {color}'

def color_spec_dia(val):
    color = 'red' if float(val) > 0.4910 or float(val) < 0.4900 else 'black'
    return f'color: {color}'




#main function 
def main():
    #this is the main loop
    config_data = Config_Data()
    report_data = Report_Data()

    sample_box = Sample_Box()
    m_box = Motion_Box()
    search_box = Search_Box()

    filename_short = ""
    img_num_x_pix = 0
    img_num_y_pix = 0

    col1 = pd.Index(['title', 'descrip', 'data file', 'size', 'date', 'pixel size', 'operator', 'search box', 'UL', 'LR'])
    col2 = pd.Index(['Mini Twisted Bones', 'production run 2020', filename_short, '150 KB','07/20/2020', str(img_num_x_pix).strip()+' x '+str(img_num_y_pix).strip(), 'Rich Budek', ' ',  str(sample_box.X1_L).strip()+' , '+str(sample_box.Y1_L).strip(), str(sample_box.X2_R).strip()+' , '+str(sample_box.Y2_R).strip()])
    df_head = pd.DataFrame(col2, columns=[''], index=col1)
    head_styler = df_head.style.applymap(color_negative_red)


    #col3 = pd.Index(['STATS REPORT:', ' ', ' ', ' ', ' ', ' ', ' ', 'TIMING:', ' ', ' ', ' ', ' ', ' '])
    #col4 = pd.Index([' ', "frames actual = {:,d}".format(report_data.frame_tot_act), "frames estim  = {:,d}".format(report_data.frame_tot_est), ' ',\
    #    "stations total  = {:,d}".format(report_data.frame_bad+report_data.frame_good), "stations bad  = {:,d}".format(report_data.frame_bad), \
    #    "stations good  = {:,d}".format(report_data.frame_good), ' ', "time total        = {:15,.3f} secs".format(report_data.time_elap_tot), \
    #    "time count frames = {:15,.3f}".format(report_data.time_elap_count_frame), "time eval         = {:15,.3f} secs".format(report_data.time_elap_eval), \
    #    "time per frame    = {:15,.1f}(mS)".format(report_data.time_frame_ms), "for 30fps movie, must be 1/30 sec (33.33ms) or less"])


    #col3 = ['FRAMES:', ' ', ' ', ' ', ' ', ' ', ' ', 'TIMES:', ' ', ' ', ' ', ' ', ' ']
    col3 = ['FRAMES:', '  ', '   ', '    ', '     ', '      ', 'TIMES:', '       ', '        ', '         ', '          ']
    col4 = [' ', "frames actual = ", "frames estim  = ",\
        "w/ stations total  = ", "w/ stations bad  = ", \
        "w/ stations good  = ", ' ', "time total        = ", \
        "time count frames = ", "time eval         = ", \
        "time per frame    = "]
    col5 = [' ', "{:,d}".format(report_data.frame_tot_act), "{:,d}".format(report_data.frame_tot_est),\
        "{:,d}".format(report_data.frame_bad+report_data.frame_good), "{:,d}".format(report_data.frame_bad), \
        "{:,d}".format(report_data.frame_good), ' ', "{:15,.3f} secs".format(report_data.time_elap_tot), \
        "{:15,.3f} secs".format(report_data.time_elap_count_frame), "{:15,.3f} secs".format(report_data.time_elap_eval), \
        "{:15,.1f}(ms)".format(report_data.time_frame_ms)]

    df_results2 = pd.DataFrame({' ':col4, '  ':col5}, index=col3)


    #df_results = pd.DataFrame(col3, columns=[''], index=col4)

    results_styler = df_results2.style.applymap(color_negative_red)

    #write the report for the movie to disk
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=''))
    template = env.get_template('template_mov_01.html')
    html = template.render(head_table=head_styler.render(), results_table=results_styler.render() )
    #html = template.render(head_table=head_styler.render(), result_table=results_styler.render() )

    # write the HTML file
    with open('report-movie.html', 'w', encoding='utf-8') as f:   #put encoding on to stop unicode errors
        f.write(html)
    f.close()



print (" ")
print (".Program start.")

if __name__ == "__main__":
    main()
    print (".Program end.")

