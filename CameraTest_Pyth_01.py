"""
This program is a test of whether on-line or real-time inspection will work on the
bones.  We are looking for hangups or pancakes on the top plate. Start with static
jpeg first, then we can inspect mpg files.

By Rich Budek 01/12/2021 in Python 3.8
"""


import pandas as pd
import numpy as np
import jinja2
import math
import re
import cv2
import imutils
from imutils.video import count_frames
import time


#global values to start with

class Config_Data:
    disp_result_on_image = True
    disp_detect_win =  False
    disp_defect_win = False
    disp_movie_subwin = False      #for a movie, display all the subwindws like search box
    disp_movie_eval = True         #display all the sub screens for evaluate the movie 
    disp_contin = True             #change waitKey to a timed interval
    eval_use_edge_detect = False   #should use edge detection instead of fixed box, not working yet
    store_movie_frame = True
    movie_frame_num = 0
    movie_image_num = 0            #number of images saved
    is_fast_movie_split = False    #do we want to fast step thru movie and store

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

    file_fps = 30.0                 #frames per second

    lim_defect = 760                #number of defects (material) allowed for ejection


#search box is where the product and plate are expected.
#this will be replaced with edge finder once machine is cleaned up
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
    frame_time = 0.0
    frame_good = 0
    frame_bad = 0
    report_by = "Rich Budek"
    movie_time = 0.0
    curr_station_num = 0


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


class File_Name_Type:
    #for still images so can return everything in one spot
    name_short = ""
    name_full = ""
    path = ""


#create the file name string
def create_still_filename(config_data):
    output_rec = File_Name_Type()

    output_rec.path = config_data.filepath + "\\" + config_data.filepath_still
    output_rec.name_short = config_data.filename_still + str(config_data.movie_image_num) + ".jpg"
    output_rec.name_full = output_rec.path + "\\" + output_rec.name_short
    return output_rec



class Bad_Image_Rec:
    #records for bad images
    #stored in a table (array) of records
    rec_num = 0
    img_num = 0
    frame_num = 0
    frame_time = 0.0
    stat_num = 0
    filename_short = ""
    filename_full = ""
    tot_defects = 0


def append_badimg_rec_to_table(badrec_table, config_data, report_data, num_defects):
    #append to incoming table
    bad_image_rec = Bad_Image_Rec()
    if  badrec_table is None:
        bad_image_rec.rec_num = 0
        badrec_table = []
    else:
        bad_image_rec.rec_num = len(badrec_table) + 1
    bad_image_rec.img_num = config_data.movie_image_num
    bad_image_rec.frame_num = report_data.frame_tot_act
    bad_image_rec.frame_time = report_data.frame_time
    bad_image_rec.stat_num = report_data.curr_station_num
    filename_rec = create_still_filename(config_data)
    bad_image_rec.filename_full = filename_rec.name_full
    bad_image_rec.filename_short = filename_rec.name_short
    bad_image_rec.tot_defects = num_defects
    badrec_table.append(bad_image_rec)
    return



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

def draw_cross(image, x_cord, y_cord, color=(0,0,255), size=5, thickness=3 ):
    y_start = y_cord+size
    y_end = y_cord-size
    x_start = x_cord-size
    x_end = x_cord+size

    cv2.line(image , (x_cord,y_start) , (x_cord,y_end) , color , thickness)
    cv2.line(image , (x_start,y_cord) , (x_end,y_cord) , color , thickness)
    return


def eval_image(image, config_data, filename_short,sample_box,df_head,head_styler,is_movie, search_box, report_data, badrec_table):
    #looks at image to find out if good or bad
    #image needs to be loaded

    img_num_x_pix = image.shape[1]
    img_num_y_pix = image.shape[0]
    img_num_channel= image.shape[2]

    #not needed in eval function it's done outside this function
    #*** leave here and remove later  ***
    #col1 = pd.Index(['title', 'descrip', 'data file', 'size', 'date', 'pixel size', 'operator', 'search box', 'UL', 'LR'])
    #col2 = pd.Index(['Mini Twisted Bones', 'production run 2020', filename_short, '150 KB','07/20/2020', str(img_num_x_pix).strip()+' x '+str(img_num_y_pix).strip(), 'Rich Budek', ' ',  str(sample_box.X1_L).strip()+' , '+str(sample_box.Y1_L).strip(), str(sample_box.X2_R).strip()+' , '+str(sample_box.Y2_R).strip()])
    #df_head = pd.DataFrame(col2, columns=[''], index=col1)

    #head_styler = df_head.style.applymap(color_negative_red)

    if (not is_movie) or (is_movie and config_data.disp_movie_eval):
        cv2.imshow("Still Image",image)
    if not config_data.is_fast_movie_split:
        cv2.imwrite("pic_full.jpg",image)
    if (not is_movie) or (is_movie and config_data.disp_movie_eval):
        if config_data.disp_contin:
            cv2.waitKey(10)
        else:
            cv2.waitKey(0)


    #do edge detection
    img_blurred = np.copy(image)
    img_blurred = cv2.cvtColor( img_blurred, cv2.COLOR_BGR2GRAY )
    #img_blurred = cv2.GaussianBlur(img_blurred, (5,5), 0)
    if config_data.disp_detect_win:
        cv2.imshow("Blurred", img_blurred)
    if not config_data.is_fast_movie_split:
        cv2.imwrite("blurred_00.jpg",img_blurred)

    img_edge = cv2.Canny(img_blurred, 100, 200)
    if config_data.disp_detect_win:
        cv2.imshow("Canny", img_edge)
    if not config_data.is_fast_movie_split:
        cv2.imwrite("edge_00.jpg",img_edge)
    if config_data.disp_detect_win:
        cv2.waitKey(0)


    #find out the crop coordinates
    crop_X_L = min([sample_box.X1_L, sample_box.X2_L, sample_box.X1_R, sample_box.X2_R])
    crop_Y_L = min([sample_box.Y1_L, sample_box.Y2_L, sample_box.Y1_R, sample_box.Y2_R])
    crop_X_R = max([sample_box.X1_L, sample_box.X2_L, sample_box.X1_R, sample_box.X2_R])
    crop_Y_R = max([sample_box.Y1_L, sample_box.Y2_L, sample_box.Y1_R, sample_box.Y2_R])

    #find the top line crop
    l1_slope = ((sample_box.Y1_R - crop_Y_L) - (sample_box.Y1_L - crop_Y_L)) / ((sample_box.X1_R - crop_X_L) - (sample_box.X1_L - crop_X_L))
    l1_int = (sample_box.Y1_L - crop_Y_L) - l1_slope*(sample_box.X1_L - crop_X_L)

    #find the bottom line crop
    l2_slope = ((sample_box.Y2_R - crop_Y_L) - (sample_box.Y2_L - crop_Y_L)) / ((sample_box.X2_R - crop_X_L) - (sample_box.X2_L - crop_X_L))
    l2_int = (sample_box.Y2_L - crop_Y_L) - l1_slope*(sample_box.X2_L - crop_X_L)

    #crop image before drawing on it
    #top area first
    img_search = np.copy(image[crop_Y_L:crop_Y_R , crop_X_L:crop_X_R ])
    img_search_clean = np.copy(image[crop_Y_L:crop_Y_R , crop_X_L:crop_X_R ])

    x_val = 0
    x_chk = 0
    white = (255,255,255)
    for Xlp1 in range(img_search_clean.shape[1]) :
        x_chk = Xlp1
        #solve the line equation
        y_lp_end = int(l1_slope * x_chk + l1_int)
        for Ylp1 in range(0 , y_lp_end):
            #need to convert X & Y from big image (WC) to image
            img_search_clean[Ylp1,x_chk]=white

    for Xlp2 in range(img_search_clean.shape[1]) :
        x_chk = Xlp2
        #solve the line equation
        y_lp_start = int(l2_slope * x_chk + l2_int)
        y_lp_end = img_search_clean.shape[0]
        for Ylp in range(y_lp_start, y_lp_end):
            #need to convert X & Y from big image (WC) to image
            img_search_clean[Ylp,x_chk]=white


    #default colors
    red=(0,0,255)

    #draw cross
    draw_cross(image, search_box.orig_x1, search_box.orig_y1)
    draw_cross(image, search_box.orig_x2, search_box.orig_y2)

    #draw rectangle
    cv2.line(image , (sample_box.X1_L, sample_box.Y1_L) , (sample_box.X2_L, sample_box.Y2_L) , red , 3)
    cv2.line(image , (sample_box.X2_L, sample_box.Y2_L) , (sample_box.X2_R, sample_box.Y2_R) , red , 3)
    cv2.line(image , (sample_box.X2_R, sample_box.Y2_R) , (sample_box.X1_R, sample_box.Y1_R) , red , 3)
    cv2.line(image , (sample_box.X1_R, sample_box.Y1_R) , (sample_box.X1_L, sample_box.Y1_L) , red , 3)

    if (not is_movie) or (is_movie and config_data.disp_movie_eval):
        cv2.imshow("Still Image", image)
    if not config_data.is_fast_movie_split:
        cv2.imwrite("pic_w_search.jpg",image)
    if (not is_movie) or (is_movie and config_data.disp_movie_eval):
        if not config_data.disp_contin:
            cv2.waitKey(0)

    #display search area
    if config_data.disp_detect_win:
        cv2.imshow("Search Area", img_search)
    if not config_data.is_fast_movie_split:
        cv2.imwrite("pic_search.jpg",img_search)
    if config_data.disp_detect_win:
        cv2.waitKey(0)

    #display search area cleaned up
    if config_data.disp_detect_win:
        cv2.imshow("Search Area", img_search_clean)
    if not config_data.is_fast_movie_split:
        cv2.imwrite("pic_search_clean.jpg",img_search_clean)
    if config_data.disp_detect_win:
        cv2.waitKey(0)


    # do the detection
    bound_lwr = config_data.prod_bound_lwr    #data now config class
    bound_upr = config_data.prod_bound_upr

    #lower_bound = boundary
    lower = np.array(bound_lwr, dtype = "uint8")
    upper = np.array(bound_upr, dtype = "uint8")
    mask = cv2.inRange(img_search_clean, lower, upper)
    img_search_detect = cv2.bitwise_and(img_search_clean, img_search_clean, mask=mask)

    #display search area mask
    if config_data.disp_detect_win:
        cv2.imshow("Search Area", mask)
    #cv2.imwrite("pic_search_detect.jpg",img_search_detect)
    if config_data.disp_detect_win:
        cv2.waitKey(0)


    #display search area detect
    if config_data.disp_defect_win:
        cv2.imshow("Search Area", img_search_detect)
    if not config_data.is_fast_movie_split:
        cv2.imwrite("pic_search_detect.jpg",img_search_detect)
    if config_data.disp_defect_win:
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    #calculate number of defects
    total_pix = 0
    total_pix = cv2.countNonZero(mask)
    print ("Total defects = {}".format(total_pix))

    pos_x_text_frame = image.shape[1] - 400
    frame_str = "f = " + "{:d}".format(report_data.frame_tot_act)
    frame_time_str = "t = " + "{:.3f} sec".format(report_data.frame_time) 
    config_data.movie_frame_num = report_data.frame_tot_act
    print("string = " + frame_str)
    if config_data.disp_result_on_image:
        cv2.putText(image, frame_str, (pos_x_text_frame , 100),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        cv2.putText(image, frame_time_str, (pos_x_text_frame , 130),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    if (total_pix >= config_data.lim_defect):
        #defect found
        if config_data.disp_result_on_image:
            cv2.putText(image, "HANG UP", (10, 100),cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
            if (not is_movie) or (is_movie and config_data.disp_movie_eval):
                cv2.imshow("Still Image", image)
        filename_result = config_data.filepath + "\\" + "result_bad.jpg"        
        append_badimg_rec_to_table(badrec_table,config_data,report_data,total_pix)
        report_data.frame_bad = report_data.frame_bad +1
    else:
        if config_data.disp_result_on_image:
            cv2.putText(image, "GOOD", (10, 100),cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)
            if (not is_movie) or (is_movie and config_data.disp_movie_eval):
                cv2.imshow("Still Image", image)
        filename_result = config_data.filepath + "\\" + "result_good.jpg"
        report_data.frame_good = report_data.frame_good + 1

    img_result = cv2.imread(filename_result)
    if not config_data.is_fast_movie_split:
        cv2.imwrite("pic_result.jpg",img_result)
    
    if config_data.disp_result_on_image != True:
        cv2.imshow("RESULT",img_result)
    if (not is_movie) or (is_movie and config_data.disp_movie_eval):
        if config_data.disp_contin:
            cv2.waitKey(10)
        else:   
            cv2.waitKey(0)
    return






#main function 
def main():
    #this is the main loop
    config_data = Config_Data()
    sample_box = Sample_Box()
    m_box = Motion_Box()
    search_box = Search_Box()
    report_data = Report_Data()
    badrec_table = []   #all the bad images
  
    print("On-line visual inspection test program")
    print("rev. 01/12/2021  by Rich Budek")
    print("  ")

    is_movie = False
    type_file = int(input("1=no defect  2=defect  3=movie  "))
    if type_file == 1 :
        const_filename = config_data.filename1
    elif type_file == 2:
        const_filename = config_data.filename2
    elif type_file == 3:
        const_filename = "Top_Only.mp4"  #this is a movie
        is_movie = True

    filename_full = config_data.filepath + "\\" + const_filename
    print("Evaluate picture: " + filename_full)
    print(" ")

    if is_movie and config_data.is_fast_movie_split:
        #if it is a movie, and fast split is on, 
        #set all the proper settings
        #config_data.disp_result_on_image = True
        config_data.disp_detect_win =  False
        config_data.disp_defect_win = False
        config_data.disp_movie_subwin = False
        config_data.disp_movie_eval = False         #evaluate the movie 
        #config_data.eval_use_edge_detect = True     #maybe check on this one
        #config_data.store_movie_frame = True


    #now get the coordinates for sample box
    #hard coded for now
    sample_box.X1_L = search_box.sb_ul_x
    sample_box.Y1_L = search_box.sb_ul_y
    sample_box.X2_L = sample_box.X1_L   #no slope for now
    sample_box.Y2_L = sample_box.Y1_L + search_box.sb_ht

    #for the right side
    sample_box.X1_R = search_box.sb_ur_x
    sample_box.Y1_R = search_box.sb_ur_y
    sample_box.X2_R = sample_box.X1_R
    sample_box.Y2_R = sample_box.Y1_R + search_box.sb_ht

    print("coordinates for search box:")
    print(sample_box.X1_L , " , ", sample_box.Y1_L)
    print(sample_box.X2_L , " , ", sample_box.Y2_L)
    print(" ")
    print(sample_box.X1_R , " , ", sample_box.Y1_R)
    print(sample_box.X2_R , " , ", sample_box.Y2_R)

    
    # summary at the top
    filename_short = re.sub(r'\\.+\\', '', filename_full)
    filename_short = re.sub(r'^(.*:)', '', filename_short)

    if is_movie:
        #count estimated number of frames
        report_data.time_start = time.time()
        #report_data.frame_tot_est = count_frames(filename_full,override=True)
        report_data.frame_tot_est = 1043
        report_data.time_count_frame = time.time()
        report_data.frame_tot_act = 0
        report_data.frame_time = 0.0
        report_data.frame_bad = 0
        report_data.frame_good = 0

        vs = cv2.VideoCapture(filename_full)

        #need to run motion detection to find first frame
        #not implemented yet, so continue with firs
        first_frame = None

        num_mb_skip_frame = 0
        while True:
            ret,frame_in = vs.read()
            if (ret is None) or (frame_in is None) or (ret == False):
                break

            report_data.frame_tot_act = report_data.frame_tot_act + 1
            report_data.frame_time = round( report_data.frame_tot_act / config_data.file_fps, 5)
            num_mb_skip_frame = num_mb_skip_frame -1

            if not config_data.is_fast_movie_split:
                cv2.imshow("Full Frame",frame_in)
            if m_box.contin == False:
                if not config_data.is_fast_movie_split:
                    if config_data.disp_contin:
                        cv2.waitKey(10)
                    else:
                        cv2.waitKey(0)

            frame = np.copy(frame_in[m_box.y1:m_box.y2 , m_box.x1:m_box.x2 ])

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #orig was (21,21)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            #if first frame is None, initialize it
            if first_frame is None:
                first_frame = gray
                continue
            if config_data.disp_movie_subwin:
                cv2.imshow("Frame",gray)
            #cv2.waitKey(0)

            #difference between previous frame and this one
            frame_delta = cv2.absdiff(first_frame, gray)
            if config_data.disp_movie_subwin:
                cv2.imshow("Delta",frame_delta)
            thresh = cv2.threshold(frame_delta, 25,255, cv2.THRESH_BINARY)[1]

            #dilate then find countours
            thresh = cv2.dilate(thresh, None, iterations=2)
            if config_data.disp_movie_subwin:
                cv2.imshow("Thresh",thresh)

            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            #count how many contours are different
            num_cnts = 0
            still_max_cnts = 6    #number of contours allowed for being 
            min_area = 300
            for c in cnts:
                if cv2.contourArea(c) > min_area:
                    num_cnts = num_cnts +1

            #see if in motion
            print("num contours = " +  str(num_cnts))
            first_frame = np.copy(gray)

            if num_cnts > still_max_cnts:
                cv2.putText(gray, "*** moving  ****", (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if config_data.disp_movie_subwin:
                    cv2.imshow("Frame",gray)
                cv2.putText(frame_in, "*** moving  ****", (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                print ("motion")
                if not config_data.is_fast_movie_split:
                    cv2.imshow("Full Frame",frame_in)
            else:
                #no motion so stop
                   
                #check if there are frames to skip
                if num_mb_skip_frame <= 0:
                    image = np.copy(frame_in)
                    if config_data.movie_image_num < 2:
                        #only do it for the first few frame(s) info stays the same after
                        img_num_x_pix = image.shape[1]
                        img_num_y_pix = image.shape[0]
                        img_num_channel= image.shape[2]
                        col1 = pd.Index(['title', 'descrip', 'data file', 'size', 'date', 'pixel size', 'operator', 'search box', 'UL', 'LR'])
                        col2 = pd.Index(['Mini Twisted Bones', 'production run 2020', filename_short, '150 KB','07/20/2020', str(img_num_x_pix).strip()+' x '+str(img_num_y_pix).strip(), 'Rich Budek', ' ',  str(sample_box.X1_L).strip()+' , '+str(sample_box.Y1_L).strip(), str(sample_box.X2_R).strip()+' , '+str(sample_box.Y2_R).strip()])
                        df_head = pd.DataFrame(col2, columns=[''], index=col1)
                        head_styler = df_head.style.applymap(color_negative_red)

                    eval_image(image, config_data, filename_short, sample_box, df_head, head_styler, True, search_box, report_data, badrec_table)

                    cv2.putText(frame_in, "STOPPED", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    if not config_data.is_fast_movie_split:
                        cv2.imshow("Full Frame",frame_in)
                    if not config_data.is_fast_movie_split:
                        if config_data.disp_contin:
                            cv2.waitKey(10)
                        else:
                            cv2.waitKey(0)

                    num_mb_skip_frame = m_box.skip_frame
                    if config_data.store_movie_frame:
                        fullfilename_rec = create_still_filename(config_data)
                        fullfilename_still = fullfilename_rec.name_full
                        print(fullfilename_still)
                        cv2.imwrite(fullfilename_still,image)

                    config_data.movie_image_num = config_data.movie_image_num + 1                        
            if not config_data.is_fast_movie_split:
                cv2.waitKey(10)

        if not config_data.is_fast_movie_split:
            cv2.waitKey(0)

        report_data.time_end = time.time()  #int(round(time.time() * 1000))
        report_data.time_elap_tot = (report_data.time_end -  report_data.time_start)
        report_data.time_elap_count_frame = (report_data.time_count_frame - report_data.time_start)
        report_data.time_elap_eval = (report_data.time_end - report_data.time_count_frame)
        report_data.time_frame_ms = ((report_data.time_elap_eval) / (report_data.frame_tot_act))*1000.0
    else:
        #load the image
        image = cv2.imread(filename_full)

        img_num_x_pix = image.shape[1]
        img_num_y_pix = image.shape[0]
        img_num_channel= image.shape[2]
        col1 = pd.Index(['title', 'descrip', 'data file', 'size', 'date', 'pixel size', 'operator', 'search box', 'UL', 'LR'])
        col2 = pd.Index(['Mini Twisted Bones', 'production run 2020', filename_short, '150 KB','07/20/2020', str(img_num_x_pix).strip()+' x '+str(img_num_y_pix).strip(), 'Rich Budek', ' ',  str(sample_box.X1_L).strip()+' , '+str(sample_box.Y1_L).strip(), str(sample_box.X2_R).strip()+' , '+str(sample_box.Y2_R).strip()])
        df_head = pd.DataFrame(col2, columns=[''], index=col1)
        head_styler = df_head.style.applymap(color_negative_red)
        eval_image(image, config_data, filename_short, sample_box, df_head, head_styler, False, search_box, report_data, badrec_table)


    print("Evaluation is done ...")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    if is_movie:
        #print to screen
        print(" ")
        print("STATS REPORT:")
        print("frames actual = {:,d}".format(report_data.frame_tot_act))
        print("frames estim  = {:,d}".format(report_data.frame_tot_est))
        print("stations bad  = {:,d}".format(report_data.frame_bad))
        print("stations good = {:,d}".format(report_data.frame_good))

        print(" ")
        print("time total        = {:15,.3f}".format(report_data.time_elap_tot))
        print("time count frames = {:15,.3f}".format(report_data.time_elap_count_frame))
        print("time eval         = {:15,.3f}".format(report_data.time_elap_eval))
        print("time per frame    = {:15,.1f}(mS)".format(report_data.time_frame_ms))

        #bad images found
        print(" ")
        for bad_rec in badrec_table:
            outstring = str(bad_rec.rec_num) + " " + str(bad_rec.img_num) + " " + str(bad_rec.frame_num) + " " + str(bad_rec.tot_defects) + " " + bad_rec.filename_short
            print(outstring)
        print ("Total = " + str(len(badrec_table)) )

        #create movie report
        img_num_x_pix = image.shape[1]
        img_num_y_pix = image.shape[0]
        img_num_channel= image.shape[2]

        #the header
        col1 = pd.Index(['title', 'descrip', 'data file', 'size', 'date', 'pixel size',\
           'operator', \
           'search box', 'UL', 'LR', \
           'motion box', 'UL ', 'LR '])
        col2 = pd.Index(['Mini Twisted Bones', 'production run 2020', filename_short, '150 KB','07/20/2020', str(img_num_x_pix).strip()+' x '+str(img_num_y_pix).strip(), \
            'Rich Budek', \
            ' ',  str(sample_box.X1_L).strip()+' , '+str(sample_box.Y1_L).strip(), str(sample_box.X2_R).strip()+' , '+str(sample_box.Y2_R).strip(), \
            ' ',  str(m_box.x1).strip()+' , '+str(m_box.y1).strip(), str(m_box.x2).strip()+' , '+str(m_box.y2).strip(), \
           ])
        df_head = pd.DataFrame(col2, columns=[''], index=col1)
        head_styler = df_head.style.applymap(color_negative_red)

        col1a = ['title', 'descrip', 'data file', 'size', 'date', 'pixel size',\
           'operator', \
           'search box', 'UL', 'LR', \
           'motion box', 'UL', 'LR ']
        col2a = ['Mini Twisted Bones', 'production run 2020', filename_short, '150 KB','07/20/2020', str(img_num_x_pix).strip()+' x '+str(img_num_y_pix).strip(), \
            'Rich Budek', \
            ' ',  str(sample_box.X1_L).strip()+' , '+str(sample_box.Y1_L).strip(), str(sample_box.X2_R).strip()+' , '+str(sample_box.Y2_R).strip(), \
            ' ',  str(m_box.x1).strip()+' , '+str(m_box.y1).strip(), str(m_box.x2).strip()+' , '+str(m_box.y2).strip(), \
           ]

        print("did the header table")
        #df_head = pd.DataFrame({'1':col1a, '2':col2a}, columns=[''])
        #head_styler = df_head.style.applymap(color_negative_red)
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
        results_styler = df_results2.style.applymap(color_negative_red)


        #now the bad pictures found table
        col1_c = [rec.rec_num for rec in badrec_table]
        col2_c = [rec.img_num for rec in badrec_table]
        col3_c = [round(rec.frame_num,3) for rec in badrec_table]
        col4_c = ["{:.3f}".format(rec.frame_time) for rec in badrec_table]
        col5_c = [rec.tot_defects for rec in badrec_table]
        col6_c = [rec.filename_short for rec in badrec_table]

        df_badimg = pd.DataFrame({'image':col2_c, 'frame':col3_c, 'time':col4_c, 'defects':col5_c, 'file name':col6_c}, index=col1_c)
        badimg_styler = df_badimg.style.applymap(color_negative_red)

        #table of image names
        col1_d = [rec.rec_num for rec in badrec_table]
        col2_d = [rec.filename_full for rec in badrec_table]
        col3_d = [round(rec.frame_num,3) for rec in badrec_table]
        col4_d = ["{:.3f}".format(rec.frame_time) for rec in badrec_table]

        images_table = []
        for i in range(len(col2_d)):
            images_rec = dict(filename_full=col2_d[i], frame=col3_d[i], time=col4_d[i])
            images_table.append(images_rec)

        #df_images not used at the moment.  Tested it 
        df_images = pd.DataFrame({'recnum':col1_d, 'filename_full':col2_d,'frame':col3_d, 'time':col4_d})

        images_styler = df_images.style.applymap(color_negative_red)

        #write the report for the movie to disk
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=''))
        template = env.get_template('template_mov_01.html')
        html = template.render(head_table=head_styler.render(), results_table=results_styler.render(), badimg_table=badimg_styler.render(), images_table=images_table )

        # write the HTML file
        with open('report-movie.html', 'w', encoding='utf-8') as f:   #put encoding on to stop unicode errors
            f.write(html)
        f.close()
    else:
        #write the report for image to disk    
        # template handling
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=''))
        template = env.get_template('template_01.html')
        html = template.render(head_table=head_styler.render() )

        # write the HTML file
        with open('report.html', 'w', encoding='utf-8') as f:   #put encoding on to stop unicode errors
            f.write(html)
        f.close()



print (" ")
print (".Program start.")

if __name__ == "__main__":
    main()
    print (".Program end.")


     