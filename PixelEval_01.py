"""
This program to evaluate pixels for their color contents.  Will be used 
for defect detection and edge detection.

By Rich Budek 01/15/2021 in Python 3.8
"""


import pandas as pd
import numpy as np
import jinja2
import math
import re
import cv2
import imutils
from matplotlib import pyplot as plt


#global values to start with
const_orig_x1 = 16
const_orig_y1 = 161
const_orig_x2 = 847
const_orig_y2 = 200

const_mb_x1 = 30
const_mb_y1 = 307
const_mb_x2 = 817
const_mb_y2 = 504
const_mb_contin = True
const_mb_skip_frame = 23  #num of frames to skip after motion stop

const_sb_ht = 71
const_sb_ul_x = 17
const_sb_ul_y = 550
const_sb_ur_x = 847
const_sb_ur_y = 504
const_img_x_pix = 1280
const_img_y_pix = 720

const_lim_defect = 500    #


class Config_Data:
    disp_result_on_image = True
    disp_detect_win =  False
    disp_defect_win = False
    disp_movie_subwin = False
    disp_movie_eval = False       #evaluate the movie 
    eval_use_edge_detect = True
    store_movie_frame = True
    movie_frame_num = 0

    #for the product detection
    prod_bound_lwr = [45, 92, 115] # in B, G, R format
    prod_bound_upr = [105, 132, 165]

    mach_bound_lwr = [180, 185, 180]  # in B, G, R format
    mach_bound_upr = [250, 250, 250]  # orig was 240, 239, 238

    filepath = "Z:\Shared Folders\Data_WCO\BHPB VisInsp OnSite\Samp Data"
    filename1 = "NoDefect_01.jpg"
    filename2 = "Defect_01.jpg"
    filename_test = "test_plate_01.jpg"



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

def draw_cross(image, x_cord, y_cord, color=(0,0,255), size=5, thickness=3 ):
    y_start = y_cord+size
    y_end = y_cord-size
    x_start = x_cord-size
    x_end = x_cord+size

    cv2.line(image , (x_cord,y_start) , (x_cord,y_end) , color , thickness)
    cv2.line(image , (x_start,y_cord) , (x_end,y_cord) , color , thickness)
    return


def eval_image(image, config_data, filename_short,sample_box,df_head,head_styler,is_movie):
    #looks at image to find out if good or bad
    #image needs to be loaded

    img_num_x_pix = image.shape[1]
    img_num_y_pix = image.shape[0]
    img_num_channel= image.shape[2]
    print ("channels: {}", img_num_channel)

    col1 = pd.Index(['title', 'descrip', 'data file', 'size', 'date', 'pixel size', 'operator', 'search box', 'UL', 'LR'])
    col2 = pd.Index(['Mini Twisted Bones', 'production run 2020', filename_short, '150 KB','07/20/2020', str(img_num_x_pix).strip()+' x '+str(img_num_y_pix).strip(), 'Rich Budek', ' ',  str(sample_box.X1_L).strip()+' , '+str(sample_box.Y1_L).strip(), str(sample_box.X2_R).strip()+' , '+str(sample_box.Y2_R).strip()])
    df_head = pd.DataFrame(col2, columns=[''], index=col1)

    head_styler = df_head.style.applymap(color_negative_red)

    if (not is_movie) or (is_movie and config_data.disp_movie_eval):
        cv2.imshow("Still Image",image)
    cv2.imwrite("pic_full.jpg",image)
    if (not is_movie) or (is_movie and config_data.disp_movie_eval):
        cv2.waitKey(0)


    #do edge detection
    img_blurred = np.copy(image)
    img_blurred = cv2.cvtColor( img_blurred, cv2.COLOR_BGR2GRAY )
    #img_blurred = cv2.GaussianBlur(img_blurred, (5,5), 0)
    if config_data.disp_detect_win:
        cv2.imshow("Blurred", img_blurred)
    cv2.imwrite("blurred_00.jpg",img_blurred)

    img_edge = cv2.Canny(img_blurred, 100, 200)
    if config_data.disp_detect_win:
        cv2.imshow("Canny", img_edge)
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
    draw_cross(image, const_orig_x1, const_orig_y1)
    draw_cross(image, const_orig_x2, const_orig_y2)

    #draw rectangle
    cv2.line(image , (sample_box.X1_L, sample_box.Y1_L) , (sample_box.X2_L, sample_box.Y2_L) , red , 3)
    cv2.line(image , (sample_box.X2_L, sample_box.Y2_L) , (sample_box.X2_R, sample_box.Y2_R) , red , 3)
    cv2.line(image , (sample_box.X2_R, sample_box.Y2_R) , (sample_box.X1_R, sample_box.Y1_R) , red , 3)
    cv2.line(image , (sample_box.X1_R, sample_box.Y1_R) , (sample_box.X1_L, sample_box.Y1_L) , red , 3)

    if (not is_movie) or (is_movie and config_data.disp_movie_eval):
        cv2.imshow("Still Image", image)
    cv2.imwrite("pic_w_search.jpg",image)
    if (not is_movie) or (is_movie and config_data.disp_movie_eval):
        cv2.waitKey(0)

    #display search area
    if config_data.disp_detect_win:
        cv2.imshow("Search Area", img_search)
    cv2.imwrite("pic_search.jpg",img_search)
    if config_data.disp_detect_win:
        cv2.waitKey(0)

    #display search area cleaned up
    if config_data.disp_detect_win:
        cv2.imshow("Search Area", img_search_clean)
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
    cv2.imwrite("pic_search_detect.jpg",img_search_detect)
    if config_data.disp_defect_win:
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    #calculate number of defects
    total_pix = 0
    total_pix = cv2.countNonZero(mask)
    print ("Total defects = {}".format(total_pix))

    if (total_pix >= const_lim_defect):
        if config_data.disp_result_on_image:
            cv2.putText(image, "HANG UP", (10, 100),cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
            if (not is_movie) or (is_movie and config_data.disp_movie_eval):
                cv2.imshow("Still Image", image)
        filename_result = config_data.filepath + "\\" + "result_bad.jpg"
    else:
        if config_data.disp_result_on_image:
            cv2.putText(image, "GOOD", (10, 100),cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)
            if (not is_movie) or (is_movie and config_data.disp_movie_eval):
                cv2.imshow("Still Image", image)
        filename_result = config_data.filepath + "\\" + "result_good.jpg"

    img_result = cv2.imread(filename_result)
    cv2.imwrite("pic_result.jpg",img_result)
    
    if config_data.disp_result_on_image != True:
        cv2.imshow("RESULT",img_result)
    if (not is_movie) or (is_movie and config_data.disp_movie_eval):
        cv2.waitKey(0)
    return









#main function 
def main():
    #this is the main loop
    config_data = Config_Data()
    sample_box = Sample_Box()
    m_box = Motion_Box()
  
    print("Pixel Test Evaluation")
    print("rev. 01/15/2021  by Rich Budek")
    print("  ")

    is_movie = False
    const_filename = config_data.filename_test

    print(is_movie)
    filename_full = config_data.filepath + "\\" + const_filename
    print("Evaluate picture: " + filename_full)
    print(" ")

    #now get the coordinates for sample box
    #hard coded for now
    sample_box.X1_L = const_sb_ul_x
    sample_box.Y1_L = const_sb_ul_y
    sample_box.X2_L = sample_box.X1_L   #no slope for now
    sample_box.Y2_L = sample_box.Y1_L + const_sb_ht

    #for the right side
    sample_box.X1_R = const_sb_ur_x
    sample_box.Y1_R = const_sb_ur_y
    sample_box.X2_R = sample_box.X1_R
    sample_box.Y2_R = sample_box.Y1_R + const_sb_ht

    
    # summary at the top
    filename_short = re.sub(r'\\.+\\', '', filename_full)
    filename_short = re.sub(r'^(.*:)', '', filename_short)


    ##histograms
    #img = cv2.imread(filename_full)
    #cv2.imshow("Full",img)


    #color = ('b', 'g', 'r')
    #for i,col in enumerate(color):
    #    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    #    plt.plot(histr,color = col)
    #    plt.xlim([0,256])
    #plt.show()
    #cv2.waitKey(0)


    cv2.destroyAllWindows()
    filename_full = config_data.filepath + "\\" + config_data.filename1
    image_in = cv2.imread(filename_full)
    cv2.imshow("Original Image",image_in)
    cv2.waitKey(0)

    image = np.copy(image_in[141:520,0:900,:])
    cv2.imshow("Cropped Image",image)
    cv2.waitKey(0)
    


    ## do the detection
    #bound_lwr = config_data.mach_bound_lwr    #data now config class
    #bound_upr = config_data.mach_bound_upr

    ##lower_bound = boundary
    #lower = np.array(bound_lwr, dtype = "uint8")
    #upper = np.array(bound_upr, dtype = "uint8")
    #mask = cv2.inRange(image, lower, upper)
    #img_masked = cv2.bitwise_and(image, image, mask=mask)
    #cv2.imshow("Color mask", img_masked)

    #cv2.waitKey(0)

    ##now do the edge detection

    ##do edge detection
    img_blurred = np.copy(image)
    img_blurred = cv2.cvtColor( img_blurred, cv2.COLOR_BGR2GRAY )
    img_gray = np.copy(img_blurred)
    img_blurred = cv2.GaussianBlur(img_blurred, (5,5), 0)  #was (5,5)
    #cv2.imshow("Blurred", img_blurred) 

    #img_edge = cv2.Canny(img_blurred, 1, 3)  # docs say 100,200  mine was  30,150
    #cv2.imshow("Canny", img_edge)

    #cv2.waitKey(0)


    #try auto tuned canny
    sigma = 0.01
    img_avg = np.median(img_blurred)
    lower = int(max(0, (1.0 - sigma) * img_avg))
    upper = int(min(255, (1.0 + sigma) * img_avg))
    img_edged = cv2.Canny(img_blurred, lower, upper, L2gradient=True)
    cv2.imshow("Auto Canny", img_edged)
    cv2.waitKey(0)


    #dilate then find countours
    thresh = cv2.threshold(img_edged, 25,255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Threshold", thresh)
    cv2.waitKey(0)

    thresh = cv2.dilate(thresh, None, iterations=4)  #orig was 2
    cv2.imshow("Dialate",thresh)
    cv2.waitKey(0)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #count how many contours are different
    num_cnts = 0
    still_max_cnts = 6    #number of contours allowed for being 
    min_area = 300
    for c in cnts:
        if cv2.contourArea(c) > min_area:
            num_cnts = num_cnts +1

    print("num contours = " +  str(num_cnts))


    return


    if is_movie:
        #need to run motion detection to find first frame
        vs = cv2.VideoCapture(filename_full)
        first_frame = None

        num_mb_skip_frame = 0
        while True:
            ret,frame_in = vs.read()
            if ret == True:
                #if frame_in is None:
                #    print("end of file")
                #    break
                num_mb_skip_frame = num_mb_skip_frame -1

                cv2.imshow("Full Frame",frame_in)
                if m_box.contin == False:
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
                    cv2.imshow("Full Frame",frame_in)
                else:
                    #no motion so stop
                   
                    #check if there are frames to skip
                    if num_mb_skip_frame <= 0:
                        image = np.copy(frame_in)
                        img_num_x_pix = image.shape[1]
                        img_num_y_pix = image.shape[0]
                        img_num_channel= image.shape[2]
                        col1 = pd.Index(['title', 'descrip', 'data file', 'size', 'date', 'pixel size', 'operator', 'search box', 'UL', 'LR'])
                        col2 = pd.Index(['Mini Twisted Bones', 'production run 2020', filename_short, '150 KB','07/20/2020', str(img_num_x_pix).strip()+' x '+str(img_num_y_pix).strip(), 'Rich Budek', ' ',  str(sample_box.X1_L).strip()+' , '+str(sample_box.Y1_L).strip(), str(sample_box.X2_R).strip()+' , '+str(sample_box.Y2_R).strip()])
                        df_head = pd.DataFrame(col2, columns=[''], index=col1)
                        head_styler = df_head.style.applymap(color_negative_red)
                        eval_image(image, config_data, filename_short, sample_box, df_head, head_styler, True)

                        cv2.putText(frame_in, "STOPPED", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        cv2.imshow("Full Frame",frame_in)
                        cv2.waitKey(0)
                        num_mb_skip_frame = m_box.skip_frame

                cv2.waitKey(10)
        cv2.waitKey(0)
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
        eval_image(image, config_data, filename_short, sample_box, df_head, head_styler, False)


    print("Image is done ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    #write the report to disk    
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


     