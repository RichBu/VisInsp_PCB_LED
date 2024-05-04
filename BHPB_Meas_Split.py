
#this program splits the BEFORE reaming and AFTER reaming into separate files

import pandas as pd
import numpy as np
import jinja2
import math
import re


# open the file
datafname = "Z:\Shared Folders\Data_WCO\BHPB Meas\Large_Top_737_754.out"
datafname_short = re.sub(r'\\.+\\', '', datafname)
datafname_short = re.sub(r'^(.*:)', '', datafname_short)

datafname_out1 = re.sub('.out', '', datafname_short) + "_B" + ".out"
datafname_out2 = re.sub('.out', '', datafname_short) + "_A" + ".out"


fin = open(datafname,"r")
fout1 = open(datafname_out1, 'w', encoding='utf-8')
fout2 = open(datafname_out2, 'w', encoding='utf-8')

after_found = 0
outfile = 3

rdplate_line = 0  #reading aline on a plate

for line_in in fin:
    line_out1 = ""
    line_out2 = ""
    if line_in.find("BEFORE REAMING")>=0:
        outfile = 1
        line_out1 = ""
        line_out2 = ""
    elif line_in.find("AFTER")>=0:
        outfile = 2
        line_out1 = ""
        line_out2 = ""
    elif not line_in.strip():
        #it was null
        outfile = 3
        line_out1 = " "
        line_out2 = " "
    elif line_in.find("HOLE")>=0:
        #it was hole line
        outfile = 3
        line_out1 = line_in.replace("\n","")
        line_out2 = line_in.replace("\n","")
    elif line_in.find("_TH")>=0:
        #it was coord
        print("coord")
        if outfile == 1:
            line_out1 = line_in.replace("\n","")
            line_out2 = ""
        elif outfile == 2:
            print("#2")
            line_out1 = ""
            line_out2 = line_in.replace("\n","")
    elif line_in.find("DIAM")>=0:
        #it was diam
        if outfile == 1:
            line_out1 = line_in.replace("\n","")
            line_out2 = ""
        elif outfile == 2:
            line_out1 = ""
            line_out2 = line_in.replace("\n","")
    else:
        if outfile == 1:
            line_out1 = line_in.replace("\n","")
            line_out2 = ""
        elif outfile == 2:
            line_out1 = ""
            line_out2 = line_in.replace("\n","")
        elif outfile == 3:
            line_out1 = line_in.replace("\n","")
            line_out2 = line_in.replace("\n","")

    if line_in.find("Z_TH")>=0:
        #it was Z coord, store it for after reaming
        ZcordStr = line_out1

    if (line_in.find("Y_TH")>=0 and outfile == 2):
        #it was Y coord file and now switching
        line_out2 = line_out2 + "\n" + ZcordStr


    #figured out what to do
    if line_out1 != "":
        fout1.write(line_out1)
        fout1.write("\n")
    if line_out2 != "":
        fout2.write(line_out2)
        fout2.write("\n")

# lines all read in
# close all the files
fout1.close()
fout2.close()
fin.close()

print ("conversion done")
