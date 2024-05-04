
import pandas as pd
import numpy as np
import jinja2
import math
import re

class Plate_Desc:
    def __init__(self, title, descrip, serialnum, date_time, datafname):
        self.title = title
        self.descrip = descrip
        self.serialnum = serialnum
        self.date_time = date_time
        self.datafname = datafname

    def clear(self):
        self.title = ""
        self.descrip = ""
        self.serialnum = ""
        self.date_time = ""
        self.datafname = ""
        return self

    def new(self):
        self.__init__("", "", "", "", "" )
        return self


# open the file
datafname = "Z:\Shared Folders\Data_WCO\BHPB Meas\Large_Top_755_772.out"
datafname_short = re.sub(r'\\.+\\', '', datafname)
datafname_short = re.sub(r'^(.*:)', '', datafname_short)


fin = open(datafname,"r")
rdplate_line = 0  #reading aline on a plate
plate_num = 0
plate_desc = Plate_Desc("","","","", datafname_short)
plate_hole_rec = []      # holds all the data for all the hole measurements on one plate
plate_hole_table = []    # all the holes for a single plate
plate_meas_table = []    # list of two dimension (plate desc + plate_holes_rec's)

hole_rec = []
hole_table = []

for line_in in fin:
    line_out = ""
    if line_in.find("%")>=0:
        #nothing
        line_out = ""
    elif not line_in.strip():
        #it was null
        line_out = ""
    elif line_in.find("()")>=0:
        #it is the third line in 
        line_out = ""
    else:
        line_out=line_in.replace("\n","")

    # anything but a blank line
    if line_out != "":

        if (rdplate_line==0):
            if (line_out.find("HOLE ")>=0):
                rdplate_line = 4   #there is another hole on the plate
            else:
                if plate_num ==0:
                    plate_num += 1
                else:
                    #if not the first plate then must push to stack
                    plate_meas_rec = (plate_desc, plate_hole_table)
                    plate_meas_table.append(plate_meas_rec)
                    plate_desc = Plate_Desc("","","","", datafname_short)
                    plate_hole_table = []
                    plate_num += 1

        # now, need to find out if a plate reading is in progress
        if rdplate_line == 0:
            #header
            plate_desc = Plate_Desc("","","","", datafname_short)
            rdplate_line = rdplate_line + 1
            plate_desc.title = line_out.strip()
        elif rdplate_line == 1:
            #descrip #2
            rdplate_line = rdplate_line + 1
            plate_desc.descrip = line_out.strip()
        elif rdplate_line == 2:
            #serial number
            if line_out.find("SERIAL")>= 0:
                #it is serial number
                plate_desc.serialnum = line_out.replace("SERIAL: ", "")
            rdplate_line = rdplate_line + 1
        elif rdplate_line == 3:
            #time and date
            tempstr = line_out.replace("  ", ",")
            split_val_list = tempstr.split(",")
            if len(split_val_list[1]) < 6:
                split_val_list[1] = "0" + split_val_list[1]
            date_str = split_val_list[0][2] + split_val_list[0][3] + "/" + split_val_list[0][4] + split_val_list[0][5] + "/" + "20" + split_val_list[0][0] + split_val_list[0][1]
            time_str = split_val_list[1][0] + split_val_list[1][1] + ":" + split_val_list[1][2] + split_val_list[1][3] + ":" + split_val_list[1][4] + split_val_list[1][5]
            plate_desc.date_time = date_str + "  " + time_str
            rdplate_line = rdplate_line + 1
        elif rdplate_line == 4:
            #hole number
            if line_out.find("HOLE")>= 0:
                #it is serial number
                tempstr = line_out.replace("HOLE ", "")
                plate_hole_rec = []
                plate_hole_rec.append(tempstr)
            rdplate_line = rdplate_line + 1
        elif rdplate_line == 5:
            #X pos
            if line_out.find("X_TH")>= 0:
                tempstr1 = line_out.replace("X_TH:", "")
                tempstr2 = tempstr1.replace("X_MEA:", "")
                tempstr3 = tempstr2.replace("X_DIF:", "")
                tempstr4 = tempstr3.replace("  ", ",")
                split_val_list = tempstr4.split(",")

                plate_hole_rec.append(split_val_list[0])
                plate_hole_rec.append(split_val_list[1])
                plate_hole_rec.append(split_val_list[2])
            rdplate_line = rdplate_line + 1
        elif rdplate_line == 6:
            #Y pos
            if line_out.find("Y_TH")>= 0:
                tempstr1 = line_out.replace("Y_TH:", "")
                tempstr2 = tempstr1.replace("Y_MEA:", "")
                tempstr3 = tempstr2.replace("Y_DIF:", "")
                tempstr4 = tempstr3.replace("  ", ",")
                split_val_list = tempstr4.split(",")

                plate_hole_rec.append(split_val_list[0])
                plate_hole_rec.append(split_val_list[1])
                plate_hole_rec.append(split_val_list[2])
            rdplate_line = rdplate_line + 1
        elif rdplate_line == 7:
            #Y pos
            if line_out.find("Z_TH")>= 0:
                tempstr1 = line_out.replace("Z_TH:", "")
                tempstr2 = tempstr1.replace("Z_MEA:", "")
                tempstr3 = tempstr2.replace("Z_DIF:", "")
                tempstr4 = tempstr3.replace("  ", ",")
                split_val_list = tempstr4.split(",")

                plate_hole_rec.append(split_val_list[0])
                plate_hole_rec.append(split_val_list[1])
                plate_hole_rec.append(split_val_list[2])
            rdplate_line = rdplate_line + 1
        elif rdplate_line == 8:
            #DIAM
            if line_out.find("DIAM")>= 0:
                tempstr1 = line_out.replace("DIAM:", "")
                tempstr2 = tempstr1.replace("DIA_ERR:", "")
                tempstr3 = tempstr2.replace("  ", ",")
                split_val_list = tempstr3.split(",")

                plate_hole_rec.append(split_val_list[0])
                plate_hole_rec.append(split_val_list[1])
                # last number read. next line blank but will be chopped up on top
                plate_hole_table.append(plate_hole_rec)
            rdplate_line = 0
        else:
            print(plate_desc)
            print(plate_meas)
            rdplate_line = 0

        #print (line_out)

# lines all read in, store the last record in memory
plate_meas_rec = (plate_desc, plate_hole_table)
plate_meas_table.append(plate_meas_rec)


num_holes = len(plate_meas_table[0][1])
num_plates = len(plate_meas_table)

# summary at the top
col1 = pd.Index(['plates', 'descrip', 'data file', 'start', 'stop', 'operator', '# plates', 'start s/n', 'stop s/n','# holes'])
col2 = pd.Index([plate_meas_table[0][0].title, plate_meas_table[0][0].descrip, plate_meas_table[0][0].datafname, plate_meas_table[0][0].date_time, plate_meas_table[len(plate_meas_table)-1][0].date_time, 'Rich Budek', str(len(plate_meas_table)).strip(), plate_meas_table[0][0].serialnum, plate_meas_table[len(plate_meas_table)-1][0].serialnum, str(num_holes).strip() ])
df_head = pd.DataFrame(col2, columns=[''], index=col1)

print(df_head)

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


head_styler = df_head.style.applymap(color_negative_red)

#create serial numbers
meas_tab_serial_num = pd.Index(['spec'], dtype='object')
meas_tab_num = pd.Index([' '])
i=0
for lp in plate_meas_table:
    meas_tab_serial_num = meas_tab_serial_num.append(pd.Index([plate_meas_table[i][0].serialnum]))
    i += 1
    meas_tab_num = meas_tab_num.append(pd.Index([i]))



# GO THRU THE PLATES NOW


#create the dataframe first so that can append without creating a complex array and eating up memory
df_table_01a = pd.DataFrame( meas_tab_serial_num, columns=['PLATE'], index=meas_tab_num )
df_table_01b = pd.DataFrame( [], columns=[], index=[])
df_table_01c = pd.DataFrame( [], columns=[], index=[])
df_table_01d = pd.DataFrame( [], columns=[], index=[])


curr_table_num = 1
#loop thru all of the holes (outside loop for columns)
for i in range(num_holes):
    Xcol=pd.Index([plate_meas_table[0][1][i][1]])  # X spec
    Ycol=pd.Index([plate_meas_table[0][1][i][4]])  # Y spec
    
    for j in range(num_plates):
        Xcol = Xcol.append(pd.Index([plate_meas_table[j][1][i][2]] ))
        Ycol = Ycol.append(pd.Index([plate_meas_table[j][1][i][5]] ))

    # all plates read append to dataframe
    if curr_table_num == 1:
        df_table_01a['X'+str(i+1)] = Xcol
        df_table_01a['Y'+str(i+1)] = Ycol
        if i>=4 :
            curr_table_num += 1
            df_table_01b = pd.DataFrame( meas_tab_serial_num, columns=['PLATE'], index=meas_tab_num )
    elif curr_table_num == 2:
        df_table_01b['X'+str(i+1)] = Xcol
        df_table_01b['Y'+str(i+1)] = Ycol
        if i>=9 :
            curr_table_num += 1
            df_table_01c = pd.DataFrame( meas_tab_serial_num, columns=['PLATE'], index=meas_tab_num )
    elif curr_table_num == 3:
        df_table_01c['X'+str(i+1)] = Xcol
        df_table_01c['Y'+str(i+1)] = Ycol
        if i>=14 :
            curr_table_num += 1
            df_table_01d = pd.DataFrame( meas_tab_serial_num, columns=['PLATE'], index=meas_tab_num )
    elif curr_table_num == 4:
        df_table_01d['X'+str(i+1)] = Xcol
        df_table_01d['Y'+str(i+1)] = Ycol


meas_01a_styler = df_table_01a.style.applymap(color_negative_red)
meas_01b_styler = df_table_01b.style.applymap(color_negative_red)
meas_01c_styler = df_table_01c.style.applymap(color_negative_red)
meas_01d_styler = df_table_01d.style.applymap(color_negative_red)


# for the diameter and Z positions
#create the dataframe first so that can append without creating a complex array and eating up memory
df_table_02a = pd.DataFrame( meas_tab_serial_num, columns=['PLATE'], index=meas_tab_num )
df_table_02b = pd.DataFrame( [], columns=[], index=[])
df_table_02c = pd.DataFrame( [], columns=[], index=[])
df_table_02d = pd.DataFrame( [], columns=[], index=[])


hole_dia_table = pd.Index([])   # clear the table here
hole_dia_rec = pd.Index([])     # numeric values

curr_table_num = 1
#loop thru all of the holes (outside loop for columns)
for i in range(num_holes):
    Xcol=pd.Index([plate_meas_table[1][1][0][8]])  # Z height spec = Z height first plate first hole
    Ycol=pd.Index(['.4900'])  # diam
    
    for j in range(num_plates):
        Xcol = Xcol.append(pd.Index([plate_meas_table[j][1][i][8]] ))
        Ycol = Ycol.append(pd.Index([plate_meas_table[j][1][i][10]] ))
        hole_dia_rec = hole_dia_rec.append(pd.Index([float(plate_meas_table[j][1][i][10])]))

    # all plates read append to dataframe
    if curr_table_num == 1:
        df_table_02a['Z'+str(i+1)] = Xcol
        df_table_02a['dia'+str(i+1)] = Ycol
        if i>=4 :
            curr_table_num += 1
            df_table_02b = pd.DataFrame( meas_tab_serial_num, columns=['PLATE'], index=meas_tab_num )
    elif curr_table_num == 2:
        df_table_02b['Z'+str(i+1)] = Xcol
        df_table_02b['dia'+str(i+1)] = Ycol
        if i>=9 :
            curr_table_num += 1
            df_table_02c = pd.DataFrame( meas_tab_serial_num, columns=['PLATE'], index=meas_tab_num )
    elif curr_table_num == 3:
        df_table_02c['Z'+str(i+1)] = Xcol
        df_table_02c['dia'+str(i+1)] = Ycol
        if i>=14 :
            curr_table_num += 1
            df_table_02d = pd.DataFrame( meas_tab_serial_num, columns=['PLATE'], index=meas_tab_num )
    elif curr_table_num == 4:
        df_table_02d['Z'+str(i+1)] = Xcol
        df_table_02d['dia'+str(i+1)] = Ycol

    # store to all holes data
    hole_dia_table = hole_dia_table.append(hole_dia_rec)
    hole_dia_rec = pd.Index([])     # clear out after adding the record


# all plates and holes done, so all data read in
df_hole_dia_table = pd.DataFrame( hole_dia_table, columns=['Dia'], index=range(hole_dia_table.size) )

meas_02a_styler = df_table_02a.style.applymap(color_negative_red)
meas_02b_styler = df_table_02b.style.applymap(color_negative_red)
meas_02c_styler = df_table_02c.style.applymap(color_negative_red)
meas_02d_styler = df_table_02d.style.applymap(color_negative_red)


# for the calculations / differences
#create the dataframe first so that can append without creating a complex array and eating up memory
df_table_03a = pd.DataFrame( meas_tab_serial_num, columns=['PLATE'], index=meas_tab_num )
hole_acc_rec = pd.Index([])   # clear the rec
hole_diam_list = []    #hole diam list
hole_acc_table = pd.Index([])   # clear the table here

curr_table_num = 1
meas_table_list = []

#loop thru all of the holes (outside loop for columns)
for i in range(num_holes):
    X1spec = float(plate_meas_table[0][1][i][1])
    Y1spec = float(plate_meas_table[0][1][i][4])
    X1len = (float(plate_meas_table[0][1][0][1]) - float(plate_meas_table[0][1][i][1]))
    Y1len = (float(plate_meas_table[0][1][0][4]) - float(plate_meas_table[0][1][i][4]))
    TotLen = math.sqrt( X1len*X1len + Y1len*Y1len )

    Xcol=pd.Index(["{0:1.4f}".format(X1spec)])    # X position theoretical
    Xdif=pd.Index(['0.0020'])                     # X diff allowed
    Ycol=pd.Index(["{0:1.4f}".format(Y1spec)])    # Y position theoretical
    Ydif=pd.Index(['0.0020'])                         # Y diff allowed
    X1dif=pd.Index(['0.0020'])                        # difference from X1 hole
    Y1dif=pd.Index(['-.0020'])                        # difference from Y1 hole
    LenDif = pd.Index(['0.0020'])                     # len from X1
    Zdif=pd.Index(['0.0020'])                         # Z diff allowed
    DiaCol=pd.Index(['0.4900'])                       # diamter spec
    DiaDif=pd.Index(['0.0010'])                        # diam diff allowed ... but really only one sides

    

    for j in range(num_plates):
        print(j)   # j value for debugging
        Xhole1 = float(plate_meas_table[j][1][0][2])     # X pos first hole
        Yhole1 = float(plate_meas_table[j][1][0][5])     # Y pos first hole
        Zhole1 = float(plate_meas_table[j][1][0][8])     # Z height at first hole

        Xpos = float(plate_meas_table[j][1][i][2])
        XdifCalc = X1spec - Xpos
        Ypos = float(plate_meas_table[j][1][i][5])
        YdifCalc = Y1spec - Ypos

        #theoretical diff - real diff
        XlenReal = Xhole1 - Xpos
        X1difCalc = (float(plate_meas_table[j][1][0][1])-float(plate_meas_table[j][1][i][1])) -  XlenReal
        YlenReal = Yhole1 - Ypos
        Y1difCalc = (float(plate_meas_table[j][1][0][4])-float(plate_meas_table[j][1][i][4])) -  YlenReal
        #len to hole1
        LenCalc = math.sqrt(XlenReal*XlenReal + YlenReal*YlenReal)
        LenDifCalc = LenCalc - TotLen

        Zpos = float(plate_meas_table[j][1][i][8])
        ZdifCalc = Zhole1 - Zpos

        DiaVal = float(plate_meas_table[j][1][i][10])
        DiaDifCalc = DiaVal - float('.4900')

        Xcol = Xcol.append(pd.Index([str(Xpos)]))
        tempStr = "{0:+1.4f}".format(XdifCalc)
        print ("Xdif Val = " + tempStr)
        #tempStr = "hello"
        Xdif = Xdif.append(pd.Index(["{0:1.4f}".format(XdifCalc)]))
        Ycol = Ycol.append(pd.Index(["{0:1.4f}".format(Ypos)]))
        Ydif = Ydif.append(pd.Index(["{0:1.4f}".format(YdifCalc)]))
        X1dif = X1dif.append(pd.Index(["{0:1.4f}".format(X1difCalc)]))
        Y1dif = Y1dif.append(pd.Index(["{0:1.4f}".format(Y1difCalc)]))
        LenDif = LenDif.append(pd.Index(["{0:1.4f}".format(LenDifCalc)]))
        Zdif = Zdif.append(pd.Index(["{0:1.4f}".format(ZdifCalc)]))
        DiaCol = DiaCol.append(pd.Index(["{0:1.4f}".format(DiaVal)]))
        DiaDif = DiaDif.append(pd.Index(["{0:1.4f}".format(DiaDifCalc)]))

        if LenDifCalc<0.200 and LenDifCalc>-0.200:  # throw out the bad values
            hole_acc_rec = hole_acc_rec.append(pd.Index([LenDifCalc]))
            hole_diam_list.append(LenDifCalc)

        print ("Xhole1=" + str(Xhole1) + " Yhole1=" + str(Yhole1))
        print ("X1diff=" + str(X1dif.size) + " Y1diff=" + str(Y1dif.size))

        # end of loop for plates
        
    # all plates read append to dataframe
    df_table_03a['X'+str(i+1)] = Xcol
    df_table_03a['Xdif'] = Xdif
    df_table_03a['Y'+str(i+1)] = Ycol
    df_table_03a['Ydif'] = Ydif
    df_table_03a['X1dif'] = X1dif
    df_table_03a['Y1dif'] = Y1dif
    df_table_03a['L1dif'] = LenDif
    df_table_03a['Z1dif'] = Zdif
    df_table_03a['Dia'] = DiaCol
    df_table_03a['D Dif'] = DiaDif

    # store to all holes accuracy
    #hole_acc_table = hole_acc_table.append( hole_acc_rec )

    df_table_03a.style.set_caption('For Hole #' + str(i+1))
    df_table_03a.style.set_caption("For Hole #" + str(i+1))


    meas_03a_styler = df_table_03a.style.applymap(color_negative_red)
    meas_03a_styler = meas_03a_styler.applymap(color_spec_dif_01, subset=['D Dif'])
    meas_03a_styler = meas_03a_styler.applymap(color_spec_dif_02, subset=['Xdif'])
    meas_03a_styler = meas_03a_styler.applymap(color_spec_dif_02, subset=['Ydif'])
    meas_03a_styler = meas_03a_styler.applymap(color_spec_dif_03, subset=['X1dif'])
    meas_03a_styler = meas_03a_styler.applymap(color_spec_dif_03, subset=['Y1dif'])
    meas_03a_styler = meas_03a_styler.applymap(color_spec_dif_03, subset=['L1dif'])
    meas_03a_styler = meas_03a_styler.applymap(color_spec_dif_02, subset=['Z1dif'])
    meas_03a_styler = meas_03a_styler.applymap(color_spec_dia, subset=['Dia'])
    #meas_03a_styler = df_table_03a.style.applymap(color_spec_high_red, subset=['D Dif'])

    meas_03a_rendered = meas_03a_styler.render(classes='data')
    meas_table_list.append(meas_03a_rendered)

    df_table_03a = pd.DataFrame( meas_tab_serial_num, columns=['PLATE'], index=meas_tab_num )

    # end of loop for holes


# print ("df__hole_acc_table size=" + df_hole_acc_table.size)
df_hole_acc_table = pd.DataFrame( hole_acc_rec, columns=['Acc'], index=range(hole_acc_rec.size) )


# now, calculate the Z heights, but go plate by plate
    
hole_zht_table = pd.Index([])   # Z heights next to hole
hole_zht_rec = pd.Index([])     # Z heights numeric values

#loop thru all of the holes (outside loop for columns)
for j in range(num_plates):
    Z1spec = float(plate_meas_table[j][1][0][8])    # Zht for hole #1
    Zdif = 0.00
    

    for i in range(num_holes):
        Zpos = float(plate_meas_table[j][1][i][8])
        ZdifCalc = Z1spec - Zpos
        hole_zht_table = hole_zht_table.append(pd.Index([ZdifCalc]))
        # end of loop for plates
# end of loop for holes

df_zht_table = pd.DataFrame( hole_zht_table, columns=['Zht'], index=range(hole_zht_table.size))


# all plates and holes done, so all do all the plots for 


#plot #01  all the hole diameters
ax01=df_hole_dia_table.plot(kind='line', title='Hole Diameters', legend='False', figsize=(11,8))
ax01.set_ylim(0.4880, 0.4920)
ax01.set(xlabel='hole number (every ' + str(num_holes).strip() + ' holes is a plate)', ylabel='hole diam (in inches)')
fig01=ax01.get_figure()
fig01.savefig('plot_01.svg')

#plot #02  hole diameter histogram
ax02=df_hole_dia_table.plot(kind='hist', title='Hole Diameter Distribution', legend='False', figsize=(11,8), bins=60)
ax02.set(xlabel='hole diam (in inches)' + '\n' + 'spec = 0.4900 - 0.4910')
#ax01.set_ylim(0.4880, 0.4920)
fig02=ax02.get_figure()
fig02.savefig('plot_02.svg')

#plot #03  all the hole accuracies
ax03=df_hole_acc_table.plot(kind='line', title='Hole Center X-Y Accuracy (distance to hole #1)' + '\n' + 'Closer to 0.0000 is best', legend='False', figsize=(11,8))
ax03.set_ylim(-0.0050, 0.0050)
ax03.set(xlabel='hole number (every ' + str(num_holes).strip() + ' holes is a plate)' , ylabel='X-Y Center Accuracy (in inches)')
fig03=ax03.get_figure()
fig03.savefig('plot_03.svg')

#plot #04  hole accuracy histogram
ax04=df_hole_acc_table.plot(kind='hist', title='Hole Accuracy Distribution', legend='False', figsize=(11,8), bins=60)
ax04.set(xlabel='hole X-Y accuracy (closer to 0.0000 is best)')
#ax04.set_xlim(-0.50, +0.50)
fig04=ax04.get_figure()
fig04.savefig('plot_04.svg')

#plot #05  hole Z heights
ax05=df_zht_table.plot(kind='line', title='Z height by each hole (height diff to hole #1)' + '\n' + 'Closer to 0.0000 is best', legend='False', figsize=(11,8))
ax05.set_ylim(-0.010, 0.010)
ax05.set(xlabel='hole number (every ' + str(num_holes).strip() + ' holes is a plate)' , ylabel='Z height difference (in inches)')
fig05=ax05.get_figure()
fig05.savefig('plot_05.svg')

#plot #06  hole Z heights histogram
ax06=df_zht_table.plot(kind='hist', title='Z height Distribution', legend='False', figsize=(11,8), bins=60)
ax06.set(xlabel='Z hole height diff from hole #1 (closer to 0.0000 is best)')
fig06=ax06.get_figure()
fig06.savefig('plot_06.svg')


# template handling
env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=''))
template = env.get_template('template_01.html')
html = template.render(head_table=head_styler.render(), meas_01a_table=meas_01a_styler.render(), meas_01b_table=meas_01b_styler.render(), meas_01c_table=meas_01c_styler.render(), meas_01d_table=meas_01d_styler.render(), meas_02a_table=meas_02a_styler.render(), meas_02b_table=meas_02b_styler.render(), meas_02c_table=meas_02c_styler.render(), meas_02d_table=meas_02d_styler.render(), meas_03_tables=meas_table_list )


# write the HTML file
with open('report.html', 'w', encoding='utf-8') as f:   #put encoding on to stop unicode errors
    f.write(html)
f.close()


print ("end of report")
