#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:34:53 2020

@author: ancey
"""
import subprocess
import os
import sys
import shutil

##################################################################
# Launcher will run AVAC many times. It should work independently.
# You just have to specify the values of mu and xi to be tested.
# The simulation parameters are changed in AddSetrun.py
##################################################################



dirName = 'runs'  #directory in which the results lie

###############################################################
clear = lambda: os.system('clear')
clear()

# Finding the home directory
from os.path import expanduser
home = expanduser("~")

def EraseFile(repertoire):
    files=os.listdir(repertoire)
    for filename in files:
        os.remove(repertoire + "/" + filename)
        
def check():
    with open(home+'/.bashrc') as f:
        datafile = f.readlines()
    for line in datafile:
        s="CLAW"
        if s in line and line.find('#')==-1:
               return line
    return False  # Because you finished the search without finding

def defineTime():
    with open('AddSetrun.py') as f:
        datafile = f.readlines()
    for line in datafile:
        if "nsim" in line:
                var=(str.split(line))[2]
                nsim=int(var)
    for line in datafile:
        if "tmax" in line:
                var=(str.split(line))[2]
                tmax=float(var)                
    return [nsim,tmax]

# defines the block to complete the file name when seeking "rasterxxxx"
def tampon(i):
    switcher={0:'', 1:'0', 2:'00', 3:'000'}
    return switcher.get(i,"xxxx")  

[nsim,tmax]=defineTime()

claw=check()
if claw == False:
    print("Error: I cannot determine the $CLAW variable...")
    print("Please modify the script and define it explicitely")
else:
    claw=(str.split(claw))[1]
    claw = home+claw.replace("CLAW=$HOME", "")
 
cwd = os.getcwd() # get current directory    
print("I am running. Here is the information")
print("$CLAW: ", claw)
print("user directory: ", cwd)

nrun=100
print("There will be ", str(nrun), " runs.")
print("Each run is conducted up to time ", str(tmax), ".")


 
try:
    # Create target Directory
    os.mkdir(dirName)
    print("I have created " , dirName ,  " where all the final outcomes will lie. ") 
except FileExistsError:
    print("The " , dirName ,  " directory already exists.")
    
EraseFile(cwd+"/" + "runs") 

import AddLoop
mu_min=AddLoop.mu*(1-AddLoop.Variation_Range_Mu)
mu_max=AddLoop.mu*(1+AddLoop.Variation_Range_Mu)
xi_min=AddLoop.xi*(1-AddLoop.Variation_Range_Xi)
xi_max=AddLoop.xi*(1+AddLoop.Variation_Range_Xi)
uc_min=AddLoop.ucr*(1-AddLoop.Variation_Range_Ucr)
uc_max=AddLoop.ucr*(1+AddLoop.Variation_Range_Ucr)
be_min=AddLoop.beta*(1-AddLoop.Variation_Range_Beta)
be_max=AddLoop.beta*(1+AddLoop.Variation_Range_Beta)
nameFile=AddLoop.outputName
if nameFile == "raster":
   bloc = tampon(4-len(str(nsim)))
elif nameFile != "raster":
   bloc=''
 
ParameterFile = open('Loop_parameters.dat','a')
ParameterFile.write("mu xi beta v_c\n")

import random
for i in range(0,AddLoop.loop_number):
        mu = random.uniform(mu_min,mu_max)
        xi = random.uniform(xi_min,xi_max)
        beta = random.uniform(be_min,be_max)
        velo = random.uniform(uc_min,uc_max)
        ParameterFile.write(str(mu)+" "+str(xi)+" "+str(beta)+" "+str(velo)+"\n")
        try:
            EraseFile(cwd+"/" + "_output")
            os.rmdir(cwd+"/" + "_output")
        except FileNotFoundError:
            pass
        subprocess.run(["make","clean", "target", "CLAW="+claw])
        # os.remove("xgeoclaw")
        VoellmyFile = open('voellmy.data','w')
        VoellmyFile.write("# Modified by Launcher\n")
        VoellmyFile.write("\n")
        VoellmyFile.write("300 =: snow_density\n")
        VoellmyFile.write(str(xi))
        VoellmyFile.write(" =: Voellmy xi\n")
        VoellmyFile.write(str(mu))
        VoellmyFile.write(" =: Voellmy mu\n")
        VoellmyFile.write(str(velo))
        VoellmyFile.write(" =: velocity threshold\n")
        VoellmyFile.write(str(beta))
        VoellmyFile.write(" =: beta_slope\n")
        VoellmyFile.close()
        print("step", str(i),"/",str(nrun),": mu = ", str(mu),", xi = ", str(xi),". Voellmy file created...")
        subprocess.run(["make",".output", "target", "CLAW="+claw,">log.txt"])
        shutil.copyfile(cwd+'/_output/'+nameFile+bloc+str(nsim), cwd+'/runs/'+'run'+str(i)) 
        
ParameterFile.close()




