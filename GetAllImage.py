#!/usr/bin/python

import os, sys

# Open a file
path = r"C:\Users\Gaurang\Desktop\Python\Image Analysis\All Images"
dirs = os.listdir( path )

# This would print all the files and directories
for file in dirs:
   print(file)
