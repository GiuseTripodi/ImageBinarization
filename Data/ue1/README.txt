===========================================
 S-MS: Synchromedia Multispectral Dataset
===========================================

Created by Rachid Hedjam and Mohamed Cheriet.2013/2014

------------

The dataset is composed of 21 folders. The name of each folder is composed 
of the letter "z" and a numerical number (eg. "z97"). Each folder contains 
one MS document image with 8 spectral bands.
The ground-truth are save in the binary form with this name template: [folder-name][GT.png](eg. "z97GT.png"). 

The eight spectral bands are named as follows: 


	Name		Wavelength(nm)	Light
        -----------------------------------------
    	F1s.png		340  		UV 
    	F2s.png 	500  		Visible 1 
    	F3s.png 	600  		Visible 2 
    	F4s.png 	700  		Visible 3 
    	F5s.png 	800  		IR 1 
    	F6s.png 	900  		IR 2 
    	F7s.png 	1000 		IR 3
    	F8s.png 	1100 		IR 4
	-----------------------------------------

---------------

Read the data


The folder <Prog/> contains a function and script matlab to read the MSI.

 Function: 
	M = synchReadMSI(pth_msi,'listing_FX.txt'); 
	
	pth_msi : the path of the MSI
	
	listing_FX.txt: a text file listing the name of different spectral 	
	band images. Each MSI's folder contain a copy of "listing_FX.txt"
 
 Script: 
	A set of instructions to read the MSI.


-----------------

Before use the dataset please download and read the file: "License of use.pdf"

------------------





