# AVAC: Computing snow avalanches

AVAC is a numerical code for simulating flowing avalanches using the ClawPack and GeoClaw libraries (www.clawpack.org). It solves the two-dimensional Saint-Venant equations on an irregular topography (in a Cartesian frame). It uses the Coulomb or Voellmy empirical equation to describe flow resistance.

It also includes additional modules for importing and exporting data:
- Mathematica notebooks for preparing the input topographic data and transforming the output files into raster files and animations.
- a Grass addon (r.avac), which prepares the input data, runs the AVAC code, and post-processes the output files.
See the pdf files for further information. We plan to provide similar routines based on Python scripts.

We provide additional files to test the code or serve as examples.

The AVAC code has been extensively tested on Linux machines. See the ClawPack information for other operating systems.


*** CONTENTS ***


ClawPack files:
- AddSetrun.py
- AddZoom.py
- b4step2.f90
- launcher_random.py
- module_voellmy.f90
- setprob.f90
- Makefile
- setrun.py
- src2.f90
- voellmy.data
- valout.f90

Documentation:
- readme (this page)
- ravac_readme_en.txt: documentation in English for using r.avac
- ravac_readme_fr.txt: documentation in French for using r.avac
- article_avac_en.pdf: documentation on AVAC in English
- article_avac_fr.pdf: documentation on AVAC in French
- r.avac_manual_en.pdf: documentation on the r.avac addon in English
- r.avac_manual_fr.pdf: documentation on the r.avac addon in French


Mathematica notebooks:
- PreProcessClawpack: creating the initial.xyz file from a shapefile
- PostProcessClawpack: visualizing and exporting the fortq.xxxx and rasterxxxx into rasters (ESRI format)

GRASS addon:
- r.avac

Application examples:
- recoin.zip (AVAC+ Mathematica)
- boussolenc.r.avac.tar.gz (AVAC + r.avac addon)

***************

Christophe Ancey, EPFL, Switzerland

Vincent Bain, Toraval, France

version 1.0 October 2019

version 1.1 June 2020
