install clawpack, having declared $CLAW as an environment variable. You need
- place the avac directory in the examples/ directory of geoclaw -
- install GNU parallel 
- install openmp for fortran 
- install a recent version of GRASS, which supports Temporary features (GRASS 7. 6 mini)

Before executing r.avac from your GRASS session. you need:
- to define the working region (in the map display, Various zoom options icon > set computationale region extent interactively) with a resolution consistent with
the resolution of the topo file (g.region -pa res=XX)
- to have a digital terrain model in the form of a raster present in the current mapset
- to have delimited the avalanche departure zones within a polygon-type vector map.
Each starting zone is represented by a polygon that has an attribute (column named h by default) of the thickness of the mobilized snow layer.

In its current version r.avac does not allow you to specify zones for the AMR option in GeoClaw.

The script provides the avac code with the topographic data (topo.asc), the initial conditions (initial.xyz) and the computational domain extension, the friction
parameters in the voellmy.data and setrun.py files. Once the computation has been done, r.avac transforms the strong.qxxxx files into rasters (flow depth h(t) and kinetic pressure p(t)) that can be imported into GRASS.
 

The user can visualize the flow by launching the g.gui.animationinteractive module.
The r.avac script offers an option to compute the maximum height and maximum pressure over the whole computational domain (-m option).
