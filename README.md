# PyBasicCellprofilerPlugin

BaSiCIlluminationCalculate and BaSiCIlluminationApply are two [CellProfiler](https://cellprofiler.org) plugins. The first one calucluates the illumination background in a [model](https://www.nature.com/articles/ncomms14836) inclduing flatfield and darkfield images. The second plugin applies the correction an a set of images using the flatfield, darkfield (optional) and baseline drift (optional) images.

# How to use the plugins:
In addition to the standard modules, the Cellprofiler loads a set of plugins. The Cellprofiler recongnizes the plugins in a given folder and load them for the user.
You can select the folder containing the plugins in <em>Preferences > Cellprofiler plugins directory</em> like below figure and then click <em>save</em> botton.



The below figures show examples of settings and outputs of the plugins in CellProfiler application. The input images are accessible at 'WSI_Brain_Uncorrected_tiles' folder in the repository.

### BaSiCIlluminationCalculate settings:
![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/BasicIlluminationCalculate_setup.png)

### BaSiCIlluminationCalculate output:
![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/BasicIlluminationCalculate_output.png)

### BaSiCIlluminationApply settings:
![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/BasicIlluminationApply_setup.png)

### BaSiCIlluminationApply output:
![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/BasicIlluminationApply_output.png)
