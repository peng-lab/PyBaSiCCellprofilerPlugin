# PyBasicCellprofilerPlugin

BaSiCIlluminationCalculate and BaSiCIlluminationApply are two [CellProfiler](https://cellprofiler.org) plugins. The first one calculates the illumination background in a [model](https://www.nature.com/articles/ncomms14836) including flatfield and darkfield images. The second plugin applies the correction on a set of images using the flatfield, darkfield (optional) and baseline drift (optional) images.

## How to load the plugins?

In addition to the standard modules, the Cellprofiler loads a set of plugins. The Cellprofiler recognizes the plugins in a given folder and loads them for the user. So you need to download `basicilluminationcalculate.py` and `basicilluminationapply.py` and save them in a folder. Then select the folder containing the plugins in <em>Preferences > Cellprofiler plugins directory</em> like below figure and click on <em>save</em> botton. Perhaps you can save the plugins in the <em>CellProfiler-plugins/CellProfiler4_AutoConvert</em> folder in the Cellprofiler package where some other plugins are accessible.

![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/Cellprofiler_setting_plugins_directory.png)

If you successfully load a plugin, the plugin should be listed in the available modules. You can explore and select available modules by clicking on **+** button on the bottom-left corner of the Cellprofiler interface.

![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/Cellprofiler_add_module_botton.png)

Because the BaSiCIlluminationCalculate and BaSiCIlluminationApply plugins are image processing plugins, they are listed in the Image Processing category.

![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/Cellprofiler_modules.png)

You can find documentation of each module by selecting the module and click on **? Module Help** button.


## How to set the output folder?

After analysing the image, Cellprofiler saves the results as images and/or tables according to your configuration. Similar to plugins folder you can select the output folder in preferences. Set the <em>Preferences > Cellprofiler plugins directory</em> like below figure:


![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/Cellprofiler_setting_output_directory.png)

## Import input images

The simplest way to import images is to drag and drop a set of selected images in the "drag and drop files" of Cellprofiler.

![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/Cellprofiler_drag_and_drop_before.png)

After dropping the files, the imported files are visually listed like the following figure.

![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/Cellprofiler_drag_and_drop_after.png)


## Creating a pipeline for PyBasicCellprofilerPlugin

After importing the input images, we first load the BaSiCIlluminationCalculate module by 1) double click on it or 2) selecting the module and clicking on **+ Add to Pipeline** button.

The below figures show examples of settings and outputs of the plugins in CellProfiler application. The input images are accessible at 'WSI_Brain_Uncorrected_tiles' folder in the repository.

### BaSiCIlluminationCalculate settings:
![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/BasicIlluminationCalculate_setup.png)

### BaSiCIlluminationCalculate output:
![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/BasicIlluminationCalculate_output.png)

### BaSiCIlluminationApply settings:
![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/BasicIlluminationApply_setup.png)

### BaSiCIlluminationApply output:
![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/BasicIlluminationApply_output.png)
