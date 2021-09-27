# PyBasicCellprofilerPlugin

BaSiCIlluminationCalculate and BaSiCIlluminationApply are two [CellProfiler](https://cellprofiler.org) plugins. The first one calculates the illumination background in a [model](https://www.nature.com/articles/ncomms14836) including flatfield, darkfield and baseline drift. The second plugin applies the correction on a set of images using the flatfield, darkfield (optional) and baseline drift (optional) images.

## BaSiCIlluminationCalculate: calculating illumination background

The plugin computes a single flatfield image, a single darkfield image for a group of input images and baseline drifts for each image in the group. The size of input images should be the same. Basically, a Cellprofiler pipeline processes images in a group one-by-one in a **cycle** per each single image. In order to decrease the running time, BaSiCIlluminationCalculate plugin computes and internally stores flatfield image, darkfield image and baseline drifts in the first cycle and in the next cycles it only retrieves them. The baseline drift is stored as an array with the same length as the number of input images. Thus each single value in the baseline drift array belongs to a single input image. In contrast to the baseline drift, the flatfield and darkfield images are the same for all images in a group of input images, thus in all cycles BaSiCIlluminationCalculate provides the same flatfield and darkfield images (computed in the first cycle). Because the baseline drift is a value that varies from one image to another, in each cycle, an image with the same shape of input images and with constant pixel value is created. The pixel value of the baseline drift image equals the baseline drift of the processed image in each cycle.

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

A pipeline for PyBasic plugins can contains **BaSiCIlluminationCalculate** and **BaSiCIlluminationApply** plugins and a few other modules for saving the results.

### Adding BaSiCIlluminationCalculate to the pipeline

After importing the input images, we first load the **BaSiCIlluminationCalculate** plugin by 1) double click on it or 2) selecting the module and clicking on **+ Add to Pipeline** button. Then we configure it like below. We set the input images to **DNA** (default image group name), and ask for calculating the drakfield and baseline drift. You can also change the name of flatfield, darkfield and baseline drift images from their default values.

![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/BaSiCIlluminationCalculate_config.png)


### Adding BaSiCIlluminationApply to the pipeline

Now we add the **BaSiCIlluminationApply** to the pipeline and confiure it like below. We select the input image group like before and choose the outputs of **BaSiCIlluminationCalculate** (flatfield, darkfield and baseline drift images) as the illumination model.

![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/BaSiCIlluminationApply_config.png)

### Adding SaveImage module to save flatfield image

The **SaveImage**, a module in <em>File Processing</em> category, saves images in different formats. We add it to the pipeline and because flatfield is a single image for a group of input images we save it once (e.g. only in the first cycle). The <em>bit-depth</em> is set to 32-bit floating point image otherwise the pixel values are normed to integer. You can name the file with an arbitrary name and it is set to "Flatfield" in the following example.

![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/Flatfield_saveimage_config.png)

### Adding SaveImage module to save darkfield image

Similar to saving the flatfield image, another **SaveImage** module is added to the pipeline in order to save the darkfield image. The only differences are that we select <em>BasicDarkfield</em> for saving and name the file e.g. <em>Darkfield</em>.

![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/Darkfield_saveimage_config.png)

### Adding SaveImage module to save baseline drift images

As explained in the beginning, flatfield, darkfield and baseline drift are calculated during the first cycle. Baseline drift is a specific value for each image that should be subtracted as a part of background correction. In each cycle, a constant image with the pixel value of baseline drift is created for each image by <em>BaSiCIlluminationCalculate</em> andit is used by <em>BaSiCIlluminationApply</em> to subtract the baseline drift from an input image. You may also need to save the baseline drift images in each cycle. Since they are created one-by-one, we need to save them in every cycle. We add a suffix to the name of input images in order to create a series of file names for baseline drift images.




The below figures show examples of settings and outputs of the plugins in CellProfiler application. The input images are accessible at 'WSI_Brain_Uncorrected_tiles' folder in the repository.

### BaSiCIlluminationCalculate settings:
![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/BasicIlluminationCalculate_setup.png)

### BaSiCIlluminationCalculate output:
![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/BasicIlluminationCalculate_output.png)

### BaSiCIlluminationApply settings:
![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/BasicIlluminationApply_setup.png)

### BaSiCIlluminationApply output:
![logs_graph](https://github.com/peng-lab/PyBasicCellprofilerPlugin/blob/main/figures/BasicIlluminationApply_output.png)
