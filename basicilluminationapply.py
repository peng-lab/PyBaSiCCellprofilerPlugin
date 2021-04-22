"""
BasicIlluminationApply
========================

**CorrectIlluBasicIlluminationApplyminationApply** applies an illumination correction
as usually created by **BasicIlluminationCalculate**, to an image in
order to correct for background and shading.

The background and shading illumination is modeled as:

I_means(x) = I_true(x) * flatfield(x) + darkfield(x).

**BasicIlluminationApply** corrects the illumination for input images using given darkfield and flatfield images.
|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

See also
^^^^^^^^

See also **BasicIlluminationCalculate**.

Technical notes
^^^^^^^^^^^^^^^

The BasicIlluminationApply plugin is implemented based 
on the structure of CorrectIlluminationApply module. 
"""

import numpy
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Divider, Binary
from cellprofiler_core.setting import SettingsGroup
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething
from cellprofiler_core.setting.do_something import RemoveSettingButton
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import ImageName

DOS_DIVIDE = "Divide"
DOS_SUBTRACT = "Subtract"

######################################
#
# Rescaling choices - deprecated
#
######################################
RE_NONE = "No rescaling"
RE_STRETCH = "Stretch 0 to 1"
RE_MATCH = "Match maximums"

######################################
#
# # of settings per image when writing pipeline
#
######################################

SETTINGS_PER_IMAGE = 4


class BasicIlluminationApply(ImageProcessing):
    category = "Image Processing"
    variable_revision_number = 1
    module_name = "BasicIlluminationApply"

    def create_settings(self):
        """Make settings here (and set the module name)"""

        self.image_name = ImageSubscriber(
            "Select the input image", "None", doc="Select the image to be corrected."
        )

        self.corrected_image_name = ImageName(
            "Name the output image",
            "BasicCorr",
            doc="Enter a name for the corrected image.",
        )

        self.flatfield_image_name = ImageSubscriber(
            "Select the flat-fielf image",
            "None",
            doc="""\
Select the flatfield illumination image that will be used to
carry out the correction.
""",
        )

        self.darkfield_image_name = ImageSubscriber(
            "Select the dark-fielf image",
            "None",
            doc="""\
Select the darkfield illumination image that will be used to
carry out the correction.
""",
        )

    def settings(self):
        return [
            self.image_name,
            self.corrected_image_name,
            self.flatfield_image_name,      
            self.darkfield_image_name,
        ]

    def visible_settings(self):
        """Return the list of displayed settings
        """
        visible_settings = [
            self.image_name,
            self.corrected_image_name,
            self.flatfield_image_name,      
            self.darkfield_image_name,            
        ]
        return visible_settings
    
    '''
    def prepare_settings(self, setting_values):
        """Do any sort of adjustment to the settings required for the given values

        setting_values - the values for the settings

        This method allows a module to specialize itself according to
        the number of settings and their value. For instance, a module that
        takes a variable number of images or objects can increase or decrease
        the number of relevant settings so they map correctly to the values.
        """
        #
        # Figure out how many images there are based on the number of setting_values
        #
        assert len(setting_values) % SETTINGS_PER_IMAGE == 2
        image_count = len(setting_values) // SETTINGS_PER_IMAGE
        del self.images[image_count:]
        while len(self.images) < image_count:
            self.add_image()
    '''

    def run(self, workspace):
        """Run the module

        workspace    - The workspace contains
            pipeline     - instance of cpp for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - the parent frame to whatever frame is created. None means don't draw.
        """
        self.run_image(
            self.image_name, 
            self.corrected_image_name,
            self.flatfield_image_name,      
            self.darkfield_image_name,               
            workspace
        )

    def run_image(self, image_name, corrected_image_name, flatfield_image_name, darkfield_image_name, workspace):
        """Perform illumination according to the parameters of one image setting group

        """
        #
        # Get images from the image set
        #
        orig_image = workspace.image_set.get_image(image_name.value)
        flatfield = workspace.image_set.get_image(flatfield_image_name.value)
        if darkfield_image_name.value is not None:
            darkfield = workspace.image_set.get_image(darkfield_image_name.value)

        # Throw an error if images are incompatible
        if orig_image.pixel_data.ndim != 2:
            raise ValueError(
                "This module requires that the image has dimension of 2.\n"
                "The %s image has the shape of %s.\n"
                % (
                    image_name.value,
                    orig_image.pixel_data.shape,
                )
            )
        if flatfield.pixel_data.ndim != 2:
            raise ValueError(
                "This module requires that the flatfield image has dimension of 2.\n"
                "The %s flatfield image has the shape of %s.\n"
                % (
                    flatfield_image_name.value,
                    flatfield.pixel_data.shape,
                )
            )
        if darkfield_image_name.value is not None and darkfield.pixel_data.ndim != 2:
            raise ValueError(
                "This module requires that the darkfield image has dimension of 2.\n"
                "The %s flatfield image has the shape of %s.\n"
                % (
                    darkfield_image_name.value,
                    darkfield.pixel_data.shape,
                )
            )
        #
        # Applying the correction:
        #
        if darkfield_image_name.value is None:
            output_pixels = orig_image.pixel_data / flatfield.pixel_data
        else:
            output_pixels = (orig_image.pixel_data  - darkfield.pixel_data) / flatfield.pixel_data
        
        
        #
        # Save the output image in the image set and have it inherit
        # mask & cropping from the original image.
        #
        output_image = Image(output_pixels)
        workspace.image_set.add(corrected_image_name.value, output_image)
        #
        # Save images for display
        #
        if self.show_window:
            if not hasattr(workspace.display_data, "images"):
                workspace.display_data.images = {}
            workspace.display_data.images[image_name.value] = orig_image.pixel_data
            workspace.display_data.images[corrected_image_name.value] = output_pixels
            workspace.display_data.images[flatfield_image_name.value] = flatfield.pixel_data
            if darkfield_image_name is not None:
                workspace.display_data.images[darkfield_image_name.value] = darkfield.pixel_data

   

    def display(self, workspace, figure):
        """ If dark field is provided display one row of darkfield / flatfieldfield / input / output"""
        if self.darkfield_image_name is None:
            figure.set_subplots((3, 1))
        else:
            figure.set_subplots((4, 1))
        figure.set_subplots((2, 2))
        #nametemplate = "Illumination function:" if len(self.images) < 3 else "Illum:" #TODO
        
        if True:
            '''
            image_name = image.image_name.value
            corrected_image_name = image.corrected_image_name.value
            orig_image = workspace.display_data.images[image_name]
            corrected_image = workspace.display_data.images[corrected_image_name]
            '''
            def imshow(x, y, image, *args, **kwargs):

                if image.ndim == 2:
                    f = figure.subplot_imshow_grayscale
                else:
                    f = figure.subplot_imshow_color
                return f(x, y, image, *args, **kwargs)

            imshow(
                0,
                0,
                workspace.display_data.images[self.image_name.value],
                "Original image: %s" % self.image_name.value,
                #sharexy=figure.subplot(0, 0),
            )
            #title = f"{nametemplate} {illum_correct_function_image_name}, " \
            #        f"min={illum_image.min():0.4f}, max={illum_image.max():0.4f}"

            imshow(
                1, 
                0, 
                workspace.display_data.images[self.corrected_image_name.value], 
                "Corrected image: %s" % self.corrected_image_name.value,
                #title, sharexy=figure.subplot(0, 0)
            )
            imshow(
                0,
                1,
                workspace.display_data.images[self.flatfield_image_name.value],
                "Flatfield image: %s" % self.flatfield_image_name.value,
                #sharexy=figure.subplot(0, 0),
            )
            if self.darkfield_image_name is not None:
                imshow(
                    1,
                    1,
                    workspace.display_data.images[self.darkfield_image_name.value],
                    "Darkfield image: %s" % self.darkfield_image_name.value,
                #sharexy=figure.subplot(0, 0),
                )
