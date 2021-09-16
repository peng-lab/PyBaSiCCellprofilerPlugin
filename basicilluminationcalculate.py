#################################
#
# Imports from useful Python libraries
#
#################################
import time
import numpy as np
from typing import List
from skimage.transform import resize as skresize
from scipy.fftpack import dct, idct
#################################
#
# Imports from CellProfiler
#
##################################

__doc__ = """\
BaSiCIlluminationCalculate
=============

**BaSiCIlluminationCalculate** caluculates the background and shading correction of optical microscopy images.
The background and shading illumination is modeled as:

I_means(x) = I_true(x) * flatfield(x) + darkfield(x).

**BaSiCIlluminationCalculate** caluculates the darkfield and flatfield images for a set of input images.

Reference: Tingying Peng, Kurt Thorn, Timm Schroeder, Lichao Wang, Fabian J Theis, Carsten Marr, Nassir Navab, Nature Communication 8:14836 (2017). doi: 10.1038/ncomms14836.



|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

See also
^^^^^^^^

See also **BasicIlluminationApply**.

Technical notes
^^^^^^^^^^^^^^^

The BasicIlluminationCalculate plugin is implemented based 
on the structure of CorrectIlluminationCalculate module. 
"""


#
# Constants
#
# It's good programming practice to replace things like strings with
# constants if they will appear more than once in your program. That way,
# if someone wants to change the text, that text will change everywhere.
# Also, you can't misspell it by accident.
#
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.text import Float
from cellprofiler_core.setting.text import Integer
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import ImageName
from cellprofiler_core.image import AbstractImage
from cellprofiler_core.image import Image
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.module import Module
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT

RESIZE_ORDER = 1
RESIZE_MODE = "symmetric"
PRESERVE_RANGE = True
OUTPUT_IMAGE = "OutputImage"
FIRST_CYCLE = "First Cycle" 
LAST_CYCLE = "Last Cycle"


class BaSiCIlluminationCalculate(ImageProcessing):

    module_name = "BaSiCIlluminationCalculate"

    variable_revision_number = 1

    #
    # "create_settings" is where you declare the user interface elements
    # (the "settings") which the user will use to customize your module.
    #
    # You can look at other modules and in cellprofiler_core.settings for
    # settings you can use.
    #
    def create_settings(self):

        super(BaSiCIlluminationCalculate, self).create_settings()

        self.image_name = ImageSubscriber(
            "Select the input image",
            "None",
            doc="Choose the image to be used to calculate the illumination function.",
        )

        self.when_calculate = Choice(
            "During which cycle the background is calculated?",
            [FIRST_CYCLE, LAST_CYCLE],
            value=FIRST_CYCLE,
            doc="""\
During which cycle the background is calculated?
-  *%(FIRST_CYCLE)s: During the first cycle.
-  *%(LAST_CYCLE)s: During the first cycle.
"""
            % globals(),
        )

        self.if_darkfield = Binary(
            text="If Darkfield",
            value=True,
            doc="""\
Whether you would like to estimate darkfield, keep 'No' if the input images are brightfield
images or bright fluorescence images, set 'Yes' only if only if the input images are fluorescence 
images are dark and have a strong darkfield contribution. (default = 'No')",
"""
        )

        self.flatfield_image_name = ImageName(
            "Name the output flat-field image",
            "BasicFlatfield",
            doc="""Enter a name for the resultant flatfield image.""",
        )

        self.darkfield_image_name = ImageName(
            "Name the output dark-field image",
            "BasicDarkfield",
            doc="""Enter a name for the resultant darkfield image.""",
        )

        self.save_flatfield_image = Binary(
            "Retain the flat-field image?",
            False,
            doc="""\
You can save the flat-field image directly in the module.
"""
            % globals(),
        )

        self.save_darkfield_image = Binary(
            "Retain the dark-field image?",
            False,
            doc="""\
You can save the Dark-field image directly in the module.
"""
            % globals(),
        )

        self.if_baseline_drift = Binary(
            text="Baseline drift",
            value=False,
            doc="""\
Set to 'Yes' if input images has temporal drift (e.g. time lapse movie)  (default = 'No')",
"""
        )

        self.lambda_flatfield = Float(
            text="Flat-field regularization parameter",
            value=0,
            doc="""\
If you set the flat-field regularization parameter to 0 or a negative value, 
an internally estimated value is used. We recommend to use internally estimated 
value. High values (eg. 9.5) increase the spatial regularization strength, 
yielding a more smooth flat-field. A default value estimated from input images.
"""
        )

        self.lambda_darkfield = Float(
            text="Dark-field regularization parameter",
            value=0,
            doc="""\
If you set the dark-field regularization parameter to 0 or a negative value, 
an internally estimated value is used. We recommend to use internally estimated 
value. High values (eg. 9.5) increase the spatial regularization strength, 
yielding a more smooth dark-field. A default value estimated from input images.
"""
        )

        self.max_iterations = Integer(
            text="Maximum Iterations",
            value=500,
            doc="""\
Specifies the maximum number of iterations allowed in the optimization (default = 500).
"""
        )

        self.optimization_tolerance = Float(
            text="Tolerance of error in the optimization",
            value=1e-6,
        )

        self.working_size = Integer(
            text="Working size",
            value=128,
            doc="""\
An internal parameter, should not be reset by user without expert knowledge.
"""            
        )            

        self.max_reweight_iterations = Integer(
            text="Maximum reweighting iterations",
            value=10,
            doc="""\
An internal parameter should not be reset by the user without expert knowledge.
"""            
        )
        
        self.eplson = Float(
            text="Reweighting parameter",
            value=0.1,
            doc="""\
An internal parameter should not be reset by the user without expert knowledge.
"""            
        )
        
        self.varying_coeff = Binary(
            text="Varying coefficients",
            value=True,
            doc="""\
An internal parameter should not be reset by the user without expert knowledge.
"""            
        )
            
        self.reweight_tolerance = Float(
            text="Reweighting tolerance",
            value=1.0e-3,
            doc="""\
An internal parameter should not be reset by the user without expert knowledge.
""" 
        )           


    #
    # The "settings" method tells CellProfiler about the settings you
    # have in your module. CellProfiler uses the list for saving
    # and restoring values for your module when it saves or loads a
    # pipeline file.
    #

    def settings(self):
        #
        # The superclass's "settings" method returns [self.x_name, self.y_name],
        # which are the input and output image settings.
        #
        #settings = super(BasicIlluminationCalculate, self).settings()

        # Append additional settings here.
        return [
            self.image_name,
            self.when_calculate,
            self.if_darkfield,      
            self.if_baseline_drift,
            self.darkfield_image_name,
            self.flatfield_image_name,
            self.save_flatfield_image,
            self.save_darkfield_image,
            self.lambda_flatfield,
            self.lambda_darkfield,
            self.max_iterations,
            self.optimization_tolerance,
            self.working_size,
            self.max_reweight_iterations,
            self.eplson,
            self.varying_coeff,
            self.reweight_tolerance 
        ]

    def visible_settings(self):
        #
        # The superclass's "visible_settings" method returns [self.x_name,
        # self.y_name], which are the input and output image settings.
        #
        #visible_settings = super(BasicIlluminationCalculate, self).visible_settings()

        # Configure the visibility of additional settings below.
        visible_settings = [
            self.image_name,
            self.if_darkfield,
            self.darkfield_image_name,
            self.flatfield_image_name,         
            self.if_baseline_drift,
            self.lambda_flatfield,
            self.lambda_darkfield,
        ]
        return visible_settings

    #
    # "visible_settings" tells CellProfiler which settings should be
    # displayed and in what order.
    #
    # You don't have to implement "visible_settings" - if you delete
    # visible_settings, CellProfiler will use "settings" to pick settings
    # for display.
    #
    def get_pybasic_settings(self):
        return dict(
            if_darkfield = self.if_darkfield.value,
            if_baseline_drift = self.if_baseline_drift.value,
            lambda_flatfield = self.lambda_flatfield.value,
            lambda_darkfield = self.lambda_darkfield.value,
            max_iterations = self.max_iterations.value,
            optimization_tolerance = self.optimization_tolerance.value,
            working_size = self.working_size.value,
            max_reweight_iterations = self.max_reweight_iterations.value,
            eplson = self.eplson.value,
            varying_coeff = self.varying_coeff.value,
            reweight_tolerance = self.reweight_tolerance.value
        )


    def prepare_group(self, workspace, grouping, image_numbers):

        pipeline = workspace.pipeline
        assert isinstance(pipeline, Pipeline)
        m = workspace.measurements
        assert isinstance(m, Measurements)

        keys, groupings = pipeline.get_groupings(
                workspace
            )
        grouping = groupings[0][0]
        image_numbers = groupings[0][1]
        image_set_list = workspace.image_set_list #= ImageSetList()

        if len(image_numbers) > 0:
            title = "#%d: BasicIlluminationCalculate for %s" % (
                self.module_num,
                self.image_name,
            )
            message = (
                "BasicIlluminationCalculate is listing %d images while "
                "preparing for run" % (len(image_numbers))
            )

            output_image_provider = BaSiCIlluminationImageProvider(
                flatfield_image_name = self.darkfield_image_name.value,
                darkfield_image_name = self.flatfield_image_name.value, 
                module = self,
                pybasic_settings = self.get_pybasic_settings()
            )
            d = self.get_dictionary(image_set_list)[OUTPUT_IMAGE] = {}


            if self.when_calculate == FIRST_CYCLE:
            # Find the module that provides the image we need
                md = workspace.pipeline.get_provider_dictionary(
                    self.image_name.group, self
                )


                src_module, src_setting = md[self.image_name.value][-1]
                modules = list(pipeline.modules())
                idx = modules.index(src_module)
                last_module = modules[idx + 1]
                for w in pipeline.run_group_with_yield(
                    workspace, grouping, image_numbers, last_module, title, message
                ):
                    image = w.image_set.get_image(self.image_name.value, cache=False)
                    output_image_provider.add_image(image)
                    w.image_set.clear_cache()

            output_image_provider.serialize(d)
        self.image_counter = 0    
        return True

    def run(self, workspace):
        d = self.get_dictionary(workspace.image_set_list)[OUTPUT_IMAGE]
        output_image_provider = BaSiCIlluminationImageProvider.deserialize(
            d, self
        )
        workspace.image_set.providers.append(output_image_provider)
        flatfield = output_image_provider.provide_flatfield()
        darkfield = output_image_provider.provide_darkfield()
        baseline_drift = output_image_provider.provide_baseline_drift()
        #workspace.image_set.add("baselineDriftImage", baseline_drift)

        if self.if_baseline_drift.value is True:
            workspace.measurements.add_image_measurement("BaseLineDrift", baseline_drift[self.image_counter])
            self.image_counter += 1

        workspace.image_set.add(self.flatfield_image_name.value, flatfield)
        workspace.image_set.add(self.darkfield_image_name.value, darkfield)

        output_image_provider.serialize(d)


        
        if self.save_flatfield_image.value:
            workspace.image_set.add(self.flatfield_image_name.value, flatfield)
        if self.save_darkfield_image.value:
            workspace.image_set.add(self.darkfield_image_name.value, darkfield)
        
        if self.show_window:
            # store images for potential display
            workspace.display_data.flatfield = flatfield.pixel_data
            workspace.display_data.darkfield = darkfield.pixel_data
            if self.if_baseline_drift.value is True:
                workspace.display_data.baseline_drift = baseline_drift
    '''
    def post_group(self, workspace, grouping):
        """Handle tasks to be performed after a group has been processed

        For BasicIllumninationCalculate, we make sure the current image
        set includes the aggregate image. "run" may not have run if an
        image was filtered out.
        """
        image_set = workspace.image_set
        d = self.get_dictionary(workspace.image_set_list)[OUTPUT_IMAGE]
        output_image_provider = BaSiCIlluminationImageProvider.deserialize(
            d, self
        )
        assert isinstance(output_image_provider, BaSiCIlluminationImageProvider)
        if not self.flatfield_image_name.value in image_set.names:
            workspace.image_set.providers.append(output_image_provider)
        if not self.darkfield_image_name.value in image_set.names:
            workspace.image_set.providers.append(output_image_provider)            
    '''


    def display(self, workspace, figure):
        # these are actually just the pixel data
        flatfield_image = workspace.display_data.flatfield
        darkfield_image = workspace.display_data.darkfield

        
        if self.if_baseline_drift.value is True:
            figure.set_subplots((2, 2))
        else:
            figure.set_subplots((2, 1))

        def imshow(x, y, image, *args, **kwargs):
            if image.ndim == 2:
                f = figure.subplot_imshow_grayscale
            else:
                f = figure.subplot_imshow_color
            return f(x, y, image, *args, **kwargs)

        imshow(0, 0, flatfield_image, "Flat-field image")
        imshow(1, 0, darkfield_image, "Dark-field image")
        if self.if_baseline_drift.value is True:
            baseline_drift = workspace.display_data.baseline_drift
            figure.subplot_scatter(
                0,
                1, 
                range(len(baseline_drift)), 
                baseline_drift,
                xlabel="Image Number",
                ylabel="Drfit",
                title="Fluorescence",                
            )


    #
    # "volumetric" indicates whether or not this module supports 3D images.
    # The "pybasic" function is inherently 2D, and we've noted this
    # in the documentation for the module. Explicitly return False here
    # to indicate that 3D images are not supported.
    #
    def volumetric(self):
        return False

class BaSiCIlluminationImageProvider(AbstractImage):
    """BaSiCIlluminationImageProvider provides the illumination correction image

    This class accumulates the image data from successive images and
    calculates the flatfield and darkfield images when asked.
    """

    def __init__(self, flatfield_image_name, darkfield_image_name, module, pybasic_settings):
        super(BaSiCIlluminationImageProvider, self).__init__()
        self.__flatfield_image_name = flatfield_image_name
        self.__darkfield_image_name = darkfield_image_name
        self.__module = module
        self.if_baseline_drift = pybasic_settings['if_baseline_drift']
        self.__dirty = True
        self.__images_list = []
        self.__flatfield_pixel_data = None
        self.__darkfield_pixel_data = None
        self.__cached_flatfield = None
        self.__cached_darkfield = None
        self.__cached_baseline_drift_data = None
        self.__pybasic_settings = pybasic_settings


    D_FLATFIELD_NAME = "flatfield_image_name"
    D_DARKFIELD_NAME = "darkfield_image_name"
    D_FLATFIELD_PIXEL_DATA = "flatfield_pixel_data"
    D_DARKFIELD_PIXEL_DATA = "darkfieldf_pixel_data"
    D_BASELINE_DRIFT_DATA = "baseline_drift_data"
    D_IMAGES_LIST = "images_list"    
    D_PYBASIC_SETTINGS = "pybasic_settings"
    D_DIRTY = "dirty"

    def serialize(self, d):
        """Save the internal state of the provider to a dictionary

        d - save to this dictionary, numpy arrays and json serializable only
        Thus we can only save the pixel_data of the flatfield and darkfield and
        in deserialize() we have to call cache_output_images() in order to create
        new Image objects.
        """

        d[self.D_FLATFIELD_NAME] = self.__flatfield_image_name
        d[self.D_DARKFIELD_NAME] = self.__darkfield_image_name        
        d[self.D_FLATFIELD_PIXEL_DATA] = self.__flatfield_pixel_data
        d[self.D_DARKFIELD_PIXEL_DATA] = self.__darkfield_pixel_data
        d[self.D_BASELINE_DRIFT_DATA] = self.__cached_baseline_drift_data
        d[self.D_IMAGES_LIST] = self.__images_list
        d[self.D_PYBASIC_SETTINGS] = self.__pybasic_settings
        d[self.D_DIRTY] = self.__dirty

    @staticmethod
    def deserialize(d, module):
        """Restore a state saved by serialize

        d - dictionary containing the state
        module - the module providing details on how to perform the correction

        returns a provider set up with the restored state
        """

        provider = BaSiCIlluminationImageProvider(
            d[BaSiCIlluminationImageProvider.D_FLATFIELD_NAME], 
            d[BaSiCIlluminationImageProvider.D_DARKFIELD_NAME],
            module,
            d[BaSiCIlluminationImageProvider.D_PYBASIC_SETTINGS]
        )

        provider.__images_list = d[BaSiCIlluminationImageProvider.D_IMAGES_LIST]
        provider.__dirty = d[BaSiCIlluminationImageProvider.D_DIRTY]
        provider.__flatfield_image_name = d[BaSiCIlluminationImageProvider.D_FLATFIELD_NAME]
        provider.__darkfield_image_name = d[BaSiCIlluminationImageProvider.D_DARKFIELD_NAME]        
        provider.__flatfield_pixel_data = d[BaSiCIlluminationImageProvider.D_FLATFIELD_PIXEL_DATA]
        provider.__darkfield_pixel_data = d[BaSiCIlluminationImageProvider.D_DARKFIELD_PIXEL_DATA]
        provider.__cached_baseline_drift_data = d[BaSiCIlluminationImageProvider.D_BASELINE_DRIFT_DATA]
        provider.__pybasic_settings = d[BaSiCIlluminationImageProvider.D_PYBASIC_SETTINGS]

        provider.cache_output_images()

        return provider

    def add_image(self, image):
        """List the images from the given image

        image - an instance of cellprofiler.cpimage.Image, including
                image data
        """
        self.__dirty = True
        self.__images_list.append(image.pixel_data)

    def provide_background(self):
        if self.__dirty:
            self.run_pybasic()
        return self.__flatfield_image_name, self.__darkfield_image_name

    def provide_images_list(self):
        if self.__dirty:
            self.run_pybasic()
        return self.__images_list

    def provide_flatfield(self):
        if self.__dirty:
            self.run_pybasic()
        return self.__cached_flatfield

    def provide_darkfield(self):
        if self.__dirty:
            self.run_pybasic()
        return self.__cached_darkfield

    def provide_baseline_drift(self):
        if self.__dirty:
            self.run_pybasic()
        return self.__cached_baseline_drift_data

    def cache_output_images(self):
        self.__cached_flatfield = Image(self.__flatfield_pixel_data)
        self.__cached_darkfield = Image(self.__darkfield_pixel_data)        

    def run_pybasic(self):
        self.__flatfield_pixel_data, self.__darkfield_pixel_data = pybasic(
            images_list = self.__images_list,
            **self.__pybasic_settings
        )
        if self.if_baseline_drift:
            _drift = baseline_drift(
                images_list = self.__images_list,
                flatfield = self.__flatfield_pixel_data,
                darkfield = self.__darkfield_pixel_data,
                working_size = self.__pybasic_settings['working_size'],
            )
            self.__cached_baseline_drift_data = _drift
        
        self.cache_output_images()
        self.__dirty = False





    def reset(self):
        self.__images_list = []
        self.__cached_flatfield = None
        self.__cached_darkfield = None
        self.__cached_baseline_drift_data = None

    def get_name(self):
        return self.__flatfield_image_name, self.__darkfield_image_name

    def release_memory(self):
        # Memory is released during reset(), so this is a no-op
        pass
#
# This is the function that gets called during "run" to create the output image.
# The first parameter must be the input image data. The remaining parameters are
# the additional settings defined in "settings", in the order they are returned.
#
# This function must return the output image data (as a numpy array).
#
def pybasic(
        images_list, 
        if_darkfield,
        if_baseline_drift,
        lambda_flatfield,
        lambda_darkfield,
        max_iterations,
        optimization_tolerance,
        working_size,
        max_reweight_iterations,
        eplson,
        varying_coeff,
        reweight_tolerance 
    ):
    
    nrows = ncols = working_size
    _saved_size = images_list[0].shape
    # resizing the images
    if (
        working_size<=0 
        or working_size is None
    ):
        _working_size = 128
    else:
        _working_size = working_size

    D = np.dstack(_resize_images_list(images_list=images_list, side_size=_working_size))

    meanD = np.mean(D, axis=2)
    meanD = meanD / np.mean(meanD)
    W_meanD = _dct2d(meanD.T) 

    # setting lambda_flatfield and lambda_darkfield if they are not set by the user
    if lambda_flatfield <= 0:
        lambda_flatfield = np.sum(np.abs(W_meanD)) / 400 * 0.5
    if lambda_darkfield <= 0:
        lambda_darkfield = lambda_flatfield * 0.2

    D = np.sort(D, axis=2)

    XAoffset = np.zeros((nrows, ncols))
    weight = np.ones(D.shape)

    reweighting_iter = 0
    flag_reweighting = True
    flatfield_last = np.ones((nrows, ncols))
    darkfield_last = np.random.randn(nrows, ncols)

    while flag_reweighting:
        reweighting_iter += 1

        initial_flatfield = False
        if initial_flatfield:
            raise IOError('Initial flatfield option not implemented yet!')
        else:
            X_k_A, X_k_E, X_k_Aoffset = _inexact_alm_rspca_l1(
                images = D, 
                lambda_flatfield = lambda_flatfield,
                if_darkfield = if_darkfield, 
                lambda_darkfield = lambda_darkfield, 
                optimization_tolerance = optimization_tolerance, 
                max_iterations = max_iterations,
                weight=weight
            )

        XA = np.reshape(X_k_A, [nrows, ncols, -1], order='F')
        XE = np.reshape(X_k_E, [nrows, ncols, -1], order='F')
        XAoffset = np.reshape(X_k_Aoffset, [nrows, ncols], order='F')
        XE_norm = XE / np.mean(XA, axis=(0, 1))

        weight = np.ones_like(XE_norm) / (np.abs(XE_norm) + eplson)

        weight = weight * weight.size / np.sum(weight)

        temp = np.mean(XA, axis=2) - XAoffset
        flatfield_current = temp / np.mean(temp)
        darkfield_current = XAoffset
        mad_flatfield = np.sum(np.abs(flatfield_current - flatfield_last)) / np.sum(np.abs(flatfield_last))
        temp_diff = np.sum(np.abs(darkfield_current - darkfield_last))
        if temp_diff < 1e-7:
            mad_darkfield = 0
        else:
            mad_darkfield = temp_diff / np.maximum(np.sum(np.abs(darkfield_last)), 1e-6)
        flatfield_last = flatfield_current
        darkfield_last = darkfield_current
        if np.maximum(mad_flatfield,
                      mad_darkfield) <= reweight_tolerance or \
                reweighting_iter >= max_reweight_iterations:
            flag_reweighting = False

    shading = np.mean(XA, 2) - XAoffset
    flatfield = _resize_image(
        image = shading, 
        x_side_size = _saved_size[0], 
        y_side_size = _saved_size[1]
    )
    flatfield = flatfield / np.mean(flatfield)

    if if_darkfield:
        darkfield = _resize_image(
            image = XAoffset, 
            x_side_size = _saved_size[0], 
            y_side_size = _saved_size[1]
        )
    else:
        darkfield = np.zeros_like(flatfield)
        
    return flatfield, darkfield


def baseline_drift(
        images_list,
        working_size = 128,
        flatfield: np.ndarray = None,
        darkfield: np.ndarray = None,
        **kwargs
        ):

    nrows = ncols = working_size

    # Preparing input images
    resized_images = np.stack(_resize_images_list(images_list = images_list, side_size = working_size))
    resized_images = resized_images.reshape([-1, nrows * nrows], order = 'F')

    # Reszing flat- and dark-field
    resized_flatfield = _resize_image(image = flatfield, side_size = working_size)
    resized_darkfield = _resize_image(image = darkfield, side_size = working_size)
            
    # reweighting     
    _weights = np.ones(resized_images.shape)
    eplson = 0.1
    tol = 1e-6
    for reweighting_iter in range(1,6):
        W_idct_hat = np.reshape(resized_flatfield, (1,-1), order='F')
        A_offset = np.reshape(resized_darkfield, (1,-1), order='F')
        A1_coeff = np.mean(resized_images, 1).reshape([-1,1])

        # main iteration loop starts:
        # The first element of the second array of np.linalg.svd
        _temp = np.linalg.svd(resized_images, full_matrices=False)[1]
        norm_two = _temp[0]

        mu = 12.5/norm_two # this one can be tuned
        mu_bar = mu * 1e7
        rho = 1.5 # this one can be tuned
        d_norm = np.linalg.norm(resized_images, ord = 'fro')
        ent1 = 1
        _iter = 0
        total_svd = 0
        converged = False;
        A1_hat = np.zeros(resized_images.shape)
        E1_hat = np.zeros(resized_images.shape)
        Y1 = 0
            
        while not converged:
            _iter = _iter + 1;
            A1_hat = W_idct_hat * A1_coeff + A_offset

            # update E1 using l0 norm
            E1_hat = E1_hat + np.divide((resized_images - A1_hat - E1_hat + (1/mu)*Y1), ent1)
            E1_hat = np.maximum(E1_hat - _weights/(ent1*mu), 0) +\
                     np.minimum(E1_hat + _weights/(ent1*mu), 0)
            # update A1_coeff, A2_coeff and A_offset
            #if coeff_flag
            
            R1 = resized_images - E1_hat
            A1_coeff = np.mean(R1,1).reshape(-1,1) - np.mean(A_offset,1)

            A1_coeff[A1_coeff<0] = 0
                
            Z1 = resized_images - A1_hat - E1_hat

            Y1 = Y1 + mu*Z1

            mu = min(mu*rho, mu_bar)
                
            # stop Criterion  
            stopCriterion = np.linalg.norm(Z1, ord = 'fro') / d_norm
            if stopCriterion < tol:
                converged = True
                
        # updating weight
        # XE_norm = E1_hat / np.mean(A1_hat)
        XE_norm = E1_hat
        mean_vec = np.mean(A1_hat, axis=1)
        XE_norm = np.transpose(np.tile(mean_vec, (nrows * ncols, 1))) / (XE_norm + 1e-6)
        _weights = 1./(abs(XE_norm)+eplson)

        _weights = np.divide( np.multiply(_weights, _weights.shape[0] * _weights.shape[1]), np.sum(_weights))
    return np.squeeze(A1_coeff) 



def _inexact_alm_rspca_l1(
    images, 
    lambda_flatfield, 
    if_darkfield, 
    lambda_darkfield, 
    optimization_tolerance, 
    max_iterations,
    weight=None, 
    ):

    if weight is not None and weight.size != images.size:
            raise IOError('weight matrix has different size than input sequence')

    # if 
    # Initialization and given default variables
    p = images.shape[0]
    q = images.shape[1]
    m = p*q
    n = images.shape[2]
    images = np.reshape(images, (m, n), order='F')

    if weight is not None:
        weight = np.reshape(weight, (m, n), order='F')
    else:
        weight = np.ones_like(images)
    _, svd, _ = np.linalg.svd(images, full_matrices=False) #TODO: Is there a more efficient implementation of SVD?
    norm_two = svd[0]
    Y1 = 0
    #Y2 = 0
    ent1 = 1
    ent2 = 10

    A1_hat = np.zeros_like(images)
    A1_coeff = np.ones((1, images.shape[1]))

    E1_hat = np.zeros_like(images)
    W_hat = _dct2d(np.zeros((p, q)).T)
    mu = 12.5 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    d_norm = np.linalg.norm(images, ord='fro')

    A_offset = np.zeros((m, 1))
    B1_uplimit = np.min(images)
    B1_offset = 0
    #A_uplimit = np.expand_dims(np.min(images, axis=1), 1)
    A_inmask = np.zeros((p, q))
    A_inmask[int(np.round(p / 6) - 1): int(np.round(p*5 / 6)), int(np.round(q / 6) - 1): int(np.round(q * 5 / 6))] = 1

    # main iteration loop starts
    iter = 0
    total_svd = 0
    converged = False

    #time_zero = time.time()
    #time_zero_it = time.time()
    while not converged:
    #    time_zero_it = time.time()
        iter += 1

        if len(A1_coeff.shape) == 1:
            A1_coeff = np.expand_dims(A1_coeff, 0)
        if len(A_offset.shape) == 1:
            A_offset = np.expand_dims(A_offset, 1)
        W_idct_hat = _idct2d(W_hat.T)
        A1_hat = np.dot(np.reshape(W_idct_hat, (-1,1), order='F'), A1_coeff) + A_offset

        temp_W = (images - A1_hat - E1_hat + (1 / mu) * Y1) / ent1
        temp_W = np.reshape(temp_W, (p, q, n), order='F')
        temp_W = np.mean(temp_W, axis=2)
        W_hat = W_hat + _dct2d(temp_W.T)
        W_hat = np.maximum(W_hat - lambda_flatfield / (ent1 * mu), 0) + np.minimum(W_hat + lambda_flatfield / (ent1 * mu), 0)
        W_idct_hat = _idct2d(W_hat.T)
        if len(A1_coeff.shape) == 1:
            A1_coeff = np.expand_dims(A1_coeff, 0)
        if len(A_offset.shape) == 1:
            A_offset = np.expand_dims(A_offset, 1)
        A1_hat = np.dot(np.reshape(W_idct_hat, (-1,1), order='F'), A1_coeff) + A_offset
        E1_hat = images - A1_hat + (1 / mu) * Y1 / ent1
        E1_hat = _shrinkageOperator(E1_hat, weight / (ent1 * mu))
        R1 = images - E1_hat
        A1_coeff = np.mean(R1, 0) / np.mean(R1)
        A1_coeff[A1_coeff < 0] = 0

        if if_darkfield:
            validA1coeff_idx = np.where(A1_coeff < 1)

            B1_coeff = (np.mean(R1[np.reshape(W_idct_hat, -1, order='F') > np.mean(W_idct_hat) - 1e-6][:, validA1coeff_idx[0]], 0) - \
            np.mean(R1[np.reshape(W_idct_hat, -1, order='F') < np.mean(W_idct_hat) + 1e-6][:, validA1coeff_idx[0]], 0)) / np.mean(R1)
            k = np.array(validA1coeff_idx).shape[1]
            temp1 = np.sum(A1_coeff[validA1coeff_idx[0]]**2)
            temp2 = np.sum(A1_coeff[validA1coeff_idx[0]])
            temp3 = np.sum(B1_coeff)
            temp4 = np.sum(A1_coeff[validA1coeff_idx[0]] * B1_coeff)
            temp5 = temp2 * temp3 - temp4 * k
            if temp5 == 0:
                B1_offset = 0
            else:
                B1_offset = (temp1 * temp3 - temp2 * temp4) / temp5
            # limit B1_offset: 0<B1_offset<B1_uplimit

            B1_offset = np.maximum(B1_offset, 0)
            B1_offset = np.minimum(B1_offset, B1_uplimit / np.mean(W_idct_hat))

            B_offset = B1_offset * np.reshape(W_idct_hat, -1, order='F') * (-1)

            B_offset = B_offset + np.ones_like(B_offset) * B1_offset * np.mean(W_idct_hat)
            A1_offset = np.mean(R1[:, validA1coeff_idx[0]], axis=1) - np.mean(A1_coeff[validA1coeff_idx[0]]) * np.reshape(W_idct_hat, -1, order='F')
            A1_offset = A1_offset - np.mean(A1_offset)
            A_offset = A1_offset - np.mean(A1_offset) - B_offset

            # smooth A_offset
            W_offset = _dct2d(np.reshape(A_offset, (p,q), order='F').T)
            W_offset = np.maximum(W_offset - lambda_darkfield / (ent2 * mu), 0) + \
                np.minimum(W_offset + lambda_darkfield / (ent2 * mu), 0)
            A_offset = _idct2d(W_offset.T)
            A_offset = np.reshape(A_offset, -1, order='F')

            # encourage sparse A_offset
            A_offset = np.maximum(A_offset - lambda_darkfield / (ent2 * mu), 0) + \
                np.minimum(A_offset + lambda_darkfield / (ent2 * mu), 0)
            A_offset = A_offset + B_offset


        Z1 = images - A1_hat - E1_hat
        Y1 = Y1 + mu * Z1
        mu = np.minimum(mu * rho, mu_bar)

        # Stop Criterion
        stopCriterion = np.linalg.norm(Z1, ord='fro') / d_norm
        if stopCriterion < optimization_tolerance:
            converged = True

        if not converged and iter >= max_iterations:
            converged = True
    A_offset = np.squeeze(A_offset)
    A_offset = A_offset + B1_offset * np.reshape(W_idct_hat, -1, order='F')

    return A1_hat, E1_hat, A_offset



def _resize_images_list(images_list: List, side_size: float = None, x_side_size: float = None, y_side_size: float = None):
    if side_size is not None:
        y_side_size = x_side_size = side_size
    resized_images_list = []
    for i, im in enumerate(images_list):
        if im.shape[0] != x_side_size or im.shape[1] != y_side_size:
            resized_images_list.append(skresize(
                im, 
                (x_side_size, y_side_size), 
                order = RESIZE_ORDER, 
                mode = RESIZE_MODE,
                preserve_range = PRESERVE_RANGE
                )
            )
        else:
            resized_images_list.append(im)
    return resized_images_list

def _resize_image(image: np.ndarray, side_size: float  = None, x_side_size: float = None, y_side_size: float = None):
    if side_size is not None:
        y_side_size = x_side_size = side_size
    if image.shape[0] != x_side_size or image.shape[1] != y_side_size:
        return skresize(
            image,
            (x_side_size, y_side_size), 
            order = RESIZE_ORDER, 
            mode = RESIZE_MODE,
            preserve_range = PRESERVE_RANGE
        )
    else:
        return image


def _shrinkageOperator(matrix, epsilon):
    temp1 = matrix - epsilon
    temp1[temp1 < 0] = 0
    temp2 = matrix + epsilon
    temp2[temp2 > 0] = 0
    res = temp1 + temp2
    return res

def _dct2d(mtrx: np.array):
    """
    Calculates 2D discrete cosine transform.
    
    Parameters
    ----------
    mtrx
        Input matrix.  
        
    Returns
    -------    
    Discrete cosine transform of the input matrix.
    """
     
    # Check if input object is 2D.
    if mtrx.ndim != 2:
        raise ValueError("Passed object should be a matrix or a numpy array with dimension of two.")

    return dct(dct(mtrx.T, norm='ortho').T, norm='ortho')

def _idct2d(mtrx: np.array):
    """
    Calculates 2D inverse discrete cosine transform.
    
    Parameters
    ----------
    mtrx
        Input matrix.  
        
    Returns
    -------    
    Inverse of discrete cosine transform of the input matrix.
    """
     
    # Check if input object is 2D.
    if mtrx.ndim != 2:
        raise ValueError("Passed object should be a matrix or a numpy array with dimension of two.")
 
    return idct(idct(mtrx.T, norm='ortho').T, norm='ortho')