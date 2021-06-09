import numpy as np
import cppcpyutils as cppc
from plantcv import plantcv as pcv
from skimage import filters
from skimage import morphology
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module='plotnine')

plt.rcParams['figure.figsize'] = [12,12]
# os.chdir('..')


class options():
    def __init__(self):
        self.image = "data/vis/A2-doi-20200531T120435-VIS0-0.png"
        self.outdir = "output/vistest"
        self.result = "output/vistest/result.json"
        self.regex = "(.{2})-(.+)-(\d{8}T\d{6})-(.+)-(\d{1})"
        self.debug = 'plot'
        self.debugdir = 'debug/vistest'
        self.writeimg = True
        self.pdfs = False
        # self.pdfs = 'data/naive_bayes_training/naive_bayes_pdfs.tsv'

# global args
# args = options()
