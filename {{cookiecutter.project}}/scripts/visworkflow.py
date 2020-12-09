#!/usr/bin/env python
"""{{ cookiecutter.project }} vis analysis."""

import os
import argparse
import cppcpyutils as cppc
from plantcv import plantcv as pcv
# import matplotlib
# matplotlib.use("Agg") #<---- make sure to explicitly set display backend to Agg
# from skimage import filters
# from skimage import morphology
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module='plotnine')


# Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(
        description="Imaging processing with opencv")
    parser.add_argument("-i",
                        "--image",
                        help="Input image file.",
                        required=True)
    parser.add_argument("-o",
                        "--outdir",
                        help="Output directory for image files.",
                        required=False)
    parser.add_argument("-r", "--result", help="result file.", required=False)
    parser.add_argument("-w",
                        "--writeimg",
                        help="write out images.",
                        default=False,
                        action="store_true")
    parser.add_argument(
        "-D",
        "--debug",
        help=
        "can be set to 'print' or None (or 'plot' if in jupyter) prints intermediate images.",
        default=None)
    parser.add_argument("-d",
                        "--debugdir",
                        help="set your debug_dir",
                        required=False)
    parser.add_argument("--regex",
                        help="Format to parse filename into metadata",
                        required=False)
    parser.add_argument("-p",
                        "--pdfs",
                        help="Naive Bayes PDF file.",
                        required=False,
                        default=False)
    args = parser.parse_args()
    return args


# Main workflow
def main():

    # Get options
    args = options()
    # plt.rcParams["font.family"] = "Arial"  # All text is Arial

    # pixel_resolution
    cppc.pixelresolution = 0.052  #mm
    # see pixel_resolution.xlsx for calibration curve for pixel to mm translation

    pcv.params.text_size = 12
    pcv.params.text_thickness = 12

    if args.debug:
        pcv.params.debug = args.debug  # set debug mode
        if args.debugdir:
            pcv.params.debug_outdir = args.debugdir  # set debug directory
            os.makedirs(args.debugdir, exist_ok=True)

    args = cppc.roi.copy_metadata(args)

    # read images and create mask
    img, _, fn = pcv.readimage(args.image)
    args.imagename = os.path.splitext(fn)[0]
    # cps=pcv.visualize.colorspaces(img)

    # create mask
    if args.pdfs:  #if not provided in run_workflows.sh then will be False
        print('naive bayes classification')

        # Classify each pixel as plant or background (background and system components)
        img_blur = pcv.gaussian_blur(img, (7, 7))
        masks = pcv.naive_bayes_classifier(rgb_img=img_blur,
                                           pdf_file=args.pdfs)
        mask = masks['Plant']

        # save masks
        colored_img = pcv.visualize.colorize_masks(
            masks=[masks['Plant'], masks['Background'], masks['Blue']],
            colors=['green', 'black', 'blue'])
        # Print out the colorized figure that got created
        imgdir = os.path.join(args.outdir, 'bayesmask_images')
        os.makedirs(imgdir, exist_ok=True)
        pcv.print_image(
            colored_img, os.path.join(imgdir,
                                      args.imagename + '-bayesmask.png'))

    else:
        print('\nthreshold masking')

        # tray mask
        # _, rm, _, _ = pcv.rectangle_mask(img, (425,350), (2100,3050),'white')
        # img_tray = pcv.apply_mask(img, rm, 'black')

        # dark green
        # imgt_h = pcv.rgb2gray_hsv(img,'h')
        mask1, img1 = pcv.threshold.custom_range(img, [15, 0, 0],
                                                 [60, 255, 255], 'hsv')
        mask1 = pcv.fill(mask1, 200)
        mask1 = pcv.closing(mask1, pcv.get_kernel((5, 5), 'rectangle'))

        mask = mask1
        # img1 = pcv.apply_mask(img, mask1, 'black')

        # # remove faint algae
        # img1_a = pcv.rgb2gray_lab(img1,'b')
        # # img1_b = pcv.rgb2gray_lab(img1,'b')
        # th = filters.threshold_otsu(img1_a)
        # algaemask = pcv.threshold.binary(img1_a,th,255,'light')
        # # bmask, _ = pcv.threshold.custom_range(img1,[0,0,100],[120,120,255], 'RGB')
        # img2 = pcv.apply_mask(img1,algaemask,'black')

        # mask = pcv.rgb2gray(img2)
        # mask[mask > 0] = 255
        # pcv.plot_image(mask)

    # find objects based on threshold mask
    c, h = pcv.find_objects(img, mask)
    # setup roi based on pot locations
    rc, rh = pcv.roi.multi(img, coord=[(1250, 1000), (1250, 2300)], radius=300)
    # Turn off debug temporarily if activated, otherwise there will be a lot of plots
    pcv.params.debug = None
    # Loop over each region of interest
    # i=0
    # rc_i = rc[i]
    # rh_i = rh[i]
    final_mask = cppc.roi.iterate_rois(img,
                                       c,
                                       h,
                                       rc,
                                       rh,
                                       args=args,
                                       masked=True,
                                       gi=True,
                                       shape=True,
                                       hist=True,
                                       hue=True)
    # pcv.plot_image(final_mask)


# end of function!

if __name__ == '__main__':
    main()
