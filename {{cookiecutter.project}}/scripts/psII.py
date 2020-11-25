# %% Setup
# Export .png to outdir from LemnaBase using LT-db_extractor.py
from plantcv import plantcv as pcv
import cppcpyutils as cppc
import importlib
import os
import cv2 as cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
import importlib
from skimage import filters
from skimage import morphology
from skimage import segmentation


# from tinydb import TinyDB, Query

warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module='plotnine')

# %% io directories
indir = os.path.join('data', 'psII')
# snapshotdir = indir
outdir = os.path.join('output', 'psII')
debugdir = os.path.join('debug', 'psII')
maskdir = os.path.join(outdir, 'masks')
fluordir = os.path.join(outdir, 'fluorescence')
os.makedirs(outdir, exist_ok=True)
os.makedirs(debugdir, exist_ok=True)
os.makedirs(maskdir, exist_ok=True)
# %% pixel pixel_resolution
# mm (this is approx and should only be used for scalebar)
cppc.pixelresolution = 0.3

# %% Import tif file information based on the filenames. If extract_frames=True it will save each frame form the multiframe TIF to a separate file in data/pimframes/ with a numeric suffix
fdf = cppc.io.import_snapshots(indir, camera='psii')
# %% Define the frames from the PSII measurements and merge this information with the filename information
pimframes = pd.read_csv(os.path.join('data', 'pimframes_map.csv'),
                        skipinitialspace=True)
# this eliminate weird whitespace around any of the character fields
fdf_dark = (pd.merge(fdf.reset_index(), pimframes, on=['frameid'],
                     how='right'))

# %% remove absorptivity measurements which are blank images
# also remove Ft_FRon measurements. THere is no Far Red light.
df = (fdf_dark.query(
    '~parameter.str.contains("Abs") and ~parameter.str.contains("FRon")',
    engine='python'))

# %% remove the duplicate Fm and Fo frames where frame = Fmp and Fp from frameid 5,6
df = (df.query(
    '(parameter!="FvFm") or (parameter=="FvFm" and (frame=="Fo" or frame=="Fm") )'
))

# %% Arrange dataframe of metadata so Fv/Fm comes first
param_order = pimframes.parameter.unique()
df['parameter'] = pd.Categorical(df.parameter,
                                 categories=param_order,
                                 ordered=True)
# %% Setup Debug parmaeters
# pcv.params.debug can be 'plot', 'print', or 'None'. 'plot' is useful if you are testing your pipeline over a few samples so you can see each step.
pcv.params.debug = 'plot'  # 'print' #'plot', 'None'
# Figures will show 9x9inches which fits my monitor well.
plt.rcParams["figure.figsize"] = (9, 9)
# plt.rcParams["font.family"] = "Arial"  # All text is Arial
ilegend=1

# %% The main analysis function

# This function takes a dataframe of metadata that was created above. We loop through each pair of images to compute photosynthetic parameters
def image_avg(fundf):
    # dn't understand why import suddently needs to be inside function
    # import cv2 as cv2
    # import numpy as np
    # import pandas as pd
    # import os
    # from matplotlib import pyplot as plt
    # from skimage import filters
    # from skimage import morphology
    # from skimage import segmentation

    # Predefine some variables
    global c, h, roi_c, roi_h, ilegend, mask_Fm, fn_Fm

    # Get the filename for minimum and maximum fluoresence
    fn_min = fundf.query('frame == "Fo" or frame == "Fp"').filename.values[0]
    fn_max = fundf.query('frame == "Fm" or frame == "Fmp"').filename.values[0]

    # Get the parameter name that links these 2 frames
    param_name = fundf['parameter'].iloc[0]

    # Create a new output filename that combines existing filename with parameter
    outfn = os.path.splitext(os.path.basename(fn_max))[0]
    outfn_split = outfn.split('-')
    # outfn_split[2] = datetime.strptime(fundf.jobdate.values[0],'%Y-%m-%d').strftime('%Y%m%d')
    outfn_split[2] = fundf.jobdate.dt.strftime('%Y%m%d').values[0]
    basefn = "-".join(outfn_split[0:-1])
    outfn_split[-1] = param_name
    outfn = "-".join(outfn_split)
    print(outfn)

    # Make some directories based on sample id to keep output organized
    plantbarcode = outfn_split[0]
    fmaxdir = os.path.join(fluordir, plantbarcode)
    os.makedirs(fmaxdir, exist_ok=True)

    # If debug mode is 'print', create a specific debug dir for each pim file
    if pcv.params.debug == 'print':
        debug_outdir = os.path.join(debugdir, outfn)
        os.makedirs(debug_outdir, exist_ok=True)
        pcv.params.debug_outdir = debug_outdir

    # read images and create mask from max fluorescence
    # read image as is. only gray values in PSII images
    imgmin, _, _ = pcv.readimage(fn_min)
    img, _, _ = pcv.readimage(fn_max)
    fdark = np.zeros_like(img)
    out_flt = fdark.astype('float32')  # <- needs to be float32 for imwrite

    if param_name == 'FvFm':
        # save max fluorescence filename
        fn_Fm = fn_max

        # create mask
        # #create black mask over lower half of image to threshold upper plant only
        # img_half, _, _, _ = pcv.rectangle_mask(img, p1=(0,321), p2=(480,640))
        # # mask1 = pcv.threshold.otsu(img_half,255)
        # algaethresh = filters.threshold_otsu(image=img_half)
        # mask0 = pcv.threshold.binary(img_half, algaethresh, 255, 'light')

        # # create black mask over upper half of image to threshold lower plant only
        # img_half, _, _, _ = pcv.rectangle_mask(img, p1=(0, 0), p2=(480, 319), color='black')
        # # mask0 = pcv.threshold.otsu(img_half,255)
        # algaethresh = filters.threshold_otsu(image=img_half)
        # mask1 = pcv.threshold.binary(img_half, algaethresh, 255, 'light')

        # mask = pcv.logical_xor(mask0, mask1)
        # # mask = pcv.dilate(mask, 2, 1)
        # mask = pcv.fill(mask, 350)
        # mask = pcv.erode(mask, 2, 2)

        # mask = pcv.erode(mask, 2, 1)
        # mask = pcv.fill(mask, 100)

        # otsuT = filters.threshold_otsu(img)
        # # sigma=(k-1)/6. This is because the length for 99 percentile of gaussian pdf is 6sigma.
        # k = int(2 * np.ceil(3 * otsuT) + 1)
        # gb = pcv.gaussian_blur(img, ksize = (k,k), sigma_x = otsuT)
        # mask = img >= gb + 10
        # pcv.plot_image(mask)

        # local_otsu = filters.rank.otsu(img, pcv.get_kernel((9,9), 'rectangle'))#morphology.disk(2))
        # thresh_image = img >= local_otsu


        #_------>
        elevation_map = filters.sobel(img)
        # pcv.plot_image(elevation_map)
        thresh = filters.threshold_otsu(image=img)
        # thresh = 50

        markers = np.zeros_like(img, dtype = 'uint8')
        markers[img > thresh+8] = 2
        markers[img <= thresh+8] = 1
        # pcv.plot_image(markers,cmap=plt.cm.nipy_spectral)

        mask = segmentation.watershed(elevation_map, markers)
        mask = mask.astype(np.uint8)
        # pcv.plot_image(mask)

        mask[mask == 1] = 0
        mask[mask == 2] = 1
        # pcv.plot_image(mask, cmap=plt.cm.nipy_spectral)

        # mask = pcv.erode(mask, 2, 1)
        mask = pcv.fill(mask, 100)
        # pcv.plot_image(mask, cmap=plt.cm.nipy_spectral)
        # <-----------
        roi_c, roi_h = pcv.roi.multi(img,
                                    coord=(250, 200),
                                    radius=70,
                                    spacing=(0, 220),
                                    ncols=1,
                                    nrows=2)

        if len(np.unique(mask)) == 1:
            c = []
            YII = mask
            NPQ = mask
            newmask = mask
        else:
            # find objects and setup roi
            c, h = pcv.find_objects(img, mask)

            # setup individual roi plant masks
            newmask = np.zeros_like(mask)

            # compute fv/fm and save to file
            YII, hist_fvfm = pcv.photosynthesis.analyze_fvfm(fdark=fdark,
                                        fmin=imgmin,
                                        fmax=img,
                                        mask=mask,
                                        bins=128)
            # YII = np.divide(Fv,
            #                 img,
            #                 out=out_flt.copy(),
            #                 where=np.logical_and(mask > 0, img > 0))

            # NPQ is 0
            NPQ = np.zeros_like(YII)

        # cv2.imwrite(os.path.join(fmaxdir, outfn + '-fvfm.tif'), YII)
        # print Fm - will need this later
        # cv2.imwrite(os.path.join(fmaxdir, outfn + '-fmax.tif'), img)
        # NPQ will always be an array of 0s

    else:  # compute YII and NPQ if parameter is other than FvFm
        newmask = mask_Fm
        # use cv2 to read image becase pcv.readimage will save as input_image.png overwriting img
        # newmask = cv2.imread(os.path.join(maskdir, basefn + '-FvFm-mask.png'),-1)
        if len(np.unique(newmask)) == 1:
            YII = np.zeros_like(newmask)
            NPQ = np.zeros_like(newmask)

        else:
            # compute YII
            YII, hist_yii = pcv.photosynthesis.analyze_fvfm(fdark,
                                        fmin=imgmin,
                                        fmax=img,
                                        mask=newmask,
                                        bins=128)
            # make sure to initialize with out=. using where= provides random values at False pixels. you will get a strange result. newmask comes from Fm instead of Fm' so they can be different
            #newmask<0, img>0 = FALSE: not part of plant but fluorescence detected.
            #newmask>0, img<=0 = FALSE: part of plant in Fm but no fluorescence detected <- this is likely the culprit because pcv.apply_mask doesn't always solve issue.
            # YII = np.divide(Fvp,
            #                 img,
            #                 out=out_flt.copy(),
            #                 where=np.logical_and(newmask > 0, img > 0))

            # compute NPQ
            # Fm = cv2.imread(os.path.join(fmaxdir, basefn + '-FvFm-fmax.tif'), -1)
            Fm = cv2.imread(fn_Fm, -1)
            NPQ = np.divide(Fm,
                            img,
                            out=out_flt.copy(),
                            where=np.logical_and(newmask > 0, img > 0))
            NPQ = np.subtract(NPQ,
                            1,
                            out=out_flt.copy(),
                            where=np.logical_and(NPQ >= 1, newmask > 0))

        # cv2.imwrite(os.path.join(fmaxdir, outfn + '-yii.tif'), YII)
        # cv2.imwrite(os.path.join(fmaxdir, outfn + '-npq.tif'), NPQ)

    # end if-else Fv/Fm

    # Make as many copies of incoming dataframe as there are ROIs so all results can be saved
    outdf = fundf.copy()
    for i in range(0, len(roi_c) - 1):
        outdf = outdf.append(fundf)
    outdf.frameid = outdf.frameid.astype('uint8')

    # Initialize lists to store variables for each ROI and iterate through each plant
    frame_avg = []
    yii_avg = []
    yii_std = []
    npq_avg = []
    npq_std = []
    plantarea = []
    ithroi = []
    inbounds = []
    if len(c) == 0:

        for i, rc in enumerate(roi_c):
            # each variable needs to be stored 2 x #roi
            frame_avg.append(np.nan)
            frame_avg.append(np.nan)
            yii_avg.append(np.nan)
            yii_avg.append(np.nan)
            yii_std.append(np.nan)
            yii_std.append(np.nan)
            npq_avg.append(np.nan)
            npq_avg.append(np.nan)
            npq_std.append(np.nan)
            npq_std.append(np.nan)
            inbounds.append(False)
            inbounds.append(False)
            plantarea.append(0)
            plantarea.append(0)
            # Store iteration Number even if there are no objects in image
            ithroi.append(int(i))
            ithroi.append(int(i))  # append twice so each image has a value.

    else:
        i = 1
        rc = roi_c[i]
        for i, rc in enumerate(roi_c):
            # Store iteration Number
            ithroi.append(int(i))
            ithroi.append(int(i))  # append twice so each image has a value.
            # extract ith hierarchy
            rh = roi_h[i]

            # Filter objects based on being in the defined ROI
            roi_obj, hierarchy_obj, submask, obj_area = pcv.roi_objects(
                img,
                roi_contour=rc,
                roi_hierarchy=rh,
                object_contour=c,
                obj_hierarchy=h,
                roi_type='partial')

            if obj_area == 0:
                print('!!! No plant detected in ROI ', str(i))

                frame_avg.append(np.nan)
                frame_avg.append(np.nan)
                yii_avg.append(np.nan)
                yii_avg.append(np.nan)
                yii_std.append(np.nan)
                yii_std.append(np.nan)
                npq_avg.append(np.nan)
                npq_avg.append(np.nan)
                npq_std.append(np.nan)
                npq_std.append(np.nan)
                inbounds.append(False)
                inbounds.append(False)
                plantarea.append(0)
                plantarea.append(0)

            else:

                # Combine multiple plant objects within an roi together
                plant_contour, plant_mask = pcv.object_composition(
                    img=img, contours=roi_obj, hierarchy=hierarchy_obj)

                #combine plant masks after roi filter
                if param_name == 'FvFm':
                    newmask = pcv.image_add(newmask, plant_mask)

                # Calc mean and std dev of fluoresence, YII, and NPQ and save to list
                frame_avg.append(cppc.utils.mean(imgmin, plant_mask))
                frame_avg.append(cppc.utils.mean(img, plant_mask))
                # need double because there are two images per loop
                yii_avg.append(cppc.utils.mean(YII, plant_mask))
                yii_avg.append(cppc.utils.mean(YII, plant_mask))
                yii_std.append(cppc.utils.std(YII, plant_mask))
                yii_std.append(cppc.utils.std(YII, plant_mask))
                npq_avg.append(cppc.utils.mean(NPQ, plant_mask))
                npq_avg.append(cppc.utils.mean(NPQ, plant_mask))
                npq_std.append(cppc.utils.std(NPQ, plant_mask))
                npq_std.append(cppc.utils.std(NPQ, plant_mask))
                plantarea.append(obj_area * cppc.pixelresolution**2)
                plantarea.append(obj_area * cppc.pixelresolution**2)

                # Check if plant is compeltely within the frame of the image
                inbounds.append(pcv.within_frame(plant_mask))
                inbounds.append(pcv.within_frame(plant_mask))

                # Output a pseudocolor of NPQ and YII for each induction period for each image
                imgdir = os.path.join(outdir, 'pseudocolor_images')
                outfn_roi = outfn + '-roi' + str(i)
                os.makedirs(imgdir, exist_ok=True)
                npq_img = pcv.visualize.pseudocolor(NPQ,
                                                    obj=None,
                                                    mask=plant_mask,
                                                    cmap='inferno',
                                                    axes=False,
                                                    min_value=0,
                                                    max_value=2.5,
                                                    background='black',
                                                    obj_padding=0)
                npq_img = cppc.viz.add_scalebar(npq_img,
                                                    pixelresolution=cppc.pixelresolution,
                                                    barwidth=10,
                                                    barlabel='1 cm',
                                                    barlocation='lower left')
                # If you change the output size and resolution you will need to adjust the timelapse video script
                npq_img.set_size_inches(6,6, forward=False)
                npq_img.savefig(os.path.join(imgdir, outfn_roi + '-NPQ.png'),
                                bbox_inches='tight',
                                dpi=100)#100 is default for matplotlib/plantcv
                if ilegend == 1:#only need to print legend once
                    npq_img.savefig(os.path.join(imgdir, 'npq_legend.pdf'), bbox_inches='tight')
                npq_img.clf()

                yii_img = pcv.visualize.pseudocolor(YII,
                                                    obj=None,
                                                    mask=plant_mask,
                                                    cmap='gist_rainbow',#custom_colormaps.get_cmap(
                                                        # 'imagingwin')#
                                                    axes=False,
                                                    min_value=0,
                                                    max_value=1,
                                                    background='black',
                                                    obj_padding=0)
                yii_img = cppc.viz.add_scalebar(yii_img,
                                                    pixelresolution=cppc.pixelresolution,
                                                    barwidth=10,
                                                    barlabel='1 cm',
                                                    barlocation='lower left')
                yii_img.set_size_inches(6, 6, forward=False)
                yii_img.savefig(os.path.join(imgdir, outfn_roi + '-YII.png'), bbox_inches='tight', dpi=100)
                if ilegend == 1:#print legend once and increment ilegend  to stop in future iterations
                    yii_img.savefig(os.path.join(imgdir, 'yii_legend.pdf'), bbox_inches='tight')
                    ilegend = ilegend+1
                yii_img.clf()

            # end try-except-else

        # end roi loop

    # end if there are objects from roi filter

    # save mask of all plants to file after roi filter
    if param_name == 'FvFm':
        mask_Fm = newmask.copy()
        # pcv.print_image(newmask, os.path.join(maskdir, outfn + '-mask.png'))

    # check YII values for uniqueness between all ROI. nonunique ROI suggests the plants grew into each other and can no longer be reliably separated in image processing.
    # a single value isn't always robust. I think because there are small independent objects that fall in one roi but not the other that change the object within the roi slightly.
    # also note, I originally designed this for trays of 2 pots. It will not detect if e.g. 2 out of 9 plants grow into each other
    rounded_avg = [round(n, 3) for n in yii_avg]
    rounded_std = [round(n, 3) for n in yii_std]
    if len(roi_c) > 1:
        isunique = not (rounded_avg.count(rounded_avg[0]) == len(yii_avg)
                        and rounded_std.count(rounded_std[0]) == len(yii_std))
    else:
        isunique = True

    # save all values to outgoing dataframe
    outdf['roi'] = ithroi
    outdf['frame_avg'] = frame_avg
    outdf['yii_avg'] = yii_avg
    outdf['npq_avg'] = npq_avg
    outdf['yii_std'] = yii_std
    outdf['npq_std'] = npq_std
    outdf['obj_in_frame'] = inbounds
    outdf['unique_roi'] = isunique

    return (outdf)


    # end of function!


# %% save histogram data

# %% Setup Debug parameters
#by default params.debug should be 'None' when you are ready to process all your images
pcv.params.debug = None
# if you choose to print debug files to disk then remove the old ones first (if they exist)
if pcv.params.debug == 'print':
    import shutil
    shutil.rmtree(os.path.join(debugdir), ignore_errors=True)

# %% Testing dataframe
# # If you need to test new function or threshold values you can subset your dataframe to analyze some images
df2 = df.query('(plantbarcode=="A4") and (parameter == "FvFm" or parameter == "t300_ALon" or parameter == "t80_ALon") and (jobdate == "2020-06-01")')
# df2 = df.query('(plantbarcode=="A1") and (parameter == "FvFm" or parameter == "t300_ALon") and (jobdate == "2020-03-13")')# | (plantbarcode == "B7" & jobdate == "2019-11-20")')
# fundf = df2.query('(plantbarcode == "A1" and parameter=="FvFm" and jobdate == "2020-05-30")')
# del fundf
# # # fundf
# # end testing

# %% Process the files
# check for subsetted dataframe
if 'df2' not in globals():
    df2 = df
else:
    print('df2 already exists!')

# # initialize db
# heterodb.insert({'plantbarcode': df2.plantbarcode.})
# heterodb.insert_multiple([{'plantbarcode': df2.plantbarcode.array}])
# heterodb.get('plantbarcode')
# heterodb.all()
# Each unique combination of treatment, plantbarcode, jobdate, parameter should result in exactly 2 rows in the dataframe that correspond to Fo/Fm or F'/Fm'
dfgrps = df2.groupby(['experiment', 'plantbarcode', 'jobdate', 'parameter'])
ilegend=1
grplist = []
for grp, grpdf in dfgrps:
    # print(grp)#'%s ---' % (grp))
    grplist.append(image_avg(grpdf))
df_avg = pd.concat(grplist)

# df_avg.to_csv('output/psII/df_avg.csv',na_rep='nan', float_format='%.4f', index=False)
#
# # %% Add genotype information
gtypeinfo = pd.read_csv(os.path.join('data', 'genotype_map.csv'),
                        skipinitialspace=True)
df_avg2 = (pd.merge(df_avg, gtypeinfo, on=['plantbarcode', 'roi'], how='inner'))



# %% Write the tabular results to file!
# df_avg2.jobdate = df_avg2.jobdate.dt.strftime('%Y-%m-%d')
(df_avg2.sort_values(['jobdate', 'plantbarcode', 'frameid']).drop(
    ['filename'], axis=1).to_csv(os.path.join(outdir,
                                              'output_psII_level0.csv'),
                                 na_rep='nan',
                                 float_format='%.4f',
                                 index=False))


# %%
