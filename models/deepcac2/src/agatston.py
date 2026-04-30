# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5487233/
# Agatston method - The Agatston method uses the weighted sum of lesions with a density above 130 HU, 
# multiplying the area of calcium by a factor related to maximum plaque attenuation: 
# 130-199 HU, factor 1; 200-299 HU, factor 2; 300-399 HU, factor 3; and ≥ 400 HU, factor 4.

# https://www-sciencedirect-com.mu.idm.oclc.org/science/article/pii/S1939865421001569 
# Agatston score is a summed value of all calcified coronary lesions, based on both the total area and the maximal density of coronary calcification.

# ----------------------------------------------------
# Calculation
# detect continuous voxels (connected shapes) with HU > 130 and minimal size of 1 mm^3 (number of voxels depend on teh spacing)
# for each individual calcified leason in all coronary arteries
#   dwf = { 1 if max HU in leasion in [130, 199]
#           2 if max HU in leasion in [200, 299]
#           3 if max HU in leasion in [300, 399]
#           4 if max HU in leasion > 400 }
#   s_i = leasion area * dwf
# overall Agatston score is the sum of all individual leasion scores
# s = \sum_i s_i

# Interpretation
# AS =   0:          indicates no identifiable atherosclerotic plaque and very low cardiovascular disease (CVD) risk (Fig. 1).
# AS \in [1, 10]:    indicates minimal plaque burden and low CVD risk.
# AS \in ]10,100]:   indicates mild plaque burden and moderate CVD risk (Fig. 2).
# AS \in ]100, 400]: indicates moderate plaque burden and high CVD risk.
# AS >    400:       indicates extensive plaque burden and very high CVD risk (Fig. 3).

# Questions
# "[..] for the detection of calcium in contiguous voxels of 1 sq mm in area to be counted as individual lesions."
# --> why area and why square not cubic?

import numpy as np
from scipy.ndimage import measurements

def agatston_score_slices2(cac_np, img_np, nrConPx, spacing, allow_diagonal_connections=True, verbose=False):
    AG_DIV = 3
    pxArea = round(spacing[0] * spacing[1] * spacing[2] / AG_DIV, 3)
    
    AG = 0

    for z_slice in range(cac_np.shape[2]):
        cac_slice_np = cac_np[:, :, z_slice]
        img_slice_np = img_np[:, :, z_slice]
    
        # NOTE: allegedly (he didn't verify if intentionally) Roman used np.ones((3, 3)) thus allowed for diagonal connections
        if allow_diagonal_connections:
            structure = np.array([
                [1, 1, 1], 
                [1, 1, 1], 
                [1, 1, 1]
            ])
        else:
            structure = np.array([
                [0, 1, 0], 
                [1, 1, 1], 
                [0, 1, 0]
            ])
        
        # extract connected shapes (objects)
        # objects_lblmask, n_objects
        labeledMask, numLabels = measurements.label(cac_slice_np, structure=structure) 

        for labelNr in range(1, numLabels + 1):
            label = np.zeros(cac_slice_np.shape)
            label[labeledMask == labelNr] = 1

            clcObject = img_slice_np * label

            # 1) Remove small objects
            if np.sum(label) <= nrConPx:  # FIXME: this is copied from Roman's code but should be < instead.
                continue

            # 3) Calculate volume
            objectArea = np.sum(label) * pxArea
            
            # Get object max HU and DFW
            objectMaxHU = clcObject.max()
            objectDFW = dfw(objectMaxHU, disable_hu_check=True)

            # 4) Calculate AG for object
            objectAG = round(objectArea * objectDFW, 3)

            # 5) Sum up scores
            AG += objectAG
    
            #if verbose: print(f"z {z_slice}, plaque {plaque_label:<4}| s:{plaque_slice_score:<8.3f} v:{plaque_slice_volume:<7.2f} hu:{plaque_slice_max_hu:<8.2f} dfw:{dfw(plaque_slice_max_hu, disable_hu_check=True)}")
    #if verbose: print(f"∑ p{pid:<8}> {patient_score:<8.3f}")

    # return cummulated sum as agatston score
    return AG


# agatston score object density factor
def dfw(max_hu, disable_hu_check=False):
    assert disable_hu_check or max_hu >= 130, "HU value below TH."
    if max_hu < 200:    # [130, 200[ 
        return 1
    elif max_hu < 300:  # [200, 300[
        return 2
    elif max_hu < 400:  # [300, 400[
        return 3
    else:               # [400, 
        return 4

# as implemented by Roman (getAGclass(AG) method)
def agatston_score_interpretation(AG): 
    AG = round(AG, 3)
    classAG = None
    if AG == 0: classAG = 0
    if AG > 0 and AG <= 100: classAG = 1
    if AG > 100 and AG <= 300: classAG = 2
    if AG > 300: classAG = 3
    return classAG

