import numpy as np
from scipy.ndimage import measurements

def register_cac_mask(img_np, cac_np, allow_diagonal_connections=True):
    """Generates a thresholded mask from the input image and removes all objects which do not overlap with the provided mask.
    The idea is, that due to resampling operations, the resulting mask might not exactly match to the thresholded objects which
    this method aims to correct for. Both, the input image and the cac mask must already be in the same spacing and dimensions.
    NOTE: Does not remove small objects, which has to be done in the agatston score calculation.

    Args:
        img_np (3d numpy array): the original input image
        cac_np (3d binary numpy array): the (auto-) generated binary cac mask
    """
    assert img_np.shape == cac_np.shape, "shape missmatch"
    TH = 130

    # re-assemble a cac mask based on the hu threshold and the provided cac mask
    registered_mask = np.zeros(cac_np.shape)
    
    # generate a mask from the thresholded input image
    thmask_np = np.zeros(img_np.shape)
    thmask_np[img_np >= TH] = 1             # >= 130, verified

    # iterate through the slices
    for z_slice in range(cac_np.shape[2]):
        cac_slice_np    =        cac_np[:, :, z_slice]
        thmask_slice_np =     thmask_np[:, :, z_slice]

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
        objects_lblmask, n_objects = measurements.label(thmask_slice_np, structure=structure) 

        # iterate over all objects, identified on the thresholded image mask
        for object_nr in range(1, n_objects + 1):

            # label mask
            #object_mask = np.zeros(cac_slice_np.shape)
            #object_mask[objects_lblmask == object_nr] = 1
            object_mask = (objects_lblmask == object_nr).astype(int)

            # overlap / registration match
            if object_mask[cac_slice_np > 0].sum() > 0:    # is present on both
                registered_mask[:, :, z_slice] += object_mask

    assert registered_mask.max() <= 1, f"overlappings detected ({registered_mask.max()})"
    return registered_mask