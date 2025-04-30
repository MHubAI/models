

import ml_collections


def get_SMIT_128_bias_True():
    '''
    A Large TransMorph Network
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 1
    config.embed_dim = 128
    config.embed_dim = 48
    config.depths = (2, 2, 8, 2)
    config.num_heads = (4, 4, 8, 16)
    
    config.window_size = (4, 4, 4)
   
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = True
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (128, 128, 128)
    return config


def get_SMIT_128_bias_True_Cross():

    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 48  # change 128 or 192
    config.depths = (2, 2, 8, 2)  # change 4 to 6,10
    config.num_heads = (4, 4, 8, 16)
    config.window_size = (4, 4, 4)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = True
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.seg_head_chan = config.embed_dim // 2
    config.img_size = (128, 128, 128)
    config.pos_embed_method = 'relative'
    return config


