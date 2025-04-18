MODEL_NAME=msk_smit_lung_gtv
WEIGHTS_HASH=H4sIADC3/mcAAwXByRGAIAwAwL/FkJFDwW4wqDCAYQwPtHp3Y++NN4DKGVHsNARSBY7+OQJw9z0hUDG30lnyG42S2UZnSwrz6tTQJy1NXN/0A15deWNIAAAA
WEIGHTS_URL=`base64 -d <<<${WEIGHTS_HASH} | gunzip`
wget $WEIGHTS_URL -O weights.tar.gz
tar xvf weights.tar.gz -C /app/models/${MODEL_NAME}/src && rm weights.tar.gz
