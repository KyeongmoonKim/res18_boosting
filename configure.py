BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DROP_OUT = 0.5
CLS_NUM = 13
ITER = 5010
SAVE_ITER = 100
OVER_RATIO = 0.7
WEIGHT_DECAY = 0.01
OVER=False
#data path
TRA_IMG_PATH = '../../Data/sb_sn_sum/Images_aug/'
VAL_IMG_PATH = '../../Data/sb_sn_sum/Images/'
TES_IMG_PATH = '../../Data/sb_sn_sum/Images/'
TRA_CSV = '../../Data/sb_sn_sum/train_aug.csv'
VAL_CSV = '../../Data/sb_sn_sum/val_cls13.csv'
TES_CSV = '../../Data/sb_sn_sum/test_cls13.csv' 
#model1 conf, but in this project, model1 is fixed
MODEL1_TRA_CKPT_PATH = './res18_sbsn.ckpt' 
MODEL1_TES_CKPT_PATH = './res18_sbsn.ckpt'
MODEL1_TRAINABLE = False
MODEL1_FINE_TUNING = True
MODEL1_HARD = True
VGG_SAVE_PATH = './checkpoint_vgg/'
VGG_SAVE_NAME = 'vgg_save_'
OVER1 = False
#model2 conf
MODEL2_TRA_CKPT_PATH = './res18_ip102.ckpt' 
MODEL2_TES_CKPT_PATH = './chekcpoint_res/res_save_600.ckpt'
MODEL2_TRAINABLE = True
MODEL2_FINE_TUNING = True
MODEL2_HARD = True
OVER2 = False
TRA_LOG = './log/tra_rev.csv'
TES_LOG = './log/tes_rev.csv'
RES_SAVE_PATH = './chekcpoint_res/'
RES_SAVE_NAME = 'res_save_'
#ENSEMBLE HYPER PARAMETER
RANK1_THRESH_HOLD = 0.7
INTERVAL_TRHESH_HOLD = 0.1
BEG_WEIGHT = 50
BEG_RATIO = 1.0