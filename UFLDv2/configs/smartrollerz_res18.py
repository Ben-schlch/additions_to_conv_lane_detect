dataset = 'Tusimple' # 'Smartrollerz'
data_root = '../../IMG_ROOTS/TUSIMPLEROOT/TUSimple/test_set/' # 'dataset/smartrollerz'  # Need to be modified before running
epoch = 200
batch_size = 32
optimizer = 'SGD'
learning_rate = 0.00625
weight_decay = 0.0001
momentum = 0.9
scheduler = 'multi'
steps = [50, 75]
gamma = 0.1
warmup = 'linear'
warmup_iters = 100
use_aux = False
griding_num = 200
backbone = '18'
sim_loss_w = 0.0
shp_loss_w = 0.0
note = ''
log_path = ''
finetune = None
resume = None
test_model = ''
test_work_dir = ''
tta = True
num_lanes = 3
var_loss_power = 2.0
auto_backup = True
num_row = 65
num_col = 40
train_width = 256  # 256 #512  # 2048  # 1024
train_height = 192  # 192 #384  # 1536  # 768
num_cell_row = 100
num_cell_col = 100
mean_loss_w = 0.05
fc_norm = False
soft_loss = True
cls_loss_col_w = 1.0
cls_ext_col_w = 1.0
mean_loss_col_w = 0.05
eval_mode = 'normal'
crop_ratio = 0.8
