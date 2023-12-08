import torch, os, cv2
from utils.dist_utils import dist_print
import torch, os
from utils.common import merge_config, get_model
import tqdm
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset


def pred2coords(pred, row_anchor, col_anchor, local_width=1, original_image_width=1640, original_image_height=590):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    # ---------  Nrow  ---------- NrDim  ----  NrLane ------ Pr
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape
    # ---------  Ncol  ---------- NcDim  ----  NcLane ------ Pc
    max_indices_row = pred['loc_row'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_row = pred['exist_row'].argmax(1).cpu()
    # n, num_cls, num_lanes

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_col = pred['exist_col'].argmax(1).cpu()
    # n, num_cls, num_lanes

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []

    row_lane_idx = [1, 2]  # Ego Lanes
    col_lane_idx = [0, 3]  # Outer lanes

    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - local_width),
                                                      min(num_grid_row - 1,
                                                          max_indices_row[0, k, i] + local_width) + 1)))

                    out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - local_width),
                                                      min(num_grid_col - 1,
                                                          max_indices_col[0, k, i] + local_width) + 1)))

                    out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5

                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()
    cfg.batch_size = 1
    print('setting batch_size to 1 for demo generation')

    dist_print('start testing...')
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = get_model(cfg)

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        if 'model.' in k:
            k = 'model'
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # vout = cv2.VideoWriter('challenge.avi', fourcc, 30.0, (1280, 720))
    img_path = "/mnt/c/Users/inf21034/source/IMG_ROOTS/CHALLENGEROOT/chal1.jpg"
    video = cv2.VideoCapture("/mnt/c/Users/inf21034/PycharmProjects/conventional_lane_detection/images/Udacity/project_video.mp4")
    # for i in range(1, 485):
    cnt = 0
    while video.isOpened():
        cnt += 1
        print(cnt)
        ret, img = video.read()
        # img = cv2.imread(f"/mnt/c/Users/inf21034/source/IMG_ROOTS/CHALLENGEROOT/chal{i}.jpg")
    # img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im0 = img.copy()
        img_h, img_w = img.shape[0], img.shape[1]
        img = img_transforms(img)
        img = img[:, -cfg.train_height:, :]
        img = img.to('cuda:0')
        img = torch.unsqueeze(img, 0)
        with torch.no_grad():
            pred = net(img)
        coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width=img_w,
                             original_image_height=img_h)
        for lane in coords:
            for coord in lane:
                cv2.circle(im0, coord, 5, (0, 255, 0), -1)
        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"./demo_images/project_vid/project{cnt}.jpg", im0)
        # vout.write(im0)
    video.release()
    os.system("ffmpeg -f image2 -i ./demo_images/project_vid/project%d.jpg project_video_inferred.mp4")
    # vout.release()
    # cv2.imshow('demo', im0)
    # cv2.waitKey(0)
