from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, cv2
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor

from adet.config import get_cfg

YML_PATH = ''
WEIGHTS = ''
OUTPATH = ''
if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)


def setup(yml_path, weights):
    cfg = get_cfg()
    cfg.merge_from_file(yml_path)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.OSFormer.UPDATE_THR = 0.5
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    return cfg, predictor


def vis_features(feat):
    feat = feat.squeeze(0)
    return feat.square().sum(0)


def visualize(im_path, cfg, predictor, out_path):
    im_name = os.path.basename(im_path).split('.')[0]
    
    model = predictor.model
    
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    conv_features = []
    trans_features = []
    camin_features = []
    mask_features = []
    
    hooks = [
        # backbone feature
        model.backbone.res2.register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.backbone.res3.register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.backbone.res4.register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.backbone.res5.register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),

        # trans feature
        model.cate_head.trans_encoder.encoder.layers[5].register_forward_hook(
            lambda self, input, output: trans_features.append(output)
        ),
        
        # mask head feature
        model.mask_head.register_forward_hook(
            lambda self, input, output: mask_features.append(output)
        ),
        
        # camin feature
        model.dcin.register_forward_hook(
            lambda self, input, output: camin_features.append(output)
        )
    ]
    
    outputs = predictor(im)

    for hook in hooks:
        hook.remove()

    # save res feats, res2-res5
    spatial_shapes = []
    spatial_sizes = []
    for idx, elem in enumerate(conv_features):
        cur_feat = vis_features(elem).cpu().numpy()
        spatial_shapes.append(tuple(cur_feat.shape))
        spatial_sizes.append(cur_feat.shape[0] * cur_feat.shape[1])
        plt.axis('off')
        plt.imshow(cur_feat)
        plt.savefig(os.path.join(out_path, 'vis_res{}_{}.pdf'.format(idx + 2, im_name)), bbox_inches='tight', pad_inches=0.0)
        print(os.path.join(out_path, 'vis_res{}_{}.pdf'.format(idx + 2, im_name)))
        
    # save trans feats, trans3-trans5
    for idx, elem, (x, y) in zip(range(len(spatial_shapes) - 1), trans_features[0].split(spatial_sizes[1:], 1), spatial_shapes[1:]):
        feat = vis_features(elem.permute(0, 2, 1).view(1, -1, x, y)).cpu().numpy()
        plt.axis('off')
        plt.imshow(feat)
        plt.savefig(os.path.join(out_path, 'vis_trans{}_{}.pdf'.format(idx + 3, im_name)), bbox_inches='tight', pad_inches=0.0)
        print(os.path.join(out_path, 'vis_trans{}_{}.pdf'.format(idx + 3, im_name)))

    # save camin output features
    camin_feats = camin_features[0].squeeze(0).cpu().numpy()
    for i in range(camin_feats.shape[0]):
        feat = camin_feats[i]
        plt.cla()  # ref https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib
        plt.axis('off')
        plt.imshow(feat)
        plt.savefig(os.path.join(out_path, 'vis_dcin{}_{}.pdf'.format(i, im_name)), bbox_inches='tight', pad_inches=0.0)
        print(os.path.join(out_path, 'vis_dcin{}_{}.pdf'.format(i, im_name)))

    # save mask features
    mask_feats = vis_features(mask_features[0][0]).cpu().numpy()
    plt.imshow(mask_feats)
    plt.savefig(os.path.join(out_path, 'vis_maskhead_{}.pdf'.format(im_name)), bbox_inches='tight', pad_inches=0.0)
    print(os.path.join(out_path, 'vis_maskhead_{}.pdf'.format(im_name)))
    
    # save rea edges
    for i in range(len(mask_features[0][1])):
        feat = mask_features[0][1][i].squeeze().cpu().numpy()
        plt.axis('off')
        plt.imshow(feat)
        plt.savefig(os.path.join(out_path, 'vis_rea_edge{}_{}.pdf'.format(i, im_name)), bbox_inches='tight', pad_inches=0.0)
        print(os.path.join(out_path, 'vis_rea_edge{}_{}.pdf'.format(i, im_name)))
