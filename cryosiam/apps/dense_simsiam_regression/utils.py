import torch
import numpy as np
import numpy.core.multiarray as np_multi
from torch.serialization import add_safe_globals
import collections

add_safe_globals([np.dtype, np_multi._reconstruct, np_multi.scalar])

from cryosiam.networks.nets import (
    CNNHead,
    DenseSimSiam
)

def load_backbone_model(checkpoint_path, device="cuda:0"):
    """Load DenseSimSiam trained model from given checkpoint
    :param checkpoint_path: path to the checkpoint
    :type checkpoint_path: str
    :param device: on which device should the model be loaded, default is cuda:0
    :type device: str
    :return: DenseSimSiam model with laoded trained weights
    :rtype: cryosiam.networks.nets.DenseSimSiam
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint['hyper_parameters']['backbone_config']
    model_backbone = DenseSimSiam(block_type=config['parameters']['network']['block_type'],
                                  spatial_dims=config['parameters']['network']['spatial_dims'],
                                  n_input_channels=config['parameters']['network']['in_channels'],
                                  num_layers=config['parameters']['network']['num_layers'],
                                  num_filters=config['parameters']['network']['num_filters'],
                                  fpn_channels=config['parameters']['network']['fpn_channels'],
                                  no_max_pool=config['parameters']['network']['no_max_pool'],
                                  dim=config['parameters']['network']['dim'],
                                  pred_dim=config['parameters']['network']['pred_dim'],
                                  dense_dim=config['parameters']['network']['dense_dim'],
                                  dense_pred_dim=config['parameters']['network']['dense_pred_dim'],
                                  include_levels=config['parameters']['network']['include_levels_loss']
                                  if 'include_levels_loss' in config['parameters']['network'] else False,
                                  add_later_conv=config['parameters']['network']['add_fpn_later_conv']
                                  if 'add_fpn_later_conv' in config['parameters']['network'] else False,
                                  decoder_type=config['parameters']['network']['decoder_type']
                                  if 'decoder_type' in config['parameters']['network'] else 'fpn',
                                  decoder_layers=config['parameters']['network']['fpn_layers']
                                  if 'fpn_layers' in config['parameters']['network'] else 2)
    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if not k.startswith('_model_backbone.'):
            continue
        name = k.replace("_model_backbone.", '')  # remove `_model_backbone.`
        new_state_dict[name] = v
    model_backbone.load_state_dict(new_state_dict)
    model_backbone.eval()
    device = torch.device(device)
    model_backbone.to(device)
    return model_backbone


def load_prediction_model(checkpoint_path, device="cuda:0"):
    """Load SemanticHeads trained model from given checkpoint
    :param checkpoint_path: path to the checkpoint
    :type checkpoint_path: str
    :param device: on which device should the model be loaded, default is cuda:0
    :type device: str
    :return: InstanceHeads model with loaded trained weights
    :rtype: cryosiam.networks.nets.InstanceHeads
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint['hyper_parameters']['config']
    model = CNNHead(n_input_channels=config['parameters']['network']['dense_dim'],
                    n_output_channels=config['parameters']['network']['n_output_channels'],
                    spatial_dims=config['parameters']['network']['spatial_dims'],
                    filters=config['parameters']['network']['filters'],
                    kernel_size=config['parameters']['network']['kernel_size'],
                    padding=config['parameters']['network']['padding'])

    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if not k.startswith('_model.'):
            continue
        name = k.replace("_model.", '')  # remove `_model.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    device = torch.device(device)
    model.to(device)
    return model

Click the green "Commit changes" button.

B. Patch predict.py
Now, navigate to cryosiam/apps/dense_simsiam_regression/predict.py in your fork.

Click the pencil icon to edit it.

Delete all the code and replace it with the fully corrected version below, which includes your file-skipping logic.

Python

# Fully Patched predict.py
import torch
import numpy as np
import numpy.core.multiarray as np_multi
from torch.serialization import add_safe_globals
import collections

add_safe_globals([np.dtype, np_multi._reconstruct, np_multi.scalar])

import os
import yaml
import h5py
from torch.utils.data import DataLoader
from monai.data import Dataset, list_data_collate, GridPatchDataset, ITKReader
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    SpatialPadd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType
)

from cryosiam.utils import parser_helper
from cryosiam.data import MrcReader, TiffReader, PatchIter, MrcWriter, TiffWriter
from cryosiam.apps.dense_simsiam_regression.utils import load_backbone_model, load_prediction_model


def main(config_file_path):

    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    if 'trained_model' in cfg and cfg['trained_model'] is not None:
        checkpoint_path = cfg['trained_model']
    else:
        checkpoint_path = os.path.join(cfg['log_dir'], 'model', 'model_best.ckpt')
    backbone = load_backbone_model(checkpoint_path)
    prediction_model = load_prediction_model(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    net_config = checkpoint['hyper_parameters']['config']

    test_folder = cfg['data_folder']
    prediction_folder = cfg['prediction_folder']
    num_output_channels = net_config['parameters']['network']['n_output_channels']
    patch_size = net_config['parameters']['data']['patch_size']
    spatial_dims = net_config['parameters']['network']['spatial_dims']
    os.makedirs(prediction_folder, exist_ok=True)

    files = cfg['test_files']
    if files is None:
        files = [x for x in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, x))]

    input_files = [f for f in files if f.endswith(cfg['file_extension'])]
    test_data = []

    for f in input_files:
        pred_file = os.path.join(prediction_folder, f.replace(cfg['file_extension'], '_preds.h5'))
        if not os.path.exists(pred_file):
            test_data.append({
                'image': os.path.join(test_folder, f),
                'file_name': os.path.join(test_folder, f)
            })

    print(f"Total input files: {len(input_files)}")
    print(f"Already denoised files: {len(input_files) - len(test_data)}")
    print(f"Files to process: {len(test_data)}")

    reader = MrcReader(read_in_mem=True) if cfg['file_extension'] in ['.mrc', '.rec'] else \
        TiffReader() if cfg['file_extension'] in ['.tiff', '.tif'] else ITKReader()

    if cfg['file_extension'] in ['.mrc', '.rec']:
        writer = MrcWriter(output_dtype=np.float32, overwrite=True)
        writer.set_metadata({'voxel_size': 1})
    elif cfg['file_extension'] in ['.tiff', '.tif']:
        writer = TiffWriter(output_dtype=np.float32)
    else:
        writer = ITKWriter()

    transforms = Compose(
        [
            LoadImaged(keys='image', reader=reader),
            EnsureChannelFirstd(keys='image'),
            ScaleIntensityRanged(keys='image', a_min=cfg['parameters']['data']['min'],
                                 a_max=cfg['parameters']['data']['max'], b_min=0, b_max=1, clip=True),
            SpatialPadd(keys='image', spatial_size=patch_size),
            NormalizeIntensityd(keys='image', subtrahend=cfg['parameters']['data']['mean'],
                                divisor=cfg['parameters']['data']['std']),
            EnsureTyped(keys='image', data_type='tensor')
        ]
    )
    if spatial_dims == 2:
        patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0), overlap=(0, 0.5, 0.5))
    else:
        patch_iter = PatchIter(patch_size=tuple(patch_size), start_pos=(0, 0, 0), overlap=(0, 0.5, 0.5, 0.5))
    post_pred = Compose([EnsureType('numpy', dtype=np.float32, device=torch.device('cpu'))])

    test_dataset = Dataset(data=test_data, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, collate_fn=list_data_collate)

    print('Prediction')
    with torch.no_grad():
        for i, test_sample in enumerate(test_loader):
            out_file = os.path.join(prediction_folder, os.path.basename(test_sample['file_name'][0]))
            print('Processing file:', test_sample['file_name'][0])
            patch_dataset = GridPatchDataset(data=[test_sample['image'][0]],
                                             patch_iter=patch_iter)
            input_size = list(test_sample['image'][0][0].shape)
            preds_out = np.zeros([num_output_channels] + input_size, dtype=np.float32)
            loader = DataLoader(patch_dataset, batch_size=cfg['hyper_parameters']['batch_size'], num_workers=2)
            for item in loader:
                img, coord = item[0], item[1].numpy().astype(int)
                z, _ = backbone.forward_predict(img.cuda())
                out = prediction_model(z)
                out = post_pred(out)
                for batch_i in range(img.shape[0]):
                    c_batch = coord[batch_i][1:]
                    o_batch = out[batch_i]
                    # avoid getting patch that is outside of the original dimensions of the image
                    if c_batch[0][0] >= input_size[0] - patch_size[0] // 4 or \
                            c_batch[1][0] >= input_size[1] - patch_size[1] // 4 or \
                            (spatial_dims == 3 and c_batch[2][0] >= input_size[2] - patch_size[2] // 4):
                        continue
                    # create slices for the coordinates in the output to get only the middle of the patch
                    # and the separate cases for the first and last patch in each dimension
                    slices = tuple(
                        slice(c[0], c[1] - p // 4) if c[0] == 0 else slice(c[0] + p // 4, c[1])
                        if c[1] >= s else slice(c[0] + p // 4, c[1] - p // 4)
                        for c, s, p in zip(c_batch, input_size, patch_size))
                    # create slices to crop the patch so we only get the middle information
                    # and the separate cases for the first and last patch in each dimension
                    slices2 = tuple(
                        slice(0, 3 * p // 4) if c[0] == 0 else slice(p // 4, p - (c[1] - s))
                        if c[1] >= s else slice(p // 4, 3 * p // 4)
                        for c, s, p in zip(c_batch, input_size, patch_size))
                    preds_out[(slice(0, num_output_channels),) + slices] = o_batch[(slice(0, num_output_channels),)
                                                                                   + slices2]

            if cfg['scale_prediction']:
                preds_out = (preds_out - preds_out.min()) / (preds_out.max() - preds_out.min())

            with h5py.File(out_file.split(cfg['file_extension'])[0] + '_preds.h5', 'w') as f:
                f.create_dataset('preds', data=preds_out)

            writer.set_data_array(preds_out[0], channel_dim=None)
            writer.write(out_file.split(cfg['file_extension'])[0] + f'{cfg["file_extension"]}')


if __name__ == "__main__":
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file)
