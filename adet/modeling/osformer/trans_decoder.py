import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from adet.modeling.ops.modules.ms_deform_attn import MSDeformAttn
from .trans_utils import _get_clones, get_reference_points, with_pos_embed
from .feed_forward import get_ffn


class CISTransformerDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 ffn_type="default", num_feature_levels=4, enc_n_points=4):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        decoder_layer = TransformerDecoderLayer(d_model, dim_feedforward,
                                                dropout, ffn_type,
                                                num_feature_levels, nhead, enc_n_points)
        self.decoder = TransformerDecoder(decoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def forward(self, srcs, pos_embeds, memorys=None, pos_memorys=None):

        # prepare input for decoder
        src_flatten = []
        memory_flatten = []
        lvl_pos_embed_flatten = []
        lvl_pos_memory_flatten = []
        spatial_shapes = []
        spatial_shape_grids = []
        for lvl, (src, pos_embed, memory, pos_memory) in enumerate(zip(srcs, pos_embeds, memorys, pos_memorys)):
            bs1, c1, h1, w1 = src.shape
            spatial_shape_src = (h1, w1)
            spatial_shape_grids.append(spatial_shape_src)
            bs, c, h, w = memory.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            memory = memory.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            pos_memory = pos_memory.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            lvl_pos_memory = pos_memory + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_memory_flatten.append(lvl_pos_memory)
            src_flatten.append(src)
            memory_flatten.append(memory)
        src_flatten = torch.cat(src_flatten, 1)
        memory_flatten = torch.cat(memory_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        lvl_pos_memory_flatten = torch.cat(lvl_pos_memory_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        spatial_shape_grids = torch.as_tensor(spatial_shape_grids, dtype=torch.long, device=src_flatten.device)
        level_start_index_grid = torch.cat((spatial_shape_grids.new_zeros((1, )), spatial_shape_grids.prod(1).cumsum(0)[:-1]))

        # decoder
        memory = self.decoder(src_flatten, memory_flatten, spatial_shapes, spatial_shape_grids, level_start_index_grid,
                              level_start_index, lvl_pos_embed_flatten, lvl_pos_memory_flatten)

        return memory, level_start_index


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, ffn_type="default",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = get_ffn(d_model, ffn_type)

    def forward(self, src, pos_embed, memorys, pos_memory, reference_points, spatial_shapes,
                level_start_index, spatial_shape_grids, level_start_index_grid):
        # self attention
        src2 = self.self_attn(with_pos_embed(src, pos_embed), reference_points,
                              with_pos_embed(memorys, pos_memory), spatial_shapes, level_start_index)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.ffn(src, spatial_shape_grids, level_start_index_grid)

        return src


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, memorys, spatial_shapes, spatial_shape_grids,
                level_start_index_grid, level_start_index, pos_embed, pos_memory):
        output = src
        batch_size = src.shape[0]
        reference_points = get_reference_points(spatial_shape_grids, batch_size, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos_embed, memorys, pos_memory, reference_points, spatial_shapes,
                           level_start_index, spatial_shape_grids, level_start_index_grid)

        return output


def build_transformer_decoder(cfg):
    return CISTransformerDecoder(
        d_model=cfg.MODEL.OSFormer.HIDDEN_DIM,
        nhead=cfg.MODEL.OSFormer.NHEAD,
        num_encoder_layers=cfg.MODEL.OSFormer.DEC_LAYERS,
        dim_feedforward=cfg.MODEL.OSFormer.DIM_FEEDFORWARD,
        dropout=0.1,
        ffn_type=cfg.MODEL.OSFormer.FFN,
        num_feature_levels=len(cfg.MODEL.OSFormer.FEAT_INSTANCE_STRIDES),
        enc_n_points=cfg.MODEL.OSFormer.ENC_POINTS)
