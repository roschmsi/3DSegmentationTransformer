import torch
import hydra
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from models.modules.common import conv
from models.position_embedding import PositionEmbeddingCoordsSine
from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from models.modules.helpers_3detr import GenericMLP
from torch_scatter import scatter_mean, scatter_max, scatter_min
from torch.cuda.amp import autocast
from models.mask3d import CrossAttentionLayer, FFNLayer, _get_activation_fn, SelfAttentionLayer, PositionalEncoding3D

class StratifiedMask3D(nn.Module):
    def __init__(self, config, hidden_dim, num_queries, num_heads, dim_feedforward,
                 sample_sizes, shared_decoder, num_classes,
                 num_decoders, dropout, pre_norm,
                 positional_encoding_type, non_parametric_queries, train_on_segments, normalize_pos_enc,
                 use_level_embed, scatter_type, hlevels,
                 use_np_features,
                 voxel_size,
                 max_sample_size,
                 random_queries,
                 gauss_scale,
                 random_query_both,
                 random_normal,
                 debug=False,
                 freeze_backbone=False,
                 ):
        super().__init__()

        self.random_normal = random_normal
        self.random_query_both = random_query_both
        self.random_queries = random_queries
        self.max_sample_size = max_sample_size
        self.gauss_scale = gauss_scale
        self.voxel_size = voxel_size
        self.scatter_type = scatter_type
        self.hlevels = hlevels
        self.use_level_embed = use_level_embed
        self.train_on_segments = train_on_segments
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_classes = num_classes
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.sample_sizes = sample_sizes
        self.non_parametric_queries = non_parametric_queries
        self.use_np_features = use_np_features
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.pos_enc_type = positional_encoding_type
        self.debug = debug
        self.freeze_backbone = freeze_backbone

        self.backbone = hydra.utils.instantiate(config.backbone)
        if self.freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

        self.num_levels = len(self.hlevels)
        
        sizes = (384, 384, 192, 96, 48)

        # self.mask_features_head = conv(
        #     48, self.mask_dim, kernel_size=1, stride=1, bias=True, D=3
        # )

        self.mask_features_head = nn.Linear(48, 128)

        if self.scatter_type == "mean":
            self.scatter_fn = scatter_mean
        elif self.scatter_type == "max":
            self.scatter_fn = lambda mask, p2s, dim: scatter_max(mask, p2s, dim=dim)[0]
        else:
            assert False, "Scatter function not known"

        assert (not use_np_features) or non_parametric_queries, "np features only with np queries"

        if self.non_parametric_queries:
            self.query_projection = GenericMLP(
                input_dim=self.mask_dim,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )

            if self.use_np_features:
                self.np_feature_projection = nn.Sequential(
                    nn.Linear(sizes[-1], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
        elif self.random_query_both:
            self.query_projection = GenericMLP(
                input_dim=2*self.mask_dim,
                hidden_dims=[2*self.mask_dim],
                output_dim=2*self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True
            )
        else:
            # PARAMETRIC QUERIES
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_pos = nn.Embedding(num_queries, hidden_dim)

        if self.use_level_embed:
            # learnable scale-level embedding
            self.level_embed = nn.Embedding(self.num_levels, hidden_dim)

        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.class_embed_head = nn.Linear(hidden_dim, self.num_classes)

        if self.pos_enc_type == "legacy":
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == "fourier":
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="fourier",
                                                       d_pos=self.mask_dim, #128
                                                       gauss_scale=self.gauss_scale, #1.0
                                                       normalize=self.normalize_pos_enc) #True
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(pos_type="sine",
                                                       d_pos=self.mask_dim,
                                                       normalize=self.normalize_pos_enc)
        else:
            assert False, 'pos enc type not known'

        self.pooling = torch.nn.AvgPool1d(kernel_size=2, stride=2)

        self.masked_transformer_decoder = nn.ModuleList()
        self.cross_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()
        self.lin_squeeze = nn.ModuleList()

        self.projection_layers = nn.ModuleList([
            nn.Linear(384, self.mask_dim),
            nn.Linear(384, self.mask_dim),
            nn.Linear(192, self.mask_dim),
            nn.Linear(96, self.mask_dim),
            nn.Linear(48, self.mask_dim),
        ])

        num_shared = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_shared):
            tmp_cross_attention = nn.ModuleList()
            tmp_self_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()
            tmp_squeeze_attention = nn.ModuleList()
            for i, hlevel in enumerate(self.hlevels):
                tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

                tmp_squeeze_attention.append(nn.Linear(sizes[hlevel], self.mask_dim))

                tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

                tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=self.mask_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm
                    )
                )

            self.cross_attention.append(tmp_cross_attention)
            self.self_attention.append(tmp_self_attention)
            self.ffn_attention.append(tmp_ffn_attention)
            self.lin_squeeze.append(tmp_squeeze_attention)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []

        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            for coords_batch in coords[i]:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(coords_batch[None, ...].float(),
                                       input_range=[scene_min, scene_max])

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd

    def forward(self, feat, coord, offset, batch, neighbor_idx, masks, point2segment=None, raw_coordinates=None, is_eval=False):
        # TODO pcd_features
        
        if self.debug:
            with torch.no_grad():
                aux = self.backbone(feat, coord, offset, batch, neighbor_idx, masks)  # (num_voxels, 96)
        else:
            aux = self.backbone(feat, coord, offset, batch, neighbor_idx, masks)
        aux_decomposed = []
        for a in aux:
            offsets = a[-1]
            coords = a[1]
            feats = a[0]
            masks = a[2]
            offset_indices = [0]+offsets.tolist()
            coords_dec = [coords[offset_indices[i]:offset_indices[i+1]] for i in range(len(offsets))]
            feats_dec = [feats[offset_indices[i]:offset_indices[i+1]] for i in range(len(offsets))]
            aux_decomposed.append((feats_dec,coords_dec,masks))
        

        pcd_features = []
        for b in range(len(offset)):
            with torch.no_grad():
                pcd_features = aux_decomposed[-1][0][b]
            
        aux_coords = [[a[1][b] for b in range(len(offset))] for a in aux_decomposed[:-1]]
        aux_pos_enc = self.get_pos_encs(aux_coords)
        batch_size = len(offset)
        sampled_coords = None

        if self.non_parametric_queries:
            fps_idx = [furthest_point_sample(x.decomposed_coordinates[i][None, ...].float(),
                                             self.num_queries).squeeze(0).long()
                       for i in range(len(x.decomposed_coordinates))]

            sampled_coords = torch.stack([coordinates.decomposed_features[i][fps_idx[i].long(), :]
                                          for i in range(len(fps_idx))])

            mins = torch.stack([coordinates.decomposed_features[i].min(dim=0)[0] for i in range(len(coordinates.decomposed_features))])
            maxs = torch.stack([coordinates.decomposed_features[i].max(dim=0)[0] for i in range(len(coordinates.decomposed_features))])

            query_pos = self.pos_enc(sampled_coords.float(),
                                     input_range=[mins, maxs]
                                     )  # Batch, Dim, queries
            query_pos = self.query_projection(query_pos)

            if not self.use_np_features:
                queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            else:
                queries = torch.stack([pcd_features.decomposed_features[i][fps_idx[i].long(), :]
                                       for i in range(len(fps_idx))])
                queries = self.np_feature_projection(queries)
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_queries:
            # TODO fix device
            query_pos = torch.rand(batch_size, self.mask_dim, self.num_queries, device='cuda:0') - 0.5

            queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_query_both:
            if not self.random_normal:
                query_pos_feat = torch.rand(batch_size, 2*self.mask_dim, self.num_queries, device=x.device) - 0.5
            else:
                query_pos_feat = torch.randn(batch_size, 2 * self.mask_dim, self.num_queries, device=x.device)

            queries = query_pos_feat[:, :self.mask_dim, :].permute((0, 2, 1))
            query_pos = query_pos_feat[:, self.mask_dim:, :].permute((2, 0, 1))
        else:
            # PARAMETRIC QUERIES
            queries = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # queries, query_pos (batch_size, num_instance_queries, D), (2, 100, 128)

        predictions_class = []
        predictions_mask = []

        for decoder_counter in range(self.num_decoders): # here 3
            if self.shared_decoder:
                decoder_counter = 0
            for i, hlevel in enumerate(self.hlevels):
                
                mask_features = [self.projection_layers[hlevel](aux_decomposed[hlevel][0][i]) for i in range(batch_size)]
                output_class, outputs_mask, attn_mask = self.mask_module(queries,
                                                        mask_features,
                                                        None,
                                                        len(aux) - hlevel - 1,
                                                        ret_attn_mask=True,
                                                        point2segment=None,
                                                        coords=None)

                # decomposed_aux = aux[hlevel][0].unsqueeze(0)
                # decomposed_attn = attn_mask

                curr_sample_size = max([pcd.shape[0] for pcd in aux_decomposed[hlevel][0]])

                if min([pcd.shape[0] for pcd in aux_decomposed[hlevel][0]]) == 1:
                    raise RuntimeError("only a single point gives nans in cross-attention")

                if not (self.max_sample_size or is_eval):
                    curr_sample_size = min(curr_sample_size, self.sample_sizes[hlevel])

                rand_idx = []
                mask_idx = []
                for k in range(len(aux_decomposed[hlevel][0])):
                    pcd_size = aux_decomposed[hlevel][0][k].shape[0]
                    if pcd_size <= curr_sample_size:
                        # we do not need to sample
                        # take all points and pad the rest with zeroes and mask it
                        idx = torch.zeros(curr_sample_size,
                                          dtype=torch.long,
                                          device=queries.device)

                        midx = torch.ones(curr_sample_size,
                                          dtype=torch.bool,
                                          device=queries.device)

                        idx[:pcd_size] = torch.arange(pcd_size,
                                                      device=queries.device)

                        midx[:pcd_size] = False  # attend to first points
                    else:
                        # breakpoint()
                        # we have more points in pcd as we like to sample
                        # take a subset (no padding or masking needed)
                        idx = torch.randperm(aux_decomposed[hlevel][0][k].shape[0],
                                             device=queries.device)[:curr_sample_size]
                        midx = torch.zeros(curr_sample_size,
                                           dtype=torch.bool,
                                           device=queries.device)  # attend to all

                    rand_idx.append(idx)
                    mask_idx.append(midx)

                batched_aux = torch.stack([
                    aux_decomposed[hlevel][0][k][rand_idx[k], :] for k in range(len(rand_idx))
                ])

                batched_attn = torch.stack([
                    attn_mask[k][rand_idx[k], :] for k in range(len(rand_idx))
                ])
                
                batched_pos_enc = torch.stack([
                    aux_pos_enc[hlevel][0][k][rand_idx[k], :] for k in range(len(rand_idx))
                ])

                batched_attn.permute((0, 2, 1))[batched_attn.sum(1) == rand_idx[0].shape[0]] = False

                m = torch.stack(mask_idx)
                batched_attn = torch.logical_or(batched_attn, m[..., None])

                src_pcd = self.lin_squeeze[decoder_counter][i](batched_aux.permute((1, 0, 2)))
                if self.use_level_embed:
                    src_pcd += self.level_embed.weight[i]

                output, cross_attention_masks = self.cross_attention[decoder_counter][i](
                    queries.permute((1, 0, 2)),
                    src_pcd,
                    memory_mask=batched_attn.repeat_interleave(self.num_heads, dim=0).permute((0, 2, 1)),
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos= batched_pos_enc.permute((1, 0, 2)),
                    query_pos=query_pos
                )

                output = self.self_attention[decoder_counter][i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos
                )

                # FFN
                queries = self.ffn_attention[decoder_counter][i](
                    output
                ).permute((1, 0, 2))

                predictions_class.append(output_class)
                predictions_mask.append(outputs_mask)

        mask_features = [self.projection_layers[-1](aux_decomposed[-1][0][i]) for i in range(batch_size)]
        
        output_class, outputs_mask = self.mask_module(queries,
                                                        mask_features,
                                                        None,
                                                        0,
                                                        ret_attn_mask=False,
                                                        point2segment=None,
                                                        coords=None)
        predictions_class.append(output_class)
        predictions_mask.append(outputs_mask)

        # print('stratified mask3d forward finished')
        aux_masks = [a[2] for a in aux_decomposed[:-1]]
        
        return {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class, predictions_mask
            ),
            "aux_masks": aux_masks,
            'sampled_coords': sampled_coords.detach().cpu().numpy() if sampled_coords is not None else None,
            'backbone_features': pcd_features
        }

    def mask_module(self, query_feat, mask_features, mask_segments, num_pooling_steps, ret_attn_mask=True,
                                 point2segment=None, coords=None):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)

        output_masks = []

        # dot product of masked module to prefilter features
        for i in range(len(mask_features)):
            output_masks.append(mask_features[i] @ mask_embed[i].T)

        output_masks = output_masks

        if ret_attn_mask:
            attn_mask = [om.detach().sigmoid() < 0.5 for om in output_masks]
            
            return outputs_class, output_masks, attn_mask

        
        return outputs_class, output_masks

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
