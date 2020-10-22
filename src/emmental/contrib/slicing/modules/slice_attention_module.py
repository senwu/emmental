"""Slicing attention module."""
from typing import Any, Dict

import torch
from torch import Tensor, nn as nn
from torch.nn import functional as F


class SliceAttentionModule(nn.Module):
    """An attention module to leverage all slice representations.

    Args:
      slice_ind_key: Slice indicator head key, defaults to "_slice_ind_".
      slice_pred_key: Slice prediction head key, defaults to "_slice_pred_",
      slice_pred_feat_key: Slice prediction feature key,
        defaults to "_slice_feat_".
    """

    def __init__(
        self,
        slice_ind_key: str = "_slice_ind_",
        slice_pred_key: str = "_slice_pred_",
        slice_pred_feat_key: str = "_slice_feat_",
    ) -> None:
        """Initialize SliceAttentionModule."""
        super().__init__()

        self.slice_ind_key = slice_ind_key
        self.slice_pred_key = slice_pred_key
        self.slice_pred_feat_key = slice_pred_feat_key

    def forward(self, intermediate_output_dict: Dict[str, Any]) -> Tensor:
        """Forward function.

        Args:
          intermediate_output_dict: output dict.

        Returns:
          Output of attention.
        """
        # Collect ordered slice indicator head names
        slice_indicator_names = sorted(
            [
                flow_name
                for flow_name in intermediate_output_dict.keys()
                if self.slice_ind_key in flow_name
            ]
        )
        # Collect ordered slice predictor head names
        slice_predictor_names = sorted(
            [
                flow_name
                for flow_name in intermediate_output_dict.keys()
                if self.slice_pred_key in flow_name
            ]
        )
        # Concat slice indicator predictions
        slice_indicator_predictions = torch.cat(
            [
                F.softmax(intermediate_output_dict[slice_indicator_name][0])[
                    :, 0
                ].unsqueeze(1)
                for slice_indicator_name in slice_indicator_names
            ],
            dim=-1,
        )
        # Concat slice predictor predictions
        slice_predictor_predictions = torch.cat(
            [
                F.softmax(intermediate_output_dict[slice_predictor_name][0])[
                    :, 0
                ].unsqueeze(1)
                for slice_predictor_name in slice_predictor_names
            ],
            dim=-1,
        )
        # Collect ordered slice feature head names
        slice_feature_names = sorted(
            [
                flow_name
                for flow_name in intermediate_output_dict.keys()
                if self.slice_pred_feat_key in flow_name
            ]
        )
        # Concat slice representations
        slice_representations = torch.cat(
            [
                intermediate_output_dict[slice_feature_name][0].unsqueeze(1)
                for slice_feature_name in slice_feature_names
            ],
            dim=1,
        )
        # Attention
        A = (
            F.softmax(slice_indicator_predictions * slice_predictor_predictions, dim=1)
            .unsqueeze(-1)
            .expand([-1, -1, slice_representations.size(-1)])
        )

        reweighted_representation = torch.sum(A * slice_representations, 1)

        return reweighted_representation
