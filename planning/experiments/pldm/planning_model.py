"""PLDM adapter for the pinned planning interface."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from vendor.pldm.pldm import PLDM


class PlanningPLDM(PLDM):
    def rollout(self, info, action_sequence):
        assert 'pixels' in info, 'pixels not in info_dict'
        batch_size, num_samples, horizon = action_sequence.shape[:3]
        history_size = int(self.predictor.num_frames)
        pixel_history = info['pixels'].size(2)

        raw_action_dim = info['action'].size(-1)
        token_action_dim = action_sequence.size(-1)
        action_stride = token_action_dim // raw_action_dim
        past_action_history = None
        if history_size > 1:
            required_action_history = (history_size - 1) * action_stride
            past_action_history = torch.nan_to_num(
                info['action'][:, :, -required_action_history:, :], 0.0
            )

        initial = {
            key: value[:, 0]
            for key, value in info.items()
            if torch.is_tensor(value)
        }
        initial.pop('action', None)
        initial = self.encode(initial)

        if history_size > 1:
            required_pixel_history = (history_size - 1) * action_stride + 1
            indices = torch.arange(
                pixel_history - required_pixel_history,
                pixel_history,
                action_stride,
                device=initial['emb'].device,
            )
            emb = initial['emb'].index_select(1, indices)
        else:
            emb = initial['emb'][:, -1:]
        emb = emb.unsqueeze(1).expand(
            batch_size, num_samples, -1, -1
        )
        emb = rearrange(emb, 'b s ... -> (b s) ...').clone()

        future_actions = rearrange(
            torch.nan_to_num(action_sequence, 0.0),
            'b s ... -> (b s) ...',
        )
        if history_size > 1:
            past_actions = past_action_history.reshape(
                batch_size, num_samples, history_size - 1, -1
            )
            past_actions = rearrange(past_actions, 'b s ... -> (b s) ...')
            actions = torch.cat([past_actions, future_actions], dim=1)
        else:
            actions = future_actions

        for step in range(horizon):
            emb_context = emb[:, -history_size:]
            act_context = actions[:, step : step + history_size]
            act_emb = self.action_encoder(act_context)
            pred_emb = self.predict(emb_context, act_emb)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

        info['predicted_emb'] = rearrange(
            emb, '(b s) ... -> b s ...', b=batch_size, s=num_samples
        )
        return info

    def criterion(self, info_dict: dict):
        pred_emb = info_dict['predicted_emb']
        goal_emb = info_dict['goal_emb']
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
        return F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction='none',
        ).sum(dim=tuple(range(2, pred_emb.ndim)))

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert 'goal' in info_dict, 'goal not in info_dict'
        device = next(self.parameters()).device
        for key in list(info_dict):
            if torch.is_tensor(info_dict[key]):
                info_dict[key] = info_dict[key].to(device)

        goal = {
            key: value[:, 0]
            for key, value in info_dict.items()
            if torch.is_tensor(value)
        }
        goal['pixels'] = goal['goal']
        for key in list(info_dict):
            if key.startswith('goal_'):
                goal[key[len('goal_') :]] = goal.pop(key)
        goal.pop('action', None)
        goal = self.encode(goal)

        info_dict['goal_emb'] = goal['emb']
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)
