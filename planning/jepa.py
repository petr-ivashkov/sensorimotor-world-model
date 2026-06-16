"""JEPA with inverse dynamics model."""

import stable_pretraining as spt
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        inverse_model=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.inverse_model = inverse_model
        image_size = getattr(getattr(self.encoder, "config", None), "image_size", None)
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        elif isinstance(image_size, (list, tuple)) and image_size:
            if len(image_size) == 1:
                self.image_size = (int(image_size[0]), int(image_size[0]))
            else:
                self.image_size = (int(image_size[0]), int(image_size[1]))
        else:
            self.image_size = None

        imagenet_stats = spt.data.dataset_stats.ImageNet
        mean = torch.tensor(imagenet_stats["mean"], dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(imagenet_stats["std"], dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def _preprocess_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dtype == torch.uint8:
            pixels = pixels.to(dtype=torch.float32).div_(255.0)
        else:
            pixels = pixels.float()

        if self.image_size is not None and tuple(pixels.shape[-2:]) != self.image_size:
            pixels = F.interpolate(
                pixels,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        return (pixels - self.pixel_mean) / self.pixel_std

    def encode(self, info):
        """Encode observations and actions into embeddings."""
        pixels = info["pixels"]
        b = pixels.size(0)
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        pixels = self._preprocess_pixels(pixels)
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding."""
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def predict_action(self, z_t, z_tp1):
        """Predict action from consecutive embeddings using the inverse model."""
        assert self.inverse_model is not None, "No inverse model configured"
        return self.inverse_model(z_t, z_tp1)

    def rollout(self, info, action_sequence):
        """Rollout the model given an initial info dict and action sequence."""
        assert "pixels" in info, "pixels not in info_dict"
        B, S, T = action_sequence.shape[:3]

        history_size = int(getattr(self.predictor, "num_frames", 1))
        pixel_history = info["pixels"].size(2)

        past_action_history = None
        action_context_stride = 1
        if "action" in info:
            raw_action_dim = info["action"].size(-1)
            token_action_dim = action_sequence.size(-1)
            action_context_stride = token_action_dim // raw_action_dim
            required_action_history = (
                (history_size - 1) * action_context_stride
                if action_context_stride > 1
                else history_size
            )
            if history_size > 1:
                past_action_history = torch.nan_to_num(
                    info["action"][:, :, -required_action_history:, :], 0.0
                )
        elif history_size > 1:
            raise ValueError(
                'info["action"] is required for rollout with predictor history_size > 1'
            )

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init.pop("action", None)
        _init = self.encode(_init)
        if action_context_stride > 1 and history_size > 1:
            required_pixel_history = (history_size - 1) * action_context_stride + 1
            emb_indices = torch.arange(
                pixel_history - required_pixel_history,
                pixel_history,
                action_context_stride,
                device=_init["emb"].device,
            )
            emb = _init["emb"].index_select(1, emb_indices)
        else:
            emb = _init["emb"][:, -history_size:]
        emb = info["emb"] = emb.unsqueeze(1).expand(B, S, -1, -1)

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        action_sequence = torch.nan_to_num(action_sequence, 0.0)
        future_actions = rearrange(action_sequence, "b s ... -> (b s) ...")
        if history_size > 1:
            if action_context_stride > 1:
                past_actions = past_action_history.reshape(
                    B, S, history_size - 1, -1
                )
                past_actions = rearrange(past_actions, "b s ... -> (b s) ...")
            else:
                past_actions = rearrange(
                    past_action_history[:, :, -(history_size - 1) :, :],
                    "b s ... -> (b s) ...",
                )
            actions = torch.cat([past_actions, future_actions], dim=1)
        else:
            actions = future_actions

        for t in range(T):
            emb_context = emb[:, -history_size:]
            act_context = actions[:, t : t + history_size]
            act_emb = self.action_encoder(act_context)
            pred_emb = self.predict(emb_context, act_emb)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info_dict: dict):
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert "goal" in info_dict, "goal not in info_dict"
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_"):]] = goal.pop(k)
        goal.pop("action", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)
        cost = self.criterion(info_dict)
        return cost
