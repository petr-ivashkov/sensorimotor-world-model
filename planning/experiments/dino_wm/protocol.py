"""Locked protocol constants for the fair DINO-WM baseline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


METHOD = 'dino_wm'
VARIANT = 'noprop'
RESULT_GROUP = 'dino_wm_noprop'
INPUT_PROTOCOL = 'pixels_actions_only'
PROTOCOL_VERSION = 'matched_batch256_pixels_actions_v3'
ACTION_EMBED_DIM = 10
EXPECTED_ENCODING = {'action': ACTION_EMBED_DIM}


def validate_input_protocol(cfg: Any) -> None:
    experiment = cfg.get('dino_wm_experiment', {})
    protocol = str(
        cfg.get('optimization_matching', {}).get('protocol_version', '')
    )
    encoding = dict(cfg.wm.encoding)
    if protocol != PROTOCOL_VERSION:
        raise RuntimeError(
            f'DINO-WM protocol {protocol!r} does not match '
            f'{PROTOCOL_VERSION!r}'
        )
    if str(experiment.get('input_protocol', '')) != INPUT_PROTOCOL:
        raise RuntimeError('DINO-WM must use the pixels/actions-only protocol')
    if bool(experiment.get('uses_privileged_state', True)):
        raise RuntimeError('DINO-WM privileged-state input must be disabled')
    if encoding != EXPECTED_ENCODING:
        raise RuntimeError(
            f'DINO-WM encoding must be exactly {EXPECTED_ENCODING}, '
            f'got {encoding}'
        )


def validate_manifest_row(row: Mapping[str, str]) -> None:
    checks = {
        'method': row.get('method') == METHOD,
        'variant': row.get('variant') == VARIANT,
        'result group': row.get('result_group') == RESULT_GROUP,
        'input protocol': row.get('input_protocol') == INPUT_PROTOCOL,
        'protocol version': row.get('protocol_version') == PROTOCOL_VERSION,
        'privileged state': row.get('uses_privileged_state') == 'false',
        'state key': row.get('state_key', '') == '',
    }
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"Manifest row {row.get('run_name', '<unknown>')}: "
            + ', '.join(f'{label} mismatch' for label in failed)
        )
