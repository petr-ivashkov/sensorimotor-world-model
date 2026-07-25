#!/usr/bin/env python3
"""Populate the shared Hugging Face cache with DINOv2-Small."""

from transformers import AutoModel


BACKBONE = 'facebook/dinov2-small'
REVISION = 'ed25f3a31f01632728cabb09d1542f84ab7b0056'


def main() -> None:
    model = AutoModel.from_pretrained(BACKBONE, revision=REVISION)
    revision = getattr(model.config, '_commit_hash', None) or 'unknown'
    if revision != REVISION:
        raise RuntimeError(f'Expected revision {REVISION}, got {revision}')
    print(f'Cached {BACKBONE} at revision {revision}')


if __name__ == '__main__':
    main()
