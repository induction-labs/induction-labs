from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

from .base import BaseClickModelTemplate
from .uitars import UITarsModelTemplate
from .venus_ground import VenusGroundModelTemplate


class ModelTemplateChoice(str, Enum):
    uitars = "uitars"
    venus_ground = "venus_ground"


MODEL_TEMPLATES: Mapping[ModelTemplateChoice, BaseClickModelTemplate] = {
    ModelTemplateChoice.uitars: UITarsModelTemplate(),
    ModelTemplateChoice.venus_ground: VenusGroundModelTemplate(),
}


__all__ = (
    "MODEL_TEMPLATES",
    "BaseClickModelTemplate",
    "ModelTemplateChoice",
    "UITarsModelTemplate",
    "VenusGroundModelTemplate",
)
