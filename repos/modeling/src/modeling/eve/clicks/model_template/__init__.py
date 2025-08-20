from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

from .base import BaseClickModelTemplate
from .uitars import UITarsModelTemplate


class ModelTemplateChoice(str, Enum):
    uitars = "uitars"


MODEL_TEMPLATES: Mapping[ModelTemplateChoice, BaseClickModelTemplate] = {
    ModelTemplateChoice.uitars: UITarsModelTemplate(),
}


__all__ = (
    "MODEL_TEMPLATES",
    "BaseClickModelTemplate",
    "ModelTemplateChoice",
    "UITarsModelTemplate",
)
