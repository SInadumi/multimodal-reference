from dataclasses import dataclass

from utils.util import CamelCaseDataClassJsonMixin, Rectangle


@dataclass(eq=True)
class BoundingBoxPrediction(CamelCaseDataClassJsonMixin):
    image_id: str
    class_id: int
    rect: Rectangle
    confidence: float


@dataclass(frozen=True, eq=True)
class RelationPrediction(CamelCaseDataClassJsonMixin):
    type: str  # ガ, ヲ, ニ, ノ, =, etc...
    bounding_boxes: list[BoundingBoxPrediction]


@dataclass
class PhrasePrediction(CamelCaseDataClassJsonMixin):
    sid: str
    text: str
    relations: list[RelationPrediction]


@dataclass
class MMRefRelPrediction(CamelCaseDataClassJsonMixin):
    doc_id: str
    image_id: str
    phrases: list[PhrasePrediction]
