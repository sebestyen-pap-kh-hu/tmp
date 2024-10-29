import logging
import re
from typing import Any, Dict, List, Text

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_TEXT,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    EXTRACTOR,
    INTENT,
    INTENT_NAME_KEY,
)
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)

plate_number_regexes = [
    r"\b([epvz])[ -]?\d{5}\b",
    r"\b(m)\d{2}[ -]?\d{4}\b",
    r"\b(ck|dt|hc|cd|hx|ma|ot|rx|rr)[ -]?\d{2}[ -]?\d{2}\b",
    r"\b(cd)[ -]?\d{3}[ -]?\d{3}\b",
    r"\b(cd)[ -]?\d{4}[ -]?[0-2][0-9]\b",
    r"\b(i)[ -]?\d{3}[ -]?[a-zA-Z]{2}\b",
    r"\b(i)[ -]?\d{2}[ -]?[a-zA-Z]{2}[ -]?[0-2][0-9]\b",
    r"\b(c-c|c-x|x-a|x-b|x-c)[ -]?\d{4}\b",
    r"\b[a-zA-Z]{2}[ -]?[a-zA-Z]{2}[ -]?\d{3}\b",
    r"\b[a-zA-Z]{3}[ -]?\d{4}\b",
    r"\b[a-zA-Z]{5}[ -]?\d{2}\b",
    r"\b[a-zA-Z]{6}[ -]?\d{1}\b",
    r"\b[a-zA-Z]{3}[ -]?\d{3}\b",
    r"\b[a-zA-Z]{4}[ -]?\d{2}\b",
    r"\b[a-zA-Z]{5}[ -]?\d{1}\b",
]


@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR], is_trainable=False
)
class PlateNumberEntityExtractor(GraphComponent, EntityExtractorMixin):
    def __init__(self, config: Dict[Text, Any]) -> None:
        self.config = config

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config)

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {"use_intent": None}

    def process(self, messages: List[Message]) -> List[Message]:
        use_intent = self.config["use_intent"]
        for message in messages:
            if (use_intent is None) or (
                use_intent and message.get(INTENT).get(INTENT_NAME_KEY) in use_intent
            ):
                existing_entities = message.get(ENTITIES, [])

                extracted = self._extract_entities(message)

                message.set(
                    ENTITIES,
                    sorted(existing_entities + extracted, key=lambda x: x["start"]),
                    add_to_output=True,
                )

        return messages

    def _extract_entities(self, message):
        text = message.get("text", "")

        extracted = []
        for regex in plate_number_regexes:
            # Ensure all digits are recognizes
            regex = regex + r"(?![ -]\d)"

            for match in re.finditer(regex, text, re.IGNORECASE):
                extracted.append(
                    {
                        ENTITY_ATTRIBUTE_START: match.start(),
                        ENTITY_ATTRIBUTE_END: match.end(),
                        ENTITY_ATTRIBUTE_TEXT: match.group(0),
                        ENTITY_ATTRIBUTE_VALUE: match.group(0),
                        ENTITY_ATTRIBUTE_TYPE: "car_hu_plate_number",
                        EXTRACTOR: self.name,
                    }
                )

        return extracted
