from typing import Any, Callable, Dict, Optional, Text

import pytest
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import ENTITIES, ENTITY_ATTRIBUTE_VALUE, INTENT, TEXT
from rasa.shared.nlu.training_data.message import Message

from nlu_utils.nlu.extractors.plate_number_extractor import PlateNumberEntityExtractor

normal_plate_number = [
    ("Ab Cd-123", True),
    ("Abcd-123", True),
    ("Ab Cd123", True),
    ("Abcd123", True),
    ("Ab Cd 123", True),
    ("Abcd 123", True),
    ("Ab-Cd-123", True),
    ("Ab-Cd 123", True),
    ("Ab-Cd123", True),
]

custom_plate_number = [
    ("Abc-1234", True),
    ("Abc 1234", True),
    ("Abc1234", True),
    ("Abcd-123", True),
    ("Abcd 123", True),
    ("Abcd123", True),
    ("Abcde-12", True),
    ("Abcde 12", True),
    ("Abcde12", True),
    ("Abcdef-1", True),
    ("Abcdef 1", True),
    ("Abcdef1", True),
]

special_plate_number = [
    ("CD 123 456", True),
    ("CD123 456", True),
    ("CD 123456", True),
    ("AAAAAA-1", True),
    ("BELA-001", True),
    ("George 1", True),
    ("Jani 007", True),
    ("I 12 AA 22", True),
    ("I 123-AA", True),
    ("CD-1234 22", True),
]

normal_plate_number_old = [
    ("Abc-123", True),
    ("Abc 123", True),
    ("Abc123", True),
]

custom_plate_number_old = [
    ("Abcd-12", True),
    ("Abcd 12", True),
    ("Abcd12", True),
    ("Abcde-1", True),
    ("Abcde 1", True),
    ("Abcde1", True),
]


@pytest.fixture()
def load_extractor(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
) -> Callable[..., PlateNumberEntityExtractor]:
    def inner(config: Dict[Text, Any] = None) -> PlateNumberEntityExtractor:
        config = config or PlateNumberEntityExtractor.get_default_config()

        return PlateNumberEntityExtractor.create(
            config=config,
            model_storage=default_model_storage,
            resource=Resource("duration"),
            execution_context=default_execution_context,
        )

    return inner


@pytest.mark.parametrize(
    "text,should_match",
    [
        *normal_plate_number,
        *custom_plate_number,
        *special_plate_number,
        *normal_plate_number_old,
        *custom_plate_number_old,
    ],
)
def test_plate_number_entity_extractor_examples(
    text: str,
    should_match: bool,
    load_extractor: Callable[..., PlateNumberEntityExtractor],
):
    extractor = load_extractor()

    message = Message(data={TEXT: text, INTENT: "any_intent"})

    result = extractor.process([message])[0]
    entities = result.get(ENTITIES, [])

    if should_match:
        assert len(entities) == 1
        assert entities[0].get(ENTITY_ATTRIBUTE_VALUE) == text
    else:
        assert len(entities) == 0


@pytest.mark.parametrize(
    "intent,use_intent,include",
    [
        ("use_intent", None, True),
        ("use_intent", "use_intent", True),
        ("drop_intent", "use_intent", False),
    ],
)
def test_plate_number_entity_extractor_exclusion(
    intent: str,
    use_intent: Optional[str],
    include: bool,
    load_extractor: Callable[..., PlateNumberEntityExtractor],
):
    extractor = load_extractor({"use_intent": [use_intent]} if use_intent else None)
    message = Message(data={TEXT: "ABC-123", INTENT: intent})

    result = extractor.process([message])[0]
    entities = result.get(ENTITIES, [])
    if include:
        assert len(entities) == 1
    else:
        assert len(entities) == 0
