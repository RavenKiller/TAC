#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Registry is central source of truth in Habitat.

Taken from Pythia, it is inspired from Redux's concept of global store.
Registry maintains mappings of various information to unique keys. Special
functions in registry can be used as decorators to register different kind of
classes.

Import the global registry object using

.. code:: py

    from habitat.core.registry import registry

Various decorators for registry different kind of classes with unique keys

-   Register a task: ``@registry.register_task``
-   Register a task action: ``@registry.register_task_action``
-   Register a simulator: ``@registry.register_simulator``
-   Register a sensor: ``@registry.register_sensor``
-   Register a measure: ``@registry.register_measure``
-   Register a dataset: ``@registry.register_dataset``
"""

import collections
from typing import Any, Callable, DefaultDict, Optional, Type

from models.base_model import BaseModel
from trainers.base_trainer import BaseTrainer
from dataloaders.base_loader import BaseLoader


class Registry:
    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: Optional[str],
        assert_type: Optional[Type] = None,
    ) -> Callable:
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(to_register, assert_type)
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def register_model(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("model", to_register, name, assert_type=BaseModel)

    @classmethod
    def register_trainer(cls, to_register: None = None, *, name: Optional[str] = None):
        return cls._register_impl("trainer", to_register, name, assert_type=BaseTrainer)

    @classmethod
    def register_dataloader(
        cls, to_register: None = None, *, name: Optional[str] = None
    ):
        return cls._register_impl(
            "dataloader", to_register, name, assert_type=BaseLoader
        )

    @classmethod
    def _get_impl(cls, _type: str, name: str) -> Type:
        return cls.mapping[_type].get(name, None)

    @classmethod
    def get_model(cls, name: str) -> Type[BaseModel]:
        return cls._get_impl("model", name)

    @classmethod
    def get_trainer(cls, name: str) -> Type[BaseTrainer]:
        return cls._get_impl("trainer", name)

    @classmethod
    def get_dataloader(cls, name: str) -> Type[BaseTrainer]:
        return cls._get_impl("dataloader", name)


registry = Registry()
