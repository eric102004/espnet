from abc import ABC, abstractmethod
from typing import Tuple, Sequence, Union


class AbsCompressor(ABC):
    @abstractmethod
    def encode(
        self,
        input: Sequence[Union[int, float]],
        ilens: Sequence[int],
    ) -> Tuple[Sequence[Union[int, float]], Sequence[int]]:
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        input: Sequence[Union[int, float]],
        ilens: Sequence[int],
    ) -> Tuple[Sequence[Union[int, float]], Sequence[int]]:
        raise NotImplementedError
