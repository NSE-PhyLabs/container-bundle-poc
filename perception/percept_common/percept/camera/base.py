"""카메라 어댑터 추상. grab() 이 Frame 을 낸다. 프레임이 더 없으면 StopIteration, 하드웨어·라이브러리가 없으면 NotAvailable."""
from abc import ABC, abstractmethod

from ..types import Frame


class CameraAdapter(ABC):
    name: str = "camera"

    @abstractmethod
    def grab(self) -> Frame:
        """다음 프레임. 없으면 StopIteration."""

    def close(self) -> None:
        pass

    def __iter__(self):
        return self

    def __next__(self) -> Frame:
        return self.grab()
