from typing import Protocol

class Action(Protocol):
    def execute(self) -> None:
        ...

    def undo(self) -> None:
        ...

    def redo(self) -> None:
        ...