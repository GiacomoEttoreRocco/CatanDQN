from dataclasses import dataclass, field
import Command.action as action

@dataclass
class ActionController:
    undoStack: list[action.Action] = field(default_factory = list)
    redoStack: list[action.Action] = field(default_factory = list)

    def execute(self, action: action.Action):
        action.execute()
        self.undoStack.clear()
        self.redoStack.append(action)

    def undo(self) -> None:
        if not self.undoStack:
            return
        action = self.undoStack.pop()
        action.undo()
        self.redoStack.append(action)

    def redo(self) -> None:
        if not self.redoStack:
            return
        action = self.redoStack.pop()
        action.redo()
        self.undoStack.append(action)
