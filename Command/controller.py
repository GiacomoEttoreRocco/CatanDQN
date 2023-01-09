from dataclasses import dataclass, field
import Command.action as action
import os

@dataclass
class ActionController:
    undoStack: list[action.Action] = field(default_factory = list)
    redoStack: list[action.Action] = field(default_factory = list)

    def execute(self, action: action.Action):
        action.execute()
        self.redoStack.clear()
        self.undoStack.append(action)

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

    def summaryUndoStack(self):
        l = []
        for action in self.undoStack:
            for x in repr(action).split("\n\t"):
                l.append(f'{action.player.id}:{x}')
        return l
    def summaryRedoStack(self):
        l = []
        for action in self.redoStack:
            for x in reversed(repr(action).split("\n\t")):
                l.append(f'{action.player.id}:{x}')
        return l
