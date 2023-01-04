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
        # os.system("cls")
        # s = "UNDO STACK, chiamato da execute: "
        # for a in self.undoStack:
        #     s += a.__class__.__name__ + " "
        # print(s)

    def undo(self) -> None:
        if not self.undoStack:
            return
        action = self.undoStack.pop()
        action.undo()
        self.redoStack.append(action)
        # os.system("cls")
        # s = "UNDO STACK, chiamato da undo, post pop(): "
        # for a in self.undoStack:
        #     s += a.__class__.__name__ + " "
        # print(s)


    def redo(self) -> None:
        if not self.redoStack:
            return
        action = self.redoStack.pop()
        action.redo()
        self.undoStack.append(action)
