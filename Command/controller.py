@dataclass
class MoveController:
    undoStack: list[Action] = field(default_factory = list)
    redoStack: list[Action] = field(default_factory = list)

    def execute(self, moveCommand: MoveCommand):
        moveCommand.execute()
        self.undoStack.clear()
        self.redoStack.append(moveCommand)

    def undo(self) -> None:
        if not self.undoStack:
            return
        moveCommand = self.undoStack.pop()
        moveCommand.undo()
        self.redoStack.append(moveCommand)

    def redo(self) -> None:
        if not self.redoStack:
            return
        moveCommand = self.redoStack.pop()
        moveCommand.redo()
        self.undoStack.append(moveCommand)
