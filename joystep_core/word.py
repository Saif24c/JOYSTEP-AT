"""Structured Microsoft Word adapter; no coordinate automation is used."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .applications import ApplicationContext
from .intents import CommandIntent, IntentKind


class WordApi(Protocol):
    def application(self, start: bool = False) -> Any | None: ...


class PyWin32WordApi:
    """Lazy pywin32 boundary so importing JOYSTEP does not start Word."""
    def application(self, start: bool = False) -> Any | None:
        try:
            import win32com.client
            if start:
                return win32com.client.Dispatch("Word.Application")
            return win32com.client.GetActiveObject("Word.Application")
        except Exception:
            return None


@dataclass
class WordAdapter:
    name: str = "microsoft-word"
    api: WordApi | None = None
    last_message: str = ""
    _pending: tuple[str, dict[str, object]] | None = None

    def __post_init__(self) -> None:
        if self.api is None:
            self.api = PyWin32WordApi()

    def supports(self, intent: CommandIntent, context: ApplicationContext) -> bool:
        if intent.kind is IntentKind.OPEN_APPLICATION:
            return "word" in str(intent.arguments.get("target", "")).lower()
        if intent.kind is IntentKind.DOCUMENT:
            return True
        if intent.kind in {IntentKind.SAVE, IntentKind.UNDO, IntentKind.REDO, IntentKind.COPY,
                           IntentKind.PASTE, IntentKind.SELECT_ALL, IntentKind.DELETE,
                           IntentKind.BACKSPACE, IntentKind.TYPE_TEXT}:
            return self.api.application(False) is not None
        return False

    def execute(self, intent: CommandIntent, context: ApplicationContext) -> bool:
        self.last_message = ""
        try:
            if intent.kind is IntentKind.OPEN_APPLICATION:
                app = self.api.application(True)
                if app is None:
                    return self._fail("Word needs help: Microsoft Word is unavailable.")
                app.Visible = True
                self.last_message = "Word is ready."
                return True
            app = self.api.application(False)
            if app is None:
                return self._fail("Word needs help: open Microsoft Word first.")
            document = self._document(app)
            if document is None:
                return self._fail("Word needs help: no active document.")
            if getattr(document, "ReadOnly", False) and self._mutates(intent):
                return self._fail("Word needs help: document is read-only.")
            if intent.kind is IntentKind.SAVE:
                return self._request("save", {})
            if intent.kind is IntentKind.UNDO:
                app.Undo()
                self.last_message = "Word undo complete."
                return True
            if intent.kind is IntentKind.REDO:
                app.Redo()
                self.last_message = "Word redo complete."
                return True
            if intent.kind is IntentKind.COPY:
                app.Selection.Copy()
                self.last_message = "Copied."
                return True
            if intent.kind is IntentKind.CUT:
                return self._delete_or_cut(app, cut=True)
            if intent.kind is IntentKind.PASTE:
                app.Selection.Paste()
                self.last_message = "Pasted."
                return True
            if intent.kind is IntentKind.SELECT_ALL:
                app.Selection.WholeStory()
                self.last_message = "Selected all document text."
                return True
            if intent.kind in {IntentKind.DELETE, IntentKind.BACKSPACE}:
                return self._delete_or_cut(app, cut=False)
            if intent.kind is IntentKind.TYPE_TEXT:
                app.Selection.TypeText(intent.arguments["text"])
                self.last_message = "Text inserted."
                return True
            return self._document_action(app, document, intent)
        except Exception:
            return self._fail("Word needs help: the document operation failed.")

    def _document_action(self, app: Any, document: Any, intent: CommandIntent) -> bool:
        action = str(intent.arguments.get("action", ""))
        selection = app.Selection
        if action.startswith("confirm "):
            return self._confirm(app, document, action)
        if action == "save_as":
            path = self._safe_path(str(intent.arguments["path"]))
            if path is None:
                return self._fail("Word needs help: save as requires an absolute .docx path.")
            return self._request("save_as", {"path": path, "overwrite": path.exists()})
        if action == "find":
            found = selection.Find.Execute(FindText=intent.arguments["text"], Forward=True, Wrap=1)
            self.last_message = "Text found." if found else "Text not found."
            return True
        if action in {"replace", "replace_all"}:
            if action == "replace_all":
                return self._request("replace_all", dict(intent.arguments))
            selection.Find.Execute(FindText=intent.arguments["find"], ReplaceWith=intent.arguments["replace"], Replace=1)
            self.last_message = "Replace complete."
            return True
        if action == "bold": selection.Font.Bold = -1
        elif action == "italic": selection.Font.Italic = -1
        elif action == "underline": selection.Font.Underline = 1
        elif action == "remove formatting": selection.ClearFormatting()
        elif action == "font_size": selection.Font.Size = intent.arguments["size"]
        elif action == "heading": selection.Style = f"Heading {intent.arguments['level']}"
        elif action == "align left": selection.ParagraphFormat.Alignment = 0
        elif action == "align center": selection.ParagraphFormat.Alignment = 1
        elif action == "align right": selection.ParagraphFormat.Alignment = 2
        elif action == "select current word": selection.Expand(2)
        elif action == "select current sentence": selection.Expand(3)
        elif action == "select current paragraph": selection.Expand(4)
        elif action == "move to beginning of document": selection.HomeKey(Unit=6)
        elif action == "move to end of document": selection.EndKey(Unit=6)
        elif action == "move next paragraph": selection.MoveDown(Unit=4, Count=1)
        elif action == "move previous paragraph": selection.MoveUp(Unit=4, Count=1)
        elif action == "move next sentence": selection.MoveDown(Unit=3, Count=1)
        elif action == "move previous sentence": selection.MoveUp(Unit=3, Count=1)
        else:
            return self._fail("Word needs help: that document command is unsupported.")
        self.last_message = "Word command complete."
        return True

    def _document(self, app: Any) -> Any | None:
        try:
            return app.ActiveDocument if app.Documents.Count else None
        except Exception:
            return None

    def _delete_or_cut(self, app: Any, cut: bool) -> bool:
        selection = app.Selection
        if abs(int(selection.End) - int(selection.Start)) >= 100:
            return self._request("cut" if cut else "delete", {})
        if cut: selection.Cut()
        else: selection.Delete()
        self.last_message = "Cut complete." if cut else "Deleted selection."
        return True

    def _request(self, action: str, arguments: dict[str, object]) -> bool:
        self._pending = (action, arguments)
        self.last_message = "Confirmation required. Say confirm save, confirm overwrite, confirm delete, or confirm replace all."
        return True

    def _confirm(self, app: Any, document: Any, action: str) -> bool:
        if self._pending is None:
            return self._fail("Word needs help: there is no pending operation to confirm.")
        pending, args = self._pending
        if pending == "save" and action == "confirm save":
            if getattr(document, "ReadOnly", False): return self._fail("Word needs help: document is read-only.")
            document.Save(); self._pending = None; self.last_message = "Document saved."; return True
        if pending == "save_as" and action in {"confirm save", "confirm overwrite"}:
            path = args["path"]
            if args["overwrite"] and action != "confirm overwrite":
                return self._fail("Confirmation required: say confirm overwrite.")
            document.SaveAs2(str(path)); self._pending = None; self.last_message = "Document saved."; return True
        if pending in {"delete", "cut"} and action == "confirm delete":
            if pending == "cut": app.Selection.Cut()
            else: app.Selection.Delete()
            self._pending = None; self.last_message = "Selection changed."; return True
        if pending == "replace_all" and action == "confirm replace all":
            app.Selection.Find.Execute(FindText=args["find"], ReplaceWith=args["replace"], Replace=2)
            self._pending = None; self.last_message = "Replace all complete."; return True
        return self._fail("Confirmation required: the confirmation did not match the pending operation.")

    def _safe_path(self, value: str) -> Path | None:
        path = Path(value)
        if not path.is_absolute() or path.suffix.lower() != ".docx": return None
        return path

    def _mutates(self, intent: CommandIntent) -> bool:
        if intent.kind in {IntentKind.SAVE, IntentKind.UNDO, IntentKind.REDO, IntentKind.CUT,
                           IntentKind.PASTE, IntentKind.DELETE, IntentKind.BACKSPACE, IntentKind.TYPE_TEXT}:
            return True
        if intent.kind is not IntentKind.DOCUMENT:
            return False
        return str(intent.arguments.get("action", "")) not in {
            "find", "select current word", "select current sentence", "select current paragraph",
            "move to beginning of document", "move to end of document", "move next paragraph",
            "move previous paragraph", "move next sentence", "move previous sentence",
        }

    def _fail(self, message: str) -> bool:
        self.last_message = message
        return True
