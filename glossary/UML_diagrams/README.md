# UML diagrams

One draw.io UML class/component diagram per template, plus a toolkit overview. Each diagram shows
the code as it is today (top) and **where new models can be added and what they are for** (the
green dashed lanes at the bottom).

| File | What it shows |
|---|---|
| `00_toolkit_overview.drawio` | All 16 templates by area, the conceptual hand-offs between them, proposed new templates |
| `tNN_<name>.drawio` | That template's package: `run.py`, `Config`, `data.py`, `methods.py`, `checks.py` + `Finding`, extra modules, spec/SQL folders, and its extension points |

## How to open

* **draw.io desktop**, or **app.diagrams.net** → *File → Open from → Device*.
* **VS Code**: the *Draw.io Integration* extension (`hediet.vscode-drawio`) opens `.drawio` files in an editor tab.
* **Claude Code**: the draw.io MCP server (`@drawio/mcp`) is registered at user scope. Restart the session to load its tools.

## How to read a template diagram

| Element | Meaning |
|---|---|
| Blue row `[S]` | Standard method: the README default |
| Orange row `[A]` | Alternative method: switch to it when the README's named caveat applies |
| Red row `[naive]` | Kept only to show the bias. Never report it |
| White row | Helper or shared step |
| Green dashed lane | **Extension point**. Its arrow points at the module the new code goes into |
| `«model»` / `«alternative»` / `«loader»` / `«check»` / `«spec»` / `«module»` | Kind of extension: new analysis / third method option / new data source / new assumption check / new YAML/CSV spec (no code) / new file in the package |
| *Pairs with* | Related template (`tNN`) or glossary entry (`gNN`) that already holds part of the method |

## Regenerating

The diagrams are generated, so they stay in step with the code:

```powershell
python glossary\UML_diagrams\build_uml_diagrams.py
```

* `build_uml_diagrams.py` reads each template's modules, public functions, `Config` and `Finding`
  fields, and spec folders with `ast`. It uses only the standard library.
* `extensions.py` holds the hand-curated parts: which functions are standard, alternative or naive
  (taken from each README's *Method choices* table), the extension points, and the proposed new
  templates. Edit this file to add or change an extension idea, then re-run the generator.

Manual edits made in draw.io are overwritten on regeneration. Put lasting changes in `extensions.py`.
