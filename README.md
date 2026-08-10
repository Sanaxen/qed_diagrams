# QED diagram studio


![qed_diagrams_all](https://github.com/Sanaxen/qed_diagrams/blob/main/qed_diagrams_all_2loop.png)

A Streamlit application for enumerating and drawing QED Feynman diagrams from
incoming and outgoing electron, positron, and photon counts. The perturbative
order can be specified by either vertex count or loop order.

## Start the application on Windows

Double-click `run_app.bat`, or run it from PowerShell:

```powershell
.\run_app.bat
```

The script creates `.venv`, installs missing Python dependencies, and starts
the application at `http://localhost:8501`.

## Graphviz installation

Graphviz is optional for the standard QED layout and required for the two
Graphviz layout options. On the first `run_app.bat` launch, the script checks
for `neato.exe`. If it is missing, the script asks:

```text
Install Graphviz now with winget? [Y/N]
```

Choose `Y` to install it automatically. Choose `N` to continue with the QED
layout. The application remains usable without Graphviz.

To install Graphviz manually:

```powershell
winget install --id Graphviz.Graphviz -e
```

Then close and reopen the terminal and verify:

```powershell
neato -V
```

If `neato` is still not found, add the Graphviz `bin` directory, normally
`C:\Program Files\Graphviz\bin`, to the Windows `PATH` environment variable.

The Python adapter `pydot` is installed automatically from `requirements.txt`.

## Layout modes

- Standard QED layout
- Graphviz `neato` layout
- Graphviz initial layout followed by QED refinement (recommended)

If Graphviz is selected but unavailable, the app displays a warning and safely
falls back to the standard QED layout.

## Diagram selection

- Enforces charge conservation and the QED vertex rules
- Supports vertex-count and loop-order input
- Supports 1PI-only or all connected diagrams, including tree-level diagrams
- Excludes vanishing odd-photon fermion loops using Furry's theorem
- Removes graph-isomorphic duplicates
- Supports up to 1000 displayed diagrams
- Exports individual PNG/SVG files and combined contact sheets
