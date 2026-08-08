# QED Diagram Studio

A Streamlit app that generates representative examples of connected Feynman diagrams based on the number of external electron, positron, and photon lines, the number of QED vertices, and the loop order.

## Launch

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

## Current Scope

- Particles: electrons (`e⁻`), positrons (`e⁺`), and photons (`γ`)
- Validates charge conservation and the specified loop order
- Includes only One-Particle Irreducible (1PI) diagrams
- Excludes closed fermion loops with an odd number of photon vertices (based on Furry's theorem)
- Removes duplicates using strict graph isomorphism checking
- Renders using the open-source `feynman` library (v2.1.1)
- Uses `pyfeyn2` (v2.4.3) for FeynML structures and automatic line-bending logic
- Assigns internal photon lines to separate upper and lower lanes to prevent overlap
- Allows input of either the number of vertices or the loop order, automatically calculating the other
- Each vertex connects two electron lines and one photon line
- Generates connected diagrams (excluding self-loops)
- For large inputs, generates up to 24 unique representative examples rather than an exhaustive list

## Definition of 1PI

Only diagrams that remain connected after the removal of any single internal propagator are included. Consequently, standard exchange-type tree diagrams—which split into two separate components when an internal line is cut—are not displayed.