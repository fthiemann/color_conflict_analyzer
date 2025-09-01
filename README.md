# Color Conflict Analyzer (QGIS Plugin)

Analyze and fix color conflicts in your QGIS project for users with color-vision deficiencies (CVD).  
The plugin simulates multiple CVD scenarios, detects risky color pairings (via ΔE2000), and **automatically proposes & applies** recolors that maximize perceptual distance—without touching your original layers.

## ✨ Key features

- **CVD simulations**: normal vision, protanomaly/deuteranomaly/tritanomaly (25–100% severity).
- **Conflict detection**: ΔE2000 distances in CIE Lab across all selected layers & categories.
- **Smart recoloring**: searches the color space around targets, enforces distance to all kept colors (incl. CVD views), stays in sRGB gamut.
- **One-click preview**: duplicates analyzed layers, applies new colors on the **clones**, and groups them under **`CVD Recolored`**.
- **Renderer support**: `single symbol`, `categorized`, `graduated` (originals remain unchanged).

---

## 🚀 Quick start

1. **Install the plugin**
   - Clone or download this repo.
   - Copy the folder into your local QGIS plugins directory:
     - **Windows**: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\ColorConflictAnalyzer`
     - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/ColorConflictAnalyzer`
     - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/ColorConflictAnalyzer`
   - Restart QGIS and enable the plugin in **Plugins → Manage and Install Plugins…**.

2. **Install Python dependencies** (inside QGIS’ Python environment)
   - Open the **OSGeo4W Shell** (Windows) or your system terminal with QGIS’ env activated (macOS/Linux).
   - Run:
     ```bash
     # Windows (OSGeo4W Shell)
     python -m pip install --user colorspacious colormath scikit-image numpy

     # macOS / Linux (inside QGIS' python environment)
     python3 -m pip install --user colorspacious colormath scikit-image numpy
     ```
   - If `numpy` is already present, pip will skip it.

3. **Use it**
   - Open **Plugins → ColorConflictAnalyzer**.
   - Select layers to analyze, set **ΔE threshold**, click **Analyze**.
   - Pick categories to recolor, set **Recolor threshold**, click **Recolor Layers**.
   - See duplicated, recolored layers grouped under **`CVD Recolored`** in the Layer Panel.

---

## 📦 Requirements

- **QGIS**: 3.28+ (tested with **3.40.5**)
- **Python packages** (inside QGIS’ Python):
  - `numpy`
  - `scikit-image` (for `deltaE_ciede2000`)
  - `colorspacious` (RGB↔CVD and RGB↔Lab conversions)
  - `colormath` (Lab object used by parts of the code)
- **Qt / PyQt**: whatever ships with your QGIS (no extra install needed)

> ⚠️ Make sure you install packages into **QGIS’ Python**, not your system Python. On Windows, always use the **OSGeo4W Shell**.

---

## 🧠 How it works (in short)

1. **Simulation**  
   For each selected layer (and each category/range), the plugin converts the symbol color to sRGB (0–1), optionally applies a CVD transform (colorspacious), then converts to CIE Lab.

2. **Conflict detection**  
   ΔE2000 distances are computed across all selected items and across all simulated views. Smallest distances below your **ΔE threshold** are reported as conflicts.

3. **Recoloring strategy**  
   For items you mark to recolor, the plugin:
   - Samples directions on a sphere around the original color in Lab.
   - Steps outwards until ΔE≥threshold to the original, then **pushes** further while ensuring ΔE≥threshold to **every** kept color in **every** view (normal + CVD severities).
   - Picks the best candidate by a small multi-criteria score (max margin vs. minimum semantic shift, etc.) and **stays in sRGB gamut**.

4. **Non-destructive apply**  
   The plugin duplicates each analyzed layer, applies new colors on the **clone** only (renderer-aware), and nests everything in a `CVD Recolored` group for side-by-side comparison.

---

## 🧭 Usage tips

- Start with **ΔE threshold** ~15 for analysis, **Recolor threshold** ~20 for recoloring.
- Recolor only categories that matter most (those with many conflicts or extremely low ΔE).
- You can run multiple times: each run creates or reuses the `CVD Recolored` group with fresh clones.
- 

---

## 🛠 Renderer details

- **Single Symbol**  
  The symbol is cloned and recolored safely; renderer is re-assigned to the clone.

- **Categorized**  
  Each `QgsRendererCategory` is **rebuilt** with a **cloned** symbol; we create a fresh `QgsCategorizedSymbolRenderer` and assign it to the clone (prevents dangling pointers and legend crashes).

- **Graduated**  
  Each `QgsRendererRange` is **rebuilt** with a **cloned** symbol; we create a fresh `QgsGraduatedSymbolRenderer` preserving the class attribute (and mode where available) and assign it to the clone.

> Other renderers (rule-based, heatmap, etc.) are currently not modified; they are cloned but left untouched.


