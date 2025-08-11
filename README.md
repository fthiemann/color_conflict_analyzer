# Color Conflict Analyzer (QGIS Plugin)

A QGIS plugin to **detect, analyze, and group color conflicts** between layers and categories in your project.  
Supports **CIELAB2000** color difference calculations, conflict scoring, and connected-group detection.

## Features
- ✅ **Layer selection** – choose which layers to analyze
- 🎯 **CIELAB2000 analysis** – detects perceptual color differences
- 📊 **Conflict summary per color** – count, average ΔE, min/max
- 🔗 **Conflict groups** – groups of colors that conflict with each other
- ⚠️ **Severity levels** – critical, major, medium, or no conflict
- Works with **single**, **graduated**, and **categorized** symbols

## How it works
1. Select layers in the plugin dialog.
2. The plugin extracts color values from each symbol.
3. All colors are compared using **ΔE CIE2000**.
4. Results are shown in the plugin’s output panel:
   - Conflict list
   - Per-color statistics
   - Conflict groups
