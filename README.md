Color Conflict Analyzer (QGIS Plugin)
A QGIS plugin to detect, analyze, and group color conflicts between layers and categories in your project.
Supports CIELAB2000 color difference calculations, conflict scoring, and connected-group detection.

Features
✅ Layer selection – choose which layers to analyze

🎯 CIELAB2000 analysis – detects perceptual color differences

📊 Conflict summary per color – count, average ΔE, min/max

🔗 Conflict groups – groups of colors that conflict with each other

⚠️ Severity levels – critical, major, medium, or no conflict

Works with single, graduated, and categorized symbols

How it works
Select layers in the plugin dialog.

The plugin extracts color values from each symbol.

All colors are compared using ΔE CIE2000.

Results are shown in the plugin’s output panel:

Conflict list

Per-color statistics

Conflict groups

Installation
Clone or download this repository.

Copy the plugin folder into your QGIS profile’s plugin directory:

Windows:
C:\Users\<username>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\

macOS/Linux:
~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/

Restart QGIS and enable the plugin via
Plugins → Manage and Install Plugins → Installed.

Requirements
QGIS 3.x

Python 3.x

colormath library (ships with QGIS or install via pip if standalone)
