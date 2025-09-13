from qgis.PyQt.QtWidgets import QMessageBox

class WarningsDialog(QMessageBox):
    def __init__(self, parent =None):
        super().__init__(parent)
        self.setWindowTitle("Graduated Renderer Warning")
        self.setIcon(QMessageBox.Warning)
        self.setText("Warning: Graduated Renderer Selected for Recoloring")
        text = """
The recoloring functionality may not produce optimal results for graduated (continuous) renderers.

Graduated renderers use color gradients to represent continuous data values. When individual colors within such gradients are recolored, the visual continuity and logical progression of the gradient may be disrupted.

This can result in:
- Broken color transitions
- Loss of visual hierarchy
- Reduced interpretability of the data

The conflict analysis remains accurate, but automatic recoloring of graduated symbols should be used with caution.

For more information please refer to the documentation.

Consider manually adjusting the color scheme for graduated renderers instead.
        
        """
        
        self.setInformativeText(text.strip())
        self.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        self.button(QMessageBox.Ok).setText("Continue")
        self.button(QMessageBox.Cancel).setText("Cancel Recoloring")
        self.setDefaultButton(QMessageBox.Ok)



def show_graduated_warning(parent = None):
    result = WarningsDialog(parent).exec_()
    return result == QMessageBox.Ok


