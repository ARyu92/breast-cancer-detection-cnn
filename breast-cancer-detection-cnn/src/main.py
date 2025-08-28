import sys
from qtpy import QtWidgets
from ui.GUI import GUI





def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec())



if __name__ == "__main__":
    main()