import sys
from qtpy import QtWidgets
from ui.GUI import GUI

#The main function of the entire application.
def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec())



if __name__ == "__main__":
    main()