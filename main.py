from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, 
        QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QPushButton, QSizePolicy, 
        QStyleFactory, QTabWidget, QPlainTextEdit,
        QVBoxLayout, QWidget, QFileDialog)
from PyQt5.QtGui import QIcon
from settings.coord_settings import CS
from settings.seg_settings import SS
import os
from coordinates import coordinates
from merge_coordinates import merge_coordinates
from clear_excess_stumps import clear_excess_stumps
import ast

from segmentation_vor import segmentation_vor
from segmentation_ram import segmentation_ram
from segmentation_clear import segmentation_clear
# from seg_after import seg_after
# from orbit_gif import orbit_gif
from predict import predict
from parameters import parameters

def str_to_bool(s):
    if s == "True": 
        return True 
    elif s=="False": 
        return False 

class WidgetGallery(QWidget):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.setWindowIcon(QIcon('logo/logo.png'))

        self.resize(750, 750)

        self.file_path_TRAJ = 'empty'
        self.file_path_SHAPE = 'empty'
        self.file_path_COORD = 'empty'
        self.file_path_SETTINGS = "settings\settings.yaml"

        self.createTopLeftGroupBox()
        self.createRightTabWidget()
        self.createTopRightGroupBox()
        self.createBottomRightGroupBox()
        self.createBottomLeftGroupBox()

        # styleLabel = QLabel("Style:")

        topLayout = QHBoxLayout()
        topLayout.addStretch(1)
        # topLayout.addWidget(styleLabel)

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addWidget(self.topLeftGroupBox, 1, 0)
        mainLayout.addWidget(self.topRightGroupBox, 1, 1)
        mainLayout.addWidget(self.bottomLeftGroupBox, 2, 0)
        mainLayout.addWidget(self.bottomRightGroupBox, 2, 1)

        mainLayout.addWidget(self.rightTabWidget, 1, 3, 2, 1)

        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)

        self.setWindowTitle("pcPCD")
        self.changeStyle('Fusion')

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))

    def advanceProgressBar(self):
        curVal = self.progressBar.value()
        maxVal = self.progressBar.maximum()
        self.progressBar.setValue(curVal + (maxVal - curVal) // 100)

    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("Coordinates")

        self.checkBoxC1 = QCheckBox("Coordinates (int = 7000)")
        self.checkBoxC2 = QCheckBox("Coordinates (int = 5000)")
        self.checkBoxC3 = QCheckBox("Coordinates (int = 1000)")
        self.checkBoxC4 = QCheckBox("Merge Coordinates")
        self.checkBoxC5 = QCheckBox("Clear Excess Stumps")

        self.checkBoxC5.stateChanged.connect(self.auto_file_coords)

        layout = QVBoxLayout()
        layout.addWidget(self.checkBoxC1)
        layout.addWidget(self.checkBoxC2)
        layout.addWidget(self.checkBoxC3)
        layout.addWidget(self.checkBoxC4)
        layout.addWidget(self.checkBoxC5)
        layout.addStretch(1)
        self.topLeftGroupBox.setLayout(layout)
    
    def createBottomLeftGroupBox(self):
        self.bottomLeftGroupBox = QGroupBox("Segmentation and Parameters")

        self.checkBoxS1 = QCheckBox("Segmentation Voronoi")
        self.checkBoxS2 = QCheckBox("Segmentation RAM")
        self.checkBoxS3 = QCheckBox("Segmentation Clear")
        self.checkBoxS4 = QCheckBox("Predict Labels")
        self.checkBoxS5 = QCheckBox("Estimate Parameters")

        self.checkBoxS1.stateChanged.connect(self.seg_file_coords)
        self.checkBoxS2.stateChanged.connect(self.seg_file_coords)
        self.checkBoxS3.stateChanged.connect(self.seg_file_coords)
        self.checkBoxS4.stateChanged.connect(self.seg_file_coords)
        self.checkBoxS5.stateChanged.connect(self.seg_file_coords)

        layout = QVBoxLayout()
        layout.addWidget(self.checkBoxS1)
        layout.addWidget(self.checkBoxS2)
        layout.addWidget(self.checkBoxS3)
        layout.addWidget(self.checkBoxS4)
        layout.addWidget(self.checkBoxS5)
        layout.addStretch(1)
        self.bottomLeftGroupBox.setLayout(layout)


    def createTopRightGroupBox(self):
        self.topRightGroupBox = QGroupBox("Control")

        self.disableWidgetsCheckBox = QCheckBox("Выбрать настройки из файла *.yaml")

        self.button0 = QPushButton('* Файл настроек', self)
        self.button0.clicked.connect(self.open_file_dialog_SETTINGS)
        self.file_path_label0 = QLabel(' ', self)
        self.button0.setDisabled(True)
        self.file_path_label0.setDisabled(True)

        self.startButton = QPushButton("НАЧАТЬ ОБРАБОТКУ")
        self.startButton.clicked.connect(self.start)


        self.labelempty000 = QLabel('')
        self.labelempty001 = QLabel('')

        self.selectFolderBtn = QPushButton('* Папка проекта', self)
        self.selectFolderBtn.clicked.connect(self.selectFolder)
        self.selectedPathLabel = QLabel(' ', self)

        self.button1 = QPushButton('* Файл с данными облака', self)
        self.button1.clicked.connect(self.open_file_dialog_POINTS)
        self.file_path_label1 = QLabel(' ', self)

        self.button2 = QPushButton('* Файл с треком человека', self)
        self.button2.clicked.connect(self.open_file_dialog_TRAJ)
        self.file_path_label2 = QLabel(' ', self)

        self.button3 = QPushButton('Файл с границами участка', self)
        self.button3.clicked.connect(self.open_file_dialog_SHAPE)
        self.file_path_label3 = QLabel('Опционально', self)
    
        self.button4 = QPushButton('* Файл с координатами', self)
        self.button4.clicked.connect(self.open_file_dialog_COORD)
        self.file_path_label4 = QLabel(' ', self)

        self.disableWidgetsCheckBox.toggled.connect(self.rightTabWidget.setDisabled)

        self.disableWidgetsCheckBox.toggled.connect(self.selectFolderBtn.setDisabled)
        self.disableWidgetsCheckBox.toggled.connect(self.selectedPathLabel.setDisabled)

        self.disableWidgetsCheckBox.toggled.connect(self.button0.setEnabled)
        self.disableWidgetsCheckBox.toggled.connect(self.button0.setEnabled)

        self.disableWidgetsCheckBox.toggled.connect(self.button1.setDisabled)
        self.disableWidgetsCheckBox.toggled.connect(self.file_path_label1.setDisabled)

        self.disableWidgetsCheckBox.toggled.connect(self.button1.setDisabled)
        self.disableWidgetsCheckBox.toggled.connect(self.file_path_label1.setDisabled)

        self.disableWidgetsCheckBox.toggled.connect(self.button2.setDisabled)
        self.disableWidgetsCheckBox.toggled.connect(self.file_path_label2.setDisabled)

        self.disableWidgetsCheckBox.toggled.connect(self.button3.setDisabled)
        self.disableWidgetsCheckBox.toggled.connect(self.file_path_label3.setDisabled)

        self.disableWidgetsCheckBox.toggled.connect(self.button4.setDisabled)
        self.disableWidgetsCheckBox.toggled.connect(self.file_path_label4.setDisabled)

        self.button2.hide()
        self.file_path_label2.hide()
    
        self.button4.hide()
        self.file_path_label4.hide()

        layout = QVBoxLayout()
        layout.addWidget(self.disableWidgetsCheckBox)  
        layout.addWidget(self.button0)
        layout.addWidget(self.file_path_label0)
        layout.addWidget(self.selectFolderBtn)
        layout.addWidget(self.selectedPathLabel)
        layout.addWidget(self.button1)
        layout.addWidget(self.file_path_label1)
        layout.addWidget(self.button2)
        layout.addWidget(self.file_path_label2)
        layout.addWidget(self.button3)
        layout.addWidget(self.file_path_label3)
        layout.addWidget(self.button4)
        layout.addWidget(self.file_path_label4)
        layout.addWidget(self.labelempty000)
        layout.addWidget(self.labelempty001)
        layout.addWidget(self.startButton)
        layout.addStretch(1)
        
        self.topRightGroupBox.setLayout(layout)
        
    
    def selectFolder(self):
        self.folderPath = QFileDialog.getExistingDirectory(self, "Выберите папку проекта", "")
        self.selectedPathLabel.setText(self.folderPath)
        self.selectFolderBtn.setText("Папка проекта")

    def open_file_dialog_POINTS(self):
        options = QFileDialog.Options()
        self.file_path_POINTS, _ = QFileDialog.getOpenFileName(self, "Выберите файл с данными облака", "", "Las Files (*.las);;PCD Files (*.pcd)", options=options)
        if self.file_path_POINTS:
            self.file_path_label1.setText(self.file_path_POINTS)
            self.button1.setText("Файл с данными облака")


    def open_file_dialog_SETTINGS(self):
        options = QFileDialog.Options()
        self.file_path_SETTINGS, _ = QFileDialog.getOpenFileName(self, "Выберите файл настроек", "", "(*.yaml);;(*.yml)", options=options)
        if self.file_path_SETTINGS:
            self.file_path_label0.setText(self.file_path_SETTINGS)
            self.button0.setText("Файл настроек")

    def open_file_dialog_TRAJ(self):
        options = QFileDialog.Options()
        self.file_path_TRAJ, _ = QFileDialog.getOpenFileName(self, "Выберите файл с треком человека", "", "Las Files (*.las);;PCD Files (*.pcd)", options=options)
        if self.file_path_TRAJ:
            self.file_path_label2.setText(self.file_path_TRAJ)
            self.button2.setText("Файл с треком человека")
    
    def open_file_dialog_SHAPE(self):
        options = QFileDialog.Options()
        self.file_path_SHAPE, _ = QFileDialog.getOpenFileName(self, "Выберите файл с границами участка", "", "Shapes Files (*.shp)", options=options)
        if self.file_path_SHAPE:
            self.file_path_label3.setText(self.file_path_SHAPE)
    
    def open_file_dialog_COORD(self):
        options = QFileDialog.Options()
        self.file_path_COORD, _ = QFileDialog.getOpenFileName(self, "Выберите файл с координатами", "", "CSV Files (*.csv)", options=options)
        if self.file_path_COORD:
            self.file_path_label4.setText(self.file_path_COORD)
            self.button4.setText("Файл с координатами")
    
    def seg_file_coords(self):
        if self.checkBoxS1.isChecked() or self.checkBoxS2.isChecked() or self.checkBoxS3.isChecked() or self.checkBoxS4.isChecked() or self.checkBoxS5.isChecked():
            self.button4.show()
            self.file_path_label4.show()
        else:
            self.button4.hide()
            self.file_path_label4.hide()

    def add_to_grid(self, layout, info, lbl, obj, i):
        layout.addWidget(info, i, 0, 1, 2)
        layout.addWidget(lbl, i+1, 0)
        layout.addWidget(obj, i+1, 1)
        i = i + 2
        return layout, i
    
    def auto_file_coords(self):
        if self.checkBoxC5.isChecked():
            self.file_path_label4.setText('Файл будет создан автоматически')
        if not self.checkBoxC5.isChecked():
            if self.file_path_COORD != 'empty':
                self.file_path_label4.setText(self.file_path_COORD)
            if self.file_path_COORD == 'empty':
                self.file_path_label4.setText(' ')
    
    def createRightTabWidget(self):
        self.rightTabWidget = QTabWidget()
        self.rightTabWidget.setSizePolicy(QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Ignored)

        tab1 = QWidget()

        self.comboinfo1 = QLabel('Обрезка данных по высоте и границам участка')
        self.labelc1 = QLabel('FLAG_cut_data')
        self.comboBox1 = QComboBox(self)
        self.comboBox1.addItem("True")
        self.comboBox1.addItem("False")
        self.comboBox1.setCurrentIndex(0)

        self.comboinfo2 = QLabel('Выделение подобластей')
        self.labelc2 = QLabel('FLAG_make_cells')
        self.comboBox2 = QComboBox(self)
        self.comboBox2.addItem("True")
        self.comboBox2.addItem("False")
        self.comboBox2.setCurrentIndex(0)

        self.comboinfo3 = QLabel('Выделение пеньков деревьев и вычисление координат')
        self.labelc3 = QLabel('FLAG_make_stumps')
        self.comboBox3 = QComboBox(self)
        self.comboBox3.addItem("True")
        self.comboBox3.addItem("False")
        self.comboBox3.setCurrentIndex(0)

        self.comboinfo4 = QLabel('Метод выделения подобластей')
        self.labelc4 = QLabel('cut_data_method')
        self.comboBox4 = QComboBox(self)
        self.comboBox4.addItem("voronoi_tessellation")
        self.comboBox4.addItem("flood_fill")
        self.comboBox4.setCurrentIndex(0)
        self.comboBox4.currentTextChanged.connect(self.on_combobox_changed)

        self.labelinfo1 = QLabel('Нижняя граница рассматриваемого слоя точек')
        self.label1 = QLabel('LOW')
        self.edit1 = QLineEdit(self)
        self.edit1.setText("0.0")

        self.labelinfo2 = QLabel('Верхняя граница рассматриваемого слоя точек')
        self.label2 = QLabel('UP')
        self.edit2 = QLineEdit(self)
        self.edit2.setText("3.0")

        self.labelinfo3 = QLabel('Сдвиг по Х облака точек')
        self.label3 = QLabel('x_shift')
        self.edit3 = QLineEdit(self)
        self.edit3.setText("0")

        self.labelinfo4 = QLabel('Сдвиг по Y облака точек')
        self.label4 = QLabel('y_shift')
        self.edit4 = QLineEdit(self)
        self.edit4.setText("0")

        self.labelinfo5 = QLabel('Сдвиг по Z облака точек')
        self.label5 = QLabel('z_shift')
        self.edit5 = QLineEdit(self)
        self.edit5.setText("0")

        self.comboinfo5 = QLabel('Метод кластеризации')
        self.labelc5 = QLabel('algo')
        self.comboBox5 = QComboBox(self)
        self.comboBox5.addItem("birch")
        self.comboBox5.addItem("spectral")
        self.comboBox5.addItem("kmeans")
        self.comboBox5.setCurrentIndex(0)

        self.labelinfo6 = QLabel('Количество кластеров при разделение участка на подобласти (voronoi_tessellation)')
        self.label6 = QLabel('n_clusters')
        self.edit6 = QLineEdit(self)
        self.edit6.setText("32")

        self.labelinfo7 = QLabel('Размерность ячейки для выделения ячеек по границам трека человека')
        self.label7 = QLabel('cell_size')
        self.edit7 = QLineEdit(self)
        self.edit7.setText("0.20")

        self.labelinfo8 = QLabel('Лимит по минимальной высоте извлеченных пеньков на первом этапе фильтрации')
        self.label8 = QLabel('height_limit_1')
        self.edit8 = QLineEdit(self)
        self.edit8.setText("1.25")
        
        self.labelinfo9 = QLabel('Лимит по минимальной высоте извлеченных пеньков на втором этапе фильтрации')
        self.label9 = QLabel('height_limit_2')
        self.edit9 = QLineEdit(self)
        self.edit9.setText("1.35")

        self.labelinfo10 = QLabel('Параметр алгоритма DBSCAN по осям XY')
        self.label10 = QLabel('eps_XY')
        self.edit10 = QLineEdit(self)
        self.edit10.setText("0.08")

        self.labelinfo11 = QLabel('Параметр алгоритма DBSCAN по оси Z')
        self.label11 = QLabel('eps_Z')
        self.edit11 = QLineEdit(self)
        self.edit11.setText("0.7")

        tab1hbox = QGridLayout()
        tab1hbox.setContentsMargins(5, 5, 5, 5)

        i = 1
        tab1hbox, i = self.add_to_grid(tab1hbox, self.comboinfo1, self.labelc1, self.comboBox1, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.comboinfo2, self.labelc2, self.comboBox2, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.comboinfo3, self.labelc3, self.comboBox3, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.comboinfo4, self.labelc4, self.comboBox4, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.labelinfo1, self.label1, self.edit1, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.labelinfo2, self.label2, self.edit2, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.labelinfo3, self.label3, self.edit3, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.labelinfo4, self.label4, self.edit4, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.labelinfo5, self.label5, self.edit5, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.comboinfo5, self.labelc5, self.comboBox5, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.labelinfo6, self.label6, self.edit6, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.labelinfo7, self.label7, self.edit7, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.labelinfo8, self.label8, self.edit8, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.labelinfo9, self.label9, self.edit9, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.labelinfo10, self.label10, self.edit10, i)
        tab1hbox, i = self.add_to_grid(tab1hbox, self.labelinfo11, self.label11, self.edit11, i)

        tab1.setLayout(tab1hbox)

        tab2 = QWidget()

        self.labelinfos1 = QLabel('Номер дерева, с которого начнется извлечение')
        self.labels1 = QLabel('first_num')
        self.edits1 = QLineEdit(self)
        self.edits1.setText("0")

        self.labelinfos2 = QLabel('Шаг просмотро по высоте')
        self.labels2 = QLabel('STEP')
        self.edits2 = QLineEdit(self)
        self.edits2.setText("2.5")

        self.labelinfos3 = QLabel('Пороги по высоте, до которых действуют eps_steps и min_pts')
        self.labels3 = QLabel('z_thresholds')
        self.edits3 = QLineEdit(self)
        self.edits3.setText("[0.5, 0.625, 0.695, 0.75, 0.875, 1]")
    
        self.labelinfos4 = QLabel('eps алгоритма DBSCAN, которые дествуют при z_thresholds (0.35 + eps_steps[i])')
        self.labels4 = QLabel('eps_steps')
        self.edits4 = QLineEdit(self)
        self.edits4.setText("[0.01, 0.15, 0.35, 0.5, 0.6, 0.7]")

        self.labelinfos5 = QLabel('minPts алгоритма DBSCAN, которые дествуют при z_thresholds')
        self.labels5 = QLabel('min_pts')
        self.edits5 = QLineEdit(self)
        self.edits5.setText("[50, 50, 50, 50, 45, 40]")

        tab2hbox = QGridLayout()
        tab2hbox.setContentsMargins(5, 5, 5, 5)

        i = 1
        tab2hbox, i = self.add_to_grid(tab2hbox, self.labelinfos1, self.labels1, self.edits1, i)
        tab2hbox, i = self.add_to_grid(tab2hbox, self.labelinfos2, self.labels2, self.edits2, i)
        tab2hbox, i = self.add_to_grid(tab2hbox, self.labelinfos3, self.labels3, self.edits3, i)
        tab2hbox, i = self.add_to_grid(tab2hbox, self.labelinfos4, self.labels4, self.edits4, i)
        tab2hbox, i = self.add_to_grid(tab2hbox, self.labelinfos5, self.labels5, self.edits5, i)

        self.labelempty = QLabel('')
        tab2hbox.addWidget(self.labelempty, i, 0, 1, 2)
        self.labelempty1 = QLabel('')
        tab2hbox.addWidget(self.labelempty1, i+2, 0, 1, 2)
        self.labelempty2 = QLabel('')
        tab2hbox.addWidget(self.labelempty2, i+4, 0, 1, 2)
        self.labelempty3 = QLabel('')
        tab2hbox.addWidget(self.labelempty3, i+6, 0, 1, 2)
        self.labelempty4 = QLabel('')
        tab2hbox.addWidget(self.labelempty4, i+8, 0, 1, 2)

        self.labelempty5 = QLabel('')
        tab2hbox.addWidget(self.labelempty5, i+10, 0, 1, 2)
        self.labelempty6 = QLabel('')
        tab2hbox.addWidget(self.labelempty6, i+12, 0, 1, 2)
        self.labelempty7 = QLabel('')
        tab2hbox.addWidget(self.labelempty7, i+14, 0, 1, 2)
        self.labelempty8 = QLabel('')
        tab2hbox.addWidget(self.labelempty8, i+16, 0, 1, 2)
        self.labelempty9 = QLabel('')
        tab2hbox.addWidget(self.labelempty9, i+18, 0, 1, 2)

     
        tab2.setLayout(tab2hbox)


        self.rightTabWidget.addTab(tab1, "Coordinates Settings")
        self.rightTabWidget.addTab(tab2, "Segmentation Settings")
    
    def on_combobox_changed(self):
        if self.comboBox4.currentText() == "flood_fill":
            self.button2.show()
            self.file_path_label2.show()
        elif self.comboBox4.currentText() == "voronoi_tessellation":
            self.button2.hide()
            self.file_path_label2.hide()

    def createBottomRightGroupBox(self):
        self.bottomRightGroupBox = QGroupBox("INFO")
        layout = QVBoxLayout()
        self.textEdit = QPlainTextEdit()
        self.textEdit.setReadOnly(True)

        layout.addWidget(self.textEdit)
        self.bottomRightGroupBox.setLayout(layout)

    def start(self):
        if self.disableWidgetsCheckBox.isChecked(): 
            cs = CS()
            cs.set(self.file_path_SETTINGS)
            ss = SS()
            ss.set(self.file_path_SETTINGS)
        if not self.disableWidgetsCheckBox.isChecked(): 
            relative_path_points = os.path.relpath(self.file_path_POINTS, self.folderPath)
            if self.file_path_TRAJ == 'empty':
                relative_path_traj = 'empty.file'
            else:
                relative_path_traj = os.path.relpath(self.file_path_TRAJ, self.folderPath)
            if self.file_path_SHAPE == 'empty':
                relative_path_shape = 'empty.file'
            else:
                relative_path_shape = os.path.relpath(self.file_path_SHAPE, self.folderPath)
            if self.file_path_COORD == 'empty':
                relative_path_coord = 'empty.file'
            else:
                relative_path_coord = os.path.relpath(self.file_path_COORD, self.folderPath)

            if self.checkBoxS1.isChecked() or self.checkBoxS2.isChecked() or self.checkBoxS3.isChecked() or self.checkBoxS4.isChecked() or self.checkBoxS5.isChecked():
                save_pth = relative_path_points.partition('.')[0] + "_Clear_Excess.csv"  
                self.file_path_COORD = os.path.join(self.folderPath, save_pth)
                relative_path_coord = os.path.relpath(self.file_path_COORD, self.folderPath)


            cs = CS(FLAG_cut_data = str_to_bool(self.comboBox1.currentText()),
            FLAG_make_cells = str_to_bool(self.comboBox2.currentText()),
            FLAG_make_stumps = str_to_bool(self.comboBox3.currentText()),
            cut_data_method = self.comboBox4.currentText(),
            LOW = float(self.edit1.text()),
            UP = float(self.edit2.text()),
            x_shift = float(self.edit3.text()),
            y_shift = float(self.edit4.text()),
            z_shift = float(self.edit5.text()),
            algo = self.comboBox5.currentText(),
            n_clusters = int(self.edit6.text()),
            cell_size = float(self.edit7.text()),
            height_limit_1 = float(self.edit8.text()),
            height_limit_2 = float(self.edit9.text()),
            eps_XY = float(self.edit10.text()),
            eps_Z = float(self.edit11.text()),
            path_base = self.folderPath,
            fname_points = relative_path_points,
            fname_traj = relative_path_traj,
            fname_shape = relative_path_shape,
            )

            ss = SS(path_base = self.folderPath,
            fname_points = relative_path_points,
            fname_shape = relative_path_shape,
            csv_name_coord = relative_path_coord,
            first_num = int(self.edits1.text()),
            STEP = float(self.edits2.text()),
            z_thresholds = ast.literal_eval(self.edits3.text()),
            eps_steps = ast.literal_eval(self.edits4.text()),
            min_pts = ast.literal_eval(self.edits5.text()),
            )

        print(f'main.py {cs.FLAG_make_cells}')

        if self.checkBoxC1.isChecked():
            coordinates(7000, cs)
            self.textEdit.appendPlainText("Done processing Coordinates(int = 7000)")
        if self.checkBoxC2.isChecked():
            coordinates(5000, cs)
            self.textEdit.appendPlainText("Done processing Coordinates(int = 5000)")
        if self.checkBoxC3.isChecked():
            coordinates(1000, cs)
            self.textEdit.appendPlainText("Done processing Coordinates(int = 1000)")
        if self.checkBoxC4.isChecked():
            merge_coordinates(cs)
            self.textEdit.appendPlainText("Done processing Merge Coordinates")
        if self.checkBoxC5.isChecked():
            clear_excess_stumps(cs)
            self.textEdit.appendPlainText("Done processing Clear Excess Stumps")

        if self.checkBoxS1.isChecked():
            segmentation_vor(ss, make_binding = True)
            self.textEdit.appendPlainText("Done processing Segmentation Voronoi")
        if self.checkBoxS2.isChecked():
            segmentation_ram(ss)
            self.textEdit.appendPlainText("Done processing Segmentation RAM")
        if self.checkBoxS3.isChecked():
            segmentation_clear(ss)
            self.textEdit.appendPlainText("Done processing Segmentation Clear")
        if self.checkBoxS4.isChecked():
            model_name = 'cpl1-1024-rp-s1024-pn2'
            path_file = os.path.join(ss.path_base, ss.step1_folder_name, ss.step2_folder_name, ss.step3_folder_name)
            predict(path_file, model_name)
            self.textEdit.appendPlainText("Done processing Predict Labels")
        if self.checkBoxS5.isChecked():
            path_file = os.path.join(ss.path_base, ss.step1_folder_name, ss.step2_folder_name, ss.step3_folder_name)
            parameters(ss, path_file)
            self.textEdit.appendPlainText("Done processing Estimate Parameters")
        
        self.textEdit.appendPlainText("All steps done")


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    sys.exit(app.exec())