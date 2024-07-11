# pcPCD - pretty classes Point Cloud Data

## Описание
Объектом разработки является программа определения координат деревьев и их характеристик с использованием данных LiDAR. 

- Первым этапом при выполнении задачи автоматизированной таксации леса является обнаружение и фиксация местоположения дерева. 
- Вторым этапом - сегментация отдельных деревьев.
- Третьим этапом - расчет параметров и характеристик дерева.

Разработанная программа позволит обнаружить деревья по результатам лазерного сканирования наземного базирования на 
участке лесного массива. **pcPCD** позволяет обрабатывать загружаемые в качестве входных данных 
файлы плотных облаков точек участка леса с целью определения местоположения и количества деревьев на участке. 

Программа предлагает надежное обнаружение стволов деревьев, а методы искусственного интеллекта позволяют с высокой 
точностью обнаружить деревья, которые находятся в паспорте объекта. Позволяет извлечь файлы отдельных деревьев, 
определить их видовой состав, рассчитать таксационные параметры.

## Возможности и особенности
- **Предобработка участка (границы участка)**
- **Интеллектуальное разделение участка без потерь информации**
- **Обнаружение, сегментация стволов деревьев и их координат**
- **Использование нескольких сценариев для лучшего результата**
- **Классификация стволов деревьев с целью удаления лишних объектов**
- **Сегментация участка на отдельные деревья на основе координат**
- **Определение качества сегментации**
- **Расчет таксационных параметров каждого дерева**
- **Определение видового состава**
- **Улучшение сегментации после определения пород деревьев**
- **Формирование множества отчетов об участке**


## Демонстрация
### Исходное облако и его предобработка
<p align="center">
  <img src="about\image7.png">
</p>

### Разделение на основе траектории движения сканера
cut_data_method: "flood_fill"
<p align="center">
  <img src="about\image9.png">
</p>

### Интеллектуальное разделение участка (анализ плотности и диаграмма Вороного)
cut_data_method: "voronoi_tessellation" #рекомендуется
<p align="center">
  <img src="about\image12.png">
</p>

<p align="center">
  <img src="about\image15.png">
</p>

### Обнаружение, сегментация стволов деревьев
<p align="center">
  <img src="about\image18.png">
</p>

### Удаление шумов
<p align="center">
  <img src="about\image20.png">
</p>

### Этапы поиска координат на примере
<p align="center">
   <img src="about\1.png">
   <img src="about\2.png">
   <img src="about\3.png">
   <img src="about\4.png">
   <img src="about\5.png">
   <img src="about\6.png">
   <img src="about\7.png">
</p>

### Визуализация результатов 
**Исходное облако**
![gif1](about/image66_1.gif)
**Срез нижней части стволов**
![gif2](about/image66_3.gif)
**Сегментация стволов**
![gif3](about/image66_4.gif)
**Сегментация деревьев**
![gif4](about/image66_2.gif)
## Связанные исследования
Подробные обзоры представлен в работах:

**[Grishin, I. A. Procedure for locating trees and estimating diameters using LiDAR data / I. A. Grishin, V. I. Terekhov // 2023 5th International Youth Conference on Radio Electronics, Electrical and Power Engineering (REEPE). – IEEE, 2023. – Т. 5. – С. 1-5.](https://ieeexplore.ieee.org/document/10086843)**

**[An Efficient Technique for Determining Tree Coordinates using LiDAR Data via Deep Learning / I. A. Grishin, B. S. Goryachkin, V. I. Terekhov,  S. I. Chumachenko // 2024 6th International Youth Conference on Radio Electronics, Electrical and Power Engineering (REEPE). – IEEE, 2024. – С. 1-6.](https://ieeexplore.ieee.org/abstract/document/10479853)**

**[Individual Tree Segmentation Quality Evaluation Using Deep Learning Models LiDAR Based / I. A. Grishin, T. Y. Krutov, A. I. Kanev, V. I. Terekhov // Optical Memory and Neural Networks. – 2023. – Т. 32. – №. Suppl 2. – С. S270-S276.](https://link.springer.com/article/10.3103/S1060992X23060061)**

**[Подход к автоматической оценке таксационных параметров деревьев с помощью данных LiDAR / С. И. Чумаченко, В. И. Терехов, Е. М. Митрофанов, И. А. Гришин // Динамика сложных систем - XXI век. – 2022. – Т. 16, № 4. – С. 63-73. – DOI 10.18127/j19997493-202204-06. – EDN XDQNDC.](https://www.elibrary.ru/item.asp?id=49989358)**

**[Классификация пород деревьев с использованием моделей глубокого обучения на основе MLP / Ч. Чжан, Е. К. Сахарова, И. А. Гришин [и др.] // Искусственный интеллект в автоматизированных системах управления и обработки данных : Сборник статей II Всероссийской научной конференции: в 5 томах, Москва, 27–28 апреля 2023 года. – Москва: КДУ, Добросвет, 2023. – С. 377-383. – EDN AXDOLZ.](https://www.elibrary.ru/item.asp?id=60234479)**

**[Гришин, И. А. Этапы обработки данных LiDAR лесных массивов /  И. А. Гришин, В.И. Терехов // Искусственный интеллект в автоматизированных системах управления и обработки данных : Сборник статей II Всероссийской научной конференции: в 5 томах, Москва, 27–28 апреля 2023 года. – Москва: КДУ, Добросвет, 2023. – С. 377-383. – EDN AXDOLZ.](https://bookonlime.ru/product-pdf/iiasu23-iskusstvennyy-intellekt-v-avtomatizirovannyh-sistemah-upravleniya-i-obrabotki-1)**

**[МЭС многоклассовой классификации видов деревьев по лидарным данным / И. А. Гришин, А. Г. Базанова, А. Г. Нищук [и др.] // Мивар'24. – Москва : Издательский Дом "Инфра-М", 2024. – С. 279-286.](https://znanium.ru/read?id=448902&pagenum=279)**



## Презентация
Обзор исследования вы можете посмотреть в **[презентации](about/Presentation.pdf)**:

[![Presentation](about/pres.png)](about/Present ation.pdf)

## Установка и использование

1. **Установка**:
    ```bash
   conda create --name pcPCDenv379 python=3.7.9
   conda activate pcPCDenv379
   pip install -r requirements_cuda.txt
   pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
    ```
2. **Запуск**
- С графическим интерфейсом:
    ```bash
    python main.py
    ```
  1. **Выберите файл settings.yaml в папке проекта (пример в test_data\settings.yaml)**
  2. **Поставьте галочки по очереди на желаемых этапах**
  3. **Запустите обработку**

- Из терминала:
  ```bash
  python pipeline_coord.py
  python pipeline_seg.py
   ```
  1. **Укажите путь до файла settings.yaml**
  2. **Возможен запуск всех этапов по отдельности с указанием каталогов в блоке**
  ```bash
  if __name__ == "__main__" :
    ss = SS()
    yml_path = "settings\settings.yaml"
    ss.set(yml_path)
    path_file = 'path\to\catalog'
    func(ss)
   ```

