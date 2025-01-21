import os
from classes.PCD import PCD
from classes.PCD_UTILS import PCD_UTILS

def process_pcd_files(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.pcd'):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_dir, file)

                try:
                    # Читаем файл .pcd
                    pc = PCD()
                    pc.open(source_path)

                    pc.points = PCD_UTILS.shift(pc.points, 11000, 22400, 147)

                    pc.save(destination_path)
                except Exception as e:
                    print(f"Ошибка обработки файла {source_path}: {e}")

if __name__ == "__main__":
    source_folder = "D:/lidar/data/orion/ex3_1/vor/ram/clear"
    destination_folder = "D:/lidar/data/orion/ex3_1/trees"

    process_pcd_files(source_folder, destination_folder)
