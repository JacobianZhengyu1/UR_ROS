from pathlib import Path
import os

from _GcodeParser import gcode

if __name__ == '__main__':

    gcodes_folder_path = "./gcodes"
    gcodes_csv_folder_path = './gcodes_csv'
    input_folder_path_pathlib = Path(gcodes_folder_path)
    output_folder_path_pathlib = Path(gcodes_csv_folder_path)

    gcodes_files_path = list(input_folder_path_pathlib.glob("*.gcode"))

    specified_files = ["162-werer-ascii"]

    for gcode_file_path in gcodes_files_path:
        if specified_files == []:
            pass
        else:
            if gcode_file_path.stem not in specified_files:
                continue

        print(gcode_file_path.name)
        gcode_file = gcode(str(gcode_file_path.parent),
                           str(gcode_file_path.name),
                           save=False
                           )

        print(gcode_file.printing_stats)
        print(gcode_file.traveling_stats)

        gcode_file.save_instance(path=gcodes_csv_folder_path)

