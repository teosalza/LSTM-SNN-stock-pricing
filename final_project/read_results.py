from os import listdir
from os.path import isfile, join, isdir
import pandas as pd

main_dir = "./final_project"

only_main_dir = [f for f in listdir(main_dir) if isdir(join(main_dir, f))]

for dir in only_main_dir:
    if dir[0] != "_":
        main_dir2 = join(main_dir,dir)
        only_main_dir2 = [f for f in listdir(main_dir2) if isdir(join(main_dir2, f))]

        for dir2 in only_main_dir2:
            main_dir3 =  join(main_dir2,dir2)
            file_list = [f for f in listdir(main_dir3) if isfile(join(main_dir3, f)) and f.split(".")[1] == "xlsx"]

            for file in file_list:
                results = pd.read_excel(join(main_dir3,file))
                print("{}".format(round(float(results["accuracy2"].values),3)))

