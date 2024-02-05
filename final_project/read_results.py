from os import listdir
from os.path import isfile, join, isdir
import pandas as pd
import numpy as np 

main_dir = "./final_project"

only_main_dir = [f for f in listdir(main_dir) if isdir(join(main_dir, f))]

for dir in only_main_dir:
    if dir[0] != "_":
        main_dir2 = join(main_dir,dir)
        only_main_dir2 = [f for f in listdir(main_dir2) if isdir(join(main_dir2, f))]

        for dir2 in only_main_dir2:
            main_dir3 =  join(main_dir2,dir2)
            file_list = [f for f in listdir(main_dir3) if isfile(join(main_dir3, f)) and f.split(".")[1] == "xlsx"]

            folder_summary_results = "summary_results.xlsx"
            dataframe_results = pd.DataFrame()
            for file in file_list:
                if "summary" not in file:
                    results = pd.read_excel(join(main_dir3,file))
                    vect_file = file.split("_")
                    
                    #window size

                    results[vect_file[2][0]] = vect_file[2][1:]
                    #hidden size
                    results[vect_file[3][0]] = vect_file[3][1:]
                    #batch size
                    results[vect_file[4][0]] = vect_file[4].split(".")[0][1:]

                    dataframe_results = pd.concat([dataframe_results,results])
                    # print(np.round(np.array(results["accuracy2"],dtype=float),2)[0])

            dataframe_results.to_excel(join(main_dir3,folder_summary_results),index=False)
                

