from os import listdir
from os.path import isfile, join, isdir
import pandas as pd
import numpy as np 
 
main_dir = "/home/matteo/Desktop/dax-stock"
final_results_file = "final_summary.xlsx"
dict_results = {}

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
            results = pd.read_excel(join(main_dir3,folder_summary_results))
                    
            dict_results[dir+"-"+dir2] = results
                    # print(np.round(np.array(results["accuracy2"],dtype=float),2)[0])

                

with pd.ExcelWriter(join(main_dir,final_results_file)) as writer:  
    for title,pd_res in dict_results.items():
        pd_res.to_excel(writer,sheet_name=title,index=False)