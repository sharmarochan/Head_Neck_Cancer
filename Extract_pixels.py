import pydicom
import pandas as pd
import csv

arr=[]
df_img_path = pd.read_csv('./new_csv_file_karishma.csv',index_col=False)
for i in range(len(df_img_path)):
    try:
        ds = pydicom.read_file(str(df_img_path.iloc[i, 0]))
        arr.append(ds)
        with open('extract_pixel.csv', "w") as output:
           writer = csv.writer(output)
           for val in arr:
               writer.writerow(df_img_path)

    except:
        i+=1
        print('a')