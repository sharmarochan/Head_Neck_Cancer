import os
import csv

arr=[]
for root,dirs,files in os.walk('.\dataset'):
    for file in files:
        if file.endswith('.dcm'):
            arr.append(os.path.join(root,file))
            with open('new_csv_file_karishma.csv',"w") as output:
                writer=csv.writer(output)
                for val in arr:
                    writer.writerow([val])
