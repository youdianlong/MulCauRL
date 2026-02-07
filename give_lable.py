import os
import pandas as pd

input_dir = r"/data1/zyc/result_data/cptac_lung/pt_files"  
records = []
for filename in os.listdir(input_dir):
    if filename.endswith(".pt"):
        slide_id_raw = os.path.splitext(filename)[0] 
        case_id = slide_id_raw[:12]                  
        slide_id = slide_id_raw        
        records.append({"case_id": case_id, "slide_id": slide_id})

df = pd.DataFrame(records)
output_dir='/data1/zyc/result_data/cptac_lung/'

output_csv = os.path.join(output_dir, "case_slide_ids_Lung.csv")
df.to_csv(output_csv, index=False)


import pandas as pd

file1 = "/data1/zyc/result_data/cptac_lung/case_slide_ids_Lung.csv"         
file2 = "/data1/zyc/result_data/cptac_lung/process_list_autogen.csv"        
file3 = "/data1/zyc/result_data/cptac_lung/process_list_autogen_luad.csv"    
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
slides2 = df2['slide_id'].astype(str).tolist()
slides3 = df3['slide_id'].astype(str).tolist()
def assign_label(slide_id):
    sid = str(slide_id)
    for s in slides2:
        if sid in s:  
            return "subtype_1"
    
    for s in slides3:
        if sid in s:
            return "subtype_2"
    
    return "unknown"

df1['label'] = df1['slide_id'].apply(assign_label)
output_file = "/data1/zyc/result_data/cptac_lung/case_slide_ids_Lung_labeled.csv"
df1.to_csv(output_file, index=False)
