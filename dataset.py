import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import python_calamine
# import multiprocessing as mp
import time

from concurrent.futures import ThreadPoolExecutor


def read_and_change_labels(singel_file):
    
    file = singel_file
    root = "./excels"

    # print(f"start processing of: {file}")    

    df = pd.read_excel(file, engine="calamine", usecols=['ردیف پرونده', 'کد طبقه اول', 'کد طبقه دوم', 'کد طبقه سوم'], dtype=str)
        
    df['im_path'] = root+ "/"+ df['ردیف پرونده'] + ".jpg"   # image add

    df['label code'] = df['کد طبقه اول'].astype(str) + '#' + df['کد طبقه دوم'].astype(str) + '#' + df['کد طبقه سوم'].astype(str)

    df.drop(columns=['ردیف پرونده', 'کد طبقه اول', 'کد طبقه دوم', 'کد طبقه سوم'], inplace=True)

    # print(f"{file} is completed")

    return df

def save_csv(name, df):
    start = time.time()
    df.to_csv(name, encoding='utf-8-sig', index=False)
    print(f"{name} saved in {time.time() - start:.2f} seconds.")



def managing_dataset(root, exel_paths):
    all_data_frames = []
    print(exel_paths)

    print("start threading")
    with ThreadPoolExecutor(max_workers=8) as executor:
        all_data_frames = list(executor.map(read_and_change_labels, exel_paths))

    # print("------------------")
    # print("ended threading")
    # print("start merging")

    merged_df = pd.concat(all_data_frames, ignore_index=True)

    unique_labels = {label_code: idx for idx, label_code in enumerate(merged_df['label code'].unique())} 
    merged_df['label'] = merged_df['label code'].map(unique_labels)

    seed = 30

    # print("--------------")
    # print("start spliting")
    train_df, temp_df = train_test_split(merged_df, test_size=0.2, random_state=seed, shuffle=True)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, shuffle=True)

    # print("--------------")
    # print("start to csv")   

    train_df.to_csv('train.csv', encoding='utf-8-sig', index=False)
    test_df.to_csv('test.csv', encoding='utf-8-sig', index=False)
    valid_df.to_csv('validation.csv', encoding='utf-8-sig', index=False)


if __name__ == "__main__":
    t1 = time.time()
    root = "./excels"
    exel_paths = list(map(os.path.normpath, list(map(str, Path(root).rglob('*.xlsx')))))
    managing_dataset(root, exel_paths)
    t2 = time.time()
    print(f"execute time: {t2 - t1}")
