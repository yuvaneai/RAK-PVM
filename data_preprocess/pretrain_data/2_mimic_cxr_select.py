import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df_all_matched = pd.read_csv(os.path.join(BASE_DIR, "./result/1_all_matched.csv"))
df_master = pd.read_csv(os.path.join(BASE_DIR, "./result/0_master.csv"))
df_all_matched.rename(columns={'img_path': 'Path'}, inplace=True)
master_select= pd.merge(df_all_matched, df_master, on= ['Path'], how='inner')
master_select.drop(columns=['txt_path'], inplace=True)
master_select.to_csv(os.path.join(BASE_DIR, "./result/2_master_select.csv"), index=False)
print()

df_train = pd.read_csv(os.path.join(BASE_DIR, "./result/0_train.csv"))
train_select= pd.merge(df_all_matched, df_train, on= ['Path'], how='inner')
train_select.drop(columns=['txt_path'], inplace=True)
train_select.to_csv(os.path.join(BASE_DIR, "./result/2_train_select.csv"), index=False)
print()

df_test = pd.read_csv(os.path.join(BASE_DIR, "./result/0_test.csv"))
test_select= pd.merge(df_all_matched, df_test, on= ['Path'], how='inner')
test_select.drop(columns=['txt_path'], inplace=True)
test_select.to_csv(os.path.join(BASE_DIR, "./result/2_test_select.csv"), index=False)
print()


