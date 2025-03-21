import pandas as pd

file_name = ['test_data.csv', 'train_data.csv', 'valid_data.csv']
for file in file_name:
    print(f'File: {file}')
    df = pd.read_csv(f'data/text/{file}')
    count = df['Answer'].value_counts()
    print(count)
    print(sum(count.values))
    print()