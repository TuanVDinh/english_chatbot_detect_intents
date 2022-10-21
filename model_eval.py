import pandas as pd
from chat import get_response

df = pd.read_csv('model_eval.csv')
length = len(df["BookRestaurant"].values)
count = 0

for col in df.columns:
    li1 = []
    if col != "sentence":
        for i in range(length):
            if df[col].values[i] == 1:
                li1.append(df["sentence"].values[i])
        for i in range(len(li1)):
            s = get_response(li1[i])
            if s == f'Your intent is "{col}"':
                count += 1

acc = (count/length)*100
print("Accuracy: ", acc)