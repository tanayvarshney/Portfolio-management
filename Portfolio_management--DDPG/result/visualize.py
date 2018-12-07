import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./results.csv")
df.Date = df.Date.astype('O')
plt.plot( 'history1', data=df, marker='', color='blue', linewidth=1, label= "Without Sentiment")

plt.plot( 'history4', data=df, marker='', color='red', linewidth=1, label= "With Sentiment")
plt.legend()
labels = ["2015-01-05","2015-08-10","2016-03-12","2016-10-28","2017-05-31","2017-12-29"]
label = []
for date in df.Date:
    if date in labels:
        label.append(date)
    else:
        label.append("")
# You can specify a rotation for the tick labels in degrees or with keywords.
x = range(len(df.Date.tolist()))
print(len(label))
print(label)
plt.xticks(x, label)
# Pad margins so that markers don't get clipp
#plt.xticks([1,2,3,4,5,6],["2015-01-05","2015-08-10","2016-03-12","2016-10-28","2017-05-31","2017-12-29"])
plt.xlabel("Days (Starting from Jan 2015)")
plt.ylabel("Asset Value")
plt.show()
