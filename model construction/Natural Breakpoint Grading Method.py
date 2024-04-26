import numpy as np
import pandas as pd
import jenkspy

df = pd.read_excel('data/train-test.xlsx')
y = np.array(df['Salience'])
breaks = jenkspy.jenks_breaks(y, 5)
print(f"breaks: {breaks}")
