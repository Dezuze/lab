# Experiment 9: retail -> RFM -> entropy/information gain -> ID3.
# Expected input is online_retail.csv with InvoiceNo, InvoiceDate, Quantity,
# UnitPrice and CustomerID. A prepared customers.csv is also accepted.
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

try:
    retail = pd.read_csv('online_retail.csv')
except FileNotFoundError:
    retail = pd.read_csv('customers.csv')

if {'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice', 'CustomerID'}.issubset(retail.columns):
    retail = retail.dropna(subset=['CustomerID', 'InvoiceDate'])
    retail = retail[retail['Quantity'] > 0]
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])
    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']
    snapshot = retail['InvoiceDate'].max() + pd.Timedelta(days=1)
    df = retail.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('Amount', 'sum'),
    ).reset_index()
    # Discretize RFM values for a categorical ID3 tree.
    for column in ['Recency', 'Frequency', 'Monetary']:
        df[f'{column}_Cat'] = pd.qcut(
            df[column].rank(method='first'), 3,
            labels=['Low', 'Medium', 'High'])
    df['Target_Segment'] = np.where(
        (df['Monetary_Cat'] == 'High') &
        (df['Frequency_Cat'] == 'High'), 'HighValue', 'Other')
else:
    df = retail.dropna()

features = ['Recency_Cat', 'Frequency_Cat', 'Monetary_Cat']
features = [feature for feature in features if feature in df.columns]
target = 'Target_Segment'
if not features or target not in df.columns:
    raise ValueError(
        'Use online_retail.csv (retail columns) or customers.csv with '
        'categorical RFM and Target_Segment columns.')

def entropy(values):
    p = values.value_counts(normalize=True)
    return -np.sum(p * np.log2(p))

def information_gain(data, feature):
    total = entropy(data[target])
    weighted = sum((len(g) / len(data)) * entropy(g[target])
                   for _, g in data.groupby(feature))
    return total - weighted

def id3(data, remaining):
    if data[target].nunique() == 1:
        return data[target].iloc[0]
    if not remaining:
        return data[target].mode()[0]
    best = max(remaining, key=lambda f: information_gain(data, f))
    tree = {best: {}}
    for value, group in data.groupby(best):
        tree[best][value] = id3(group, [f for f in remaining if f != best])
    return tree

for f in features:
    print(f, 'Information Gain:', round(information_gain(df, f), 4))
print('ID3 Decision Tree:')
tree = id3(df, features)
pprint(tree, sort_dicts=False)
importance = {f: information_gain(df, f) for f in features}
print('Feature importance (information gain):', importance)

def tree_lines(node, prefix=''):
    lines = []
    if isinstance(node, dict):
        for feature, branches in node.items():
            lines.append(prefix + feature)
            for value, child in branches.items():
                lines.append(prefix + f'|- {value} ->')
                lines.extend(tree_lines(child, prefix + '   '))
    else:
        lines.append(prefix + str(node))
    return lines

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(importance.keys(), importance.values())
axes[0].set_ylabel('Information gain')
axes[0].set_title('ID3 feature importance')
axes[0].tick_params(axis='x', rotation=30)
axes[1].axis('off')
axes[1].text(0, 1, '\n'.join(tree_lines(tree)), va='top', family='monospace')
axes[1].set_title('ID3 tree')
plt.tight_layout()
plt.show()
