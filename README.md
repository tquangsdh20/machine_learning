## Part 2: Decision Tree

```python
import pandas as pd
import math
```


```python
dat = pd.read_csv('./dataset/DecisionTree', sep='\t')
```


```python
dat.head()
```




<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Ownership</th>
      <th>Travel Cost</th>
      <th>Income</th>
      <th>Mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>0</td>
      <td>Cheap</td>
      <td>Low</td>
      <td>Bus</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>1</td>
      <td>Cheap</td>
      <td>Medium</td>
      <td>Bus</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Female</td>
      <td>0</td>
      <td>Cheap</td>
      <td>Low</td>
      <td>Bus</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>1</td>
      <td>Cheap</td>
      <td>Medium</td>
      <td>Bus</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>1</td>
      <td>Expensive</td>
      <td>High</td>
      <td>Car</td>
    </tr>
  </tbody>
</table>
</div>



### Caculation for Entropy of special column - Mode


```python
# Dem so truong hop
items = pd.unique(dat['Mode'])
n_total = dat['Mode'].count()
p = dict()
for item in items:
    n_i = dat['Mode'].loc[dat['Mode'] == item].count()
    p[item] = n_i/n_total

print(p)
```

    {'Bus': 0.4, 'Car': 0.3, 'Train': 0.3}
    


```python
# Entropy(S)
entropy_s = 0
for item in p.keys():
    pi = p[item]
    ent_i = -pi*math.log2(pi)
    entropy_s += ent_i
print(entropy_s)
```

    1.5709505944546684
    

### Caculation for Entropy of Gender with Male value


```python
# Lua ra toan bo male
gender = pd.unique(dat['Gender'])
males = dat.loc[dat['Gender'] == gender[0]]
males
```




<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Ownership</th>
      <th>Travel Cost</th>
      <th>Income</th>
      <th>Mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>0</td>
      <td>Cheap</td>
      <td>Low</td>
      <td>Bus</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>1</td>
      <td>Cheap</td>
      <td>Medium</td>
      <td>Bus</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>1</td>
      <td>Cheap</td>
      <td>Medium</td>
      <td>Bus</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Male</td>
      <td>2</td>
      <td>Expensive</td>
      <td>Medium</td>
      <td>Car</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Male</td>
      <td>0</td>
      <td>Standard</td>
      <td>Medium</td>
      <td>Train</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dem so truong hop
items_male = pd.unique(males['Mode'])
n_total_male = males['Mode'].count()
p_male = dict()
for item in items_male:
    n_i = males['Mode'].loc[males['Mode'] == item].count()
    p_male[item] = n_i/n_total_male
print(p_male)
# Entropy(S)
entropy_male = 0
for item in p_male.keys():
    pi = p_male[item]
    ent_i = -pi*math.log2(pi)
    entropy_male += ent_i
print(f'Entropy(gender=\'Male\') = {entropy_male:.3}')
```

    {'Bus': 0.6, 'Car': 0.2, 'Train': 0.2}
    Entropy(gender='Male') = 1.37
    

### Caculation for Entropy of Gender with Female value


```python
# Lua ra toan bo male
gender = pd.unique(dat['Gender'])
females = dat.loc[dat['Gender'] == gender[1]]
females
```




<div align="center">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Ownership</th>
      <th>Travel Cost</th>
      <th>Income</th>
      <th>Mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>Female</td>
      <td>0</td>
      <td>Cheap</td>
      <td>Low</td>
      <td>Bus</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>1</td>
      <td>Expensive</td>
      <td>High</td>
      <td>Car</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Female</td>
      <td>2</td>
      <td>Expensive</td>
      <td>High</td>
      <td>Car</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Female</td>
      <td>1</td>
      <td>Cheap</td>
      <td>Medium</td>
      <td>Train</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Female</td>
      <td>1</td>
      <td>Standard</td>
      <td>Medium</td>
      <td>Train</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dem so truong hop
items_female = pd.unique(females['Mode'])
n_total_female = females['Mode'].count()
p_female = dict()
for item in items_female:
    n_i = females['Mode'].loc[females['Mode'] == item].count()
    p_female[item] = n_i/n_total_female
print(p_female)
# Entropy(S)
entropy_female = 0
for item in p_female.keys():
    pi = p_female[item]
    ent_i = -pi*math.log2(pi)
    entropy_female += ent_i
print(f'Entropy(gender=\'Female\') = {entropy_female:.3}')
```

    {'Bus': 0.2, 'Car': 0.4, 'Train': 0.4}
    Entropy(gender='Female') = 1.52
    

### Entropy(Gender)


```python
n_gender = dat['Gender'].count()
n_male = dat['Gender'].loc[dat['Gender'] == 'Male'].count()
p_male = n_male/n_gender
p_female = 1-p_male
entropy_gender = p_male*entropy_male + p_female*entropy_female
print(entropy_gender)
```

    1.4464393446710155
    

### Gain(S,Gender)


```python
## Gain(S,Gender)
gain = entropy_s - entropy_gender
print(f'Gain(S,Gender) = {gain:.3}')
```

    Gain(S,Gender) = 0.125
    

### Build *get_gain()* function


```python
from source.Gain import get_gain
```


```python
a = get_gain(dat,'Mode','Gender')
print(f'Gain(S,Gender) = {a:.3}')
```

    Gain(S,Gender) = 0.125
    


```python
# Get all the names of the features
features = dat.drop('Mode',axis=1).keys()
print(features)
```

    Index(['Gender', 'Ownership', 'Travel Cost', 'Income'], dtype='object')
    


```python
gain = dict()
fmax = ''
gmax = -1
for feature in features:
    gain[feature] = get_gain(dat,'Mode',feature)
    if gmax < gain[feature]: 
        gmax = gain[feature]
        fmax = feature 
    print(f'Gain(S,{feature}) = {gain[feature]:.3}')
print(f'The most meaning feature is "{fmax}"')
```

    Gain(S,Gender) = 0.125
    Gain(S,Ownership) = 0.534
    Gain(S,Travel Cost) = 1.21
    Gain(S,Income) = 0.695
    The most meaning feature is "Travel Cost"
    
