Great — here’s a clean **template collection** for common data visualizations in Python using **Matplotlib** and **Seaborn**.

You can treat this as a ready-to-paste block in your projects or make it into a **README.md** or `.ipynb` notebook with examples.

---

# 📊 Visualization Templates (Bar, Scatter, Histogram, Boxplot, etc.)

Make sure to first import:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

### 1️⃣ Bar Graph

```python
# Counts by category
sns.countplot(x='column_name', data=df)
plt.title('Count of Categories')
plt.show()

# Bar plot with mean value
sns.barplot(x='category_column', y='numeric_column', data=df)
plt.title('Mean Value per Category')
plt.show()
```

---

### 2️⃣ Scatter Plot

```python
# Simple scatter
sns.scatterplot(x='col_x', y='col_y', data=df)
plt.title('Scatter Plot')
plt.show()

# Scatter with color by category
sns.scatterplot(x='col_x', y='col_y', hue='category_col', data=df)
plt.title('Scatter with Category Hue')
plt.show()
```

---

### 3️⃣ Histogram

```python
# Histogram using pandas
df['numeric_column'].hist(bins=20)
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Histogram using seaborn
sns.histplot(df['numeric_column'], bins=20, kde=True)
plt.title('Histogram with KDE')
plt.show()
```

---

### 4️⃣ Box Plot

```python
# Single numeric column
sns.boxplot(x=df['numeric_column'])
plt.title('Boxplot of Numeric Column')
plt.show()

# Grouped boxplot
sns.boxplot(x='category_column', y='numeric_column', data=df)
plt.title('Boxplot by Category')
plt.show()
```

---

### 5️⃣ Pairplot

```python
# Pairwise relationships between numeric columns
sns.pairplot(df)
plt.show()

# Pairplot with category hue
sns.pairplot(df, hue='category_column')
plt.show()
```

---

### 6️⃣ Heatmap (Correlation Matrix)

```python
# Correlation heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

---

### 7️⃣ Violin Plot (advanced)

```python
sns.violinplot(x='category_column', y='numeric_column', data=df)
plt.title('Violin Plot by Category')
plt.show()
```

---

### 📦 Example of Combined Layout (Multiple Plots Together)

```python
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['numeric_column'], bins=20, kde=True)
plt.title('Histogram')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['numeric_column'])
plt.title('Boxplot')

plt.tight_layout()
plt.show()
```

---

### ⚡ Notes

✅ Always call `plt.show()` at the end when mixing matplotlib and seaborn.
✅ You can customize seaborn’s theme with `sns.set_style('whitegrid')`.
✅ For very large data, consider sampling: `df.sample(1000)`.

---

If you want, I can also prepare:

✅ A full `.py` script
✅ A `.ipynb` notebook with dummy data and plots

Would you like me to set that up? 🚀
