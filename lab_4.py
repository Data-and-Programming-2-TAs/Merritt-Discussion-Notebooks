#lab_4.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

def main():
	wine = load_wine()
	df = clean(wine)
	plot_alcohol_color(df)
	plot_2d_kde(df)
	plot_dists(df)

def clean(wine):
	df = pd.DataFrame(wine.data, columns=wine.feature_names)
	df['class'] = wine.target
	color_dict = dict(zip([0,1,2], ['red', 'blue', 'orange']))
	df['color'] = df['class'].map(color_dict)
	translation_dict = dict(zip([0,1,2], wine.target_names))
	df['class'] = df['class'].map(translation_dict)
	return df

def plot_alcohol_color(df):
	plt.scatter(df['alcohol'], df['color_intensity'], alpha=0.5, edgecolors='none', c=df['color'])
	plt.title('Alcohol and Color Intensity in Certain Wines')
	plt.ylabel('Color Intensity')
	plt.xlabel('Alcohol')
	plt.show()
	plt.savefig('plots/Alcohol_vs_Color.png')

def plot_dists(df):
	sns.displot(data = df, x='alcohol', hue='class', kind='kde', alpha = 0.3, fill = True)
	plt.show()
	plt.savefig('plots/distplot_comparison.png')


def plot_2d_kde(df):
	sns.displot(data=df, x='alcohol', y='color_intensity', 
				hue='class', kind='kde', rug = True, alpha=0.5)
	plt.show()
	plt.savefig('plots/2dkde.png')



if __name__ == '__main__':
	main()