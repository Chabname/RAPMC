# RAPMC

## Reader of Article for Personalized Medecine for Cancer 

### Kaggle - M2 Bioinformatics - Université de Paris

**Sujet :** [Personalized Medicine: Redefining Cancer Treatment.](https://www.kaggle.com/c/msk-redefining-cancer-treatment)

**Objectif :** Predict the effect of Genetic Variants to enable Personalized Medicine

**Références:**



## Data set

- [test_text](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data?select=test_text.zip)
- [test_variants](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data?select=test_variants.zip)
- [training_text](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data?select=training_text.zipar)
- [training_variants](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data?select=training_variants.zip))

## Installation

### Requierments

Install [**miniconda**](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and [**git**](https://git-scm.com/).

### RAPMC environment

Clone [RAPMC repository](https://github.com/Chabname/RAPMC).

```
% git clone https://github.com/Chabname/RAPMC.git
```

Move in your local repository and run conda with `RAPMC.yml` file.

```
% conda env create -f RAPMC.yml
```

And activate it.

```
% conda activate RAPMC
```

### Modules

- wordcloud
- numpy
- pandas
- seaborn
- matplotlib
- nltk
- gensim
- tensorflow
- keras
- sklearn
- scikit-plot
- jupyter

### Memo For creating env manually

```
$ conda install numpy pandas seaborn matplotlib tensorflow keras nltk gensim jupyter
$ pip install wordcloud scikit-plot sklearn

```
