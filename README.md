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

<br><br>

## Installation

### Requierements

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

- python 3.8.11
- python-levenshtein
- wordcloud
- numpy
- pandas
- seaborn
- matplotlib
- nltk
- gensim
- tensorflow
- tensorflow_hub
- keras
- sklearn
- scikit-plot
- jupyter
- transformers
- sentencepiece
- pydot
- graphviz

### Creating manually the environment

> ℹ️ **Info**
>
> If using the yaml, file these steps are not necessary
```
$ conda create env -n RAPMC python=3.8
$ conda activate RAPMC
$ conda install numpy pandas seaborn matplotlib tensorflow keras nltk gensim jupyter transformers python-levenshtein
$ pip install wordcloud scikit-plot sklearn tensorflow_hub sentencepiece pydot graphviz

```

<br><br>

## Create a Word2Vec Model

This create a new model which will learn  by running `train_embed.py`

> ⚠️ **Warning!**
>
> Run the script <u>**only** from the project's parent directory</u>:
> 
> `% python src/train_embed.py`

 Options | Description | Default value |
|:-------:|-------------|---------------|
| `-caf` | Input **c**lean **a**rticle **f**ile | `datas/all_data_clean.txt` |
|`-t`, `--type` | Input **t**ype of the model to create | `both` |
| `-ws`, `--winsize` | **w**indow **s**ize of the context for the model | `20` |
| `-e`, `--epoch` | Number of **e**poch for training the model | `20` |
| `-b`, `--batch` | Number of **b**atch for training the model | `10000` |
| `-sw`, `--stopword` | Deleting **s**top**w**ords | `True` |
| `-r`, `--repeat` | **r**epeat the article vector to amplify datas | `2000` |
| `-c`, `--concat` | **c**oncat all the articles before training | `True` |

> ℹ️ **Info**
>
> `--type` takes only thrree values :
> - cbow
> - skipgram
> - both

> **Example**
>
> ```
> $ python src/train_embed.py --type skipgram 
> ```


<br><br>

## Train Word2Vec Model with new articles

This create a choosen model which will learn new aticles by running `new_learn_embed.py`

> ⚠️ **Warning!**
>
> Run the script <u>**only** from the project's parent directory</u>:
> 
> `% python src/train_embed.py`

 Options | Description | Default value |
|:-------:|-------------|---------------|
| `-tc`, `-tc` | Input **c**lean **a**rticle **f**ile | `datas/all_data_clean.txt` |
|`-tm`, `--testclean`| Input **c**lean **a**rticle **f**ile | None |
| `-ws`, `--winsize` | **w**indow **s**ize of the context for the model | `20` |
| `-e`, `--epoch` | Number of **e**poch for training the model | `20` |
| `-b`, `--batch` | Number of **b**atch for training the model | `10000` |
| `-sw`, `--stopword` | Deleting **s**top**w**ords | `True` |
| `-r`, `--repeat` | **r**epeat the article vector to amplify datas | `2000` |
| `-c`, `--concat` | **c**oncat all the articles before training | `True` |


> **Example 1**
>
> ```
> $ python src/new_learn_embed.py -tm datas/skipgram_3284.model
> ```
>
>**Example 2**
>
> ```
> $ python src/new_learn_embed.py -tm datas/cbow_3284.model
> ```
>
>**Example 3**
>
> ```
> $ python src/new_learn_embed.py -tm datas/cbow_A3316_WS20_E20_B10000_R2000_CTrue.model
> ```
>
>**Example 4**
>
> ```
> $ python src/new_learn_embed.py -tm datas/skipgram_A3316_WS20_E15_B10000_R2000_CTrue.model
> ```
>
>**Example 5**
>
> ```
> $ python src/new_learn_embed.py -tm datas/Copy_cbow_A3316_WS20_E20_B10000_R2000_CTrue_full.model -tc datas/stage2_test_clean
> ```
>
>**Example 5**
>
> ```
> $ python src/new_learn_embed.py -tm datas/skipgram_A3316_WS20_E20_B10000_R2000_CTrue.model -tc datas/stage2_test_clean
> 