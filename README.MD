# Restaurant Comparison using Aspect-Based Sentiment Analysis and Topic Clustering using BERT

Often times, a review for a restaurant cannot capture the true semantic message behind the reviewer's words. While a rating of 3/5 might mean that the person found the place rather mediocre, the details in the review they write could provide a greater understanding of the restaurant, its service, ambience, food quality and price. 

In the example:
```
Chinatown definitely has better quality with cheaper prices.
```
The review might give us a 1/5 or 2/5, but the aspect analysis gives us sentiment analysis specific to aspects:
```
[{'term': 'quality', 'polarity': 'positive'}, {'term': 'prices', 'polarity': 'positive'}]
```

This restaurant comparison tool aims to look beyond a general review rating and automate the sentiment analysis of aspects that a reviewer speaks about. The aspects are then rated into one of the following categories, with their subsequent positive/total rating: `food, ambience, price, service` The categories were determined from the list of unique labelled categories in the [SemEval Restaurant 14, 15 and 16 dataset](#datasets).


The project consists of <strong>4 key stages</strong>:

1. Extracting resturant reviews from Tripadvisor.

2. Running ABSA tasks (Aspect Term Extraction + Aspect Term Sentiment Classification / Joint)

3. Clustering aspects into aspect categories/topics.
4. Create scoring metric per category for different restaurants

## Table of Contents  
---
1. [Datasets](#datasets) 

2. [Requirements](#requirements)  
3. [Hyperparameter Optimization](#hyperparameter-optimization) 

4. [Training Models](#training)  
5. [Solution](#solution)  
6. [Testing with a trained model and a TripAdvisor Post](#testing)  
7. [Evaluation](#evaluation)  

## Datasets
---
<strong>TripAdvisor:</strong> The initial dataset used to gather the TripAdvisor restaurant reviews has been scraped from TripAdvisor pages. The scraper goes through a set number of pages and scrapes 15 reviews per page. The review creation should be done before to save time, or it can be performed live to demonstrate real-time restaurant comparison with new restaurant URLs. It can be found at `data\trip_advisor_reviews.py`

<strong>ABSA:</strong> The dataset used to train the ABSA model(s) was a reformatted version of the SemEval Restaurant [2014](https://alt.qcri.org/semeval2014/task4/), [2015](https://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools) and [2016](https://alt.qcri.org/semeval2016/task5/) data.

## Requirements
---
The requirements can be found in the `requirements.txt` file. 

## Hyperparameter Optimization
---
While the memory allocation space on the GPUs available to use meant that we were limited in the amount of testing we could do, we were able to alter the learning rate to record F1, precision, recall and accuracy scores.

The statistics of InstructABSA's hyperparameters are below:
```
Model: Tk-Instruct-base-def-pos 2 , 
GPU: 1xNvidia GeForce RTX 3070, 
Train Batch Size for ATE and Joint were set to 2,
Gradient Accumulation Steps:2, 
Initial learning rate: 5e-5,
Number of Epochs: 4
```

## Training
---
Checkpoints for the InstructABSA ATE and Joint models can be found here:
https://drive.google.com/file/d/143JO87-eoOrjkDYBgT5noHw_IPmf2kfe/view?usp=sharing

To train InstructABSA yourself, the `models/InstructABSA/InstructABSA` folder holds training scripts for both the ATE and Joint tasks that can be used in the CLI.

The checkpoints for the Sentiment Attention NN can be found in the `models/Checkpoints` folder. 

## Solution
---
### Model Selection and Fine-tuning

There are three key definitions involved in the following descriptions:

1. ATE: Aspect Term Extraction
2. ATSC: Aspect Term Sentiment Classification
3. Joint: ATE + ATSC simultaneously. 

For this project, two different approaches towards the problem were explored.

The models used for the ATE and Joint task was InstructABSA, covered in [this paper](https://arxiv.org/pdf/2302.08624.pdf), and the model used for the ATSC task was a segmentation attention based LSTM model covered in [this paper](https://ojs.aaai.org/index.php/AAAI/article/view/12020/11879).

InstructABSA's code can be found here: https://github.com/kevinscaria/InstructABSA
Learning Latent Opinions for Aspect-Level Sentiment Classification's official code can be found here: https://github.com/berlino/SA-Sent

The project adapts the models above.

#### Approach 1: ATE + ATSC

The first approach involved first performing the ATE task to extract a list of polarities from a sentence. The list of polarities were then passed into a dedicated ATSC model that classifies the aspect based on its sentiment. 

#### Approach 2: Joint

The second approach utilises the combination of ATE and ATSC, allowing for the task to be done simultaneously. 

With both the approaches above, the list of aspects were grouped into the following pre-determined restaurant aspect-related categories: `food, ambience, price, service`.

## Testing

To begin, you may either use the example reviews CSV that has been scraped found at `data/reviews.csv`, or you may use the `data/trip_advisor_reviews.py` to scrape custom restaurant review data.

After doing so, use the `notebooks\InstructATE_SA_Sent.ipynb` notebook to perform ATE + ATSC, or the `notebooks\InstructJoint.ipynb` notebook to perform the Joint task on a given review.

The output for a given scraped TripAdvisor restaurant review should look like the following:

```
Analysis:
--------
Food: 33/42
Ambience: 7/18
Price: 24/25
Service: 3/8
```

## Evaluation

InstructABSA can be evaluated using F1, Recall and Precision Scores, as per the paper linked above (that does not use accuracy in their evaluation steps). Accuracy can also be used, which was provided as an optional add-on.

The scripts for evaluating both InstructABSA and the SA model are also available in their respective folders in the `models` directory.





