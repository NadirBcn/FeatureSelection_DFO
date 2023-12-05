# FeatureSelection_DFO

![huge-fly](https://github.com/NadirBcn/FeatureSelection_DFO/assets/94077842/4390793e-1e1a-4b80-bbaa-57689234a0c8)

## Feature Selection with Dispersive Flies Optimisation : Project Overview

The Dispersive Flies Optimisation (DFO) algorithm was utilized as the primary feature selection method in this research project. DFO is a nature-inspired optimization technique that emulates the pattern of foraging demonstrated by dispersive flies. The optimization algorithm functions by exploring the feature space and identifying the subset of features that contains the most information, therefore improving the overall performance of machine learning models.

The DFO algorithm was implemented using the Python programming language, with the incorporation of the NumPy library to enhance the efficiency of numerical computations. The algorithm requires multiple input parameters, such as the population size (N), the dimensionality of the feature space (D), and the disturbance threshold (delta). To find the best parameters for each dataset, an iterative approach was applied, where different values have been
used for experimentation to reach the best results. In order to assess the effectiveness of DFO, three distinct datasets were employed, each characterized by different levels of feature dimensionality. DFO was utilized to conduct feature selection on each dataset, aiming to identify the most optimal feature subsets that maximize the performance of the chosen machine learning models.





## Code and Ressource used 

* **Packages:** Pandas, Numpy, google.colab, matplotlib, seaborn, sickit-learn.
* **URL Base Algorithm Implementation:** (https://github.com/mohmaj/DFO)


## Data Cleaning
Data cleaning, a crucial step in the preparation of data, specializes in locating and fixing
problems with data quality. This involves processes such as locating and sorting out missing
values, handling outliers, and discovering and fixing data mistakes. By cleaning the data we
make the dataset as accurate and thorough as feasible while eliminating any biases or distortions
that can influence the analysis findings.

### Dataset 1 (Low/Medium Dimension - 74 Features) 
* All the features with more than 20% missing values has been deleted. Which lowered the data from 74 features to 52.
* No duplicates
* Mapping of the target features (Also called dependant variable). Moving this data from multi-class classification to binary (With labels 'Paid' or 'Default')
  
### Dataset 2 (Medium dimension - 378 Features)
* This dataset did not contains any null values or duplicates.
* Contains mostly binary features. 

### Dataset 3 (High dimension - 1559 Features)
* No presence of null values.
* Duplicates were removed.
* Presence of outliers in the data that also have been removed as they can impact the model.
* Presence of a class imbalance similar to the first dataset.


## Data preprocessing 
Data preprocessing is a set of processes done on raw data to resolve problems including missing
values, outliers, inconsistent formats, and noisy or erroneous entries. The purpose is to prepare
the data for analysis by eliminating inconsistencies, reducing noise, and standardizing the
format, allowing for more accurate and useful data insights.

### Dataset 1 
* Encoded categorical features using Label Encoder using LabelEncoder. Scaled the numerical features using StandardScaler.
* Divided the data to three different sets. Training (60% of data), Validation (20% of data) and Test (20% of data).
* Applied an oversampling technique to solve the class imbalance issue present within the dependant variable.
  
### Dataset 2
* Mapping of the target feature from linear to binary. (Two distinct categories)
* Scaled numerical features using StandardScaler. 
* Divided the data into two different sets where the first set being the training and representing 80% of data, test having the remaining 20%.
  
### Dataset 3
* Presence of only numerical features, therefore only a scaling was necessary.
* Splitting the data into three sets, training/validation/test as seen previously.
* Application of an oversampling approach to deal with the class imbalance issue



## Classification models chosen

### Dataset 1 - Decision Tree Classifier 
* Applied decision tree as it can handle a mixture of numerical and categorical data quite well
* The large amount of training sample generated from oversampling (900.000) is handled well by the algorithm
* Reached a performance of 0.95 F1 score after tuning hyperparameters such as the maximum depth or minimum samples. 

### Dataset 2 - k-NN Classifier
* Applied kNN on this dataset as it works well on data with a medium amount of training samples, which is true in this case.
* Iterated the algorithm over a range of k values (The most crucial hyperparameter to choose) to find the one that perform the best.
* Reached a performance of 0.83 F1 Score.
  
### Dataset 3 - Support Vector Machine (SVM)
* Applied SVM on this data as it contained the lowest amount of training samples. (Around 1500)
* Reached a performanced of 0.63 F1 Score after iterating the algorithm with various hyperparameters value such as the 'C' value.


## Implementation of Dispersive Flies Optimisation (DFO) for feature selection

### Parameter tuning 
In order to find the best hyperparameters for this nature-based algorithm, the same approach of iterating over a range of different values has been applied. One of this parameters is the population size 'N'. This parameter can affect DFO performance in choosing the most relevant subset of features. A large population of flies can find better solutions, but can slow down the algorithm, on the opposite side, a smaller population size might fasten the algorithm but may present a less satisfactory solution. Another paramater is the The disturbance threshold 'detla', it is a
critical parameter that influences the exploration and exploitation balance during the optimization process. 

## Results 

## Dataset 1 
![Capture](https://github.com/NadirBcn/FeatureSelection_DFO/assets/94077842/31616b5e-562a-4753-93c5-a821e8a4de2e)

The convergence analysis for the first dataset, comprising 49 features, revealed promising patterns during the feature selection process with DFO, as shown the plots. The line plot highlighting the increase in model performance over iterations showed a consistent upward trend, signifying the algorithm's ability to refine the feature subset. As DFO explored the solution space, the performance metric, such as the F1 score, steadily improved.

In parallel, the line plot illustrating the reduction in the number of selected features exhibited a clear downward trend. This demonstrated DFO's capability to identify and save the most informative features while eliminating irrelevant ones. The process of dimensionality reduction led to a redefined and more efficient feature subset, enhancing the model's interpretability and performance.


## Dataset 2 
![Capture2](https://github.com/NadirBcn/FeatureSelection_DFO/assets/94077842/5dd3d5e3-3262-4236-8932-ab237a77ba4d)

It can be seen that the convergence analysis for the second dataset, containing 364 features, also demonstrated promising outcomes. The line plot displaying the performance improvement over iterations exhibited a sharp upward trajectory, indicating DFO's effectiveness in improving the feature subset. The model performance, measured by an accuracy metric, improved rapidly for the first few iterations then stagnate as DFO iteratively selected relevant features.

Simultaneously, the line plot representing the reduction in the number of selected features showcased a gradual decline. 

DFO successfully spotted irrelevant features, leading to a compact and efficient feature subset. The dimensionality reduction enhanced the model's computational efficiency and interpretability.

## Dataset 3
![Capture3](https://github.com/NadirBcn/FeatureSelection_DFO/assets/94077842/38482839-149a-4b51-a313-176451df10ba)

The figure shows the convergence of the third and final dataset, thus the performance improvement and dimensionality reduction trends with the Dispersive Flies Optimization (DFO) algorithm. The line plot representing the model F1 score displayed a gradual increase from the initial value of 0.66. However, the plot showed rapid fluctuations with intermittent increases and decreases. Despite these fluctuations, the overall trajectory of the line displayed a consistent upward trend, eventually stabilizing around an F1 score of approximately 0.75. This indicates that DFO effectively refined the feature subset, leading to enhanced model performance for this dataset.

Similarly, the line plot depicting the feature count showcased a sharp decline, by rapidly reducing the number of features from 1522 to significantly lower values (Around 260). Furthermore, the plot exhibited slight fluctuations with occasional small bumps, but the overall trend continued downwards. 

DFO demonstrated its proficiency in selecting relevant features 53 and performing dimensionality reduction. As a result, the feature count was streamlined to around 260, significantly reducing the dataset dimensionality while retaining crucial information for model training.











