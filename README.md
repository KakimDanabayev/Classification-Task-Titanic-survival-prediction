## Titanic-Machine-Learning-Classification-Task-

- In this project, I used Machine Learning models to predict passengers' survival on Titanic. 
This project is about the Titanic disaster of 1912. The main goal of this notebook is to try to present a complete approach to modeling problems, that goes from Exploratory Data Analysis to applying Supervised learning techniques to our data. This notebook's content is mainly directed to data scientists, data science students, or people interested in how these techniques can be applied to data.

![загрузка (1)](https://user-images.githubusercontent.com/127029668/232261189-14e750de-b519-4508-9594-c8c14318aa03.jpg)


- The Titanic was launched on May 31, 1911. As it prepared to embark on its maiden voyage, the Titanic was one of the largest and most opulent ships in the world. It had a gross registered tonnage (i.e., carrying capacity) of 46,328 tons, and when fully laden the ship displaced (weighed) more than 52,000 tons. The Titanic was approximately 882.5 feet (269 meters) long and about 92.5 feet (28.2 meters) wide at its widest point. The Titanic was 882 feet 9 inches (269.06 m) long with a maximum breadth of 92 feet 6 inches (28.19 m). Titanic's total height, measured from the base of the keel to the top of the bridge, was 104 feet (32 m). She measured 46,329 GRT and 21,831 NRT and with a draught of 34 feet 7 inches (10.54 m), she displaced 52,310 tons. All three of the Olympic-class ships had ten decks (excluding the top of the officers' quarters), eight of which were for passenger use. From top to bottom, the decks were:

- The Boat Deck, on which the lifeboats were housed. It was from here during the early hours of 15 April 1912 that Titanic's lifeboats were lowered into the North Atlantic. The bridge and wheelhouse were at the forward end, in front of the captain's and officers' quarters. The bridge stood 8 feet (2.4 m) above the deck, extending out to either side so that the ship could be controlled while docking. The wheelhouse stood within the bridge. The entrance to the First Class Grand Staircase and gymnasium were located midship along with the raised roof of the First Class lounge, while at the rear of the deck were the roof of the First Class smoke room and the relatively modest Second Class entrance. The wood-covered deck was divided into four segregated promenades: for officers, First Class passengers, engineers, and Second Class passengers respectively. Lifeboats lined the side of the deck except in the First Class area, where there was a gap so that the view would not be spoiled.

Source: Wikipedia & Britannica (https://en.wikipedia.org/wiki/Titanic)/(https://www.britannica.com/topic/Titanic)

## Information about the dataset

- Titanic Data - Contains demographics and passenger information from 891 of the 2224 passengers and crew on board the Titanic. The Titanic data contains a mix of textual, Boolean, continuous, and categorical variables. It exhibits interesting characteristics such as missing values, outliers, and text variables ripe for text mining–a rich database that will allow us to demonstrate data transformations. We will use the classic Titanic dataset. The data consists of demographic and traveling information for1,309 of the Titanic passengers, and the goal is to predict the survival of these passengers. The Titanic dataset is also the subject of the introductory competition on Kaggle.com

- We have 1,309 records and 14 attributes, three of which we will discard. The homedest attribute has too few existing values, the boat attribute is only present for passengers who have survived, and the body attribute is only for passengers who have not survived. We will discard these three columns later on while using the data schema.

## Dataset Attribution
Features After getting a better perception of the different aspects of the dataset, I started exploring the features and the part they played in the survival or demise of a traveler.

- Survived - the first feature reported if a traveler lived or died. A comparison revealed that more than 60% of the passengers had died.
- Pclass - this feature renders the passenger division. The tourists could opt from three distinct sections, namely class-1, class-2, and class-3. The third class had the highest number of commuters, followed by class-2 and class-1. The number of tourists in the third class was more than the number of passengers in the first and second classes combined. The survival chances of a class-1 traveler were higher than a of class-2 and class-3 travelers.
- Sex - approximately 65% of the tourists were male while the remaining 35% were female. Nonetheless, the percentage of female survivors was higher than the number of male survivors. More than 80% of male commuters died, as compared to around 70% of female commuters.
- Age - the youngest traveler onboard was aged around two months and the oldest traveler was 80 years. The average age of tourists onboard was just under 30 years. Clearly, a larger fraction of children under 10 survived than died. or every other age group, the number of casualties was higher than the number of survivors. More than 140 people within the age group 20 and 30 were dead as compared to just around 80 people of the same age range sustained.
- SibSp- sibSp is the number of siblings or spouses of a person onboard. A maximum of 8 siblings and spouses traveled along with one of the travelers. More than 90% of people traveled alone or with one of their siblings or spouse. The chances of survival dropped drastically if someone traveled with more than 2 siblings or a spouse.
- Parch- similar to the SibSp, this feature contained the number of parents or children each passenger was touring with. A maximum of 9 parents/children traveled along with one of the travelers.
- Pclass: A proxy for socio-economic status (SES) 1st = Upper 2nd = Middle 3rd = Lower
- Fare - by splitting the fare amount into four categories, it was obvious that there was a strong association between the charge and the survival. The higher a tourist paid, the higher would be his chances to survive.
- Embarked - embarked implies where the traveler mounted from. There are three possible values for Embark — Southampton, Cherbourg, and Queenstown. More than 70% of the people boarded from Southampton. Just under 20% boarded from Cherbourg and the rest boarded from Queenstown. People who boarded from Cherbourg had a higher chance of survival than people who boarded from Southampton or Queenstown.
- Cabin - is the cabin number of the passenger

## Study Objective
- Problem understanding and definition: understand the problem and how the potential solution would look. Also, define the requirements for solving the problem
Data collection and preparation: get a dataset that is ready for analysis
Data understanding using Exploratory Data Analysis (EDA): understand your dataset
Feature Engineering and Data Processing: a process of using raw data to create features that will be used for predictive modeling.
Model building: produce some predictive models that solve the problem
Model evaluation: choose the best model among a subset of the most promising ones and determine how good the model is in providing the solution
Communication and/or deployment: use the predictive model and its results

## Research process
- Worked 21 hours for 2 weeks as a data scientist and developed classification models that could predict the survival rate among Titanic passengers predicted by study factors. Data Analysis included using different analyzing techniques, fact-checking, extracting new data, patterns & data visualization, feature engineering, building pipelines, tuning hyperparameters, and final evaluation & reporting.

## Conclusion
In this classification task, I used XGBClassifier, RandomForestClassifier, Decision Tree Classifier, Gradient Boosting Classifier, SVC, LogisticRegression, AdaBoost Classifier, and KNeighbors Classifier. XGBClassifier and RandomForestClassifier showed better scores in accuracy compared with other algorithms. I also added a description of each hyperparameter that I tuned. In the pipeline, I added imputation methods, encoding methods, and scaling methods (for some), and additionally added functions to drop constant, duplicated, and correlated features (though eventually, I ended up keeping only a few factors, I decided not to delete and keep these functions). Working 21 hours for 2 weeks as a data scientist I developed classification models that could predict the survival rate among Titanic passengers predicted by the research factors.  

- First, I used different analyzing techniques to see the relationship between features to the target. Some of the outputs from the analysis needed fact-checking, so I compared my findings to the real facts from the external resources.
- Second, I extracted new data to see some patterns in the feature engineering part by merging variables and creating new factors.
- Third, in this work, I put my efforts to visualize all of my findings in the 'EDA' and 'Feature Engineering' parts.
- Fourth, I used pipelines and grid search to tune models' hyperparameters and find the best estimators for them.
- Fifth, I added five different types of classification metrics and also used a confusion matrix to visualize models predicting errors.
- Six, I separately built a testing sample where you could type information about passengers and forecast whether he/she would survive or not. 

