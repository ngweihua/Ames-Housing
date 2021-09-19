# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2 - Ames Housing Data and Kaggle Challenge

# Problem Statement
Our client is an overseas property investment firm that specialised in purchasing existing residential housing, performing necessary cost-effective renovations and selling them to yield a better profits.

As a data scientist, our task is to build a good regression model based on the Ames housing dataset to help client to have a good prediction on the price of the house which helps them in their decision making. We also aimed to identify top features to look out for when investing in Ames residential housing such as valuable residential location and important features of the residential houses that can be cost effectively renovated to increase its value.

The given Ames housing dataset consists of 81 features for residential houses that were sold between 2006 and 2010.

The following are the general workflow for this project:
Understanding the data
Exploratory data analysis (EDA)
Data cleaning
Pre-processing and feature engineering
Modelling and evaluation
Conclusion and recommendation

The model will then be evaluated by two metrics - coefficient of determination (R2) and root mean square error (RMSE). The objective is to get a high R2 and a low RMSE score.

# Data Dictionary
|Attribute|Variable Type |Dataset|Description|
|---|---|---|---|
|**Id**|*Discrete*|Ames Housing|Unique ID for each property|
|**PID**|*Nominal*|Ames Housing|Parcel identification number  - can be used with city web site for parcel review|
|**MS SubClass**|*Nominal*|Ames Housing|Identifies the type of dwelling involved in the sale|
|**MS Zoning**|*Nominal*|Ames Housing|Identifies the general zoning classification of the sale|
|**Lot Frontage**|*Continuous*|Ames Housing|Linear feet of street connected to property|
|**Lot Area**|*Continuous*|Ames Housing|Lot size in square feet|
|**Street**|*Nominal*|Ames Housing|Type of road access to property|
|**Alley**|*Nominal*|Ames Housing|Type of alley access to property|
|**Lot Shape**|*Ordinal*|Ames Housing|General shape of property|
|**Land Contour**|*Nominal*|Ames Housing|Flatness of the property|
|**Utilities**|*Ordinal*|Ames Housing|Type of utilities available|
|**Lot Config**|*Nominal*|Ames Housing|Lot configuration|
|**Land Slope**|*Ordinal*|Ames Housing|Slope of property|
|**Neighborhood**|*Nominal*|Ames Housing|Physical locations within Ames city limits (map available)|
|**Condition 1**|*Nominal*|Ames Housing|Proximity to various conditions|
|**Condition 2**|*Nominal*|Ames Housing|Proximity to various conditions (if more than one is present)|
|**Bldg Type**|*Nominal*|Ames Housing|Type of dwelling|
|**House Style**|*Nominal*|Ames Housing|Style of dwelling|
|**Overall Qual**|*Ordinal*|Ames Housing|Rates the overall material and finish of the house|
|**Overall Cond**|*Ordinal*|Ames Housing|Rates the overall condition of the house|
|**Year Built**|*Discrete*|Ames Housing|Original construction date|
|**Year Remod/Add**|*Discrete*|Ames Housing|Remodel date (same as construction date if no remodeling or additions)|
|**Roof Style**|*Nominal*|Ames Housing|Type of roof|
|**Roof Matl**|*Nominal*|Ames Housing|Roof material|
|**Exterior 1st**|*Nominal*|Ames Housing|Exterior covering on house|
|**Exterior 2nd**|*Nominal*|Ames Housing|Exterior covering on house (if more than one material)|
|**Mas Vnr Type**|*Nominal*|Ames Housing|Masonry veneer type|
|**Mas Vnr Area**|*Continuous*|Ames Housing|Masonry veneer area in square feet|
|**Exter Qual**|*Ordinal*|Ames Housing|Evaluates the quality of the material on the exterior |
|**Exter Cond**|*Ordinal*|Ames Housing|Evaluates the present condition of the material on the exterior|
|**Foundation**|*Nominal*|Ames Housing|Type of foundation|
|**Bsmt Qual**|*Ordinal*|Ames Housing|Evaluates the height of the basement|
|**Bsmt Cond**|*Ordinal*|Ames Housing|Evaluates the general condition of the basement|
|**Bsmt Exposure**|*Ordinal*|Ames Housing|Refers to walkout or garden level walls|
|**BsmtFin Type 1**|*Ordinal*|Ames Housing|Rating of basement finished area|
|**BsmtFin SF 1**|*Continuous*|Ames Housing|Type 1 finished square feet|
|**BsmtFin Type 2**|*Ordinal*|Ames Housing|Rating of basement finished area (if multiple types)|
|**BsmtFin SF 2**|*Continuous*|Ames Housing|Type 2 finished square feet|
|**Bsmt Unf SF**|*Continuous*|Ames Housing|Unfinished square feet of basement area|
|**Total Bsmt SF**|*Continuous*|Ames Housing|Total square feet of basement area|
|**Heating**|*Nominal*|Ames Housing|Type of heating|
|**Heating QC**|*Ordinal*|Ames Housing|Heating quality and condition|
|**Central Air**|*Nominal*|Ames Housing|Central air conditioning|
|**Electrical**|*Ordinal*|Ames Housing|Electrical system|
|**1st Flr SF**|*Continuous*|Ames Housing|First Floor square feet|
|**2nd Flr SF**|*Continuous*|Ames Housing|Second floor square feet|
|**Low Qual Fin SF**|*Continuous*|Ames Housing|Low quality finished square feet (all floors)|
|**Gr Liv Area**|*Continuous*|Ames Housing|Above grade (ground) living area square feet|
|**Bsmt Full Bath**|*Discrete*|Ames Housing|Basement full bathrooms|
|**Bsmt Half Bath**|*Discrete*|Ames Housing|Basement half bathrooms|
|**Full Bath**|*Discrete*|Ames Housing|Full bathrooms above grade|
|**Half Bath**|*Discrete*|Ames Housing|Half baths above grade|
|**Bedroom AbvGr**|*Discrete*|Ames Housing|Bedrooms above grade (does NOT include basement bedrooms)|
|**Kitchen AbvGr**|*Discrete*|Ames Housing|Kitchens above grade|
|**Kitchen Qual**|*Ordinal*|Ames Housing|Kitchen quality|
|**TotRms AbvGrd**|*Discrete*|Ames Housing|Total rooms above grade (does not include bathrooms)|
|**Functional**|*Ordinal*|Ames Housing|Home functionality (Assume typical unless deductions are warranted)|
|**Fireplaces**|*Discrete*|Ames Housing|Number of fireplaces|
|**Fireplace Qu**|*Ordinal*|Ames Housing|Number of fireplaces|
|**Garage Type**|*Nominal*|Ames Housing|Garage location|
|**Garage Yr Blt**|*Discrete*|Ames Housing|Year garage was built|
|**Garage Finish**|*Ordinal*|Ames Housing|Interior finish of the garage|
|**Garage Cars**|*Discrete*|Ames Housing|Size of garage in car capacity|
|**Garage Area**|*Continuous*|Ames Housing|Size of garage in square feet|
|**Garage Qual**|*Ordinal*|Ames Housing|Garage quality|
|**Garage Cond**|*Ordinal*|Ames Housing|Garage condition|
|**Paved Drive**|*Ordinal*|Ames Housing|Paved driveway|
|**Wood Deck SF**|*Continuous*|Ames Housing|Wood deck area in square feet|
|**Open Porch SF**|*Continuous*|Ames Housing|Open porch area in square feet|
|**Enclosed Porch**|*Continuous*|Ames Housing|Enclosed porch area in square feet|
|**3Ssn Porch**|*Continuous*|Ames Housing|Three season porch area in square feet|
|**Screen Porch**|*Continuous*|Ames Housing|Screen porch area in square feet|
|**Pool Area**|*Continuous*|Ames Housing|Pool area in square feet|
|**Pool QC**|*Ordinal*|Ames Housing|Pool quality|
|**Fence**|*Ordinal*|Ames Housing|Fence quality|
|**Misc Feature**|*Nominal*|Ames Housing|Miscellaneous feature not covered in other categories|
|**Misc Val**|*Continuous*|Ames Housing|Dollar-Value of miscellaneous feature|
|**Mo Sold**|*Discrete*|Ames Housing|Month Sold (MM)|
|**Yr Sold**|*Discrete*|Ames Housing|Year Sold (YYYY)|
|**Sale Type**|*Nominal*|Ames Housing|Type of sale|
|**SalePrice**|*Continuous*|Ames Housing|Sale price $$ (target)|

# Directory Structure
|S/N|Filename in Main Folder|Filename in Sub folder|
|--|---|---|
|1.|01_EDA_Datacleaning_Preprocessing.ipynb|
|2.|02_Feature_Engineering_Modelling.ipynb|
|-|datasets|
|3.|-|train.csv|
|4.|-|test.csv|
|5.|-|df_X.csv|
|6.|-|df_X_test.csv|
|7.|-|kaggle_sub.csv|
|8.|-|Ames Housing Description.txt|
|9|Kaggle_score.png|
|10.|presentation.pdf|


# Conclusion and Recommendation
The sale price of the house can be modelled with much greater accuracy by using the feature identified by lasso regression. 
<br>The following are the summary of steps that was taken to derive at the final model:
+ Exploratory data analysis (EDA)
+ Data Cleaning 
    - Imputing missing data
    - Outlier removal
    - One hot encoded variables (get_dummies)
+ Pre-processing and Feature Engineering
    - Create new column (house_age)
    - using lasso coefficient for feature selection
+ Modelling and evaluation 
    - Hyperparameter tuning
    
<br>In conclusion, by applying regularization, there is a great improvement in the performance of the model. The best regression model has a R2 score of 0.88 which accounts for 88% of the variance and a RMSE score of $20,508. 
<br>
<br>The following are the top 20 important features of the residential housing that can potentially help to increase its value, not in ranking order: 
+ Size of the house (area of 1st floor, 2nd floor, basement, ground living, lot, basement type 1 finished) 
+ Having an overall house quality of 7,8,9,10
+ Excellent Kitchen quality 
+ Residential houses in the neighbourhood of Greenhill, Northridge Heights,Crawford, Northridge and Somerset
+ Excellent basement Quality which evaluates the height of the basement
+ Building type - single-family detached
+ Good exposure to walkout or garden level basement walls
+ 1st exterior covering on house - brick face

With that, the recommended location will be :
- Greenhill
- Northridge Heights
- Crawford
- Northridge
- Somerset

The features that can be renovated to increase value are: 
- Increase the overall house material and finish quality
- Kitchen quality 
- Brick face house exterior 

The model limits to Ames residential housing and the housing value is not up to date as the database was taken from 2006 to 2010. 
# Future steps

Here are some of the steps to improve the model: 
- Log transformation of the house saleprice instead of dropping them as outlier as the graph shows that the model cannot predict well after house sale price of $300,000 
- Grouping the features such as area to create one feature to reduce noise and multicollinearity
- Try different models to predict the saleprice such as elastic net



# Appendix
1. http://jse.amstat.org/v19n3/decock/DataDocumentation.txt