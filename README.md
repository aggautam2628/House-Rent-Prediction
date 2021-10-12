# House-Rent-Prediction
A project discussing the best suitable method for prediction of house price in Melbourne Australia.



Step1: Data Collection

Source: kaggle.com
Link: https://www.kaggle.com/anthonypino/melbourne-housing-market?select=Melbourne_housing_FULL.csv

Sample Raw Data:

![image](https://user-images.githubusercontent.com/46028802/136984260-25bf9631-4943-4dd4-8634-95bac7e6603e.png)

Sample size: Around 34000
Total of 21 columns, which means that we have 21 features

![image](https://user-images.githubusercontent.com/46028802/136984384-fdc74d36-c071-4139-8c63-bafd094244b5.png)

Step 2: Data Cleaning

The raw data which we have is far from perfect and can batter our ML algorithm by producing highly inaccurate results which can be disastrous 
What are the imperfections:
1. Missing values
2. Outliers
3. Unnecessary features


1. Missing Values

![image](https://user-images.githubusercontent.com/46028802/136984771-6334e694-70b3-4443-b51e-862a5daf2971.png)

each category with missing values

![image](https://user-images.githubusercontent.com/46028802/136984849-bb2479bd-410d-457d-9076-2706435ff8f6.png)

Missing data % wise

2. Outliers

We will use Seaborn API for data visualization.
Scatter plot will be used on some features. (features are chosen randomly)

3. Unnecessary features

Three features are absolutely unnecessary out of 21. 

They are:
1. Postcode: No contribution in determining price of a property 
2. Bedroom2: Redundant data as we already have ‘room’ feature
3. Building Area: Amount of missing data for this feature is 55%, after removing null rows this feature will greatly reduce sample data. So, we will drop it despite the fact that building area is important for determining price.

df2 = df1.drop(['Postcode', 'Bedroom2', 'BuildingArea'], axis='columns')

Step 3: Data Analysis

For analysis we will divide features into numerical variables and categorical variables.

Categorical variables:

A categorical variable is a category or type. For example, hair color is a categorical value or hometown is a categorical variable. Species, treatment type, and gender are all categorical variables.

In our database the categorical variables are:

Suburb, Address, Type, Method, SellerG, Postcode, CouncilArea, Regionname

We are using boxplot graph combined with Seaborn API

Numerical variable:
A numerical variable is a variable where the measurement or number has a numerical meaning. For example, total rainfall measured in inches is a numerical value, heart rate is a numerical value, number of cheeseburgers consumed in an hour is a numerical value.

In our database the numerical variables are:
Rooms, Price, Distance, Bathroom, Car, Landsize, YearBuilt, Lattitude, Longtitude, Propertycount

We will use Scatter plot for analysing numerical variables.

Correlation heatmap
We will use this very useful feature SeaBorn API to get final insight into how features are affecting price.
We will decide that using correlation coefficient.



Outcome:
⚫ Our model can predict price of a house in particular neighbourhood in Melbourne suburb. Accuracy achieved by our model is around 80% when suitable learning model is used (random forest in our case).
⚫ We were able to discover best suited regression models for these kinds of tasks.
⚫ Importance of data cleaning in a machine learning model

Analysis:

![image](https://user-images.githubusercontent.com/46028802/136985390-6c5cb91e-9208-47e5-9f27-25de15bd14cc.png)

![image](https://user-images.githubusercontent.com/46028802/136985446-caed417d-2336-4e91-b316-28b82aac7733.png)

Looking at all the scatter plots above we can see here are very few outliers in data in the first place. But for better functioning of our algorithm we will remove any property which is above 75th percentile score.

![image](https://user-images.githubusercontent.com/46028802/136985954-03d3f912-db5f-4843-b4ea-f694e752006c.png)

Scatter plot after removing outliers

Box-plot:
Note: The black line inside each box represents median of respective data. Box edges represent 25th percentile of median value.

![image](https://user-images.githubusercontent.com/46028802/136986097-a6a50084-67b6-47d9-9b88-5b6fe26563c8.png)

Insights:
⚫ Median prices for houses are over $1M, townhomes are $800k -$900k and units are approx $500k.
⚫ Payment method doesn’t make much of a difference in house prices so it is not much of our interest
⚫ Median prices in the Metropolitan Region are higher than than that of Victoria Region - with Southern Metro being the area with the highest median home price
⚫ Price fluctuates very much with CouncilArea

![image](https://user-images.githubusercontent.com/46028802/136986310-df83b83a-e122-4611-8070-87c286e0c29d.png)

Insights:
⚫ The majority of homes in the dataset have 4 or 5 rooms.
⚫ The most prominent trend is that there is a negative correlation between Distance from Melbourne's Central Business District (CBD) and Price.

HeatMap:

Note: Focus on ‘price’ row and column

![image](https://user-images.githubusercontent.com/46028802/136986463-ba5c23cb-b83c-4209-a28c-04726efed57d.png)

Insights:

1. Weak Positive correlation (weakly directly proportional to price)
⚫ Car
⚫ Landsize 
⚫ Longitude

2. Moderate Positive Correlation (moderately directly proportional to price)
⚫ Rooms
⚫ Bathrooms

3. Strong Positive Correlation
None, and that is a good thing!!

4. Negative correlation (inversely proportional to price)
⚫ Distance from city centre
⚫ YearBuilt
⚫ Latitude
⚫ Property count

Function to predict price:

![image](https://user-images.githubusercontent.com/46028802/136986867-291d4703-e0af-4bcc-bd32-f15ca0be23ed.png)

Random Forest:

Showing accuracy just above 80%

![image](https://user-images.githubusercontent.com/46028802/136987045-64086918-330d-4326-9536-e6e3c100f7c1.png)

SVM:

Showing poor accuracy

![image](https://user-images.githubusercontent.com/46028802/136987102-cf849c02-18a7-4da3-9597-44e162175d53.png)


Logistic regression:

Showing poor accuracy

![image](https://user-images.githubusercontent.com/46028802/136987150-e6721329-f74c-43b9-9274-451289d97434.png)

Result:

Lets’ predict price of a house in Abbotsford suburb having 2 bedrooms at a distance of 2.5 km from Melbourne city centre having 1 bathroom and having space of 1 car.
Predicted price is: AUS $ 814300

![image](https://user-images.githubusercontent.com/46028802/136987327-2bda2563-839b-44fb-a6ab-b16279e79783.png)

![image](https://user-images.githubusercontent.com/46028802/136987355-b6fc2fb2-5d80-4921-9a49-54b010ee9e7f.png)

Conclusion:
Applications of AI and machine learning (ML) are ever increasing in our daily lives. Consumer market is no exception.ML models are finding great use in price prediction models such as property, share prices etc. We used supervised learning technique to find a fairly accurate house price prediction model. We have achieved a very accurate model and this model can be used websites and webapps as a backend model as this model is fairly reliable.




