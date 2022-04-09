# Big data project in Apache Spark using Spark MLlib 

# Data Overview

![e1](https://user-images.githubusercontent.com/70089857/160276669-9e5c057d-4cf0-429d-9f3c-51055f5cd583.PNG)

From the overview above, we can see that UserId is a unique ID for the user and movieId unique ID for the movie. The rating column is the prediction that we want to make and the rest of the columns can be used as predictors.

We can see that the movies dataset has 27278 rows and 3 columns. Ratings dataset has 20000263 rows and 4 columns.
* userId : Unique Id for user
* movieId: unique Id for movie, consistent between ratings,  tags ,movies and links. movieId refers to the same movies across these four data files.
* rating: Rating between 0 and 5 for the movie
* title: The movie title, titles are entered manually and imported from https://www.themoviedb.org/.
* timestamp: Date and time the rating was given
* genre: genres  list separated by pipeline |  includes the following : Action, Adventure, Animation, Children’s, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War , Western.

Do a left join between ratings and movies using movieID. We will use this to get better correlation and visualization. 
![e1](https://user-images.githubusercontent.com/70089857/160276890-dea9e54a-64ed-4929-bfe6-419a70189324.PNG)

# Data Analysis
Before we build the prediction model , let’s analyze and visualize each predictor to gain insight on the distribution.

![Capture](https://user-images.githubusercontent.com/70089857/160277007-cd3bf1e5-5032-4cd4-9f79-b486785c13de.PNG)
This histogram actually shows that the ratings are normally distributed . There are certain movies that aren’t rated much at all while some are rated numerous times by users.

![Capture](https://user-images.githubusercontent.com/70089857/160277049-5ae1fb68-7f1d-4e99-8711-36bc1ceee7d9.PNG)
The histogram shows some users are more active in rating the movies. The ratings are approximately normally distributed but skewed to the left.

![Capture](https://user-images.githubusercontent.com/70089857/160277102-e4d5777b-5044-4ec6-9fc5-79fe01f071f8.PNG)
The ratings are approximately negatively skewed.

The genre is a pipeline of list separated by ‘|’. Lets get all the genres into a data frame for preprocessing.
```
#Get all the genres (Adventure|Animation|Children|Comedy|Fantasy) into a dataframe
genres <- as.data.frame(movies$genres,stringsASFactors = FALSE)
genre <- as.data.frame(tstrsplit(genres[,1], '[|]', 
                                        type.convert=TRUE), 
                              stringsAsFactors=FALSE) 

#list out all genre so i can create list
unique(genre[c(1)]) 
unique(genre[c(2)])

head(genres)
colnames(genre) <- c(1:10)
genrelist <- c("Action","Adventure","Animation","Comedy","Children","Crime","Drama","Documentary","Fantasy","Film-Noir",
               "Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","war","western")

genrelist
```
Let's take a look at our target rating. We won’t be using it for predictions but it is worth analyzing.
![Capture](https://user-images.githubusercontent.com/70089857/160277345-4aa86001-f7bb-498f-8afa-eb46214e1993.PNG)
The graph shows that users have a higher tendency to give round numbers for the rating. We can also see that users tend to give higher ratings.

# Building the ML model for linear regression
```
set.seed(123)
new_edx <- movielens %>%
  transmute(year=year(as_datetime(timestamp)),
            week=week(as_datetime(timestamp)),
            genres=genres,
            userId=userId,
            movieId=movieId,
            rating=rating)

# take a random sample of size 100000 from a dataset
# sample without replacement
mysample <- new_edx[sample(1:nrow(new_edx), 100000,
                          replace=FALSE),] 

#spilt dataset into 8:2 ratio
samp <- createDataPartition(mysample$rating, p = 0.8, list = FALSE)
train <- mysample[samp,]
test <- mysample[-samp,]
```
Transforming the timestamp into week and year variables. R is not very good at dealing with big data thus we will do random sampling of 100000 data. Dataset is also split into 8:2 ratio. 80% for training, 20% for testing.

Genres column is one-hot encoded and removed.
```
#one-hot encode the genres column
train$genres <- str_split(train$genres, pattern="\\|")
one_hot_genres <- enframe(train$genres) %>%
  unnest(value) %>%
  mutate(temp = 1) %>%
  pivot_wider(names_from = value, values_from = temp, values_fill = list(temp = 0))
train <- cbind(train, one_hot_genres) %>% select(-name)
train$genres <- NULL
```
Two attributes were added. movie score will be the average rating for the movieId minus the total average rating. User score will be the average rating for the userID minus the average rating and movie score. These two variables will help in plotting the DT. We need to have more predictor variables. userID and movieID are irrelevant, too little predictors will lead to underfitting or generalizing problems. Timestamp is already transformed into week and year. Same is done to the testing set as well.

![Capture](https://user-images.githubusercontent.com/70089857/160277444-83bad680-797a-4c59-b5a1-5c424b0e8487.PNG)

This is a naive prediction of all the ratings with the average rating on the training set. The result RMSE is  1.051201. The prediction is off by 1, which is not acceptable.

Linear model with week,year, user score and movie score
![Capture](https://user-images.githubusercontent.com/70089857/160277558-ad1b44b6-44c7-45bc-b2c6-412c2e3e280a.PNG)
![Capture](https://user-images.githubusercontent.com/70089857/160277586-c90c2049-503d-43f5-9259-5b5802254801.PNG)
RMSE score seems to be off by 1. Lets try using the caret package. Caret package(Classification and Regression Training) contains functions to streamline the model training process for complex regression and classification problems. It is a powerful and popular package.

![Capture](https://user-images.githubusercontent.com/70089857/160277621-747cd753-7376-432e-94fd-c07e4bfc06a3.PNG)

The result is worse than the baseline model.

We will now add the hot encoded genres into the predictors set. We will do linear regression with regularization. 
```
not_genres <- c("userId", "movieId", "rating", "week", "title", "year", "user_score", "movie_score")
genres <- colnames(train)[!colnames(train) %in% not_genres]
genres

genre_scores <- data.frame(genre="",m=0, sd=0)
for(genre in genres){
  results <- train %>% filter(train[colnames(train)==genre]==1) %>%
    summarise(m=mean(rating), sd=sd(rating))
  genre_scores <- genre_scores %>% add_row(genre=genre, m=results$m, sd=results$sd)
}
genre_scores <- genre_scores[-1,]
genre_scores[is.na(genre_scores)] <- 0
genre_scores
```

![Capture](https://user-images.githubusercontent.com/70089857/160277712-080b7b36-de71-4bf7-898f-09e6d829a545.PNG)

We include a lambda parameter in the cost function to prevent overfitting. This lambda is then used to update the theta parameters in the gradient descent algorithm.We can see from the plot that the lambda which minimizes the RMSE is 3.5.

![Capture](https://user-images.githubusercontent.com/70089857/160277741-f6307c1f-7aa5-47d8-87e3-b1e47e9e78c9.PNG)

We will use lambda 3.5 to train and predict the test set.

```
lambda <- 3.5
avg_rating <- mean(train$rating)
movie_score <- train %>%
  group_by(movieId) %>%
  summarise(b_m = sum(rating - avg_rating)/(n()+lambda))
user_score <- train %>% 
  left_join(movie_score, by="movieId") %>%
  mutate(b_m = ifelse(is.na(b_m), 0, b_m)) %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - b_m - avg_rating)/(n()+lambda))
genre_score <- as.matrix(test[, genres]) %*% genre_scores$m
n_genres <- rowSums(test[,genres])
genre_score <- genre_score / n_genres

predicted_ratings <- 
  test %>% 
  left_join(movie_score, by = "movieId") %>%
  left_join(user_score, by = "userId") %>%
  cbind(genre_score) %>%
  mutate(pred = genre_score) %>%
  mutate(pred = ifelse(!is.na(b_m)|!is.na(b_u), 
                       avg_rating + replace_na(b_m,0) + replace_na(b_u,0), 
                       pred))

result_6 <- RMSE(test$rating, predicted_ratings$pred)
cat("RMSE :", result_6)
```
The final RMSE score is 0.9468128. This seems to be the lowest but however it is still not optimal. We will move on to other models.


# Random Forest model
Random forest model is based on generating a large number of decision trees where each is constructed on a different subset of the training dataset. The subsets are usually selected by sampling at random and with replacement from the original dataset. The decision trees are then used to identify a classification by selecting the most common output.

I also use the caret package which makes the process of training, tuning and evaluating machine learning models in R consistent and easy. It has features like data splitting and Modeling tuning which include the random forest algorithm.

To implement the random forest model, I tried different methods to get the best model with low RMSE. Lower values of RMSE indicate better fit.

On the first try, I loaded all the variables with ntree = 500. The first parameter ‘rating ~ .’ is what we want to predict by using each of the remaining columns of the data. The ntree defines the number of trees to be generated.  Once the model is done training, we are able to see the output which include details like the mtry. Mtry is the number of variables randomly sampled as candidates at each split. 

For each different value of mtry, the tuning parameter, caret performs a separate bootstrap. Once caret determines the best value of the tuning parameter, in this case the value of 25 is best for mtry (highest R-squared) it runs the model one more time using this value for the tuning. 

![Capture](https://user-images.githubusercontent.com/70089857/160277863-3a69ac12-e03e-4839-9ccb-33c89e8e06e2.PNG)

However, the RMSE score was not very good and could be better. The ideal RMSE score should be between 0.2 and 0.5 as it shows that the model can relatively predict the data accurately.

On the second try, I used only selected variables (user_score + movie_score + year + week) with ntree = 500 to see whether it can improve the rmse score. 

![Capture](https://user-images.githubusercontent.com/70089857/160277885-f0a3fffb-5e00-496b-8147-1e2315b3fc3f.PNG)

The results show an improvement by 0.1 where mtry = 3.

On the third try, I used a set of variables that I did not use in the previous method which only includes the genre one-hot encoding to see whether there is any improvement made. I first filter out the columns I did not want in the dataset and then train the model. 


![Capture](https://user-images.githubusercontent.com/70089857/160277929-728dc5ff-f648-4f86-ba84-6c98e2dfc8da.PNG)

It also improved by 0.1 but still wasn’t good enough. 

On the fourth try, I tried to do some data tuning with all the variables as the input parameter. It uses Grid Search which is available in the caret package. It uses each axis of the grid as an algorithm parameter, and points in the grid are specific combinations of parameters. Because we are only tuning one parameter, the grid search is a linear search through a vector of candidate values.


![Capture](https://user-images.githubusercontent.com/70089857/160277963-1fd44a73-3cb4-4566-bec3-d226e1f62655.PNG)

The RMSE improved from 1.085382 to 1.07085 where mtry = 15. However, it is still not close to 0.5. 
Therefore, I concluded that the Random Forest algorithm is also a good choice for anyone who needs to develop a model quickly. It is easy to use and flexible. However, as we increase the number of trees in a random forest, the time taken to train each of them also increases. 
With the final rmse score not close to 0.5, Random Forest algorithm is not suitable for this dataset.

# Support Vector Machines Model

Support Vector Machines models try to find groups that can be somewhat separated and put in their own category(classes). They are similar to Linear/Logistic Regression models as they can perform regression tasks as well.
Using caret package(train function specified to svm_Linear mode):

```
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
svm_Linear <- train(rating~.,data=train,method='svmLinear',trControl=trctrl,preProcess=c("center",scale"),tuneLength = 10)
svm_Linear
```

![Capture](https://user-images.githubusercontent.com/70089857/160278040-f117a83a-63f7-4381-a86d-5b3f64f5db51.PNG)

From the start, it has achieved an RMSE of 0.2 within the train set itself. This could be the model I am looking for. This RMSE however is comparing the rating column with other columns and not itself.

```
test_pred <- predict(svm_Linear, newdata = test)
head(test_pred)
test_predi <- round_any(test_pred,0.5)
head(test_predi)
```

![Capture](https://user-images.githubusercontent.com/70089857/160278158-7c5be877-5824-4d5e-964e-ba611f2ddc63.PNG)

I use the predict() function to test the model with the test set and received some data with many decimal places. To compare the results, I rounded values to any closest 0.5 value.

![Capture](https://user-images.githubusercontent.com/70089857/160278195-fac693ef-689f-4a21-b6f6-459ee4ea081f.PNG)

Putting the predictions and the rating column into a table, we notice that there are predicted values ABOVE 5, which are not part of the original rating of 0.5 to 5, therefore I converted any value above 5 into 5 as below:

```
n = length(test_predi)
for (i in 1:n)  {
  if(test_predi[i] > 5) test_predi[i] = 5
  }
  confusionMatrix(test_df)
  RMSE(test$rating, test_predi)
  
  ```
  With the confusionMatrix however, the result only returned an accuracy of 0.2 as well.
  
A final RMSE calculation shows a score of 1.114, which is the worst the computer has calculated so far.
I will now attempt to use svm() function in the package e1071:

```
test_svm <- svm(rating~., data=train)
summary(test_svm)
rmse = RMSE(test$rating, predi)
rmse
```
Following the same steps, we achieved a similar result, that is a 0.2 accuracy and 1.118 rmse. This could be a possible sign of overfitting.
As with the Random Forest model, we will attempt to build the model on just the 4 selected predictors: year, week, movie_score and user_score.

```
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
svm_Linear <- train(rating~user_score+movie_score+year+week,data=train,method='svmLinear',trControl=trctrl,
preProcess =c("center","scale"),tuneLength = 10)
svm_Linear
rmse = RMSE(test$rating,test_predi)
rmse
```

![Capture](https://user-images.githubusercontent.com/70089857/160278530-3809d634-b42a-4dab-a86d-75584d339a22.PNG)

Again, both original train RMSE, test RMSE and accuracy show similar results regardless of the less used predictors. To avoid overfitting, we will attempt to avoid overfitting by doubling the data input from 10000 to 20000.

By increasing the data input, the final RMSE has lowered by 0.01. This possibly proves the theory of overfitting in this case further. Since the original data set has 20 million data, that amount might make the difference in obtaining a much lower RMSE and better accuracy along with better predictions, alas we do not have the computer capacity to attempt such a task.

In conclusion, Support Vector Machines might not be the model we are looking for as well due to resulting in the highest final RMSE, with a probable cause being overfitting. Taking a look back at the data obtained, most of the ratings are clustered around 2.5-4.5, thus the model being unable to set definite classes, disallowing the model to perform on new data.

# Discussion

The final RMSE score is 0.9468128. There is a lot of room for improvement, but we have not found a better solution yet. One idea is to stratify the user_scores by genres, since the same user may love some genres but hate others.I believe that logistic regression will be better for categorical data than linear regression. Linear regression deals with continuous values whereas classification problems mandate discrete values. It is more sensitive to the shift in threshold value when new data points are added.

We can consider running Lasso before Random Forest for a better prediction. Random Forest is a fully nonparametric predictive algorithm, it may not efficiently incorporate known relationships between the response and the predictors. Random Forest is also unable to discover trends that would enable it in extrapolation values that fall outside of the training set. The same user might not give as high of a rating to a movie genre in the future. We deal with extrapolation by using linear models such as SVM or Linear Regression.

Moving on to SVM, results obtained were the worst with an RMSE of above 1.1, however it seems likely that overfitting is the main cause of this issue. More tests or experiments need to be conducted for a better estimation of the models’ performance. Also of note that ratings cluster around a certain score as expected of normal ratings, meaning that the model is unable to set definitive classes as it might look at it as one single class, or have many different classes and thus having trouble relating them to one another, finally having a very undefined hyperplane.




