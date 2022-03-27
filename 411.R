#setwd("D:/data/ml-20m")
library(lubridate)
library(tidyverse)
library(caret)
library(data.table)
library(stringr)
library(caTools)
library(Hmisc)
library(caret)


movies <- read.csv("movies.csv",stringsAsFactors=FALSE)
ratings <- read.csv("ratings.csv")

head(movies)
head(ratings)
head(tags)

str(movies)
str(ratings)

data.table(movies)
data.table(ratings)

summary(movies)
summary(ratings)

#Lets see mean rating for each movie ID
meanRating = aggregate( ratings$rating ~ ratings$movieId,data=ratings, FUN= mean) 
meanRating


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

movielens <- left_join(ratings, movies, by = "movieId")
head(movielens)
apply(movielens, 2, function(x) any(is.na(x)))

movielens %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins=20, color = "blue") +
  scale_x_log10() + 
  ggtitle("Number of Ratings for each Movies") +
  xlab("Movie ID") +
  ylab("Number of Ratings")


movielens %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins=20, color = "blue") +
  scale_x_log10() + 
  ggtitle("Number of Ratings given by each Users") +
  xlab("User ID") +
  ylab("Number of Ratings")


movielens %>% 
  count(genres) %>% 
  ggplot(aes(n)) + 
  geom_histogram( bins=20, color = "blue") +
  scale_x_log10()+ 
  ggtitle("Number of Ratings for each Genres") +
  xlab("Genres") +
  ylab("Number of Ratings")

movielens %>% count(rating) %>%
  ggplot(aes(rating,n)) + 
  geom_line() +
  geom_point() +
  ggtitle("Amount for each Rating") +
  xlab("Rating") +
  ylab("Number of Ratings")


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


#one-hot encode the genres column
train$genres <- str_split(train$genres, pattern="\\|")
one_hot_genres <- enframe(train$genres) %>%
  unnest(value) %>%
  mutate(temp = 1) %>%
  pivot_wider(names_from = value, values_from = temp, values_fill = list(temp = 0))
train <- cbind(train, one_hot_genres) %>% select(-name)
train$genres <- NULL

#adding the average rating  for each movie minus the total average rating
avg_rating <- mean(train$rating)
movie_score <- train %>% group_by(movieId) %>%
  summarise(movie_score = mean(rating-avg_rating))

#adding the average rating for each user minus the total average rating and movie score
user_score <- train %>% left_join(movie_score, by="movieId") %>%
  mutate(movie_score = ifelse(is.na(movie_score), 0, movie_score)) %>%
  group_by(userId) %>%
  summarise(user_score = mean(rating-avg_rating-movie_score)) 

train <- train %>% left_join(user_score) %>% left_join(movie_score)

head(train)


#Apply the same wrangling to the test set

#one-hot encode the genres column
test$genres <- str_split(test$genres, pattern="\\|")
one_hot_genres <- enframe(test$genres) %>%
  unnest(value) %>%
  mutate(temp = 1) %>%
  pivot_wider(names_from = value, values_from = temp, values_fill = list(temp = 0))
test <- cbind(test, one_hot_genres) %>% select(-name)
train$genres <- NULL

#adding columns of genres that are not present in test set, and removing those that are not in the train set
for(col in names(train)){
  if(!col %in% names(test)){
    test$newcol <- 0
    names(test)[names(test)=="newcol"] <- col
  }
}
for(col in names(test)){
  if(!col %in% names(train)){
    test[,col] <- NULL
  }
}

#adding the average scores on the train set of each user and movie
test$user_score <- NULL
test$movie_score <- NULL
test <- test %>% left_join(user_score, by="userId") %>% left_join(movie_score, by="movieId")

#if there are users or movies in the test set that are not in the train set, assign the score of the user and movie as 0
test <- test %>% mutate(user_score = ifelse(is.na(user_score), 0, user_score)) %>% mutate(movie_score = ifelse(is.na(movie_score), 0, movie_score))

#reorder the columns to follow the train set
test <- test %>% select(names(train))
head(test)

#For baseline comparison, predict ratings as the mean of ratings in train
y_hat <- mean(train$rating)
result_1 <- RMSE(test$rating, y_hat)
cat("RMSE :", result_1)


#Building Machine Learning Models

#Linear Model

#Linear Model with year, week, user_score, and movie_score
control <- trainControl(method = "none")
fit_lm <- train(rating~user_score+movie_score+year+week, data=train, method="lm", trControl=control)
print(fit_lm$finalModel)

y_hat <- predict(fit_lm, test)
result_2 <- RMSE(test$rating, y_hat)
cat("RMSE :", result_2)


#Linear Model with all features except userId, movieId, year and week
train_2 <- train %>% select(-c("userId", "movieId", "week","year"))
control <- trainControl(method = "none")
fit_lm <- train(rating~., data=train_2, method="lm", trControl=control)
print(fit_lm$finalModel)

y_hat <- predict(fit_lm, test)
result_3 <- RMSE(test$rating, y_hat)
cat("RMSE :", result_3)


#Using caret 
fit_tree <- train(rating~user_score+movie_score+year+week, data=train, method="rpart")
print(fit_tree$results)
plot(fit_tree$finalModel)
text(fit_tree$finalModel)

y_hat <- predict(fit_tree, test)
result_4 <- RMSE(test$rating, y_hat)
cat("RMSE :", result_4)



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
idx <- createDataPartition(train$rating, times=1, p=0.8, list=FALSE)
train_part_1 <- train[idx, ]
train_part_2 <- train[-idx, ]

#calculating the best lambda
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  avg_rating <- mean(train_part_1$rating)
  movie_score <- train_part_1 %>%
    group_by(movieId) %>%
    summarise(b_m = sum(rating - avg_rating)/(n()+l))
  user_score <- train_part_1 %>% 
    left_join(movie_score, by="movieId") %>%
    mutate(b_m = ifelse(is.na(b_m), 0, b_m)) %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_m - avg_rating)/(n()+l))
  predicted_ratings <- 
    train_part_2 %>% 
    left_join(movie_score, by = "movieId") %>%
    left_join(user_score, by = "userId") %>%
    mutate(b_m = ifelse(is.na(b_m), 0, b_m)) %>%
    mutate(b_u = ifelse(is.na(b_u), 0, b_u)) %>%
    mutate(pred = avg_rating + b_m + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, train_part_2$rating))
})

lambda <- lambdas[which.min(rmses)]
qplot(lambdas, rmses)
print(lambda)

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