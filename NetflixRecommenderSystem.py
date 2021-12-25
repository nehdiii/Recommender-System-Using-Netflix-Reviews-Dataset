"""
credit by : Nehdi Taha Mustapha 25/12/2021
"""
"""
"""
"""
1)
importations:
os :  used for manipulating directory paths
numpy : Scientific and vector computation for python
matplotlib : Plotting library
scipy : Optimization module in scipy
pandas: is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool
"""
import os
import numpy as np
from matplotlib import pyplot
import matplotlib as mpl
from scipy import optimize
import pandas as pd
import random

"""
2)
loading the data 
Netflix_Dataset_Rating.csv : dataset containing ratings/id of user/id of  movie on netflix data base 
Netflix_Dataset_Movie.csv : dataset containing the way movies sorted in netflix database id of movie/name of movie
data source :  https://www.kaggle.com/netflix-inc/netflix-prize-data
"""

df11 = pd.read_csv('C:/Users/DELL/PycharmProjects/multiprocessing gardient desecnt/Netflix_Dataset_Rating.csv')
df2 = pd.read_csv('C:/Users/DELL/PycharmProjects/multiprocessing gardient desecnt/Netflix_Dataset_Movie.csv')

# the data is very large to process in (takes much time on training) so i tried to reduce it by sampling 1000 rows
#from data set randomly
df1=df11.sample(n=1000)
"""
 3)
 data understanding
 
"""
print("columns df1 :")
print(list(df1.columns))
print("columns df2 :")
print(list(df2.columns))
print('-----------------------------')
print('first 5 lins of dataset 1:')
print(df1.head())
print('first 5 lins of dataset 2:')
print(df2.head())
print("-----------------------------")
print("max and min of ratings")
print("max of ratings : "+str(df1['Rating'].max()))
print("min of ratings : "+str(df1['Rating'].min()))
print("mean of ratings : "+str(df1['Rating'].mean()))
print("-----------------------------")
avalable_users_id = df1['User_ID'].unique()
rated_movies_id = df1['Movie_ID'].unique()
num_of_movies = len(rated_movies_id)
num_of_users = len(avalable_users_id)
print("Number of movies : "+str(num_of_movies))
print("Number of users : "+str(num_of_users))
y_size = (num_of_movies,num_of_users)
print("Size of matrix y  : "+str(y_size))
num_of_movies_on_netflix = len(list(df2['Movie_ID']))
print("Number of movies on Netflix DataBase : "+str(num_of_movies_on_netflix))


"""
4)
Data Preprocessing:
We will creat the variables Y and R. The matrix Y (a num_movies num_users matrix) stores the ratings Y(i , j) (from 1 to 5). 
The matrix R is an binary-valued indicator matrix, where R(i , j) = 1 if user j gave a rating to movie i , and R(i , j) = 0 otherwise. The objective of collaborative 
filtering is to predict movie ratings for the movies that users have not yet rated, that is, the entries 
with R(i , j) = 0 . This will allow us to recommend the movies with the highest predicted ratings to the user.

"""




Y=pd.DataFrame(np.zeros(y_size),index=rated_movies_id,columns=avalable_users_id)
R=pd.DataFrame(np.zeros(y_size),index=rated_movies_id,columns=avalable_users_id)
#filling the Matrix Y
for ind in df1.index:
        Y[df1['User_ID'][ind]][df1['Movie_ID'][ind]] = df1['Rating'][ind]

print("-------------------------------")
print("How y looks like")
print(Y)
##filling the Matrix R
for ind in df1.index:
 if( Y[df1['User_ID'][ind]][df1['Movie_ID'][ind]] != 0):
   R[df1['User_ID'][ind]][df1['Movie_ID'][ind]] =1
print("-------------------------------")
print("How y looks like")
print(R)

"""
5)
Collaborative Filtering algo 
to get more  knowledge abaouth this algo and how it used in recomander systems u can check this url :
https://realpython.com/build-recommendation-engine-collaborative-filtering/
"""

def CollaborativeFiltering(params, Y, R, num_users, num_movies,num_features, lambda_=0.0):
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features:].reshape(num_users, num_features)
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    #------------------------------------------
    J = (1 / 2) * np.sum(np.square(np.dot(X, Theta.T) - Y) * R) + (lambda_ / 2) * np.sum(np.square(Theta)) + (lambda_ / 2) * np.sum(np.square(X))
    X_grad = np.dot(((np.dot(X, Theta.T) - Y) * R), Theta) + (lambda_ * X)
    Theta_grad = np.dot(((np.dot(X, Theta.T) - Y) * R).T, X) + (lambda_ * Theta)
    grad = np.concatenate([X_grad.ravel(), Theta_grad.ravel()])
    return J, grad
"""
6)
Normalize Ratings is function that help us to deal with non reating users case to recomand the avrage rated movie for this type 
of users  
to get more  knowledge abaouth this algo and how it used in recomander systems u can check this url :
https://realpython.com/build-recommendation-engine-collaborative-filtering/
"""
def normalizeRatings(Y, R):
    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = R[i, :] == 1
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return Ynorm, Ymean


""""

7) let's create our new user and randomly give it some ratings on some movies 

"""

my_ratings = np.zeros(num_of_movies)
my_ratings[3] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

#out put the rated moves
print('New user ratings:')
print('-----------------')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0 :
        print('Rated %d stars: %s' % (my_ratings[i], df2['Movie_ID'][rated_movies_id[i]]))


"""
8) defining some hyperparamters 
"""
#  Add our own ratings to the data matrix
Y = np.hstack([my_ratings[:, None], Y])
R = np.hstack([(my_ratings > 0)[:, None], R])
#  Normalize Ratings
Ynorm, Ymean = normalizeRatings(Y, R)
#  Useful Values
num_movies, num_users = Y.shape
num_features = 10
# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
initial_parameters = np.concatenate([X.ravel(), Theta.ravel()])
# Set options for scipy.optimize.minimize
options = {'maxiter': 100}

"""
9)Training
I used optimize.minimize from scipy to reduce work on the gradient desecent and in general the optimazier 

"""
# Set Regularization
lambda_ = 10
res = optimize.minimize(lambda x: CollaborativeFiltering(x, Ynorm, R, num_users,
                                               num_movies, num_features, lambda_),
                        initial_parameters,
                        method='TNC',
                        jac=True,
                        options=options)
theta = res.x
# Unfold the returned theta back into X and theta
X = theta[:num_movies*num_features].reshape(num_movies, num_features)
Theta = theta[num_movies*num_features:].reshape(num_users, num_features)
print('Recommender system learning completed.')


"""
10) the predition phase  
"""

p = np.dot(X, Theta.T)
my_predictions = p[:, 0] + Ymean
# we sorte the predicted movie with highest movie
ix = np.argsort(my_predictions)[::-1]

"""
11) final phase checking out the results 
"""

print('Top recommendations for you:')
print('----------------------------')
for i in range(10):
    j = ix[i]
    print('Predicting rating %.1f for movie %s' % (my_predictions[j],df2['Name'][rated_movies_id[j]]))
print('\nOriginal ratings provided:')
print('--------------------------')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d stars: %s' % (my_ratings[i], df2['Name'][rated_movies_id[i]]))
"""
output example of the code 
columns df1 :
['User_ID', 'Rating', 'Movie_ID']
columns df2 :
['Movie_ID', 'Year', 'Name']
-----------------------------
first 5 lins of dataset 1:
          User_ID  Rating  Movie_ID
16722745  1572201       3      4356
12435346   760072       4      3320
12724649  2344419       2      3371
7160919   1948153       4      1905
3526263   2348237       4       963
first 5 lins of dataset 2:
   Movie_ID  Year                          Name
0         1  2003               Dinosaur Planet
1         2  2004    Isle of Man TT 2004 Review
2         3  1997                     Character
3         4  1994  Paula Abdul's Get Up & Dance
4         5  2004      The Rise and Fall of ECW
-----------------------------
max and min of ratings
max of ratings : 5
min of ratings : 1
mean of ratings : 3.591
-----------------------------
Number of movies : 512
Number of users : 996
Size of matrix y  : (512, 996)
Number of movies on Netflix DataBase : 17770
-------------------------------
How y looks like
      1572201  760072   2344419  1948153  ...  383344   612980   1738108  1905658
4356      3.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0
3320      0.0      4.0      0.0      0.0  ...      0.0      0.0      0.0      0.0
3371      0.0      0.0      2.0      0.0  ...      0.0      0.0      0.0      0.0
1905      0.0      0.0      0.0      4.0  ...      0.0      0.0      0.0      0.0
963       0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0
...       ...      ...      ...      ...  ...      ...      ...      ...      ...
97        0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0
528       0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0
1618      0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0
3433      0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0
3579      0.0      0.0      0.0      0.0  ...      0.0      4.0      0.0      0.0

[512 rows x 996 columns]
-------------------------------
How y looks like
      1572201  760072   2344419  1948153  ...  383344   612980   1738108  1905658
4356      1.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0
3320      0.0      1.0      0.0      0.0  ...      0.0      0.0      0.0      0.0
3371      0.0      0.0      1.0      0.0  ...      0.0      0.0      0.0      0.0
1905      0.0      0.0      0.0      1.0  ...      0.0      0.0      0.0      0.0
963       0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0
...       ...      ...      ...      ...  ...      ...      ...      ...      ...
97        0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0
528       0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0
1618      0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0
3433      0.0      0.0      0.0      0.0  ...      0.0      0.0      0.0      0.0
3579      0.0      0.0      0.0      0.0  ...      0.0      1.0      0.0      0.0

[512 rows x 996 columns]
New user ratings:
-----------------
Rated 3 stars: 1906
Rated 5 stars: 2458
Rated 4 stars: 149
Rated 5 stars: 3080
Rated 3 stars: 1471
Rated 5 stars: 2496
Rated 4 stars: 1956
Rated 5 stars: 1643
Rated 5 stars: 820
Recommender system learning completed.
Top recommendations for you:
----------------------------
Predicting rating 5.0 for movie America's Most Haunted Town
Predicting rating 5.0 for movie Go Fish
Predicting rating 5.0 for movie Lone Wolf and Cub: Baby Cart in the Land of Demons
Predicting rating 5.0 for movie Muppets From Space
Predicting rating 5.0 for movie Chopin: Desire for Love
Predicting rating 5.0 for movie Mean Machine
Predicting rating 5.0 for movie No Ordinary Love
Predicting rating 5.0 for movie Sphere
Predicting rating 5.0 for movie Black Rainbow
Predicting rating 5.0 for movie Tick Tock

Original ratings provided:
--------------------------
Rated 3 stars: The Knights Templar
Rated 5 stars: Plenty
Rated 4 stars: The Edward R. Murrow Collection
Rated 5 stars: Duel at Diablo
Rated 3 stars: Voices of Iraq
Rated 5 stars: Solas
Rated 4 stars: Cirque du Soleil: Dralion
Rated 5 stars: Rules: Pyaar Ka Superhit Formula
Rated 5 stars: Predator Island
"""