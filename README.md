# Recommender-System-Using-Netflix-Reviews-Dataset



<img src="https://c.tenor.com/NerN41mjgV0AAAAC/netflix-intro.gif" width="1100" height="400" />

data from : https://www.kaggle.com/netflix-inc/netflix-prize-data


output example of the code :


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
