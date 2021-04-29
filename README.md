# Sources
The data was sourced from three places:
- The point-by-point data that was crucial to this project was collected by Jeff Sackmann and is posted on a github repository: https://github.com/JeffSackmann/tennis_slam_pointbypoint 
- The historical ELO ranking data was collected from the Ultimate Tennis Statistics Website: https://www.ultimatetennisstatistics.com/rankingsTable 
# Code Index
- eda.ipynb: Just a data exploration, I’m keeping it there for my own purposes
- Point_probs.ipynb: This is the notebook where I make the dataset and train the logistic model for predicted probabilities of success, the code saves the model as ‘point_prob_model.sav’ for use in streaks.py
- Streaks.py: This is the script where the majority of the simulation and calculation goes on. The results of the simulation are stored in streaks_points.csv, which are analyzed in results_analysis.ipynb
- Results_analysis.ipynb: This is where the results are analyzed and plots produced. Two of them are nicely interactive.
