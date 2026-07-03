# Random Forest Model for Wimbledon Gentlemen's Singles, 2026

## Goal:
My overall goal with this was to predict the winner of Wimbledon's gentlemen's singles tournament via a random forest model.

## Model Type:
I chose a random forest instead of a regression or neural network model for the following reasons.

### Regressions:
Regressions assume linearity, no multicollinearity, independence of observations, homoscedasticity of residuals, and a normal distribution for residuals. The ones we'll look at will be linearity, no multicollinearity, and independence of observations.
- Linearity: The random forest is superior because certain features simply aren't linear. For example, the 'ace_rate' importance plateaus, because the difference between a bad serve and a good serve is not the same as a good serve to a great serve. Along the same lines, features like 'grass_serve_quality', 'wimb_formula_dff', 'peak_rank', and some other features fail to be linear.
- No multicollinearity: Necesarrily, there will be multicollinearity. All features related to winning, like 'grass_win_rate', 'ytd_win_rate', and 'top10_win_rate' will be __, and in a similat vein, so will all the features related to serving. Because of this, a regression would fail with these correlated features.
- Independence of observations: A given match isn't independent from all other matches. Player's momentum or fatigue in a given tournament mean that the matches prior affect the next ones, meaning that every match can actually be thought of as dependent on the prior one, to an extent. Because of this, we can't assume independence of observations.

### Neural Networks:
Neural networks, though very powerful, have requirements and tradeoffs which aren't practical with predicting a tennis tournamet outcome. A neural network would require significant data and many more grass matches than actually exist, as grass is the shortest season for the ATP. Furthermore, all interpretability would be lost, meaning  it would be much harder to see if the model would be overfitting or not. 

### Random Forests:
Because of the clear limitations of regressions and neural networks, the random forest model would allow for non-linearity and multicorrelation while still being semi-interpretable. The parameter choices were as follows:
- 'n_estimators = 1000': By increasing the total number of decision trees, the variance of ___ decreases.
- 'max_depth = 15': With 11 features, more decisions need to be made, and so greater depth is enabled to pick up any deeper relationships.
- 'min_samples_leaf = 25': By forcing 25 historical samples to exist for a given decision, this works to regulate the 'max_depth' by ensuring there is backing behind every decision.
- 'criterion = "log_loss"': By using "log_loss" as opposed to "gini" or other criterion, the model penalizes extremes and overconfidence, thereby allowing for ___. 

## Features:
As an avid tennis fan, I decided to look at both the basic features, and some more nuanced ones as well. The features I chose incldue:
- 'rank_diff'
  - This is calculated via the log difference between ranks, because the difference from 1 to 50 is very different than 50 to 100 w.r.t. ranking.
- 'peak_rank'
- 'grass_win_rate'
  - ??
- 'peak_wimb_rate'
  - The best a player has ever done at Wimbledon.
- 'ytd_win_rate'
- 'top10_win_rate'
  - Player's who do better against the top 10 tend to have greater odds of upsets / going deep.
- 'ace_rate'
- 'first_serve_pct'
- 'second_serve_won_pct'
- 'grass_serve_quality'
  - Right now this only shows serve quality by taking the % of points won on serve. However, I look to add a speed metric to it, because speed is very important on the fast grass surface.
- 'wimb_formula_diff'
  - Wimbledon used to employ a formula to seed players using both ATP ranking and their own developed "grass ranking." Now it has been removed, but through this we can see whether it was valuable or not.

## Output:
???

## Limitations:
One of the bigger limitations I faced was in the data. Ideally, having data around the average speeds of forehands / backhands, the width / depth of serve, and spatial data like if points were won from the baseline or from dropshots would have helped immensely, I believe. That being said, such granular data is currently being generated, but will take time in order to have it ready for a tournament before the tournament starts. Some other limitations faced include ???

## Going Forward:
???

## Last Note:
Thank you to tennisabstract.com for generating the data I used for the majority of this project.
