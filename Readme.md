# DLS and Random Forest + XG Boost ML model for Kohli's performance in test cricket , ball-ball

## Overview
The **Virat Kohli Performance Analyzer** is a sophisticated machine learning application built using Streamlit that predicts the remaining runs for Virat Kohli based on various match conditions during Test matches. By leveraging advanced statistical methods and machine learning models, this application provides valuable insights into Kohli's performance, helping users understand the nuances of his batting under different circumstances.

## Data Extraction
We extracted a comprehensive dataset containing **839 Test match ball-by-ball data points**. This data was meticulously gathered to encapsulate all relevant information about each ball bowled, including runs scored, balls faced, wickets taken, and match context. The Python script written for this purpose processes the raw data to create a focused dataset on **Virat Kohli's** batting performance, specifically tailored to evaluate his effectiveness in Test cricket.

## Model Explanation

### Mathematical Foundation

1. **Duckworth-Lewis-Stern (DLS) Method**: 
   The DLS method is a complex statistical approach used in cricket to recalibrate the target score when rain or other interruptions affect a match. The methodology focuses on resource allocation based on the number of overs and wickets remaining.

   The primary equation used in DLS can be simplified as follows:
   
   $$ \[
   R = \text{Resources Left} \times \text{Par Score}
   \] $$

   - **Resources Left**: Calculated based on the current overs remaining and wickets in hand.
   - **Par Score**: The expected score at any given point in the match based on historical data.

   The formula for calculating resources can be derived as follows:

   $$ \[
   \text{Resources} = 100 - \left( \frac{Wickets \, Lost}{Total \, Wickets} \times 100 \right) - \left( \frac{Overs \, Bowled}{Total \, Overs} \times 100 \right)
   \] $$

2. **Random Forest Regressor**:
   The Random Forest model is an ensemble learning technique that constructs multiple decision trees during training and outputs the mean prediction of the individual trees. 

   The prediction can be expressed mathematically as follows:

   $$ \[
   \hat{Y} = \frac{1}{N} \sum_{i=1}^{N} T_i(X)
   \] $$

   - $$ \( \hat{Y} \) $$ : Predicted output.
   - $$ \( N \) $$: Number of trees.
   - $$ \( T_i(X) \) $$: Prediction from the \(i^{th}\) tree.

   The feature importance can also be derived using the mean decrease impurity (MDI):

   $$ \[
   \text{Importance}(j) = \sum_{t \in \mathcal{T}_j} \left( \frac{N_t}{N} \times \Delta \text{impurity}_t \right)
   \] $$

3. **XGBoost Regressor**:
   XGBoost is another powerful machine learning algorithm that employs gradient boosting. It aims to minimize the loss function \(L\) by combining weak learners. The optimization can be mathematically framed as follows:

   $$ \[
   L = \sum_{i=1}^{N} l(y_i, \hat{y}_i) + \Omega(f)
   \] $$

   - $$ \( y_i \) $$: Actual target value.
   - \( \hat{y}_i \): Predicted value.
   - \( \Omega(f) \): Regularization term to control complexity, given by:

   \[
   \Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
   \]

   Where:
   - \(T\) = number of leaves in the tree,
   - \(w_j\) = weight of the \(j^{th}\) leaf.

### Feature Engineering
To enhance the predictive capability of our models, we engineered several advanced features from the raw dataset:

- **Rolling Averages**: 
   \[
   \text{rolling\_avg}(n) = \frac{1}{n} \sum_{t=1}^{n} R_t
   \]
   Where \(R_t\) is the runs scored in the last \(n\) balls.

- **Dot Ball Pressure**:
   \[
   \text{dot\_ball\_pressure} = \frac{\text{Number of Dot Balls in Last 10 Balls}}{10}
   \]

- **Innings Progress**:
   \[
   \text{innings\_progress} = 1 - \frac{\text{balls remaining}}{90 \times 6}
   \]

### Machine Learning Implementation
The implemented machine learning framework consists of two main models:
1. **Random Forest Regressor** for initial predictions.
2. **XGBoost Regressor** to refine predictions based on the insights from the Random Forest model.

The final predicted score is a weighted combination of the predictions from both models:

\[
\text{Final Prediction} = 0.4 \times \text{RF Prediction} + 0.6 \times \text{XGBoost Prediction}
\]

## Conclusion
The **Virat Kohli Performance Analyzer** harnesses the power of machine learning and sophisticated statistical methods to provide predictions about Kohli's performance in real-time during Test matches. By utilizing a robust dataset and advanced feature engineering, our model can accurately assess various match conditions, offering insights that are valuable to cricket analysts, fans, and teams alike.

For further inquiries or suggestions, please feel free to reach out!
