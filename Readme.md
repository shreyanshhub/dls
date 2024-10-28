# DLS and Random Forest + XG Boost ML model for Kohli's performance in test cricket , ball-ball

## Overview
This is a sophisticated machine learning application built using Streamlit that predicts the remaining runs for Virat Kohli based on various match conditions during Test matches. By leveraging advanced statistical methods and machine learning models, this application provides valuable insights into Kohli's performance, helping users understand the nuances of his batting under different circumstances.

## Data Extraction
We extracted a comprehensive dataset containing **839 Test match ball-by-ball data points**. This data was meticulously gathered to encapsulate all relevant information about each ball bowled, including runs scored, balls faced, wickets taken, and match context. The Python script written for this purpose processes the raw data to create a focused dataset on **Virat Kohli's** batting performance, specifically tailored to evaluate his effectiveness in Test cricket.

## Model Explanation

### Mathematical Foundation

1. **Duckworth-Lewis-Stern (DLS) Method**: 
   The DLS method is a complex statistical approach used in cricket to recalibrate the target score when rain or other interruptions affect a match. The methodology focuses on resource allocation based on the number of overs and wickets remaining.

   The primary equation used in DLS can be simplified as follows:
   
   ```
   R = Resources Left × Par Score
   ```

   - **Resources Left**: Calculated based on the current overs remaining and wickets in hand.
   - **Par Score**: The expected score at any given point in the match based on historical data.

   The formula for calculating resources can be derived as follows:

   ```
   Resources = 100 - (Wickets Lost / Total Wickets × 100) - (Overs Bowled / Total Overs × 100)
   ```

2. **Random Forest Regressor**:
   The Random Forest model is an ensemble learning technique that constructs multiple decision trees during training and outputs the mean prediction of the individual trees. 

   The prediction can be expressed mathematically as follows:

   ```
   Ŷ = (1/N) ∑(i=1 to N) Ti(X)
   ```

   Where:
   - Ŷ: Predicted output
   - N: Number of trees
   - Ti(X): Prediction from the i-th tree

   The feature importance can also be derived using the mean decrease impurity (MDI):

   ```
   Importance(j) = ∑(t ∈ Tj) (Nt/N × Δimpurityt)
   ```

3. **XGBoost Regressor**:
   XGBoost is another powerful machine learning algorithm that employs gradient boosting. It aims to minimize the loss function L by combining weak learners. The optimization can be mathematically framed as follows:

   ```
   L = ∑(i=1 to N) l(yi, ŷi) + Ω(f)
   ```

   Where:
   - yi: Actual target value
   - ŷi: Predicted value
   - Ω(f): Regularization term to control complexity, given by:

   ```
   Ω(f) = γT + (1/2)λ ∑(j=1 to T) wj²
   ```

   Where:
   - T = number of leaves in the tree
   - wj = weight of the j-th leaf

### Feature Engineering
To enhance the predictive capability of our models, we engineered several advanced features from the raw dataset:

- **Rolling Averages**: 
   ```
   rolling_avg(n) = (1/n) ∑(t=1 to n) Rt
   ```
   Where Rt is the runs scored in the last n balls.

- **Dot Ball Pressure**:
   ```
   dot_ball_pressure = Number of Dot Balls in Last 10 Balls / 10
   ```

- **Innings Progress**:
   ```
   innings_progress = 1 - (balls remaining / (90 × 6))
   ```

### Machine Learning Implementation
The implemented machine learning framework consists of two main models:
1. **Random Forest Regressor** for initial predictions.
2. **XGBoost Regressor** to refine predictions based on the insights from the Random Forest model.

The final predicted score is a weighted combination of the predictions from both models:

```
Final Prediction = 0.4 × RF Prediction + 0.6 × XGBoost Prediction
```

## Conclusion
This harnesses the power of machine learning and sophisticated statistical methods to provide predictions about Kohli's performance in real-time during Test matches. By utilizing a robust dataset and advanced feature engineering, our model can accurately assess various match conditions, offering insights that are valuable to cricket analysts, fans, and teams alike.

For further inquiries or suggestions, please feel free to reach out!
