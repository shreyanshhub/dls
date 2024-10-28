import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Cache data and model loading to speed up performance
@st.cache_data
def load_data():
    return pd.read_csv('kohli_batting_data.csv')

@st.cache_resource
def initialize_model():
    return KohliDLSModel()

class KohliDLSModel:
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()

    def create_advanced_features(self, df):
        df['rolling_avg_5'] = df.groupby('batter')['runs_batter'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        df['rolling_strike_rate_10'] = (df.groupby('batter')['runs_batter'].rolling(10, min_periods=1).sum() / 10 * 100).reset_index(0, drop=True)
        df['dot_ball_pressure'] = df.groupby('batter')['runs_batter'].apply(lambda x: x.eq(0).rolling(10, min_periods=1).mean()).reset_index(0, drop=True)
        df['innings_progress'] = 1 - (df['balls_remaining'] / (90 * 6))
        df['resources_remaining'] = self.calculate_resources(df['balls_remaining'], df['wickets_in_hand'])
        
        bowler_stats = df.groupby('bowler').agg({
            'runs_batter': ['mean', 'std'],
            'wicket': 'mean'
        }).reset_index()
        bowler_stats.columns = ['bowler', 'avg_runs_conceded', 'std_runs', 'wicket_rate']
        df = df.merge(bowler_stats, on='bowler', how='left')

        venue_stats = df.groupby('venue').agg({
            'runs_batter': 'mean',
            'strike_rate': 'mean'
        }).reset_index()
        df = df.merge(venue_stats, on='venue', how='left', suffixes=('', '_venue_avg'))

        team_stats = df.groupby('bowling_team').agg({
            'runs_batter': 'mean',
            'wicket': 'mean'
        }).reset_index()
        df = df.merge(team_stats, on='bowling_team', how='left', suffixes=('', '_team_avg'))

        return df

    def calculate_resources(self, balls_remaining, wickets):
        max_resources = 100
        overs_remaining = balls_remaining / 6
        wicket_factor = (wickets / 10) ** 1.5
        decay_rate = 0.02
        resources = max_resources * wicket_factor * np.exp(-decay_rate * (90 - overs_remaining))
        return resources

    def prepare_features(self, df):
        feature_cols = [
            'balls_remaining', 'wickets_in_hand', 'resources_remaining',
            'innings_progress', 'rolling_avg_5', 'rolling_strike_rate_10',
            'dot_ball_pressure', 'avg_runs_conceded', 'wicket_rate'
        ]
        return self.scaler.transform(df[feature_cols])

    def train_model(self, df):
        df['runs_remaining'] = df.groupby('match_date')['runs_batter'].transform(lambda x: x.iloc[::-1].cumsum().iloc[::-1])
        features = df[['balls_remaining', 'wickets_in_hand', 'resources_remaining', 'innings_progress', 
                       'rolling_avg_5', 'rolling_strike_rate_10', 'dot_ball_pressure', 
                       'avg_runs_conceded', 'wicket_rate']]
        
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        target = df['runs_remaining']

        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

        self.rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
        self.rf_model.fit(X_train, y_train)

        self.xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42)
        self.xgb_model.fit(X_train, y_train)

    def calculate_par_score(self, df):
        features = self.prepare_features(df)
        rf_pred = self.rf_model.predict(features)
        xgb_pred = self.xgb_model.predict(features)
        return 0.4 * rf_pred + 0.6 * xgb_pred

    def calculate_match_impact(self, df):
        df['expected_runs'] = self.calculate_par_score(df)
        df['runs_impact'] = df['runs_batter'] - (df['expected_runs'] / df['balls_remaining'])
        df['pressure_impact'] = df['runs_impact'] * df['resources_remaining'] / 100
        return df

def main():
    df = load_data()
    model = initialize_model()
    df_processed = model.create_advanced_features(df)
    model.train_model(df_processed)
    df_with_impact = model.calculate_match_impact(df_processed)
    return model, df_with_impact

model, results = main()

# Streamlit Interface
st.title("Virat Kohli Performance Analyzer")

# Sidebar for filtering
st.sidebar.header("Filter Options")
venue = st.sidebar.selectbox("Select Venue", results['venue'].unique())
filtered_data = results[results['venue'] == venue]

opposition = st.sidebar.selectbox("Select Opposing Team", filtered_data['bowling_team'].unique())
filtered_data = filtered_data[filtered_data['bowling_team'] == opposition]

match_date = st.sidebar.selectbox("Select Match Date", filtered_data['match_date'].unique())
filtered_data = filtered_data[filtered_data['match_date'] == match_date]

st.subheader(f"Performance Summary for {match_date} at {venue} against {opposition}")
st.write(filtered_data[['batter', 'runs_batter', 'runs_remaining', 'expected_runs', 'runs_impact', 'pressure_impact']])

# Explanation of columns
st.markdown("### Column Descriptions")
st.write("""
- **Batter**: Name of the batter.
- **Runs Scored**: Total runs scored by the batter in the match.
- **Runs Remaining**: Remaining runs as per DLS , refer down in python file kohli.py on Github
- **Expected Runs**: Predicted runs based on current match situation.
- **Runs Impact**: Difference between actual runs scored and expected runs.
- **Pressure Impact**: Indicates how runs scored relate to remaining resources in the match.
""")

# Explanation Section for Models
st.markdown("### Model Explanation")
st.markdown("""
#### DLS Model Mathematics
The **DLS** (Duckworth-Lewis-Stern) formula adapts to predict remaining runs based on:
- Remaining balls
- Wickets in hand
- Adjusted resource ratios
The DLS method is a complex statistical approach used in cricket to recalibrate the target score when rain or other interruptions affect a match. The methodology focuses on resource allocation based on the number of overs and wickets remaining.

The primary equation used in DLS can be simplified as follows:

$$
R = \\text{Resources Left} \\times \\text{Par Score}
$$

Where:
- **Resources Left**: Calculated based on the current overs remaining and wickets in hand.
- **Par Score**: The expected score at any given point in the match based on historical data.

The formula for calculating resources can be derived as follows:

$$
\\text{Resources} = 100 - \\left( \\frac{Wickets \\, Lost}{Total \\, Wickets} \\times 100 \\right) - \\left( \\frac{Overs \\, Bowled}{Total \\, Overs} \\times 100 \\right)
$$


#### Random Forest
- An ensemble method using multiple decision trees.
- Each tree predicts an outcome, and the final prediction is the average (for regression).
The prediction can be expressed mathematically as follows:

$$
\\hat{Y} = \\frac{1}{N} \\sum_{i=1}^{N} T_i(X)
$$

Where:
- \( \\hat{Y} \): Predicted output.
- \( N \): Number of trees.
- \( T_i(X) \): Prediction from the \(i^{th}\\) tree.

The feature importance can also be derived using the mean decrease impurity (MDI):

$$
\\text{Importance}(j) = \\sum_{t \\in \\mathcal{T}_j} \\left( \\frac{N_t}{N} \\times \\Delta \\text{impurity}_t \\right)
$$

#### XGBoost
- A gradient boosting Machine Learning framework that optimizes model performance.
- Incorporates regularization to prevent overfitting.
- **Key parameters**:
  - **n_estimators**: Number of trees.
  - **max_depth**: Limits tree depth.
  - **learning_rate**: Step size for model updates.
$$
L = \\sum_{i=1}^{N} l(y_i, \\hat{y}_i) + \\Omega(f)
$$

Where:
- \( y_i \): Actual target value.
- \( \\hat{y}_i \): Predicted value.
- \( \\Omega(f) \): Regularization term to control complexity, given by:

$$
\\Omega(f) = \\gamma T + \\frac{1}{2} \\lambda \\sum_{j=1}^{T} w_j^2
$$

Where:
- \(T\) = number of leaves in the tree,
- \(w_j\) = weight of the \(j^{th}\\) leaf.

### Pipeline
- Data preprocessed using `StandardScaler`.
- Models trained on split data.
- Predictions are combined using weighted average.
""")

# Graphical Analysis
st.subheader("Graphical Analysis")
st.markdown("### Run Analysis Over Time")
st.line_chart(data=filtered_data[['runs_impact', 'pressure_impact']], use_container_width=True)
st.line_chart(data=filtered_data[['rolling_avg_5']], use_container_width=True)
st.line_chart(data=filtered_data[['innings_progress']], use_container_width=True)
st.line_chart(data=filtered_data[['runs_remaining']], use_container_width=True)
st.line_chart(data=filtered_data[['strike_rate']], use_container_width=True)

# Label axes for clarity
st.write("""
### Y-Axis:
- `runs_impact` & `pressure_impact`: Deviation from expected score.
- `rolling_avg_5`: Five-over rolling average of runs scored.
- `innings_progress`: Match's progress proportion.
- `runs_remaining`: Cumulative runs remaining.
- `strike_rate`: Batting strike rate.

### X-Axis:
The x-axis represents the ball-by-ball sequence of events throughout the match.
""")

if __name__ == '__main__':
    main()

