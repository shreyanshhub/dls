import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import streamlit as st

def load_data():
    df = pd.read_csv('kohli_batting_data.csv')
    return df

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
        features = df[['balls_remaining', 'wickets_in_hand', 'resources_remaining', 'innings_progress', 'rolling_avg_5', 'rolling_strike_rate_10', 'dot_ball_pressure', 'avg_runs_conceded', 'wicket_rate']]
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
    model = KohliDLSModel()
    df_processed = model.create_advanced_features(df)
    model.train_model(df_processed)
    df_with_impact = model.calculate_match_impact(df_processed)
    return model, df_with_impact

model, results = main()

# Streamlit Interface
st.title("Virat Kohli Performance Analyzer")

st.sidebar.header("Filter Options")
venue = st.sidebar.selectbox("Select Venue", results['venue'].unique())
filtered_data = results[results['venue'] == venue]

opposition = st.sidebar.selectbox("Select Opposing Team", filtered_data['bowling_team'].unique())
filtered_data = filtered_data[filtered_data['bowling_team'] == opposition]

match_date = st.sidebar.selectbox("Select Match Date", filtered_data['match_date'].unique())
filtered_data = filtered_data[filtered_data['match_date'] == match_date]

st.subheader(f"Performance Summary for {match_date} at {venue} against {opposition}")
st.write(filtered_data[['batter', 'runs_batter', 'runs_remaining', 'expected_runs', 'runs_impact', 'pressure_impact']])

# Explanation Section
st.markdown("### Model Explanation")
st.markdown("""
- **DLS Model Mathematics**: The **DLS** (Duckworth-Lewis-Stern) formula was adapted to predict remaining runs using remaining balls, wickets, and adjusted resource ratios.
  - Resources are calculated based on balls remaining and wickets in hand, then used to estimate match scenarios.
- **XGBoost Model**:
  - `n_estimators`: Number of trees in the model.
  - `max_depth`: Limits tree depth to avoid overfitting.
  - `learning_rate`: Controls step size during learning.
- **Pipeline**:
  - Data preprocessed with `StandardScaler` for feature scaling.
  - Models are trained on split data, and predictions are combined using a weighted average.
""")

# Graphical Analysis
st.subheader("Graphical Analysis")

st.line_chart(data=filtered_data[['runs_impact', 'pressure_impact']], use_container_width=True)
st.line_chart(data=filtered_data[['rolling_avg_5']], y="rolling_avg_5", use_container_width=True)
st.line_chart(data=filtered_data[['innings_progress']], y="innings_progress", use_container_width=True)
st.line_chart(data=filtered_data[['runs_remaining']], y="runs_remaining", use_container_width=True)
st.line_chart(data=filtered_data[['strike_rate']], y="strike_rate", use_container_width=True)

st.write("""
### Y-Axis:
- `runs_impact` & `pressure_impact`: Measures the deviation from the expected score.
- `rolling_avg_5`: Five-overs rolling average of runs scored.
- `innings_progress`: Shows the match’s progress as a proportion.
- `runs_remaining`: Cumulative runs remaining.
- `strike_rate`: Strike rate of the batter.

### X-Axis:
The x-axis across graphs represents the ball-by-ball sequence of events throughout the match.
""")

