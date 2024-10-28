import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split

class KohliDLSModel:
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()

    def create_advanced_features(self, df):
        """Create sophisticated features for the model"""
        df['rolling_avg_5'] = df.groupby('match_date')['runs_batter'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        df['rolling_strike_rate_10'] = (df.groupby('match_date')['runs_batter'].rolling(10, min_periods=1).sum() / 10 * 100).reset_index(0, drop=True)

        df['dot_ball_pressure'] = df.groupby('match_date')['runs_batter'].apply(
            lambda x: x.eq(0).rolling(10, min_periods=1).mean()
        ).reset_index(0, drop=True)

        df['innings_progress'] = 1 - (df['balls_remaining'] / (90 * 6))
        df['resources_remaining'] = self.calculate_resources(df['balls_remaining'], df['wickets_in_hand'])

        bowler_stats = df.groupby('bowler').agg({
            'runs_batter': ['mean', 'std'],
            'wicket': 'mean'
        }).reset_index()
        bowler_stats.columns = ['bowler', 'avg_runs_conceded', 'std_runs', 'wicket_rate']
        df = df.merge(bowler_stats, on='bowler', how='left')

        venue_stats = df.groupby('venue').agg({
            'runs_batter': 'mean'
        }).reset_index()
        venue_stats.columns = ['venue', 'venue_avg_runs']
        df = df.merge(venue_stats, on='venue', how='left')

        team_stats = df.groupby('bowling_team').agg({
            'runs_batter': 'mean',
            'wicket': 'mean'
        }).reset_index()
        team_stats.columns = ['bowling_team', 'team_avg_runs', 'team_wicket_rate']
        df = df.merge(team_stats, on='bowling_team', how='left')

        return df

    def calculate_resources(self, balls_remaining, wickets):
        """Calculate resource percentage using exponential decay"""
        max_resources = 100
        overs_remaining = balls_remaining / 6

        # Parameters for the exponential decay function
        wicket_factor = (wickets / 10) ** 1.5
        decay_rate = 0.02

        resources = max_resources * wicket_factor * np.exp(-decay_rate * (90 - overs_remaining))
        return resources

    def prepare_features(self, df):
        """Prepare features for model input"""
        feature_cols = [
            'balls_remaining', 'wickets_in_hand', 'resources_remaining',
            'innings_progress', 'rolling_avg_5', 'rolling_strike_rate_10',
            'dot_ball_pressure', 'avg_runs_conceded', 'wicket_rate',
            'venue_avg_runs', 'team_avg_runs'
        ]
        
        return self.scaler.transform(df[feature_cols])

    def train_model(self, df):
        """Train the DLS-like model"""
        df['runs_remaining'] = df.groupby('match_date')['runs_batter'].transform(
            lambda x: x.iloc[::-1].cumsum().iloc[::-1]
        )

        features = self.prepare_features(df)
        target = df['runs_remaining']

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        self.scaler.fit(X_train)  # Fit the scaler here
        features = self.scaler.transform(X_train)  # Transform the training features

        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.rf_model.fit(features, y_train)

        self.xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            random_state=42
        )
        self.xgb_model.fit(features, y_train)
    
    def calculate_par_score(self, df_situation):
        """Calculate par score for given match situation"""
        if self.rf_model is None or self.xgb_model is None:
            raise ValueError("Models not trained yet")

        features = self.prepare_features(df_situation)

        rf_pred = self.rf_model.predict(features)
        xgb_pred = self.xgb_model.predict(features)

        par_score = 0.4 * rf_pred + 0.6 * xgb_pred

        return par_score

    def calculate_match_impact(self, df):
        """Calculate match impact score for each ball"""
        df['expected_runs'] = self.calculate_par_score(df)
        df['runs_impact'] = df['runs_batter'] - (df['expected_runs'] / df['balls_remaining'])
        df['pressure_impact'] = df['runs_impact'] * df['resources_remaining'] / 100

        return df

def main():
    # Load your data
    df = pd.read_csv('kohli_batting_data.csv')

    # Initialize and train model
    model = KohliDLSModel()

    # Create advanced features
    df_processed = model.create_advanced_features(df)

    # Train the model
    model.train_model(df_processed)

    # Calculate match impact
    df_with_impact = model.calculate_match_impact(df_processed)

    # Save results
    df_with_impact.to_csv('kohli_dls_analysis.csv', index=False)

    return model, df_with_impact

# Streamlit interface
def run_streamlit():
    st.title("Virat Kohli DLS Analysis")

    # Load data
    df = pd.read_csv('kohli_batting_data.csv')
    model = KohliDLSModel()
    df_processed = model.create_advanced_features(df)
    model.train_model(df_processed)

    st.sidebar.header("Match Situation")
    balls_remaining = st.sidebar.number_input("Balls Remaining", min_value=0)
    wickets_in_hand = st.sidebar.number_input("Wickets in Hand", min_value=0, max_value=10)
    innings_progress = st.sidebar.number_input("Innings Progress (0-1)", min_value=0.0, max_value=1.0)
    
    # Create a DataFrame for the input situation
    input_df = pd.DataFrame({
        'balls_remaining': [balls_remaining],
        'wickets_in_hand': [wickets_in_hand],
        'innings_progress': [innings_progress]
    })

    # Process features
    input_df_processed = model.create_advanced_features(input_df)

    if st.sidebar.button("Calculate Par Score"):
        par_score = model.calculate_par_score(input_df_processed)
        st.write(f"Calculated Par Score: {par_score[0]}")

    # Display the processed DataFrame
    st.subheader("Processed Data")
    st.write(df_processed)

if __name__ == "__main__":
    # Run the Streamlit app
    run_streamlit()
