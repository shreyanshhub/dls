import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Load and prepare the model
@st.cache_resource
def load_model_data():
    """Load the trained model and necessary data"""
    df = pd.read_csv('kohli_batting_data.csv')
    model = KohliDLSModel()
    df_processed = model.create_advanced_features(df)
    model.train_model(df_processed)
    return model, df_processed

class KohliDLSModel:
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        
    def create_advanced_features(self, df):
        """Create sophisticated features for the model"""
        # Basic rolling statistics
        df['rolling_avg_5'] = df.groupby('match_date')['runs_batter'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        df['rolling_strike_rate_10'] = (df.groupby('match_date')['runs_batter'].rolling(10, min_periods=1).sum() / 10 * 100).reset_index(0, drop=True)
        
        # Pressure metrics
        df['dot_ball_pressure'] = df.groupby('match_date')['runs_batter'].apply(
            lambda x: x.eq(0).rolling(10, min_periods=1).mean()
        ).reset_index(0, drop=True)
        
        # Batting phase indicators
        df['innings_progress'] = 1 - (df['balls_remaining'] / (90 * 6))
        df['resources_remaining'] = self.calculate_resources(
            df['balls_remaining'],
            df['wickets_in_hand']
        )
        
        return df
    
    def calculate_resources(self, balls_remaining, wickets):
        """Calculate resource percentage using exponential decay"""
        max_resources = 100
        overs_remaining = balls_remaining / 6
        wicket_factor = (wickets / 10) ** 1.5
        decay_rate = 0.02
        resources = max_resources * wicket_factor * np.exp(-decay_rate * (90 - overs_remaining))
        return resources

    def predict_score(self, situation_dict):
        """Predict score for given match situation"""
        df_situation = pd.DataFrame([situation_dict])
        features = self.prepare_features(df_situation)
        
        rf_pred = self.rf_model.predict(features)[0]
        xgb_pred = self.xgb_model.predict(features)[0]
        
        return 0.4 * rf_pred + 0.6 * xgb_pred

def main():
    st.set_page_config(
        page_title="Kohli DLS Predictor",
        page_icon="🏏",
        layout="wide"
    )

    st.title("🏏 Virat Kohli DLS-style Performance Predictor")
    st.markdown("""
    This app predicts Virat Kohli's expected performance based on match situations using a sophisticated DLS-style model.
    Enter the match situation below to get predictions and insights.
    """)

    try:
        model, historical_data = load_model_data()
    except FileNotFoundError:
        st.error("Error: Required data files not found. Please ensure 'kohli_batting_data.csv' exists in the current directory.")
        return

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Match Situation")
        
        # Basic match inputs
        innings = st.selectbox("Innings", [1, 2, 3, 4], index=0)
        overs_completed = st.slider("Overs Completed", 0, 89, 20)
        wickets_lost = st.slider("Wickets Lost", 0, 9, 2)
        current_score = st.number_input("Current Score", 0, 1000, 100)
        
        # Advanced inputs
        st.subheader("Advanced Parameters")
        bowling_team = st.selectbox(
            "Opposition Team",
            ["Australia", "England", "South Africa", "New Zealand", "Pakistan", "West Indies", "Sri Lanka", "Bangladesh"]
        )
        
        venue_type = st.selectbox(
            "Venue Type",
            ["Home", "Away", "Neutral"]
        )

    with col2:
        st.subheader("Recent Form")
        last_5_avg = st.slider("Last 5 Innings Average", 0.0, 100.0, 45.0)
        current_strike_rate = st.slider("Current Strike Rate", 0.0, 200.0, 65.0)
        
        st.subheader("Match Context")
        pressure_situation = st.select_slider(
            "Pressure Situation",
            options=["Low", "Medium", "High"],
            value="Medium"
        )
        
        pitch_condition = st.select_slider(
            "Pitch Condition",
            options=["Batting Friendly", "Neutral", "Bowling Friendly"],
            value="Neutral"
        )

    # Create prediction button
    if st.button("Predict Performance"):
        # Prepare situation dictionary
        situation = {
            'balls_remaining': (90 - overs_completed) * 6,
            'wickets_in_hand': 10 - wickets_lost,
            'resources_remaining': model.calculate_resources((90 - overs_completed) * 6, 10 - wickets_lost),
            'innings_progress': overs_completed / 90,
            'rolling_avg_5': last_5_avg,
            'rolling_strike_rate_10': current_strike_rate,
            'dot_ball_pressure': 0.4 if pressure_situation == "High" else 0.3 if pressure_situation == "Medium" else 0.2
        }

        # Get prediction
        predicted_runs = model.predict_score(situation)

        # Display results
        st.header("Prediction Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Predicted Runs",
                value=f"{predicted_runs:.0f}",
                delta=f"{predicted_runs - current_score:.0f} more runs"
            )

        with col2:
            st.metric(
                label="Predicted Strike Rate",
                value=f"{(predicted_runs / ((90 - overs_completed) * 6)) * 100:.1f}",
                delta=None
            )

        with col3:
            resources = model.calculate_resources((90 - overs_completed) * 6, 10 - wickets_lost)
            st.metric(
                label="Resources Remaining",
                value=f"{resources:.1f}%",
                delta=None
            )

        # Create visualizations
        st.subheader("Performance Projections")
        
        # Create projection chart
        remaining_overs = np.arange(overs_completed, 90)
        projected_scores = [current_score + (i-overs_completed) * (predicted_runs-current_score)/(90-overs_completed) for i in remaining_overs]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=remaining_overs,
            y=projected_scores,
            mode='lines',
            name='Projected Score',
            line=dict(color='royalblue')
        ))
        
        fig.update_layout(
            title='Projected Score Progression',
            xaxis_title='Overs',
            yaxis_title='Score',
            hovermode='x'
        )
        
        st.plotly_chart(fig)

        # Historical comparison
        st.subheader("Historical Comparison")
        similar_situations = historical_data[
            (historical_data['innings'] == innings) &
            (abs(historical_data['balls_remaining'] - situation['balls_remaining']) < 30)
        ]
        
        if not similar_situations.empty:
            avg_historical = similar_situations['runs_batter'].mean()
            st.write(f"In similar situations, Kohli has averaged {avg_historical:.1f} runs")
        
            # Create historical distribution plot
            fig = px.histogram(
                similar_situations,
                x='runs_batter',
                title='Distribution of Runs in Similar Situations',
                nbins=20
            )
            fig.add_vline(x=predicted_runs, line_dash="dash", line_color="red", annotation_text="Predicted")
            st.plotly_chart(fig)
        else:
            st.write("Not enough historical data for similar situations")

if __name__ == "__main__":
    main()
