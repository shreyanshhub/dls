import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

class KohliAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.data = None
        self.processed_data = None
        
    def load_data(self, file_path='kohli_batting_data.csv'):
        """Load and preprocess Kohli's batting data"""
        self.data = pd.read_csv(file_path)
        self.data['match_date'] = pd.to_datetime(self.data['match_date'])
        return self.data
    
    def process_data(self):
        """Create features from raw data"""
        df = self.data.copy()
        
        # Calculate basic stats per match
        match_stats = df.groupby(['match_date', 'venue', 'innings', 'batting_team', 'bowling_team']).agg({
            'runs_batter': 'sum',
            'balls_faced': 'max',
            'balls_remaining': 'first',
            'wickets_in_hand': 'first'
        }).reset_index()
        
        # Calculate rolling averages
        match_stats = match_stats.sort_values('match_date')
        match_stats['rolling_avg_5'] = match_stats['runs_batter'].rolling(5, min_periods=1).mean()
        match_stats['rolling_sr'] = (match_stats['runs_batter'] / match_stats['balls_faced'] * 100).rolling(5, min_periods=1).mean()
        
        # Calculate venue and opposition stats
        venue_stats = match_stats.groupby('venue')['runs_batter'].agg(['mean', 'count']).reset_index()
        opposition_stats = match_stats.groupby('bowling_team')['runs_batter'].agg(['mean', 'count']).reset_index()
        
        self.processed_data = match_stats
        self.venue_stats = venue_stats
        self.opposition_stats = opposition_stats
        
        return match_stats
        
    def get_recent_form(self, match_date):
        """Get performance metrics from recent matches"""
        recent_matches = self.processed_data[self.processed_data['match_date'] < match_date].tail(3)
        return {
            'avg_score': recent_matches['runs_batter'].mean(),
            'avg_sr': recent_matches['rolling_sr'].mean(),
            'scores': recent_matches['runs_batter'].tolist()
        }
    
    def train_model(self):
        """Train prediction model"""
        df = self.processed_data.copy()
        
        features = ['balls_remaining', 'wickets_in_hand', 'rolling_avg_5', 'rolling_sr']
        target = 'runs_batter'
        
        # Scale features
        self.scaler.fit(df[features])
        X_scaled = self.scaler.transform(df[features])
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_scaled, df[target])
    
    def predict_score(self, situation_dict):
        """Predict score for given match situation"""
        features = ['balls_remaining', 'wickets_in_hand', 'rolling_avg_5', 'rolling_sr']
        df_situation = pd.DataFrame([situation_dict])
        
        X_scaled = self.scaler.transform(df_situation[features])
        prediction = self.model.predict(X_scaled)[0]
        
        return prediction

def main():
    st.set_page_config(page_title="Kohli Performance Analyzer", page_icon="🏏", layout="wide")
    
    st.title("🏏 Virat Kohli Performance Analyzer")
    st.markdown("""
    This application analyzes Virat Kohli's batting performance using historical data and predicts
    potential outcomes based on match situations.
    """)
    
    try:
        # Initialize analyzer and load data
        analyzer = KohliAnalyzer()
        raw_data = analyzer.load_data()
        processed_data = analyzer.process_data()
        analyzer.train_model()
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Performance Analysis", "Match Predictor", "Historical Insights"])
        
        with tab1:
            st.subheader("Overall Performance Metrics")
            
            # Calculate career stats
            career_stats = {
                'matches': len(processed_data),
                'total_runs': processed_data['runs_batter'].sum(),
                'average': processed_data['runs_batter'].mean(),
                'strike_rate': (processed_data['runs_batter'].sum() / processed_data['balls_faced'].sum() * 100)
            }
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Matches", f"{career_stats['matches']}")
            col2.metric("Total Runs", f"{career_stats['total_runs']:.0f}")
            col3.metric("Average", f"{career_stats['average']:.2f}")
            col4.metric("Strike Rate", f"{career_stats['strike_rate']:.2f}")
            
            # Performance trends
            st.subheader("Performance Trends")
            fig_trends = go.Figure()
            
            # Running average
            fig_trends.add_trace(go.Scatter(
                x=processed_data['match_date'],
                y=processed_data['rolling_avg_5'],
                name='5-Match Average',
                line=dict(color='blue')
            ))
            
            # Strike rate trend
            fig_trends.add_trace(go.Scatter(
                x=processed_data['match_date'],
                y=processed_data['rolling_sr'],
                name='Strike Rate Trend',
                line=dict(color='red')
            ))
            
            fig_trends.update_layout(
                title='Performance Trends Over Time',
                xaxis_title='Date',
                yaxis_title='Runs/Strike Rate',
                hovermode='x unified'
            )
            st.plotly_chart(fig_trends)
            
            # Venue Analysis
            st.subheader("Venue Performance")
            fig_venue = px.bar(
                analyzer.venue_stats,
                x='venue',
                y='mean',
                color='count',
                title='Average Score by Venue',
                labels={'mean': 'Average Score', 'count': 'Matches Played'}
            )
            st.plotly_chart(fig_venue)
            
        with tab2:
            st.subheader("Match Situation Predictor")
            
            col1, col2 = st.columns(2)
            
            with col1:
                overs_completed = st.slider("Overs Completed", 0, 49, 20)
                wickets_lost = st.slider("Wickets Lost", 0, 9, 2)
                current_score = st.number_input("Current Score", 0, 1000, 50)
                
                # Get user's last three scores for form calculation
                st.subheader("Recent Form")
                score1 = st.number_input("Last Innings Score", 0, 200, 45)
                score2 = st.number_input("Second Last Innings Score", 0, 200, 32)
                score3 = st.number_input("Third Last Innings Score", 0, 200, 67)
                
            with col2:
                bowling_team = st.selectbox(
                    "Opposition Team",
                    sorted(processed_data['bowling_team'].unique())
                )
                venue = st.selectbox(
                    "Venue",
                    sorted(processed_data['venue'].unique())
                )
            
            if st.button("Generate Prediction"):
                # Prepare prediction inputs
                recent_avg = (score1 + score2 + score3) / 3
                recent_sr = 85  # Default strike rate if not available
                
                situation = {
                    'balls_remaining': (50 - overs_completed) * 6,
                    'wickets_in_hand': 10 - wickets_lost,
                    'rolling_avg_5': recent_avg,
                    'rolling_sr': recent_sr
                }
                
                # Get prediction
                predicted_score = analyzer.predict_score(situation)
                
                # Display prediction results
                st.subheader("Prediction Results")
                col1, col2, col3 = st.columns(3)
                
                col1.metric(
                    "Predicted Final Score",
                    f"{predicted_score:.0f}",
                    f"{predicted_score - current_score:.0f} more runs"
                )
                
                projected_sr = (predicted_score - current_score) / ((50 - overs_completed) * 6) * 100
                col2.metric("Required Strike Rate", f"{projected_sr:.1f}")
                
                # Opposition analysis
                opp_stats = analyzer.opposition_stats[analyzer.opposition_stats['bowling_team'] == bowling_team].iloc[0]
                col3.metric(
                    f"Average vs {bowling_team}",
                    f"{opp_stats['mean']:.1f}",
                    f"{opp_stats['mean'] - career_stats['average']:.1f} vs career average"
                )
                
                # Projection chart
                fig_proj = go.Figure()
                remaining_overs = np.arange(overs_completed, 50)
                projected_scores = [current_score + (i-overs_completed) * (predicted_score-current_score)/(50-overs_completed) for i in remaining_overs]
                
                fig_proj.add_trace(go.Scatter(
                    x=remaining_overs,
                    y=projected_scores,
                    mode='lines',
                    name='Projected Score'
                ))
                
                fig_proj.update_layout(
                    title='Projected Score Progression',
                    xaxis_title='Overs',
                    yaxis_title='Score'
                )
                st.plotly_chart(fig_proj)
                
        with tab3:
            st.subheader("Historical Performance Analysis")
            
            # Score distribution
            fig_dist = px.histogram(
                processed_data,
                x='runs_batter',
                title='Score Distribution',
                nbins=30
            )
            st.plotly_chart(fig_dist)
            
            # Opposition analysis
            fig_opp = px.bar(
                analyzer.opposition_stats,
                x='bowling_team',
                y='mean',
                color='count',
                title='Performance Against Different Teams',
                labels={'mean': 'Average Score', 'count': 'Matches Played'}
            )
            st.plotly_chart(fig_opp)
            
            # Innings progression
            avg_progression = raw_data.groupby('over')['runs_batter'].mean().reset_index()
            fig_prog = px.line(
                avg_progression,
                x='over',
                y='runs_batter',
                title='Average Runs per Over'
            )
            st.plotly_chart(fig_prog)
            
    except FileNotFoundError:
        st.error("Error: Could not find kohli_batting_data.csv. Please ensure the file is in the correct location.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
