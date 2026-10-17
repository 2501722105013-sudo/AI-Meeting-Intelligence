"""
EDA (Exploratory Data Analysis) Module for AI Meeting Intelligence.
Analyzes synthetic/stored meeting datasets and produces interactive Plotly & Matplotlib visualizations.
Key Visualizations:
1. Meeting Duration Distribution
2. Speaker Contribution Breakdown
3. Topic Frequency Analysis
4. Decision Count Trends
5. Task Completion & Priority Analysis
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import os
import json
from typing import Dict, Any, Tuple

class MeetingDatasetEDA:
    def __init__(self, dataset_path=None):
        if dataset_path is None:
            dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "full_dataset.csv")
        self.dataset_path = dataset_path
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        if os.path.exists(self.dataset_path):
            try:
                return pd.read_csv(self.dataset_path)
            except Exception:
                pass
        
        # Mock dataframe fallback if CSV not pre-generated
        data = []
        depts = ["Software Development", "Project Planning", "Marketing", "HR", "Finance", "Product Management"]
        for i in range(100):
            dept = depts[i % len(depts)]
            data.append({
                "meeting_id": f"MTG_{i:04d}",
                "date": f"2025-{(i%12)+1:02d}-15",
                "department": dept,
                "duration_minutes": np.random.randint(20, 60),
                "speaker_count": np.random.randint(3, 6),
                "sentence_count": np.random.randint(15, 45),
                "decision_count": np.random.randint(1, 5),
                "task_count": np.random.randint(1, 6),
                "primary_topic": f"{dept} Sync"
            })
        return pd.DataFrame(data)

    def plot_duration_distribution(self) -> go.Figure:
        """1. Meeting Duration Analysis Histogram & Box Plot."""
        fig = px.histogram(
            self.df, 
            x="duration_minutes", 
            color="department",
            marginal="box",
            title="Meeting Duration Distribution Across Departments (Minutes)",
            labels={"duration_minutes": "Duration (min)", "department": "Department"},
            color_discrete_sequence=px.colors.qualitative.Blues
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Inter, sans-serif", size=13),
            title_font=dict(size=18, color="#1E3A8A")
        )
        return fig

    def plot_speaker_contribution(self) -> go.Figure:
        """2. Speaker Contribution & Sentence Count Analysis."""
        avg_speakers = self.df.groupby("department")[["speaker_count", "sentence_count"]].mean().reset_index()
        fig = px.bar(
            avg_speakers,
            x="department",
            y=["speaker_count", "sentence_count"],
            barmode="group",
            title="Average Speaker Count & Conversation Sentences by Department",
            labels={"value": "Count", "department": "Department", "variable": "Metric"},
            color_discrete_sequence=["#1E3A8A", "#2563EB"]
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Inter, sans-serif", size=13),
            title_font=dict(size=18, color="#1E3A8A")
        )
        return fig

    def plot_topic_frequency(self) -> go.Figure:
        """3. Topic Frequency Analysis."""
        topic_counts = self.df["primary_topic"].value_counts().reset_index()
        topic_counts.columns = ["Topic", "Count"]
        
        fig = px.pie(
            topic_counts.head(10),
            names="Topic",
            values="Count",
            title="Top 10 Most Frequently Discussed Meeting Topics",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            hole=0.4
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Inter, sans-serif", size=13),
            title_font=dict(size=18, color="#1E3A8A")
        )
        return fig

    def plot_decision_trends(self) -> go.Figure:
        """4. Decision Count Trends by Department."""
        dept_decisions = self.df.groupby("department")["decision_count"].sum().reset_index()
        fig = px.bar(
            dept_decisions,
            x="department",
            y="decision_count",
            color="decision_count",
            title="Total Decisions Extracted by Department",
            labels={"decision_count": "Total Decisions", "department": "Department"},
            color_continuous_scale="Blues"
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Inter, sans-serif", size=13),
            title_font=dict(size=18, color="#1E3A8A")
        )
        return fig

    def plot_task_completion_analysis(self) -> go.Figure:
        """5. Task Count vs Decision Ratio Scatter Analysis."""
        fig = px.scatter(
            self.df,
            x="task_count",
            y="decision_count",
            size="duration_minutes",
            color="department",
            hover_name="meeting_id",
            title="Task Assignment vs Decision Density Matrix",
            labels={"task_count": "Tasks Assigned", "decision_count": "Decisions Made"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Inter, sans-serif", size=13),
            title_font=dict(size=18, color="#1E3A8A")
        )
        return fig

    def get_summary_kpis(self) -> Dict[str, Any]:
        """Summary KPI metrics for dataset dashboard."""
        return {
            "total_meetings": len(self.df),
            "avg_duration": round(self.df["duration_minutes"].mean(), 1),
            "total_decisions": int(self.df["decision_count"].sum()),
            "total_tasks": int(self.df["task_count"].sum()),
            "avg_speakers": round(self.df["speaker_count"].mean(), 1)
        }
