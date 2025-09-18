import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TensorBoardLogExtractor:
    """Extract and analyze TensorBoard logs from MASCS experiments"""

    def __init__(self, log_dir="./experiments"):
        self.log_dir = Path(log_dir)
        self.experiments = {}
        self.summary_data = defaultdict(dict)

    def discover_experiments(self):
        """Discover all experiments in the log directory"""
        experiment_dirs = [d for d in self.log_dir.iterdir() if d.is_dir()]

        for exp_dir in experiment_dirs:
            # Check if this is a valid experiment directory with tensorboard logs
            tb_log_path = exp_dir / "tensorboard"

            if tb_log_path.exists():
                exp_name = exp_dir.name

                # Try to extract parameters from directory name
                config = self._parse_config_from_name(exp_name)

                self.experiments[exp_name] = {
                    "path": exp_dir,
                    "config": config,
                    "tb_log_path": tb_log_path
                }

        print(f"Found {len(self.experiments)} experiments")
        return self.experiments

    def _parse_config_from_name(self, exp_name):
        """Parse experiment configuration from directory name"""
        config = {}

        # Try to extract model name
        model_match = re.search(r'([a-zA-Z]+?\d*)', exp_name)
        if model_match:
            config['model'] = model_match.group(1)

        # Try to extract dataset name
        dataset_match = re.search(r'(cifar10|cifar100|imagenet)', exp_name, re.IGNORECASE)
        if dataset_match:
            config['dataset'] = dataset_match.group(1).lower()

        # Try to extract budget (numbers in the name)
        budget_matches = re.findall(r'_(\d+)_', exp_name)
        if budget_matches:
            # Assume the largest number is the budget
            config['budget'] = max([int(x) for x in budget_matches])

        # Try to extract date/timestamp
        date_match = re.search(r'(\d{8}_\d{6})', exp_name)
        if date_match:
            config['timestamp'] = date_match.group(1)

        return config

    def extract_scalars_from_event_file(self, event_file):
        """Extract scalar data from a single event file"""
        try:
            event_acc = EventAccumulator(str(event_file))
            event_acc.Reload()

            scalars = {}
            for tag in event_acc.Tags()['scalars']:
                events = event_acc.Scalars(tag)
                scalars[tag] = {
                    'steps': [e.step for e in events],
                    'values': [e.value for e in events],
                    'wall_times': [e.wall_time for e in events]
                }

            return scalars
        except Exception as e:
            print(f"Error processing event file {event_file}: {e}")
            return {}

    def extract_all_scalars(self):
        """Extract scalar data from all experiments"""
        for exp_name, exp_info in self.experiments.items():
            print(f"Processing {exp_name}...")

            # Find all event files in the tensorboard directory
            event_files = list(exp_info['tb_log_path'].glob("events.out.tfevents.*"))

            if not event_files:
                print(f"  No event files found in {exp_info['tb_log_path']}")
                continue

            # Use the most recent event file
            event_file = sorted(event_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]

            # Extract scalars
            scalars = self.extract_scalars_from_event_file(event_file)
            self.summary_data[exp_name] = scalars

            # Also store experiment metadata
            self.summary_data[exp_name]['_metadata'] = {
                'config': exp_info['config']
            }

        return self.summary_data

    def create_summary_dataframe(self):
        """Create a summary DataFrame with key metrics from all experiments"""
        rows = []

        for exp_name, data in self.summary_data.items():
            if not data or exp_name.startswith('_'):
                continue

            # Get metadata
            metadata = data.get('_metadata', {})
            config = metadata.get('config', {})

            # Find the final values of key metrics with defaults
            final_train_loss = data.get('Loss/train', {}).get('values', [0])[-1] if 'Loss/train' in data else 0
            final_val_loss = data.get('Loss/val', {}).get('values', [0])[-1] if 'Loss/val' in data else 0
            final_train_acc = data.get('Accuracy/train', {}).get('values', [0])[-1] if 'Accuracy/train' in data else 0
            final_val_acc = data.get('Accuracy/val', {}).get('values', [0])[-1] if 'Accuracy/val' in data else 0
            best_val_acc = data.get('Accuracy/best_val', {}).get('values', [0])[-1] if 'Accuracy/best_val' in data else max(data.get('Accuracy/val', {}).get('values', [0]))

            # Count epochs
            num_epochs = len(data.get('Accuracy/val', {}).get('values', [])) or len(data.get('Loss/val', {}).get('values', []))

            # Create row for DataFrame
            row = {
                'experiment': exp_name,
                'model': config.get('model', 'unknown'),
                'dataset': config.get('dataset', 'unknown'),
                'budget': config.get('budget', 0),
                'timestamp': config.get('timestamp', ''),
                'epochs': num_epochs,
                'best_val_accuracy': best_val_acc,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'final_train_accuracy': final_train_acc,
                'final_val_accuracy': final_val_acc
            }

            rows.append(row)

        # Create DataFrame
        if rows:
            df = pd.DataFrame(rows)

            # Sort by best validation accuracy if the column exists
            if 'best_val_accuracy' in df.columns:
                df = df.sort_values('best_val_accuracy', ascending=False)
            else:
                print("Warning: 'best_val_accuracy' column not found, skipping sort")
        else:
            # Create empty DataFrame with expected columns
            df = pd.DataFrame(columns=[
                'experiment', 'model', 'dataset', 'budget', 'timestamp', 'epochs',
                'best_val_accuracy', 'final_train_loss', 'final_val_loss',
                'final_train_accuracy', 'final_val_accuracy'
            ])

        return df

    def export_to_csv(self, output_dir="./analysis"):
        """Export summary data to CSV files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Export summary DataFrame
        summary_df = self.create_summary_dataframe()
        summary_df.to_csv(output_dir / "experiment_summary.csv", index=False)

        # Export detailed data for each experiment
        for exp_name, data in self.summary_data.items():
            if exp_name.startswith('_'):
                continue

            exp_data = {}
            for metric, values in data.items():
                if isinstance(values, dict) and 'values' in values:
                    exp_data[f"{metric}_values"] = values['values']
                    exp_data[f"{metric}_steps"] = values['steps']

            if exp_data:
                # Find the maximum length to pad shorter lists
                max_len = max([len(v) if isinstance(v, list) else 1 for v in exp_data.values()])

                # Pad all lists to the same length
                for key, values in exp_data.items():
                    if isinstance(values, list):
                        exp_data[key] = values + [None] * (max_len - len(values))
                    else:
                        exp_data[key] = [values] + [None] * (max_len - 1)

                exp_df = pd.DataFrame(exp_data)
                exp_df.to_csv(output_dir / f"{exp_name}_detailed.csv", index=False)

        print(f"Data exported to {output_dir}")

    def generate_report(self, output_dir="./analysis"):
        """Generate an HTML report with visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        summary_df = self.create_summary_dataframe()

        html_content = f"""
        <html>
        <head>
            <title>MASCS Experiment Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>MASCS Experiment Analysis Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Summary Statistics</h2>
            <p>Total Experiments: {len(summary_df)}</p>
            <p>Average Best Validation Accuracy: {summary_df['best_val_accuracy'].mean():.4f}</p>
            <p>Best Performance: {summary_df['best_val_accuracy'].max():.4f}</p>

            <h2>Experiment Summary</h2>
            {summary_df.to_html(index=False, classes='table')}

        </body>
        </html>
        """

        with open(output_dir / "experiment_report.html", "w") as f:
            f.write(html_content)

        print(f"Report generated at {output_dir / 'experiment_report.html'}")

# Streamlit App
def main():
    st.set_page_config(page_title="TensorBoard Log Analyzer", layout="wide", page_icon="📊")

    st.title("📊 TensorBoard Log Analysis Dashboard")
    st.markdown("Analyze and visualize MASCS experiment results from TensorBoard logs")

    # Initialize session state
    if 'extractor' not in st.session_state:
        st.session_state.extractor = None
    if 'summary_df' not in st.session_state:
        st.session_state.summary_df = None

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    log_dir = st.sidebar.text_input("Log Directory", value="./experiments")

    if st.sidebar.button("Discover Experiments"):
        with st.spinner("Discovering experiments..."):
            extractor = TensorBoardLogExtractor(log_dir)
            extractor.discover_experiments()
            st.session_state.extractor = extractor
            st.sidebar.success(f"Found {len(extractor.experiments)} experiments")

    if st.session_state.extractor is not None:
        extractor = st.session_state.extractor

        if st.sidebar.button("Extract Data"):
            with st.spinner("Extracting data from TensorBoard logs..."):
                extractor.extract_all_scalars()
                summary_df = extractor.create_summary_dataframe()
                st.session_state.summary_df = summary_df
                st.sidebar.success("Data extraction complete!")

    # Main content area
    if st.session_state.extractor is not None and st.session_state.summary_df is not None:
        extractor = st.session_state.extractor
        summary_df = st.session_state.summary_df

        # Display summary table
        st.header("Experiment Summary")
        st.dataframe(summary_df, use_container_width=True)

        # Model comparison
        st.header("Model Comparison")

        col1, col2 = st.columns(2)

        with col1:
            if 'best_val_accuracy' in summary_df.columns and len(summary_df) > 0:
                fig = px.bar(summary_df, x='experiment', y='best_val_accuracy',
                            title='Best Validation Accuracy by Experiment',
                            color='budget' if 'budget' in summary_df.columns else None)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'final_val_accuracy' in summary_df.columns and 'final_train_accuracy' in summary_df.columns and len(summary_df) > 0:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=summary_df['experiment'], y=summary_df['final_train_accuracy'],
                                    name='Final Train Accuracy'))
                fig.add_trace(go.Bar(x=summary_df['experiment'], y=summary_df['final_val_accuracy'],
                                    name='Final Validation Accuracy'))
                fig.update_layout(title='Final Train vs Validation Accuracy',
                                barmode='group',
                                xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

        # Budget vs Performance Analysis
        if 'budget' in summary_df.columns and len(summary_df) > 0:
            st.header("Budget vs Performance Analysis")
            fig = px.scatter(summary_df, x='budget', y='best_val_accuracy',
                           title='Budget vs Best Validation Accuracy',
                           hover_data=['experiment'])
            st.plotly_chart(fig, use_container_width=True)

        # Experiment selection for detailed view
        st.header("Detailed Experiment Analysis")
        selected_experiment = st.selectbox("Select Experiment", list(extractor.summary_data.keys()))

        if selected_experiment and selected_experiment in extractor.summary_data:
            data = extractor.summary_data[selected_experiment]

            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Accuracy Metrics", "Loss Metrics", "Strategy Weights"])

            with tab1:
                accuracy_metrics = [tag for tag in data.keys() if 'Accuracy' in tag and not tag.startswith('_')]
                if accuracy_metrics:
                    fig = go.Figure()
                    for metric in accuracy_metrics:
                        steps = data[metric]['steps']
                        values = data[metric]['values']
                        fig.add_trace(go.Scatter(x=steps, y=values, mode='lines+markers', name=metric))

                    fig.update_layout(title=f'Accuracy Metrics - {selected_experiment}',
                                    xaxis_title='Epoch', yaxis_title='Accuracy')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No accuracy metrics found for this experiment")

            with tab2:
                loss_metrics = [tag for tag in data.keys() if 'Loss' in tag and not tag.startswith('_')]
                if loss_metrics:
                    fig = go.Figure()
                    for metric in loss_metrics:
                        steps = data[metric]['steps']
                        values = data[metric]['values']
                        fig.add_trace(go.Scatter(x=steps, y=values, mode='lines+markers', name=metric))

                    fig.update_layout(title=f'Loss Metrics - {selected_experiment}',
                                    xaxis_title='Epoch', yaxis_title='Loss')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No loss metrics found for this experiment")

            with tab3:
                strategy_metrics = [tag for tag in data.keys() if 'Strategy_weights' in tag and not tag.startswith('_')]
                if strategy_metrics:
                    fig = go.Figure()
                    for metric in strategy_metrics:
                        strategy_name = metric.replace('Strategy_weights/', '')
                        steps = data[metric]['steps']
                        values = data[metric]['values']
                        fig.add_trace(go.Scatter(x=steps, y=values, mode='lines+markers', name=strategy_name))

                    fig.update_layout(title=f'Strategy Weights - {selected_experiment}',
                                    xaxis_title='Epoch', yaxis_title='Weight')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No strategy weights found for this experiment")

        # Comparison across experiments
        st.header("Cross-Experiment Comparison")
        available_metrics = set()
        for data in extractor.summary_data.values():
            available_metrics.update([k for k in data.keys() if not k.startswith('_')])

        metric_to_compare = st.selectbox("Select Metric to Compare",
                                        sorted(list(available_metrics)))

        if metric_to_compare:
            fig = go.Figure()
            for exp_name, data in extractor.summary_data.items():
                if exp_name.startswith('_'):
                    continue

                if metric_to_compare in data:
                    steps = data[metric_to_compare]['steps']
                    values = data[metric_to_compare]['values']

                    # Create label with key experiment info
                    config = data.get('_metadata', {}).get('config', {})
                    label = f"{config.get('model', 'unknown')}-{config.get('dataset', '')}-budget{config.get('budget', 0)}"

                    fig.add_trace(go.Scatter(x=steps, y=values, mode='lines+markers', name=label))

            fig.update_layout(title=f'Comparison of {metric_to_compare} across experiments',
                            xaxis_title='Epoch', yaxis_title=metric_to_compare)
            st.plotly_chart(fig, use_container_width=True)

        # Export options
        st.sidebar.header("Export Options")
        if st.sidebar.button("Export to CSV"):
            with st.spinner("Exporting data..."):
                extractor.export_to_csv("./analysis")
                st.sidebar.success("Data exported to ./analysis directory")

        if st.sidebar.button("Generate Report"):
            with st.spinner("Generating report..."):
                extractor.generate_report("./analysis")
                st.sidebar.success("Report generated in ./analysis directory")

    else:
        st.info("Please specify a log directory and click 'Discover Experiments' to begin analysis.")

        # Example of what the app can do
        st.subheader("About this Tool")
        st.markdown("""
        This dashboard helps you analyze TensorBoard logs from MASCS experiments by providing:

        - **Experiment Discovery**: Automatically finds all experiments in your log directory
        - **Summary Statistics**: Key metrics like best validation accuracy, final losses, etc.
        - **Detailed Visualizations**: Interactive charts for accuracy, loss, and strategy weights
        - **Comparison Tools**: Compare metrics across different experiments
        - **Export Functionality**: Download data as CSV or generate HTML reports

        To get started:
        1. Ensure your TensorBoard logs are in the specified directory (default: ./experiments)
        2. Click 'Discover Experiments' to find all available experiments
        3. Click 'Extract Data' to load the metrics from TensorBoard event files
        4. Use the interactive charts to analyze your results
        """)

if __name__ == "__main__":
    main()