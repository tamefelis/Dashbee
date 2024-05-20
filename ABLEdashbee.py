import streamlit as st
import pandas as pd
from plotly import graph_objects as go
import numpy as np
from datetime import datetime
import plotly.io as pio
from io import BytesIO
from streamlit.components.v1 import html

st.set_page_config(layout="wide")

@st.cache_data
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = convert_to_datetime(df)
    return df

@st.cache_data
def convert_to_datetime(df):
    date_columns = ['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']
    df[date_columns] = df[date_columns].apply(
        lambda col: pd.to_datetime(col.astype(str).str.strip(), format='%d/%m/%Y', errors='coerce'))
    return df

@st.cache_data
def reshape_dataframe(df):
    df_long = df.melt(id_vars='Study ID', value_vars=['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4'], var_name='Visit', value_name='Date')
    df_long = df_long.dropna(subset=['Date'])
    df_long = df_long.sort_values('Date').reset_index(drop=True)
    return df_long

@st.cache_data
def add_count_columns(df_long):
    df_long['Visits per Day'] = df_long.groupby('Date')['Study ID'].transform('count')
    df_long['Cumulative Count'] = np.arange(1, len(df_long) + 1)
    return df_long.sort_values('Date')

@st.cache_data
def add_day_of_week(df_long):
    df_long['Day of Week'] = df_long['Date'].dt.day_name()
    return df_long

@st.cache_data
def load_dropout_data(file_path):
    dropout_df = pd.read_excel(file_path, sheet_name='dropout & withdrawn sheet')

    def determine_dropout_after(remark):
        match remark:
            case 'drop out after randomisation' | 'withdraw after randomisation' | 'withdraw after visit 1':
                return 'Visit 1'
            case 'withdraw after visit 2':
                return 'Visit 2'
            case 'withdraw after visit 3':
                return 'Visit 3'
            case _:
                return None

    dropout_df['Dropout After'] = dropout_df['remarks'].apply(determine_dropout_after)
    total_dropouts = dropout_df['Dropout After'].notnull().sum()
    return dropout_df, total_dropouts

def calculate_progression(df_screening, df_actual, df_dropout):
    current_date = datetime.now()
    completed_screening = df_screening[df_screening['Date for Screening'] < current_date].shape[0]
    completed_actual = df_actual[['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']].apply(lambda x: x < current_date).sum().sum()
    num_dropouts = df_dropout[df_dropout['remarks'].notnull() & (df_dropout['remarks'] != '')].shape[0]

    total_visits = 730
    progression = ((completed_screening + completed_actual + num_dropouts) / total_visits) * 100
    return progression

def create_progress_bar(progression):
    progress_html = f"""
    <style>
    .progress-container {{
        width: 100%;
        background-color: #eee;
        border-radius: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        overflow: hidden;
    }}
    .progress-bar {{
        width: {progression}%;
        background: linear-gradient(90deg, #ECF4D6, #9AD0C2, #2D9596, #265073);
        background-size: 200% 200%;
        animation: gradientShift 2s ease infinite;
        text-align: center;
        line-height: 30px;
        color: black;
        font-weight: bold;
        border-radius: 20px;
        transition: width 1s ease-out;
    }}
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    </style>
    <div class="progress-container">
        <div class="progress-bar">{progression:.2f}%</div>
    </div>
    """
    return progress_html

def generate_trace(df_long, visit, color_map):
    df_subset = df_long[df_long['Visit'] == visit]
    df_subset['Visit Count'] = df_subset.groupby('Visit').cumcount() + 1

    return go.Scatter(
        x=df_subset['Date'],
        y=df_subset['Cumulative Count'],
        mode='markers',
        name=visit,
        marker=dict(color=color_map.get(visit, 'black')),
        hovertext=[
            f"Study ID: {row['Study ID']}<br>" +
            f"Day of Week: {row['Day of Week']}<br>" +
            f"Date: {row['Date'].strftime('%d/%m/%y')}<br>" +
            f"Visit: {row['Visit']}<br>" +
            f"Total Visits on This Day: {row['Visits per Day']}<br>" +
            f"Trials Cumulative Count: {row['Cumulative Count']}<br>" +
            f"Count for {row['Visit']}: {row['Visit Count']}<br>" +
            f"Plot Label: {visit}"
            for _, row in df_subset.iterrows()
        ]
    )

@st.cache_data
def plot_cumulative_trials(df):
    df_long = reshape_dataframe(df)
    df_long = add_count_columns(df_long)
    df_long = add_day_of_week(df_long)
    
    color_map = {
        'Visit 1': '#3B5BA5',
        'Visit 2': '#c6d7eb',
        'Visit 3': '#F3B941',
        'Visit 4': '#B2456E'
    }

    fig = go.Figure(data=[generate_trace(df_long, visit, color_map) for visit in df_long['Visit'].unique()])

    fig.update_layout(
        title='Current of Cumulative Trials Conducted vs Date',
        xaxis_title='Date',
        yaxis_title='Cumulative Trials',
        autosize=True
    )

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Actual data up-to-date: {current_time}")
    return fig

def plot_visit_status(df_long, dropout_df):
    visit_types = ['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']
    max_visits = 120

    completed_counts = df_long[df_long['Date'].dt.date <= datetime.now().date()].groupby('Visit').size()

    dropout_df_actual = dropout_df[0]
    dropout_counts = dropout_df_actual.groupby('Dropout After').size()
    completed = [completed_counts.get(visit, 0) for visit in visit_types]
    cumulative_dropout_count = 0
    dropout_data = []
    remaining_data = []

    color_map = {
        'Visit 1': '#3B5BA5',
        'Visit 2': '#c6d7eb',
        'Visit 3': '#F3B941',
        'Visit 4': '#B2456E'
    }
    color_map_dark = {visit: darken_color(color) for visit, color in color_map.items()}

    for visit in visit_types:
        cumulative_dropout_count += dropout_counts.get(visit, 0)
        completed_count = completed_counts.get(visit, 0)
        total_count = cumulative_dropout_count + completed_count
        remaining_count = max(max_visits - total_count, 0)

        dropout_data.append(go.Bar(
            name=f'Cumulative Dropouts by {visit}',
            x=[visit],
            y=[cumulative_dropout_count],
            marker_color='red',
            opacity=0.36,
            text=[f"Total Dropouts after {visit}: {cumulative_dropout_count}"],
            hoverinfo="text"
        ))
        remaining_data.append(go.Bar(
            name=f'Remaining After {visit}', x=[visit], y=[remaining_count],
            marker_color=color_map_dark.get(visit, '#000000'),
            opacity=0.369
        ))

    completed_colors = [color_map.get(visit, '#000000') for visit in visit_types]
    fig = go.Figure(data=[
        go.Bar(name='Completed', x=visit_types, y=completed, marker_color=completed_colors)
    ] + dropout_data + remaining_data)

    current_date = datetime.now().strftime('%Y-%m-%d')
    title_with_subheading = f"Visit Status: Completed vs Remaining<br><sub>Data up-to-date: {current_date}</sub>"

    fig.update_layout(
        barmode='stack',
        title=title_with_subheading,
        xaxis_title='Visit Type',
        yaxis_title='Number of Visits',
        autosize=True,
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white')
    )
    return fig

def plot_gender_age_table(df_actual, df_dropout):
    merged_df = pd.merge(df_actual[['Study ID', 'Gender', 'age-tier']], 
                         df_dropout[['Study ID', 'Gender', 'age-tier']], 
                         on='Study ID', 
                         how='outer', 
                         suffixes=('_actual', '_dropout'))
    merged_df['Gender'] = merged_df['Gender_actual'].combine_first(merged_df['Gender_dropout'])
    merged_df['age-tier'] = merged_df['age-tier_actual'].combine_first(merged_df['age-tier_dropout'])
    merged_df.drop(columns=['Gender_actual', 'Gender_dropout', 'age-tier_actual', 'age-tier_dropout'], inplace=True)

    filtered_df = merged_df[merged_df['Gender'].isin(['Female', 'Male']) & merged_df['age-tier'].isin(['40-50', '51-60'])]
    pivot_table = pd.pivot_table(filtered_df, values='Study ID', index=['Gender'], columns=['age-tier'], aggfunc='count', fill_value=0, margins=True, margins_name='Grand Total')
    pivot_table = pivot_table.reindex(columns=['40-50', '51-60', 'Grand Total'], index=['Female', 'Male', 'Grand Total'])

    html_string = """
    <table style="width:100%; max-width:400px; border:1px solid black; border-collapse:collapse; margin: auto;">
        <thead>
            <tr style="background-color:#3B5BA5; color:black;">
                <th style="border:1px solid black; text-align:center;" rowspan="2">Gender</th>
                <th style="border:1px solid black; text-align:center;" colspan="3">Age Tier (years old)</th>
            </tr>
            <tr style="background-color:#3B5BA5; color:black;">
                <th style="border:1px solid black; text-align:center;">40-50</th>
                <th style="border:1px solid black; text-align:center;">51-60</th>
                <th style="border:1px solid black; text-align:center;">Grand Total</th>
            </tr>
        </thead>
        <tbody>
    """

    for i, row in pivot_table.iterrows():
        html_string += f"<tr style='background-color:#c6d7eb; color:black;'>"
        html_string += f"<td style='border:1px solid black; text-align:center;'>{i}</td>"
        for col in ['40-50', '51-60', 'Grand Total']:
            html_string += f"<td style='border:1px solid black; text-align:center;'>{row[col]}</td>"
        html_string += "</tr>"
    
    html_string += "</tbody></table>"
    return html_string

def plot_gender_age_distribution_visit1_dropout(df_actual, df_dropout):
    current_date = datetime.now().date()
    df_visit1_completed = df_actual[df_actual['Visit 1'].dt.date < current_date]
    df_dropout_filtered = df_dropout[df_dropout['remarks'].str.contains('drop out', na=False)]
    df_combined = pd.concat([df_visit1_completed, df_dropout_filtered], ignore_index=True, sort=False)
    df_combined = df_combined[['Study ID', 'Gender', 'age-tier']].drop_duplicates()
    filtered_df = df_combined[df_combined['Gender'].isin(['Female', 'Male']) & df_combined['age-tier'].isin(['40-50', '51-60'])]
    pivot_table = pd.pivot_table(filtered_df, values='Study ID', index=['Gender'], columns=['age-tier'], aggfunc='count', fill_value=0, margins=True, margins_name='Total')

    html_string = pivot_table.to_html(classes='table table-striped')
    return html_string

def darken_color(color, factor=0.7):
    color = color.lstrip('#')
    r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = [max(int(comp * factor), 0) for comp in (r, g, b)]
    return f'#{r:02x}{g:02x}{b:02x}'

def run_cumulative_trials_plot():
    st.title('ABLE Visits Progression')
    uploaded_file = st.sidebar.file_uploader("Drop the file here", type="xlsx")

    if uploaded_file is not None:
        df_screening = load_data(uploaded_file, 'Calendared Screening Visit')
        df_actual = load_data(uploaded_file, 'Calendared Study Visit')
        df_dropout, total_dropouts = load_dropout_data(uploaded_file)

        progression = calculate_progression(df_screening, df_actual, df_dropout)
        st.write("Total Progression of the Study (*Screening* + *Actual Visit*):")
        progress_bar_html = create_progress_bar(progression)
        html(progress_bar_html)

        df_long = reshape_dataframe(df_actual)
        df_visits_long = reshape_dataframe(df_actual)
        current_date = datetime.now().date()

        display_progress_bar(df_long, df_dropout)

        st.title('Projection of Current ABLE Participant')
        data_filters = {
            f"Full projection<br><sub>Data up-to-date: {current_date}</sub>": df_long,
            "Completed Visits": df_long[df_long['Date'].dt.date <= current_date],
            "Upcoming Visits": df_long[df_long['Date'].dt.date > current_date],
        }

        cols = st.columns([1, 1, 1])
        with st.expander("Download Plots"):
            for i, (filter_name, filtered_data) in enumerate(data_filters.items()):
                filtered_data = add_count_columns(filtered_data)
                filtered_data = add_day_of_week(filtered_data)
                fig = go.Figure(data=[generate_trace(filtered_data, visit, color_map) for visit in filtered_data['Visit'].unique()])

                fig.update_xaxes(dtick="M1", tickformat="%b\n%Y", title="Date")
                fig.update_layout(
                    title=f'{filter_name}', 
                    xaxis_title='Date', 
                    yaxis_title='Cumulative Trials', 
                    autosize=True,
                    paper_bgcolor='#111111',
                    plot_bgcolor='#111111',
                    font=dict(color='white')
                )

                cols[i].plotly_chart(fig, use_container_width=True)

                with cols[i].expander("Show Data"):
                    st.dataframe(filtered_data)

                html_string = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
                html_out = BytesIO(html_string.encode())
                st.download_button(
                    label=f"Download {filter_name} Plot as HTML",
                    data=html_out,
                    file_name=f'{filter_name}.html',
                    mime='text/html',
                )

        status_col, gender_age_col = st.columns(2)
        
        with status_col:
            visit_status_fig = plot_visit_status(df_long, df_dropout)
            st.title('Visit Status of Project ABLE')
            st.plotly_chart(visit_status_fig)

        with gender_age_col:
            st.title('Gender and Age-Tier distribution of Actual Participant after Scheduling')
            st.caption('Following table shows the count of individuals including dropout, by Gender and Age Tier...')
            st.caption(f"Data UTD: {current_date}")
            html_string = plot_gender_age_table(df_actual, df_dropout)
            st.markdown(html_string, unsafe_allow_html=True)

            st.title('Distribution of Gender and Age Group among Dropouts and Participants Completed Visit 1')
            st.markdown(plot_gender_age_distribution_visit1_dropout(df_actual, df_dropout), unsafe_allow_html=True)

if __name__ == "__main__":
    run_cumulative_trials_plot()
