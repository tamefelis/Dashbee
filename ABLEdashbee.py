import streamlit as st
import pandas as pd
from plotly import graph_objects as go
import numpy as np
from datetime import datetime
import plotly.io
from io import BytesIO
from streamlit.components.v1 import html

st.set_page_config(layout="wide")

# Load data functions
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Calendared Study Visit')
    df = convert_to_datetime(df)
    return df

@st.cache_data
def load_screening_data(file_path):
    df_screen = pd.read_excel(file_path, sheet_name='Calendared Screening Visit')
    df_screen = convert_to_datetime(df_screen)
    return df_screen

@st.cache_data
def load_dropout_data(file_path):
    dropout_df = pd.read_excel(file_path, sheet_name='dropout & withdrawn sheet')

    def determine_dropout_after(remark):
        match remark:
            case 'withdraw after visit 1':
                return 'Visit 2'
            case 'withdraw after visit 2':
                return 'Visit 3'
            case 'withdraw after visit 3':
                return 'Visit 4'
            case _:
                return None

    dropout_df['Dropout After'] = dropout_df['remarks'].apply(determine_dropout_after)
    total_dropouts = dropout_df['Dropout After'].notnull().sum()
    return dropout_df, total_dropouts

# Preprocess data functions
@st.cache_data
def convert_to_datetime(df):
    df[['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']] = df[['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']].apply(
        lambda col: pd.to_datetime(col.astype(str).str.strip(), format='%d/%m/%Y', errors='coerce'))
    return df

@st.cache_data
def convert_to_datetimeRE(df, columns_to_convert=None):
    if columns_to_convert is None:
        columns_to_convert = ['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']
    
    df[columns_to_convert] = df[columns_to_convert].apply(
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

def load_excel_data(uploaded_file):
    with pd.ExcelFile(uploaded_file) as xls:
        df_screening = pd.read_excel(xls, 'Calendared Screening Visit')
        df_actual = pd.read_excel(xls, 'Calendared Study Visit')
        df_dropout = pd.read_excel(xls, 'dropout & withdrawn sheet')

        df_screening = convert_to_datetimeRE(df_screening, ['Date for Screening'])
        df_actual = convert_to_datetimeRE(df_actual)
        return df_screening, df_actual, df_dropout

# Add the missing calculate_progression function
# Improved progress calculation function
def calculate_total_progress(df_actual, df_dropout, target_visits=480):
    current_date = datetime.now().date()
    completed_visits = df_actual[df_actual[['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']].lt(current_date).any(axis=1)].shape[0]
    dropouts = df_dropout.shape[0]
    total_visits_counted = completed_visits + dropouts
    progress_percentage = (total_visits_counted / target_visits) * 100
    return progress_percentage, total_visits_counted, target_visits, dropouts

# Add the missing create_progress_bar function
def create_progress_bar(progression):
    progress_html = f"""
    <style>
    .progress-container {{
        width: 100%;
        background-color: #eee;  /* Light grey background for better visibility */
        border-radius: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        overflow: hidden;  /* Ensures the inner bar does not overflow the container */
    }}

    .progress-bar {{
        width: {progression}%;
        background: linear-gradient(90deg, #ECF4D6, #9AD0C2, #2D9596, #265073);
        background-size: 200% 200%;
        animation: gradientShift 2s ease infinite;
        text-align: center;
        line-height: 30px;  /* Adjust as needed for text alignment */
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

# Revised calculate_total_progress function
def determine_missed_visits(remark):
    visit_map = {
        'withdraw after visit 1': 3,  # Missed Visit 2, Visit 3, Visit 4
        'withdraw after visit 2': 2,  # Missed Visit 3, Visit 4
        'withdraw after visit 3': 1,  # Missed Visit 4
        'withdraw after visit 4': 0   # No missed visits
    }
    return visit_map.get(remark, 0)

def calculate_missed_visits(dropout_df):
    dropout_df['Missed Visits'] = dropout_df['remarks'].apply(determine_missed_visits)
    total_missed_visits = dropout_df['Missed Visits'].sum()
    return total_missed_visits

def calculate_total_progress(df_long, dropout_df):
    current_date = datetime.now().date()
    completed_visits = df_long[df_long['Date'].dt.date <= current_date]

    total_completed_visits = len(completed_visits)
    total_missed_visits = calculate_missed_visits(dropout_df)

    total_visits_counted = total_completed_visits + total_missed_visits

    total_visits_expected = 480
    progress_percentage = (total_visits_counted / total_visits_expected) * 100

    return progress_percentage, total_visits_counted, total_visits_expected, total_missed_visits

def display_progress_bar(df_long, dropout_df, style='default'):
    progress_percentage, total_visits_counted, total_visits_expected, total_missed_visits = calculate_total_progress(df_long, dropout_df)
    
    completed_color = "#76C7C0"
    dropout_color = "#FF6F61"
    
    progress_bar_width = progress_percentage

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
        width: {progress_bar_width}%;
        background: linear-gradient(to right, {completed_color} {progress_percentage}%, {dropout_color} {progress_percentage}%);
        text-align: center;
        line-height: 30px;
        color: black;
        font-weight: bold;
        border-radius: 20px;
        transition: width 1s ease-out;
    }}
    </style>

    <div class="progress-container">
        <div class="progress-bar">{round(progress_percentage, 2)}%</div>
    </div>
    """
    st.write('Total Actual Visit Progression')
    st.markdown(progress_html, unsafe_allow_html=True)

    caption = f"Progress: {total_visits_counted} of {total_visits_expected} visits accounted (including {total_missed_visits} withdrawns)..."
    st.caption(caption)
    st.text(f"Current status including withdrawns: {progress_percentage:.2f}%")

# Plot functions
@st.cache_data
def generate_trace(df_long, visit):
    color_map = {
        'Visit 1': '#3B5BA5',
        'Visit 2': '#c6d7eb',
        'Visit 3': '#F3B941',
        'Visit 4': '#B2456E'
    }
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
    
    fig = go.Figure(data=[generate_trace(df_long, visit) for visit in df_long['Visit'].unique()])

    fig.update_layout(
        title='Current of Cumulative Trials Conducted vs Date',
        xaxis_title='Date',
        yaxis_title='Cumulative Trials',
        autosize=True
    )

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Actual data up-to-date: {current_time}")
    return fig

color_map = {
    'Visit 1': '#3B5BA5',
    'Visit 2': '#c6d7eb',
    'Visit 3': '#F3B941',
    'Visit 4': '#B2456E'
}

def darken_color(color, factor=0.7):
    color = color.lstrip('#')
    r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = [max(int(comp * factor), 0) for comp in (r, g, b)]
    return f'#{r:02x}{g:02x}{b:02x}'

color_map_dark = {visit: darken_color(color) for visit, color in color_map.items()}

@st.cache_data
def plot_visit_status(df_long, dropout_df):
    visit_types = ['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']
    max_visits = 120

    completed_counts = df_long[df_long['Date'].dt.date <= datetime.now().date()].groupby('Visit').size()
    dropout_counts = dropout_df.groupby('Dropout After').size()

    completed = [completed_counts.get(visit, 0) for visit in visit_types]
    cumulative_dropout_count = 0
    dropout_data = []
    remaining_data = []

    for visit in visit_types:
        cumulative_dropout_count += dropout_counts.get(visit, 0)
        completed_count = completed_counts.get(visit, 0)
        total_count = cumulative_dropout_count + completed_count
        remaining_count = max(max_visits - total_count, 0)

        dropout_percentage = (cumulative_dropout_count / 480) * 100

        dropout_data.append(go.Bar(
            name=f'Cumulative Dropouts by {visit}',
            x=[visit],
            y=[cumulative_dropout_count],
            marker_color='red',
            opacity=0.36,
            text=[f"Total Dropouts after {visit}: {cumulative_dropout_count} ({dropout_percentage:.2f}%)"],
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

@st.cache_data
def plot_gender_age_table(df_actual, df_dropout):
    try:
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

    except Exception as e:
        return f"An error occurred: {e}"

@st.cache_data
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

def generate_html(fig):
    return plotly.io.to_html(fig, include_plotlyjs='cdn', full_html=True)

def plot_visit_status_section(df_long, current_date, file_path):
    st.title('Visit Status')
    st.caption('Following chart demonstrates the status of our Visit Status...')
    st.caption(f"Data UTD: {current_date}")
    dropout_df, _ = load_dropout_data(file_path)
    visit_status_fig = plot_visit_status(df_long, dropout_df)
    st.plotly_chart(visit_status_fig)

    html_string_status = plotly.io.to_html(visit_status_fig, full_html=True, include_plotlyjs='cdn')
    html_out_status = BytesIO(html_string_status.encode())
    st.download_button(
        label="Download Visit Status Plot as HTML",
        data=html_out_status,
        file_name='Visit_Status_Plot.html',
        mime='text/html',
    )
    return plot_visit_status(df_long)

def plot_gender_age_distribution_section(df_actual, df_dropout, current_date):
    st.title('Gender and Age-Tier distribution of Actual Participant after Scheduling')
    st.caption('Following table shows the count of individuals including dropout, by Gender and Age Tier...')
    st.caption(f"Data UTD: {current_date}")
    html_string = plot_gender_age_table(df_actual, df_dropout)
    st.markdown(html_string, unsafe_allow_html=True)
    return plot_gender_age_table(df_actual, df_dropout)

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def download_dataframe_button(df, button_label, file_name):
    excel_data = to_excel(df)
    st.download_button(label=button_label, data=excel_data, file_name=file_name, mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

def run_cumulative_trials_plot():
    st.title('ABLE Visits Progression')
    uploaded_file = st.sidebar.file_uploader("drop that dope here", type="xlsx")

    if uploaded_file is not None:
        df_screening, df_actual, df_dropout = load_excel_data(uploaded_file)
        df_dropout, total_dropouts = load_dropout_data(uploaded_file)
        
        # Merge dropout data with actual data
        df_long = reshape_dataframe(df_actual)
        df_long = add_count_columns(df_long)
        df_long = add_day_of_week(df_long)
        
        current_date = datetime.now().date()

        display_progress_bar(df_long, df_dropout, style='tralalala')
            
        st.title('Projection of Current ABLE participant')
        data_filters = {
            f"Full projection<br><sub>Data up-to-date: {current_date}</sub>": df_long,
            "Completed Visits": df_long[df_long['Date'].dt.date <= current_date],
            "Upcoming Visits": df_long[df_long['Date'].dt.date > current_date],
        }
        
        cols = st.columns([1, 1, 1])
        with st.expander("Download Plots and DataFrames"):
            for i, (filter_name, filtered_data) in enumerate(data_filters.items()):

                filtered_data = add_count_columns(filtered_data)
                filtered_data = add_day_of_week(filtered_data)
                fig = go.Figure(data=[generate_trace(filtered_data, visit) for visit in filtered_data['Visit'].unique()])

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

                html_string = plotly.io.to_html(fig, full_html=True, include_plotlyjs='cdn')
                html_out = BytesIO(html_string.encode())
                download_button_label = f"Download {filter_name} Plot as HTML"
                st.download_button(
                    label=download_button_label,
                    data=html_out,
                    file_name=f'{filter_name}.html',
                    mime='text/html',
                )

                download_dataframe_button(filtered_data, f"Download {filter_name} Data as Excel", f'{filter_name}.xlsx')

        status_col, gender_age_col = st.columns(2)
        
        with status_col:
            visit_status_fig = plot_visit_status(df_long, df_dropout)
            st.title('Visit Status of Project ABLE')
            st.plotly_chart(visit_status_fig)

        with gender_age_col:
            plot_gender_age_distribution_section(df_actual, df_dropout, current_date)
            st.markdown("---")
            st.title('Distribution of Gender and Age Group among Dropouts and Participants Completed Visit 1')
            st.markdown(plot_gender_age_distribution_visit1_dropout(df_actual, df_dropout), unsafe_allow_html=True)

if __name__ == "__main__":
    run_cumulative_trials_plot()
