import streamlit as st
import pandas as pd
from plotly import graph_objects as go
import numpy as np
from datetime import datetime
import plotly.io 
from io import BytesIO
from streamlit.components.v1 import html

st.set_page_config(layout="wide")

@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Calendared Study Visit')
    df = convert_to_datetime(df)
    return df

def load_screening_data(file_path):
    df_screen = pd.read_excel(file_path, sheet_name='Calendared Screening Visit')
    df_screen = convert_to_datetime(df_screen)
    return df_screen

@st.cache_data
def load_dropout_data(file_path):
    dropout_df = pd.read_excel(file_path, sheet_name='dropout sheet')
    # Process to identify the visit after which participants dropped out
    dropout_df['Dropout After'] = np.where(dropout_df['remarks'] == 'drop out after randomisation', 'Visit 1', None)
    return dropout_df

#preprocess of actual visit data
@st.cache_data
def convert_to_datetime(df):
    df[['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']] = df[['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']].apply(
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
def convert_to_datetimeRE(df, columns_to_convert=None):
    if columns_to_convert is None:
        columns_to_convert = ['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']
    
    df[columns_to_convert] = df[columns_to_convert].apply(
        lambda col: pd.to_datetime(col.astype(str).str.strip(), format='%d/%m/%Y', errors='coerce'))
    return df

def load_excel_data(uploaded_file):
    with pd.ExcelFile(uploaded_file) as xls:
        df_screening = pd.read_excel(xls, 'Calendared Screening Visit')
        df_actual = pd.read_excel(xls, 'Calendared Study Visit')
        df_dropout = pd.read_excel(xls, 'dropout sheet')

        df_screening = convert_to_datetimeRE(df_screening, ['Date for Screening'])
        df_actual = convert_to_datetimeRE(df_actual)  # Default columns are used here
        return df_screening, df_actual, df_dropout


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
        background-color: #fff;
        padding: 3px;
        border-radius: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }}
    
    .progress-bar {{
        width: {progression}%;
        background: linear-gradient(-45deg, #ECF4D6, #9AD0C2, #2D9596, #265073);
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

#calculation of total progress
def calculate_total_progress(df_long, dropout_df):
    current_date = datetime.now().date()
    completed_visits = df_long[df_long['Date'].dt.date <= current_date]

    total_visits_so_far = len(completed_visits) + len(dropout_df)
    total_visits_expected = 480  # 480 visits + 250 screenings
    progress_percentage = total_visits_so_far / total_visits_expected
    return progress_percentage * 100  # Convert to percentage and round to two decimal points


def display_progress_bar(df_long, dropout_df, style='default'):
    progress_percentage = calculate_total_progress(df_long, dropout_df)
    progress_bar_width = progress_percentage

    # Define background color and gradient effect
    background_color = "linear-gradient(90deg, rgba(76,175,80,1) 0%, rgba(139,195,74,1) 100%)"
    if style == 'tralalala':
        background_color = "linear-gradient(90deg, rgb(62,106,187) 0%, rgb(106,120,192) 25%, rgb(155, 190, 200) 50%, rgb(221, 242, 253) 100%)"

    progress_html = f"""
    <style>
    .progress-container {{
        width: 100%;
        background-color: #fff;
        padding: 3px;
        border-radius: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        margin: 20px 0;
    }}

    .progress-bar {{
        width: {progress_bar_width}%;
        background: {background_color};
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
        <div class="progress-bar">{round(progress_bar_width, 2)}%</div>
    </div>
    """
    st.write('Total Actual Visit Progression')
    st.markdown(progress_html, unsafe_allow_html=True)

    # Caption
    current_date = datetime.now().date()
    completed_visits = df_long[df_long['Date'].dt.date <= current_date]
    total_visits_so_far = len(completed_visits) + len(dropout_df)
    total_visits_expected = 480
    caption = f"Progress: {total_visits_so_far} of {total_visits_expected} visits (including dropouts) completed... "
    st.caption(caption)



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
            f"Count for {row['Visit']}: {row['Visit Count']}<br>" +  # Enhanced detail
            f"Plot Label: {visit}"  # Added plot label
            for _, row in df_subset.iterrows()
        ]
    )


@st.cache_data
def plot_cumulative_trials(df):
    df_long = reshape_dataframe(df)
    df_long = add_count_columns(df_long)
    df_long = add_day_of_week(df_long)
    
    # Initialize the figure first
    fig = go.Figure(data=[generate_trace(df_long, visit, color_map) for visit in df_long['Visit'].unique()])

    # Then update the layout
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
    """ Darken a color by a given factor """
    # Convert hex color to RGB
    color = color.lstrip('#')
    r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    
    # Darken
    r, g, b = [max(int(comp * factor), 0) for comp in (r, g, b)]
    
    # Convert back to hex
    return f'#{r:02x}{g:02x}{b:02x}'
color_map_dark = {visit: darken_color(color) for visit, color in color_map.items()}

@st.cache_data
def plot_visit_status(df_long, dropout_df):
    visit_types = ['Visit 1', 'Visit 2', 'Visit 3', 'Visit 4']
    max_visits = 120

    # Count the number of completed visits for each visit type
    completed_counts = df_long[df_long['Date'].dt.date <= datetime.now().date()].groupby('Visit').size()

    # Count dropouts
    dropout_counts = dropout_df.groupby('Dropout After').size()

    # Prepare data for the plot
    completed = [completed_counts.get(visit, 0) for visit in visit_types]
    cumulative_dropout_count = 0
    dropout_data = []
    remaining_data = []

    for visit in visit_types:
        # Increase cumulative dropout count if there are dropouts after this visit
        cumulative_dropout_count += dropout_counts.get(visit, 0)

        completed_count = completed_counts.get(visit, 0)
        
        # Adjust remaining count so total doesn't exceed max_visits
        total_count = cumulative_dropout_count + completed_count
        remaining_count = max(max_visits - total_count, 0)

        # Append dropout and remaining data
        dropout_data.append(go.Bar(
            name=f'Cumulative Dropouts by {visit}', x=[visit], y=[cumulative_dropout_count],
            marker_color='red',  # Distinct color for dropouts
            opacity=0.36
        ))
        remaining_data.append(go.Bar(
            name=f'Remaining After {visit}', x=[visit], y=[remaining_count],
            marker_color=color_map_dark.get(visit, '#000000'),  # Darker color for remaining
            opacity=0.369
        ))

    # Create the bar chart
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
        # Merging and cleaning data
        merged_df = pd.merge(df_actual[['Study ID', 'Gender', 'age-tier']], 
                             df_dropout[['Study ID', 'Gender', 'age-tier']], 
                             on='Study ID', 
                             how='outer', 
                             suffixes=('_actual', '_dropout'))
        merged_df['Gender'] = merged_df['Gender_actual'].combine_first(merged_df['Gender_dropout'])
        merged_df['age-tier'] = merged_df['age-tier_actual'].combine_first(merged_df['age-tier_dropout'])
        merged_df.drop(columns=['Gender_actual', 'Gender_dropout', 'age-tier_actual', 'age-tier_dropout'], inplace=True)

        # Filtering
        filtered_df = merged_df[merged_df['Gender'].isin(['Female', 'Male']) & merged_df['age-tier'].isin(['40-50', '51-60'])]
        
        # Pivot table
        pivot_table = pd.pivot_table(filtered_df, values='Study ID', index=['Gender'], columns=['age-tier'], aggfunc='count', fill_value=0, margins=True, margins_name='Grand Total')
        pivot_table = pivot_table.reindex(columns=['40-50', '51-60', 'Grand Total'], index=['Female', 'Male', 'Grand Total'])

        # HTML table
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

    # Convert pivot table to HTML
    html_string = pivot_table.to_html(classes='table table-striped')
    return html_string



def generate_html(fig):
    return plotly.io.to_html(fig, include_plotlyjs='cdn', full_html=True)


def plot_visit_status_section(df_long,current_date):
    st.title('Visit Status')
    st.caption('Following chart demonstrates the status of our Visit Status...')
    st.caption(f"Data UTD: {current_date}")
    visit_status_fig = plot_visit_status(df_long)  # Assign the returned figure to a variable
    st.plotly_chart(visit_status_fig)  # Plot the figure
    # Generate HTML string for the "Visit Status" plot
    html_string_status = plotly.io.to_html(visit_status_fig, full_html=True, include_plotlyjs='cdn')
    # Create a BytesIO object from the HTML string
    html_out_status = BytesIO(html_string_status.encode())
    # Create a download button for the "Visit Status" plot
    st.download_button(
        label="Download Visit Status Plot as HTML",
        data=html_out_status,
        file_name='Visit_Status_Plot.html',
        mime='text/html',
    )
    return plot_visit_status(df_long)

def plot_gender_age_distribution_section(df_actual, df_dropout,current_date):
    st.title('Gender and Age-Tier distribution of Actual Participant after Scheduling')
    st.caption('Following table shows the count of individuals including dropout, by Gender and Age Tier...')
    st.caption(f"Data UTD: {current_date}")
    html_string = plot_gender_age_table(df_actual, df_dropout)
    st.markdown(html_string, unsafe_allow_html=True)
    return plot_gender_age_table(df_actual, df_dropout)

def run_cumulative_trials_plot():
    st.title('ABLE Visits Progression')
    uploaded_file = st.sidebar.file_uploader("drop that dope here", type="xlsx")

    if uploaded_file is not None:
        df_screening, df_actual, df_dropout = load_excel_data(uploaded_file)
        progression = calculate_progression(df_screening, df_actual, df_dropout)
        st.write("Total Progression of the Study (*Screening* + *Actual Visit*):")
        progress_bar_html = create_progress_bar(progression)
        html(progress_bar_html)
        
        df = load_data(uploaded_file)
        df_visits = load_data(uploaded_file)
        dropout_df = load_dropout_data(uploaded_file)  # Load dropout data
        
        #df_screening = load_screening_data(uploaded_file)
        current_date = datetime.now().date()
        df_long = reshape_dataframe(df)
        df_visits_long = reshape_dataframe(df_visits)

        
        # Display the progress bar with dropouts included
        display_progress_bar(df_long, dropout_df, style='tralalala')
            
        st.title('Projection of Current ABLE participant')
        data_filters = {
            f"Full projection<br><sub>Data up-to-date: {current_date}</sub>": df_long,
            "Completed Visits": df_long[df_long['Date'].dt.date <= current_date],
            "Upcoming Visits": df_long[df_long['Date'].dt.date > current_date],
        }
        
        # Create columns to display the plots side by side
        cols = st.columns([1, 1, 1])
        with st.expander("Download Plots"):
            for i, (filter_name, filtered_data) in enumerate(data_filters.items()):

                # Generate the plot based on the filtered data
                filtered_data = add_count_columns(filtered_data)
                filtered_data = add_day_of_week(filtered_data)
                fig = go.Figure(data=[generate_trace(filtered_data, visit) for visit in filtered_data['Visit'].unique()])

                # Set up the plot layout
                fig.update_xaxes(dtick="M1", tickformat="%b\n%Y", title="Date")
                fig.update_layout(
                    title=f'{filter_name}', 
                    xaxis_title='Date', 
                    yaxis_title='Cumulative Trials', 
                    autosize=True,  # Set autosize to True to make the plot responsive
                    paper_bgcolor='#111111',  # Set paper background color here
                    plot_bgcolor='#111111',  # Set plot background color here
                    font=dict(color='white')
                )

                # Display the plot in the respective column
                cols[i].plotly_chart(fig, use_container_width=True)  # Set use_container_width to True to make the plot fill the column width

                # Create an expander for the data frame and display the data frame in the respective column
                with cols[i].expander("Show Data"):
                    st.dataframe(filtered_data)

                html_string = plotly.io.to_html(fig, full_html=True, include_plotlyjs='cdn')
                # Create a download button for the plot
                html_out = BytesIO(html_string.encode())
                download_button_label = f"Download {filter_name} Plot as HTML"
                st.download_button(
                    label=download_button_label,
                    data=html_out,
                    file_name=f'{filter_name}.html',
                    mime='text/html',
                )
            
        # Adjusting layout for Visit Status and Gender/Age Distribution
        status_col, gender_age_col = st.columns(2)
        
        with status_col:
            visit_status_fig = plot_visit_status(df_long, dropout_df)  # Pass dropout_df here
            st.title('Visit Status of Project ABLE')
            st.plotly_chart(visit_status_fig)

        with gender_age_col:
            plot_gender_age_distribution_section(df_actual, df_dropout, current_date)
            st.markdown("---")
            st.title('Distribution of Gender and Age Group among Dropouts and Participants Completed Visit 1')
            st.markdown(plot_gender_age_distribution_visit1_dropout(df_actual, df_dropout), unsafe_allow_html=True)


if __name__ == "__main__":
    run_cumulative_trials_plot()
