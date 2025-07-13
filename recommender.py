import streamlit as st
import numpy as np
import pandas as pd


def wide_space_default():
    st.set_page_config(layout="wide")

wide_space_default()


@st.cache_data
def get_course_content_dataset() -> pd.DataFrame:
    course_content_df=pd.read_csv(r"course_content.csv")
    return course_content_df[['COURSE_ID','TITLE','DESCRIPTION']]

@st.cache_data
def get_item_bow_sim_dataset() -> pd.DataFrame:
    item_bow_sim_df=pd.read_csv(r"item_bow_sim.csv")
    return item_bow_sim_df

@st.cache_data
def get_Bow_dataset() -> pd.DataFrame:
    Bow_df=pd.read_csv(r"Bow.csv")

    # Group the DataFrame by course index and ID, and get the maximum value for each group
    grouped_df = Bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    # Create a dictionary mapping indices to course IDs
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    # Create a dictionary mapping course IDs to indices
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    # Clean up temporary DataFrame
    del grouped_df
    
    return idx_id_dict, id_idx_dict


def generate_recommendations_for_one_user(enrolled_course_ids, unselected_course_ids, id_idx_dict, sim_matrix):
    # Create a dictionary to store your recommendation results
    res = {}
    # Set a threshold for similarity
    # Set to zero so that the user can set number of results manually in streamlit app
    threshold = 0.0 
    # Iterate over enrolled courses
    for enrolled_course in enrolled_course_ids:
        # Iterate over unselected courses
        for unselect_course in unselected_course_ids:
            # Check if both enrolled and unselected courses exist in the id_idx_dict
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                # Initialize similarity value
                sim = 0
                # Find the two indices for each enrolled_course and unselect_course, based on their two ids
                course1=id_idx_dict[enrolled_course]
                course2=id_idx_dict[unselect_course]
                # Calculate the similarity between an enrolled_course and an unselect_course
                # e.g., Course ML0151EN's index is 200 and Course ML0101ENv3's index is 158
                # Find the similarity value from the sim_matrix
                # sim = sim_matrix[200][158]
                sim=sim_matrix[course1][course2]
                 # Check if the similarity exceeds the threshold
                if sim > threshold:
                    # Update recommendation dictionary with course ID and similarity score
                    if unselect_course not in res:
                        # If the unselected course is not already in the recommendation dictionary (`res`), add it.
                        res[unselect_course] = sim
                    else:
                        # If the unselected course is already in the recommendation dictionary (`res`), compare the similarity score.
                        # If the current similarity score is greater than or equal to the existing similarity score for the course,
                        # update the similarity score in the recommendation dictionary (`res`) with the current similarity score.
                        if sim >= res[unselect_course]:
                            res[unselect_course] = sim
                            
    # Sort the results by similarity
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
     # Return the recommendation dictionary
    return res



column_configuration = {
    "COURSE_ID": st.column_config.TextColumn(
        "Course ID", help="The ID of the course", max_chars=100, width="small"
    ),
    "TITLE": st.column_config.TextColumn(
        "Course Title", help="The title of the course", max_chars=500, width="medium"
    ),
    "DESCRIPTION": st.column_config.TextColumn(
        "Course Description", help="The description of the course", width="large"
    ),
}


column_configuration_recc = {
    "Similarity_Score": st.column_config.TextColumn(
        "Similarity Score", help="How similar the course is to your selections", max_chars=100, width="small"
    ),
    "Course": st.column_config.TextColumn(
        "Course ID", help="The ID of the course", max_chars=100, width="small"
    ),
    "TITLE": st.column_config.TextColumn(
        "Course Title", help="The title of the course", max_chars=500, width="medium"
    ),
    "DESCRIPTION": st.column_config.TextColumn(
        "Course Description", help="The description of the course", width="large"
    ),
}

st.header("Course Recommender system using course title and description BOW similarity scores")

select, compare = st.tabs(["Select Courses","See similar courses"])

with select:
    st.header("All Courses")

    course_content_df = get_course_content_dataset()

    event = st.dataframe(
        course_content_df,
        column_config=column_configuration,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        width=1000,
        )

    st.header("Selected Courses")
    courses = event.selection.rows
    filtered_df = course_content_df.iloc[courses]
    st.dataframe(
        filtered_df,
        column_config=column_configuration,
        use_container_width=True,
        )
    
with compare:
    
    

    if len(courses) > 0:
        res_to_filter = st.slider('Number of results', 100, 1, 100)
        idx_id_dict, id_idx_dict=get_Bow_dataset()
        item_bow_sim_df=get_item_bow_sim_dataset()
        item_bow_sim_matrix=item_bow_sim_df.to_numpy()
    
        all_courses = set(course_content_df['COURSE_ID'])
        enrolled_course_ids=filtered_df['COURSE_ID']
        unselected_course_ids = all_courses.difference(enrolled_course_ids)
        
        Recommend_courses=generate_recommendations_for_one_user(enrolled_course_ids, unselected_course_ids, id_idx_dict, item_bow_sim_matrix)
        Recommend_courses=pd.DataFrame(Recommend_courses,index=[0]).T.reset_index().rename(columns={'index':'Course',0:'Similarity_Score'})
        Recommend_courses=Recommend_courses.merge(course_content_df[['COURSE_ID','TITLE','DESCRIPTION']],left_on='Course',right_on='COURSE_ID').drop(['COURSE_ID'],axis=1)
        
        st.header("Courses you may enjoy")
        st.dataframe(
            Recommend_courses.loc[0:res_to_filter,['Similarity_Score','Course','TITLE','DESCRIPTION']],
            column_config=column_configuration_recc,
            use_container_width=True,
            height=800
            )

    else:
        st.markdown("No Courses selected.")
