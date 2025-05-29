import streamlit as st
import sqlite3
import pandas as pd
import json
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq
import os
from datetime import datetime
import tempfile
import io

# Initialize session state
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

# Database initialization
def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect('hr_recruitment.db')
    cursor = conn.cursor()

    # Create job_posts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT NOT NULL,
            job_description TEXT NOT NULL,
            required_skills TEXT NOT NULL,
            min_cgpa REAL NOT NULL,
            min_experience INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create candidates table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            candidate_name TEXT NOT NULL,
            full_name TEXT,
            email TEXT,
            phone TEXT,
            skills TEXT,
            cgpa REAL,
            experience_years INTEGER,
            education TEXT,
            projects TEXT,
            certifications TEXT,
            resume_text TEXT,
            similarity_score REAL,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES job_posts (id)
        )
    ''')

    conn.commit()
    conn.close()

# Resume text extraction
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using PyMuPDF"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        # Extract text
        doc = fitz.open(tmp_file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        # Clean up temporary file
        os.unlink(tmp_file_path)

        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# Groq LLM integration
def parse_resume_with_groq(resume_text, groq_api_key):
    """Parse resume text using Groq LLM and return structured JSON"""
    try:
        client = Groq(api_key=groq_api_key)

        prompt = f"""
        You are an expert resume parser. Extract the following information from the given resume text and return it as a valid JSON object with these exact keys:

        {{
            "full_name": "",
            "email": "",
            "phone": "",
            "skills": [],
            "cgpa": null,
            "experience_years": null,
            "education": [],
            "projects": [],
            "certifications": []
        }}

        Rules:
        - For skills, extract technical skills, programming languages, tools, frameworks, etc. as a list
        - For CGPA/GPA, look for numerical values (e.g., 8.5, 3.8, etc.). If percentage, convert to 10-point scale
        - For experience_years, calculate total years of work experience as an integer
        - For education, include degree, university/college, and year as objects
        - For projects, include title and brief description as objects
        - For certifications, list certification names
        - If any field is not found, use "Not Mentioned" for strings, empty array for lists, or null for numbers
        - Return only valid JSON, no additional text

        Resume Text:
        {resume_text[:4000]}  # Limit text to avoid token limits
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
            temperature=0.1,
        )

        response_text = chat_completion.choices[0].message.content

        # Try to parse JSON
        try:
            parsed_data = json.loads(response_text)
            return parsed_data
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                st.error("Failed to parse resume data as JSON")
                return None

    except Exception as e:
        st.error(f"Error parsing resume with Groq: {str(e)}")
        return None

# Skill similarity calculation
@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def calculate_skill_similarity(candidate_skills, job_skills, model):
    """Calculate cosine similarity between candidate and job skills"""
    try:
        if not candidate_skills or not job_skills:
            return 0.0

        # Convert skills to strings if they're lists
        candidate_skills_text = " ".join(candidate_skills) if isinstance(candidate_skills, list) else str(candidate_skills)
        job_skills_text = " ".join(job_skills) if isinstance(job_skills, list) else str(job_skills)

        # Get embeddings
        embeddings = model.encode([candidate_skills_text, job_skills_text])

        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return 0.0

# Database operations
def save_job_post(job_title, job_description, required_skills, min_cgpa, min_experience):
    """Save job post to database"""
    conn = sqlite3.connect('hr_recruitment.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO job_posts (job_title, job_description, required_skills, min_cgpa, min_experience)
        VALUES (?, ?, ?, ?, ?)
    ''', (job_title, job_description, required_skills, min_cgpa, min_experience))
    conn.commit()
    job_id = cursor.lastrowid
    conn.close()
    return job_id

def get_job_posts():
    """Retrieve all job posts"""
    conn = sqlite3.connect('hr_recruitment.db')
    df = pd.read_sql_query("SELECT * FROM job_posts ORDER BY created_at DESC", conn)
    conn.close()
    return df

def delete_job_post(job_id):
    """Delete job post and associated candidates"""
    conn = sqlite3.connect('hr_recruitment.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM candidates WHERE job_id = ?", (job_id,))
    cursor.execute("DELETE FROM job_posts WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()

def save_candidate(job_id, candidate_data, resume_text):
    """Save candidate to database"""
    conn = sqlite3.connect('hr_recruitment.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO candidates (job_id, candidate_name, full_name, email, phone, skills, cgpa, 
                              experience_years, education, projects, certifications, resume_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        job_id,
        candidate_data.get('candidate_name', ''),
        candidate_data.get('full_name', 'Not Mentioned'),
        candidate_data.get('email', 'Not Mentioned'),
        candidate_data.get('phone', 'Not Mentioned'),
        json.dumps(candidate_data.get('skills', [])),
        candidate_data.get('cgpa'),
        candidate_data.get('experience_years'),
        json.dumps(candidate_data.get('education', [])),
        json.dumps(candidate_data.get('projects', [])),
        json.dumps(candidate_data.get('certifications', [])),
        resume_text
    ))
    conn.commit()
    candidate_id = cursor.lastrowid
    conn.close()
    return candidate_id

def get_candidates(job_id):
    """Retrieve candidates for a job"""
    conn = sqlite3.connect('hr_recruitment.db')
    df = pd.read_sql_query("SELECT * FROM candidates WHERE job_id = ? ORDER BY created_at DESC", conn, params=(job_id,))
    conn.close()
    return df

def update_candidate_status(candidate_id, similarity_score, status):
    """Update candidate shortlisting status"""
    conn = sqlite3.connect('hr_recruitment.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE candidates SET similarity_score = ?, status = ? WHERE id = ?
    ''', (similarity_score, status, candidate_id))
    conn.commit()
    conn.close()

def delete_candidate(candidate_id):
    """Delete candidate"""
    conn = sqlite3.connect('hr_recruitment.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
    conn.commit()
    conn.close()

# Streamlit app
def main():
    st.set_page_config(
        page_title="Smart Resume Shortlisting System",
        page_icon="📄",
        layout="wide"
    )

    st.title("🎯 Smart Resume Shortlisting System")
    st.markdown("---")

    # Initialize database
    if not st.session_state.db_initialized:
        init_database()
        st.session_state.db_initialized = True

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Create Job Post",
        "View Job Posts",
        "Upload Resumes",
        "Shortlist Candidates",
        "View Results"
    ])

    # Groq API Key input
    st.sidebar.markdown("---")
    # groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API key")
    groq_api_key = 'YOUR API KEY'

    if page == "Create Job Post":
        create_job_post_page()
    elif page == "View Job Posts":
        view_job_posts_page()
    elif page == "Upload Resumes":
        upload_resumes_page(groq_api_key)
    elif page == "Shortlist Candidates":
        shortlist_candidates_page()
    elif page == "View Results":
        view_results_page()

def create_job_post_page():
    """Page for creating job posts"""
    st.header("📋 Create New Job Post")

    with st.form("job_post_form"):
        col1, col2 = st.columns(2)

        with col1:
            job_title = st.text_input("Job Title", placeholder="e.g., Software Engineer")
            required_skills = st.text_area("Required Skills (comma-separated)",
                                         placeholder="Python, Machine Learning, SQL, Git")
            min_cgpa = st.number_input("Minimum CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

        with col2:
            job_description = st.text_area("Job Description", height=100,
                                         placeholder="Describe the role, responsibilities, and requirements...")
            min_experience = st.number_input("Minimum Experience (years)", min_value=0, value=2, step=1)

        submitted = st.form_submit_button("Create Job Post", type="primary")

        if submitted:
            if job_title and job_description and required_skills:
                job_id = save_job_post(job_title, job_description, required_skills, min_cgpa, min_experience)
                st.success(f"✅ Job post created successfully! Job ID: {job_id}")
                st.balloons()
            else:
                st.error("❌ Please fill in all required fields")

def view_job_posts_page():
    """Page for viewing and managing job posts"""
    st.header("📑 View Job Posts")

    job_posts = get_job_posts()

    if job_posts.empty:
        st.info("No job posts found. Create your first job post!")
        return

    for _, job in job_posts.iterrows():
        with st.expander(f"{job['job_title']} (ID: {job['id']})"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**Description:** {job['job_description']}")
                st.write(f"**Required Skills:** {job['required_skills']}")
                st.write(f"**Min CGPA:** {job['min_cgpa']} | **Min Experience:** {job['min_experience']} years")
                st.write(f"**Created:** {job['created_at']}")

            with col2:
                if st.button(f"Delete", key=f"delete_{job['id']}", type="secondary"):
                    delete_job_post(job['id'])
                    st.rerun()

def upload_resumes_page(groq_api_key):
    """Page for uploading and processing resumes"""
    st.header("📤 Upload Resumes")

    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to continue")
        return

    # Select job post
    job_posts = get_job_posts()
    if job_posts.empty:
        st.info("No job posts found. Please create a job post first.")
        return

    job_options = {f"{row['job_title']} (ID: {row['id']})": row['id'] for _, row in job_posts.iterrows()}
    selected_job = st.selectbox("Select Job Post", options=list(job_options.keys()))
    job_id = job_options[selected_job]

    st.markdown("---")

    # File upload
    uploaded_files = st.file_uploader(
        "Upload Resume PDFs",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF resumes"
    )

    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")

            # Extract text from PDF
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error(f"Failed to extract text from {uploaded_file.name}")
                continue

            # Parse with Groq
            candidate_data = parse_resume_with_groq(resume_text, groq_api_key)
            if not candidate_data:
                st.error(f"Failed to parse resume {uploaded_file.name}")
                continue

            # Add candidate name from filename
            candidate_data['candidate_name'] = uploaded_file.name.replace('.pdf', '')

            # Save to database
            candidate_id = save_candidate(job_id, candidate_data, resume_text)

            progress_bar.progress((i + 1) / len(uploaded_files))

        status_text.text("✅ All resumes processed successfully!")
        st.success(f"Processed {len(uploaded_files)} resumes for job: {selected_job}")

def shortlist_candidates_page():
    """Page for shortlisting candidates based on criteria"""
    st.header("🎯 Shortlist Candidates")

    # Select job post
    job_posts = get_job_posts()
    if job_posts.empty:
        st.info("No job posts found. Please create a job post first.")
        return

    job_options = {f"{row['job_title']} (ID: {row['id']})": row['id'] for _, row in job_posts.iterrows()}
    selected_job = st.selectbox("Select Job Post", options=list(job_options.keys()))
    job_id = job_options[selected_job]

    # Get job details
    job_details = job_posts[job_posts['id'] == job_id].iloc[0]

    st.markdown("### Job Requirements")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min CGPA", job_details['min_cgpa'])
    with col2:
        st.metric("Min Experience", f"{job_details['min_experience']} years")
    with col3:
        similarity_threshold = st.slider("Skill Similarity Threshold", 0.0, 1.0, 0.65, 0.05)

    st.markdown("---")

    if st.button("🚀 Start Shortlisting Process", type="primary"):
        candidates = get_candidates(job_id)

        if candidates.empty:
            st.info("No candidates found for this job post.")
            return

        # Load sentence transformer model
        with st.spinner("Loading AI model..."):
            model = load_sentence_transformer()

        progress_bar = st.progress(0)
        status_text = st.empty()

        job_skills = [skill.strip() for skill in job_details['required_skills'].split(',')]

        for i, (_, candidate) in enumerate(candidates.iterrows()):
            status_text.text(f"Evaluating {candidate['candidate_name']}...")

            # Parse candidate skills
            try:
                candidate_skills = json.loads(candidate['skills']) if candidate['skills'] else []
            except:
                candidate_skills = []

            # Calculate similarity
            similarity_score = calculate_skill_similarity(candidate_skills, job_skills, model)

            # Check eligibility
            cgpa_ok = candidate['cgpa'] >= job_details['min_cgpa'] if candidate['cgpa'] is not None else False
            exp_ok = candidate['experience_years'] >= job_details['min_experience'] if candidate['experience_years'] is not None else False
            skill_ok = similarity_score >= similarity_threshold

            # Determine status
            if cgpa_ok and exp_ok and skill_ok:
                status = "Shortlisted"
            else:
                status = "Rejected"

            # Update database
            update_candidate_status(candidate['id'], similarity_score, status)

            progress_bar.progress((i + 1) / len(candidates))

        status_text.text("✅ Shortlisting completed!")
        st.success("Shortlisting process completed successfully!")
        st.balloons()

def view_results_page():
    """Page for viewing shortlisting results"""
    st.header("📊 Shortlisting Results")

    # Select job post
    job_posts = get_job_posts()
    if job_posts.empty:
        st.info("No job posts found.")
        return

    job_options = {f"{row['job_title']} (ID: {row['id']})": row['id'] for _, row in job_posts.iterrows()}
    selected_job = st.selectbox("Select Job Post", options=list(job_options.keys()))
    job_id = job_options[selected_job]

    candidates = get_candidates(job_id)

    if candidates.empty:
        st.info("No candidates found for this job post.")
        return

    # Summary metrics
    total_candidates = len(candidates)
    shortlisted = len(candidates[candidates['status'] == 'Shortlisted'])
    rejected = len(candidates[candidates['status'] == 'Rejected'])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Candidates", total_candidates)
    with col2:
        st.metric("Shortlisted", shortlisted, delta=f"{shortlisted/total_candidates*100:.1f}%" if total_candidates > 0 else "0%")
    with col3:
        st.metric("Rejected", rejected)

    st.markdown("---")

    # Candidates table
    for _, candidate in candidates.iterrows():
        status_emoji = "✅" if candidate['status'] == 'Shortlisted' else "❌"

        with st.expander(f"{status_emoji} {candidate['candidate_name']} - {candidate['status']}"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**Full Name:** {candidate['full_name']}")
                st.write(f"**Email:** {candidate['email']}")
                st.write(f"**Phone:** {candidate['phone']}")

                if candidate['cgpa']:
                    st.write(f"**CGPA:** {candidate['cgpa']}")
                if candidate['experience_years']:
                    st.write(f"**Experience:** {candidate['experience_years']} years")

                if candidate['skills']:
                    try:
                        skills = json.loads(candidate['skills'])
                        st.write(f"**Skills:** {', '.join(skills[:10])}...")  # Show first 10 skills
                    except:
                        st.write(f"**Skills:** {candidate['skills']}")

                if candidate['similarity_score']:
                    st.write(f"**Skill Match Score:** {candidate['similarity_score']:.3f}")

            with col2:
                if st.button(f"Delete", key=f"del_{candidate['id']}", type="secondary"):
                    delete_candidate(candidate['id'])
                    st.rerun()

    # Export functionality
    st.markdown("---")
    if st.button("📥 Download Shortlisted Candidates CSV", type="primary"):
        shortlisted_candidates = candidates[candidates['status'] == 'Shortlisted']

        if not shortlisted_candidates.empty:
            # Prepare CSV data
            csv_data = shortlisted_candidates[['candidate_name', 'full_name', 'email', 'phone',
                                            'cgpa', 'experience_years', 'similarity_score']].copy()

            # Convert to CSV
            csv_buffer = io.StringIO()
            csv_data.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()

            st.download_button(
                label="Download CSV",
                data=csv_string,
                file_name=f"shortlisted_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No shortlisted candidates to export.")

if __name__ == "__main__":
    main()
