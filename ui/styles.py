import streamlit as st

def inject_style():
    st.markdown("""
    <style>

    .stApp{
        background:#F4F7FC;
    }

    .hero-container{
        background:linear-gradient(135deg,#2563EB,#4F46E5);
        padding:35px;
        border-radius:18px;
        color:white;
        text-align:center;
        margin-bottom:25px;
        box-shadow:0 10px 25px rgba(0,0,0,.18);
    }

    .hero-container h1{
        font-size:45px;
        margin-bottom:10px;
    }

    .hero-container p{
        font-size:18px;
        opacity:.95;
    }

    div[data-testid="stMetric"]{
        background:white;
        padding:18px;
        border-radius:15px;
        border:1px solid #ddd;
        box-shadow:0 5px 15px rgba(0,0,0,.08);
    }

    .stButton>button{
        width:100%;
        border-radius:12px;
        background:#2563EB;
        color:white;
        font-weight:600;
        height:48px;
        border:none;
    }

    .stButton>button:hover{
        background:#1D4ED8;
    }

    div[data-testid="stFileUploader"]{
        background:white;
        border-radius:15px;
        padding:15px;
        border:1px solid #ddd;
    }

    textarea{
        border-radius:12px !important;
    }

    </style>
    """, unsafe_allow_html=True)