"""Shared visual styles for the MindCompass Streamlit interface."""

import streamlit as st


def apply_global_styles():
    """
    Apply the shared MindCompass light blue design system.
    """

    st.markdown(
        """

        <style>

        /* =========================================================
           Main Application
           ========================================================= */

        .stApp {
            background:
                radial-gradient(
                    circle at top right,
                    rgba(217, 236, 255, 0.65),
                    transparent 32%
                ),
                #F8FBFF;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }


        /* =========================================================
           Typography
           ========================================================= */

        h1,
        h2,
        h3 {
            color: #2F5F8F !important;
            font-weight: 650 !important;
        }

        p,
        label {
            color: #6886A5;
        }


        /* =========================================================
           Sidebar
           ========================================================= */

        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #DCECFB;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #2F5F8F !important;
        }


        /* =========================================================
        Buttons
        ========================================================= */

        /* Default / secondary buttons */
        div.stButton > button,
        button[data-testid="stBaseButton-secondary"] {
            width: 100% !important;
            min-height: 44px !important;

            background: #FFFFFF !important;
            color: #5C8DB8 !important;

            border: 1px solid #BFDDF6 !important;
            border-radius: 8px !important;

            font-weight: 500 !important;
            font-size: 0.98rem !important;

            box-shadow: none !important;

            transition: all 0.18s ease;
        }

        /* Default button hover */
        div.stButton > button:hover,
        button[data-testid="stBaseButton-secondary"]:hover {
            background: #F0F7FE !important;
            color: #477FAF !important;

            border-color: #9CCAF0 !important;

            box-shadow:
                0 3px 10px rgba(106, 174, 242, 0.10) !important;

            transform: translateY(-1px);
        }

        /* Primary action buttons */
        div.stButton > button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {
            background: #79B7EE !important;
            color: #FFFFFF !important;

            border: 1px solid #79B7EE !important;
            border-radius: 8px !important;

            font-weight: 600 !important;

            box-shadow:
                0 3px 10px rgba(121, 183, 238, 0.16) !important;
        }

        /* Force text inside primary buttons to stay white */
        div.stButton > button[kind="primary"] p,
        button[data-testid="stBaseButton-primary"] p {
            color: #FFFFFF !important;
        }

        /* Primary button hover */
        div.stButton > button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover {
            background: #68AAE4 !important;
            color: #FFFFFF !important;
            border-color: #68AAE4 !important;

            box-shadow:
                0 4px 12px rgba(104, 170, 228, 0.20) !important;

            transform: translateY(-1px);
        }

        /* Keep primary button text white on hover */
        div.stButton > button[kind="primary"]:hover p,
        button[data-testid="stBaseButton-primary"]:hover p {
            color: #FFFFFF !important;
        }

        /* Button focus */
        div.stButton > button:focus,
        button[data-testid="stBaseButton-primary"]:focus,
        button[data-testid="stBaseButton-secondary"]:focus {
            box-shadow:
                0 0 0 3px rgba(121, 183, 238, 0.15) !important;
        }

        /* =========================================================
           Text Inputs and Select Boxes
           ========================================================= */

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF;

            border-color: #DCECFB;
            border-radius: 10px;
        }

        div[data-baseweb="input"] > div:focus-within {
            border-color: #78B9F7;
        }


        /* =========================================================
           Number Input
           ========================================================= */

        div[data-testid="stNumberInput"] input {
            background-color: #FFFFFF;
        }


        /* =========================================================
           Cards
           ========================================================= */

        .mc-card {
            background: rgba(255, 255, 255, 0.96);

            border: 1px solid #DCECFB;
            border-radius: 16px;

            padding: 24px;
            margin-bottom: 18px;

            box-shadow:
                0 6px 22px
                rgba(47, 95, 143, 0.05);
        }

        .mc-card-title {
            color: #3B6B99;

            font-size: 1.15rem;
            font-weight: 650;

            margin-bottom: 6px;
        }

        .mc-card-text {
            color: #718CA7;

            font-size: 0.95rem;
            line-height: 1.6;
        }


        /* =========================================================
           Page Heading
           ========================================================= */

        .mc-eyebrow {
            color: #6AAEF2;

            font-size: 0.82rem;
            font-weight: 650;

            letter-spacing: 0.08em;
            text-transform: uppercase;

            margin-bottom: 8px;
        }

        .mc-page-title {
            color: #2F5F8F;

            font-size: 2.4rem;
            font-weight: 650;

            line-height: 1.15;

            margin-bottom: 10px;
        }

        .mc-page-description {
            color: #718CA7;

            font-size: 1.05rem;
            line-height: 1.65;

            margin-bottom: 28px;

            max-width: 760px;
        }


        /* =========================================================
           Information Box
           ========================================================= */

        .mc-info {
            background: #EDF6FF;

            border: 1px solid #DCECFB;
            border-radius: 12px;

            padding: 18px 20px;
            margin-top: 20px;

            color: #718CA7;
        }

        .mc-info strong {
            color: #3B6B99;
        }


        /* =========================================================
           Metric Cards
           ========================================================= */

        div[data-testid="stMetric"] {
            background: #FFFFFF;

            border: 1px solid #DCECFB;
            border-radius: 14px;

            padding: 18px;

            box-shadow:
                0 4px 16px
                rgba(47, 95, 143, 0.05);
        }

        div[data-testid="stMetricLabel"] {
            color: #718CA7;
        }

        div[data-testid="stMetricValue"] {
            color: #3B6B99;
        }


        /* =========================================================
           Radio Buttons
           ========================================================= */

        div[role="radiogroup"] label {
            color: #6886A5 !important;
        }


        /* =========================================================
           Captions
           ========================================================= */

        div[data-testid="stCaptionContainer"] {
            color: #718CA7;
        }


        /* =========================================================
           Alerts
           ========================================================= */

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }


        /* =========================================================
           Divider
           ========================================================= */

        hr {
            border-color: #DCECFB !important;
        }

        
        /* =========================================================
        Learning Path Cards
        ========================================================= */

        .mc-path-card {
            min-height: 215px;

            background: #FFFFFF;

            border: 1px solid #DCECFB;
            border-radius: 14px;

            padding: 24px;

            box-shadow:
                0 4px 16px rgba(47, 95, 143, 0.04);

            margin-bottom: 12px;
        }

        .mc-path-card:hover {
            border-color: #A9D2F5;

            box-shadow:
                0 6px 20px rgba(106, 174, 242, 0.10);
        }

        .mc-path-icon {
            font-size: 2rem;
            margin-bottom: 16px;
        }

        .mc-path-title {
            color: #3B6B99;

            font-size: 1.15rem;
            font-weight: 650;

            margin-bottom: 10px;
        }

        .mc-path-description {
            color: #718CA7;

            font-size: 0.92rem;
            line-height: 1.55;
        }

        .mc-selected-path {
            margin-top: 28px;

            background: #EDF6FF;

            border: 1px solid #CFE5F8;
            border-radius: 12px;

            padding: 18px 20px;

            color: #718CA7;
        }

        .mc-selected-path strong {
            color: #6886A5;
            font-size: 0.85rem;
        }

        .mc-selected-path-name {
            color: #3B6B99;

            font-size: 1.1rem;
            font-weight: 650;

            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )