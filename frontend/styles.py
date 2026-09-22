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
        Welcome Page
        ========================================================= */

        .mc-welcome-content {
            max-width: 760px;
            margin: -10px auto 26px auto;
            text-align: center;
        }

        .mc-welcome-content h1 {
            color: #2F5F8F;
            font-size: 2.7rem;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .mc-welcome-content h2 {
            color: #6AAEF2;
            font-size: 1.35rem;
            font-weight: 500;
            margin-top: 0;
            margin-bottom: 18px;
        }

        .mc-welcome-description {
            max-width: 650px;
            margin: 0 auto;
            color: #718CA7;
            font-size: 1rem;
            line-height: 1.7;
        }

        .mc-welcome-footer {
            margin-top: 22px;
            text-align: center;
            color: #9AAFC2;
            font-size: 0.82rem;
        }



        /* =========================================================
        Authentication
        ========================================================= */

        .mc-auth-header {
            text-align: center;
            margin: 20px 0 28px 0;
        }

        .mc-auth-header h1 {
            color: #2F5F8F;
            font-size: 2rem;
            font-weight: 650;
            margin-bottom: 6px;
        }

        .mc-auth-header p {
            color: #7894AD;
            font-size: 0.95rem;
            margin: 0;
        }

        .mc-auth-switch-text {
            text-align: center;
            color: #829AAF;
            font-size: 0.85rem;
            margin-top: 22px;
            margin-bottom: 8px;
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

        /* =========================
        Sidebar Navigation
        ========================= */

        /* Hide the default radio circles. */
        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }

        /* Style navigation items. */
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            width: 100%;
            padding: 10px 14px !important;
            margin-bottom: 5px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.18s ease;
        }

        /* Navigation text. */
        section[data-testid="stSidebar"] div[role="radiogroup"] label p {
            color: #5F7891 !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
        }

        /* Hover state. */
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: #F0F7FE !important;
        }

        /* Selected navigation item. */
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: #E7F3FE !important;
        }

        /* Selected navigation text. */
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
            color: #4E8FC7 !important;
            font-weight: 600 !important;
        }

        /* =========================
        Setup Sidebar Navigation
        ========================= */

        .mc-disabled-nav {
            margin-top: 8px;
        }

        .mc-disabled-nav div {
            padding: 10px 14px;
            margin-bottom: 5px;
            border-radius: 8px;

            color: #9AAFC2;
            font-size: 0.95rem;
            font-weight: 500;

            cursor: default;
            user-select: none;
        }

        .mc-setup-note {
            margin-top: 18px;
            padding: 12px 14px;

            background: #F5FAFE;
            border: 1px solid #E1EFFB;
            border-radius: 8px;

            color: #7894AD;
            font-size: 0.82rem;
            line-height: 1.5;
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


        /* ---------- Dashboard ---------- */

        .mc-dashboard-path,
        .mc-dashboard-panel {
            background: #FFFFFF;
            border: 1px solid #DCECFB;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(54, 90, 125, 0.06);
        }

        .mc-dashboard-path {
            padding: 24px 28px;
            margin-top: 28px;
            margin-bottom: 24px;
        }

        .mc-dashboard-panel {
            padding: 22px 24px;
            margin-bottom: 12px;
        }

        .mc-dashboard-path-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 24px;
        }

        .mc-dashboard-label {
            color: #7B9AB7;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.07em;
            margin-bottom: 7px;
        }

        .mc-dashboard-path-title {
            color: #2F5F8F;
            font-size: 1.35rem;
            font-weight: 650;
            line-height: 1.3;
        }

        .mc-dashboard-status {
            background: #EDF6FF;
            color: #5C8DB8;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.72rem;
            font-weight: 600;
        }

        .mc-dashboard-progress-header,
        .mc-dashboard-mastery-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #4E6F8E;
            font-size: 0.9rem;
            margin-bottom: 8px;
        }

        .mc-dashboard-progress-track {
            width: 100%;
            height: 8px;
            background: #EDF4FA;
            border-radius: 999px;
            overflow: hidden;
        }

        .mc-dashboard-progress-fill {
            height: 100%;
            background: #78B9F7;
            border-radius: 999px;
        }

        .mc-dashboard-progress-caption {
            color: #8AA2B8;
            font-size: 0.82rem;
            margin-top: 8px;
        }

        .mc-dashboard-topic-title {
            color: #2F5F8F;
            font-size: 1.25rem;
            font-weight: 650;
            line-height: 1.35;
            margin: 8px 0 16px;
        }

        .mc-dashboard-muted {
            color: #8AA2B8;
            font-size: 0.86rem;
        }

        .mc-dashboard-action {
            display: inline-block;
            margin-top: 6px;
            color: #4E8FC7;
            background: #EDF6FF;
            border-radius: 7px;
            padding: 5px 10px;
            font-size: 0.88rem;
            font-weight: 600;
        }

        .mc-dashboard-stat-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #EDF3F8;
            color: #66839F;
            font-size: 0.9rem;
        }

        .mc-dashboard-stat-row:last-child {
            border-bottom: none;
        }

        .mc-dashboard-stat-row strong {
            color: #365A7D;
            font-size: 0.98rem;
            font-weight: 650;
        }

        .mc-dashboard-section-title {
            color: #2F5F8F;
            font-size: 1.35rem;
            font-weight: 650;
            margin-top: 30px;
            margin-bottom: 14px;
        }

        .mc-dashboard-mastery-panel {
            padding-top: 8px;
            padding-bottom: 8px;
        }

        .mc-dashboard-mastery-item {
            padding: 14px 0;
        }

        .mc-dashboard-mastery-header {
            color: #527492;
        }

        .mc-dashboard-mastery-header strong {
            color: #365A7D;
            font-size: 0.9rem;
        }

        .mc-dashboard-weak-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .mc-dashboard-weak-area {
            display: inline-block;
            background: #F2F7FC;
            color: #5C7893;
            border: 1px solid #DCEAF6;
            border-radius: 999px;
            padding: 6px 11px;
            font-size: 0.84rem;
        }


        /* ---------- Learning Path ---------- */

        .mc-path-summary {
            background: #FFFFFF;
            border: 1px solid #DCECFB;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(54, 90, 125, 0.06);
            padding: 24px 28px;
            margin-top: 28px;
            margin-bottom: 24px;
        }

        .mc-path-summary-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 24px;
        }

        .mc-path-name {
            color: #2F5F8F;
            font-size: 1.35rem;
            font-weight: 650;
            line-height: 1.3;
        }

        .mc-path-journey {
            background: #FFFFFF;
            border: 1px solid #DCECFB;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(54, 90, 125, 0.06);
            padding: 12px 26px;
        }

        .mc-path-item {
            display: flex;
            min-height: 118px;
        }

        .mc-path-marker-column {
            width: 52px;
            display: flex;
            flex-direction: column;
            align-items: center;
            flex-shrink: 0;
        }

        .mc-path-marker {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            font-weight: 650;
            z-index: 1;
        }

        .mc-path-line {
            width: 2px;
            flex: 1;
            background: #E2EDF6;
        }

        .mc-path-item:last-child .mc-path-line {
            display: none;
        }

        .mc-path-content {
            flex: 1;
            margin-left: 14px;
            padding: 4px 0 28px;
        }

        .mc-path-item-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }

        .mc-path-topic {
            color: #365A7D;
            font-size: 1.05rem;
            font-weight: 650;
        }

        .mc-path-state {
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 0.74rem;
            font-weight: 600;
        }

        .mc-path-action {
            color: #7B94AB;
            font-size: 0.86rem;
            margin-top: 10px;
        }

        .mc-path-action strong {
            color: #527FA7;
            font-weight: 600;
        }

        /* Completed topic */

        .mc-path-item.completed .mc-path-marker {
            background: #EAF6F0;
            color: #5A9577;
        }

        .mc-path-item.completed .mc-path-state {
            background: #EAF6F0;
            color: #5A9577;
        }

        /* Current topic */

        .mc-path-item.current .mc-path-marker {
            background: #78B9F7;
            color: #FFFFFF;
            box-shadow: 0 0 0 5px #EDF6FF;
        }

        .mc-path-item.current .mc-path-state {
            background: #EDF6FF;
            color: #4E8FC7;
        }

        .mc-path-item.current .mc-path-topic {
            color: #2F5F8F;
        }

        /* Upcoming topic */

        .mc-path-item.upcoming .mc-path-marker {
            background: #F2F6FA;
            color: #8AA2B8;
            border: 1px solid #DCE8F2;
        }

        .mc-path-item.upcoming .mc-path-state {
            background: #F3F6F9;
            color: #8AA2B8;
        }

        .mc-path-item.upcoming .mc-path-topic {
            color: #7890A7;
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


        /* =========================================================
        Diagnostic Questions
        ========================================================= */

        .mc-question-card {
            background: #FFFFFF;

            border: 1px solid #DCECFB;
            border-radius: 14px;

            padding: 26px;

            margin-top: 22px;
            margin-bottom: 22px;

            box-shadow:
                0 4px 16px rgba(47, 95, 143, 0.04);
        }

        .mc-question-number {
            color: #6AAEF2;

            font-size: 0.82rem;
            font-weight: 650;

            text-transform: uppercase;
            letter-spacing: 0.05em;

            margin-bottom: 10px;
        }

        .mc-question-text {
            color: #365F87;

            font-size: 1.15rem;
            font-weight: 600;

            line-height: 1.55;
        }


        /* =========================================================
        Diagnostic Answer Grid
        ========================================================= */

        div[data-testid="stHorizontalBlock"] {
            gap: 14px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
