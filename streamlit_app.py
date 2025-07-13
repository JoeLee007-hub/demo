import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="AI儿童故事创作系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 语言设置
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'en'

def T(zh, en):
    return zh if st.session_state['lang']=='zh' else en

# 主页面标题
st.markdown(
    f"<h1 style='text-align: center; color: #2E86AB; margin-bottom: 2rem;'>{T('AI增强儿童故事创作系统', 'AI Children Story Creation System')}</h1>",
    unsafe_allow_html=True
)

# 语言选择
lang_col1, lang_col2, lang_col3 = st.columns([1, 1, 1])
with lang_col2:
    lang_map = {"中文": "zh", "English": "en"}
    lang_display = st.selectbox(
        T("选择语言", "Select Language"),
        options=list(lang_map.keys()),
        index=1 if st.session_state.get('lang', 'en') == 'en' else 0
    )
    st.session_state['lang'] = lang_map[lang_display]

# 系统介绍
st.markdown("---")
st.markdown(
    f"### {T('系统简介', 'System Overview')}"
)
st.markdown(
    T(
        """本系统是一个基于人工智能的儿童故事创作平台，旨在为5-8岁儿童提供个性化、教育性的故事内容。
        系统支持多种AI模型，提供实时可读性分析，并收集用户反馈以持续改进故事质量。""",
        """This system is an AI-powered children's story creation platform designed to provide personalized and educational story content for children aged 5-8.
        The system supports multiple AI models, provides real-time readability analysis, and collects user feedback to continuously improve story quality."""
    )
)

# 功能导航
st.markdown("---")
st.markdown(
    f"### {T('功能导航', 'Feature Navigation')}"
)

# 两个功能按钮
func_col1, func_col2 = st.columns(2)

with func_col1:
    if st.button(T("故事创作", "Story Generation"), use_container_width=True, type="primary"):
        st.switch_page("pages/story_generation.py")
    st.markdown(
        T(
            "创作个性化的儿童故事，支持多种主题和角色定制",
            "Create personalized children's stories with various themes and character customization"
        )
    )

with func_col2:
    if st.button(T("数据分析", "Data Analysis"), use_container_width=True, type="secondary"):
        st.switch_page("pages/data_analysis.py")
    st.markdown(
        T(
            "分析故事数据，查看用户反馈和系统性能统计",
            "Analyze story data, view user feedback and system performance statistics"
        )
    )

# 系统特性
st.markdown("---")
st.markdown(
    f"### {T('系统特性', 'System Features')}"
)

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    st.markdown(
        f"""
        #### {T('个性化定制', 'Personalized Customization')}
        {T('- 年龄适配（5-8岁）', '- Age adaptation (5-8 years)')}
        {T('- 角色自定义', '- Character customization')}
        {T('- 主题选择', '- Theme selection')}
        {T('- 字数控制', '- Word count control')}
        """
    )

with feature_col2:
    st.markdown(
        f"""
        #### {T('AI模型支持', 'AI Model Support')}
        {T('- GPT-4o 模型', '- GPT-4o Model')}
        {T('- 智能故事生成', '- Intelligent story generation')}
        {T('- 可读性分析', '- Readability analysis')}
        {T('- 教育内容优化', '- Educational content optimization')}
        
        """
    )

with feature_col3:
    st.markdown(
        f"""
        #### {T('智能分析', 'Intelligent Analysis')}
        {T('- Flesch可读性评分', '- Flesch readability scoring')}
        {T('- 年龄适宜性评估', '- Age appropriateness assessment')}
        {T('- 语言复杂度分析', '- Language complexity analysis')}
        {T('- 实时文本分析', '- Real-time text analysis')}
        """
    )

# 页脚
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #666; margin-top: 2rem;'>{T('AI增强儿童故事创作系统 - 让每个孩子都能享受个性化的故事体验', 'AI Children Story Creation System - Personalized Story Experience for Every Child')}</div>",
    unsafe_allow_html=True
)
