import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from src.prompt_builder import PromptBuilder
from src.story_generator import StoryGenerator
from src.readability import ReadabilityAnalyzer
from src.feedback import FeedbackCollector
import uuid

st.set_page_config(page_title="AI儿童故事创作系统", layout="wide")

if 'lang' not in st.session_state:
    st.session_state['lang'] = 'en'

def T(zh, en):
    return zh if st.session_state['lang']=='zh' else en

title_col, lang_col = st.columns([8, 1])
with title_col:
    st.markdown(
        f"<h1 style='font-size:2.3em;margin-bottom:0.2em;'>{T('AI增强儿童故事创作系统','AI Children Story Creation System')}</h1>",
        unsafe_allow_html=True
    )
with lang_col:
    lang_map = {"中文 🇨🇳": "zh", "English 🇬🇧": "en"}
    lang_display = st.selectbox(
        "", 
        options=list(lang_map.keys()),
        index=1 if st.session_state.get('lang', 'en') == 'en' else 0
    )
    st.session_state['lang'] = lang_map[lang_display]

st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 3, 1.5], gap="large")
st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

with col1:
    age = st.slider(T('目标年龄','Target Age'), 5, 8, 6)
    character = st.text_input(T('主角（如小狐狸、小熊等）','Main Character (e.g. Little Fox, Little Bear, etc.)'), T('小狐狸','Little Fox'))
    theme = st.text_input(T('教育主题（如合作、诚实、环保等）','Educational Theme (e.g. Cooperation, Honesty, Environmental Protection, etc.)'), T('合作','Cooperation'))
    prompt_style = st.selectbox(T('Prompt风格','Prompt Style'), [T('模板式','Template'), T('结构式','Structured'), T('问句式','Question')])
    word_limit = st.number_input(T('字数上限','Word Limit'), min_value=50, max_value=500, value=300, step=10)
    model_choice = st.selectbox(T('生成模型','Model'), [T('gpt-4o','gpt-4o'), T('claude-3','claude-3'), T('gemini-1.5-pro','gemini-1.5-pro')], index=0, help=T('选择不同大模型进行故事生成','Choose different LLMs to generate stories'))
    if model_choice == "gpt-4o":
        openai_model = "gpt-4o"
        backend = "openai"
    elif model_choice == "claude-3":
        openai_model = None
        backend = "claude"
    elif model_choice == "gemini-1.5-pro":
        openai_model = None
        backend = "gemini"
    else:
        openai_model = None
        backend = None
    st.markdown("---")
    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

with col2:
    prompt_builder = PromptBuilder(age=age, lang=st.session_state['lang'])
    if prompt_style == T('模板式','Template'):
        prompt = prompt_builder.build_template_prompt(character, theme, word_limit)
    elif prompt_style == T('结构式','Structured'):
        prompt = prompt_builder.build_structured_prompt(character, theme)
    else:
        prompt = prompt_builder.build_question_prompt(theme)
    st.subheader(T('生成的Prompt','Generated Prompt'))
    st.markdown(f"{prompt}", unsafe_allow_html=True)
    story = ""
    btn_label = T('再生成故事','Regenerate Story') if "story" in st.session_state and st.session_state["story"] else T('生成故事','Generate Story')
    story_generated = False
    if st.button(btn_label, help=T('点击生成AI故事','Click to generate AI story'), use_container_width=True):
        with st.spinner(T('故事生成中，请稍候...','Generating story, please wait...')):
            if backend == "openai":
                generator = StoryGenerator(model="openai", openai_model=openai_model)
            else:
                generator = StoryGenerator(model=backend)
            story = generator.generate_story(prompt)
        st.session_state["story"] = story
        story_generated = True
    if "story" in st.session_state and st.session_state["story"]:
        st.markdown(f"<h3 style='font-size:1.15em;margin-bottom:0.5em;'>{T('AI生成的故事','AI Generated Story')}</h3>", unsafe_allow_html=True)
        indent_style = "text-indent:2em;" if st.session_state['lang'] == 'zh' else "text-indent:0;"
        for para in st.session_state["story"].split("\n"):
            if para.strip():
                st.markdown(
                    f"<p style='{indent_style}font-size:1.08em;line-height:1.7;margin-bottom:10px;color:#333;'>{para.strip()}</p>",
                    unsafe_allow_html=True
                )
    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

with col3:
    story = st.session_state.get("story", "")

    # 可读性分析区块
    st.subheader(T('可读性分析','Readability Analysis'))
    if story:
        analyzer = ReadabilityAnalyzer()
        readability = analyzer.analyze(story)
        met1, met2 = st.columns(2)
        with met1:
            st.metric("Flesch Reading Ease", f"{readability['Flesch Reading Ease']:.1f}")
            st.metric(T('推荐年龄段','Recommended Age Range'), readability["Recommended Age Range"])
        with met2:
            st.metric(T('句子数','Sentence Count'), readability["Sentence Count"])
            st.metric(T('词数','Word Count'), readability["Word Count"])
        with st.expander(T('详细可读性分析JSON','Detailed Readability JSON')):
            st.json(readability)
    else:
        met1, met2 = st.columns(2)
        with met1:
            st.metric("Flesch Reading Ease", "--")
            st.metric(T('推荐年龄段','Recommended Age Range'), "--")
        with met2:
            st.metric(T('句子数','Sentence Count'), "--")
            st.metric(T('词数','Word Count'), "--")
        st.caption(T('生成故事后可查看可读性分析','You can view readability analysis after generating a story'))
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # 故事反馈区块
    st.subheader(T('故事反馈','Story Feedback'))
    if story:
        rating = st.select_slider(
            T('为本故事打分（1-5星）','Rate this story (1-5 stars)'),
            options=[1,2,3,4,5],
            value=5,
            format_func=lambda x: "⭐"*x
        )
        comment = st.text_area(T('您的建议（可选）','Your suggestions (optional)'), "")
        if st.button(T('提交反馈','Submit Feedback'), key="feedback_btn"):
            feedback_collector = FeedbackCollector()
            story_id = str(uuid.uuid4())
            path = feedback_collector.collect_feedback(
                story_id=story_id,
                rating=rating,
                comment=comment,
                extra={"theme": theme, "character": character, "age": age, "model": model_choice}
            )
            st.success(T('感谢您的反馈！(文件: {} )','Thank you for your feedback! (file: {} )').format(path))
            st.experimental_rerun()
    else:
        st.select_slider(
            T('为本故事打分（1-5星）','Rate this story (1-5 stars)'),
            options=[1,2,3,4,5],
            value=5,
            format_func=lambda x: "⭐"*x,
            disabled=True
        )
        st.text_area(T('您的建议（可选）','Your suggestions (optional)'), "", disabled=True)
        st.button(T('提交反馈','Submit Feedback'), key="feedback_btn", disabled=True)
    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
