import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any

from src.core.data_manager import DataManager
from src.core.readability import ReadabilityAnalyzer
from src.utils.helpers import *
from src.utils.formatters import *

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(page_title="数据分析 - AI儿童故事创作系统", layout="wide")

# 语言设置
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'en'

def T(zh, en):
    return zh if st.session_state['lang']=='zh' else en

# 统计分析模块
class StatisticalAnalysis:
    """统计分析类，提供完整的数据分析功能"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_stories(self, df: pd.DataFrame) -> Dict[str, Any]:
        """完整统计分析功能
        
        Args:
            df: 故事数据DataFrame
            
        Returns:
            包含各种统计分析结果的字典
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for analysis")
            return {}
        
        try:
            results = {
                'descriptive_stats': self._descriptive_analysis(df),
                'correlation_analysis': self._correlation_analysis(df),
                'hypothesis_testing': self._hypothesis_testing(df),
                'distribution_analysis': self._analyze_distributions(df)
            }
            self.logger.info(f"Statistical analysis completed for {len(df)} records")
            return results
        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {str(e)}")
            return {}
    
    def _descriptive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """描述性统计分析
        
        Args:
            df: 输入数据框
            
        Returns:
            包含基本统计信息和异常值检测结果的字典
        """
        numeric_cols = ['flesch_score', 'word_count', 'sentence_count']
        available_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not available_cols:
            self.logger.warning("No numeric columns available for descriptive analysis")
            return {}
        
        try:
            return {
                'basic_stats': df[available_cols].describe(),
                'outliers': self._detect_outliers(df, available_cols),
                'missing_values': df[available_cols].isnull().sum().to_dict()
            }
        except Exception as e:
            self.logger.error(f"Error in descriptive analysis: {str(e)}")
            return {}
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """相关性分析
        
        Args:
            df: 输入数据框
            
        Returns:
            包含Pearson和Spearman相关系数的字典
        """
        numeric_cols = ['flesch_score', 'word_count', 'sentence_count', 'age']
        available_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(available_cols) < 2:
            self.logger.warning("Insufficient numeric columns for correlation analysis")
            return {}
        
        try:
            # 移除包含NaN的行
            clean_df = df[available_cols].dropna()
            if clean_df.empty:
                return {}
                
            return {
                'pearson_correlations': clean_df.corr(method='pearson'),
                'spearman_correlations': clean_df.corr(method='spearman'),
                'sample_size': len(clean_df)
            }
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {str(e)}")
            return {}
    
    def _hypothesis_testing(self, df):
        """假设检验"""
        results = {}
        
        # 比较不同prompt风格的可读性
        if 'prompt_style' in df.columns and 'flesch_score' in df.columns:
            results['prompt_style_anova'] = self._compare_prompt_styles(df)
        
        # 比较不同年龄组的可读性
        if 'age' in df.columns and 'flesch_score' in df.columns:
            results['age_group_comparison'] = self._compare_age_groups(df)
            
        return results
    
    def _compare_prompt_styles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """比较不同prompt风格的效果
        
        Args:
            df: 输入数据框
            
        Returns:
            ANOVA检验结果字典
        """
        try:
            # 数据验证
            if 'prompt_style' not in df.columns or 'flesch_score' not in df.columns:
                return {}
            
            # 清理数据
            clean_df = df[['prompt_style', 'flesch_score']].dropna()
            if clean_df.empty:
                return {}
            
            # 确保有足够的数据进行比较
            style_groups = clean_df.groupby('prompt_style')['flesch_score'].apply(lambda x: list(x))
            valid_groups = [group for group in style_groups.values if len(group) >= 2]  # 每组至少2个样本
            
            if len(valid_groups) >= 2:
                # 转换为numpy数组以确保数据类型正确
                valid_groups = [np.array(group, dtype=float) for group in valid_groups]
                
                f_stat, p_value = stats.f_oneway(*valid_groups)
                
                # 计算效应量 (eta squared)
                all_data = np.concatenate(valid_groups)
                total_var = np.var(all_data, ddof=1)
                group_means = [np.mean(group) for group in valid_groups]
                group_sizes = [len(group) for group in valid_groups]
                overall_mean = np.mean(all_data)
                
                between_var = sum(size * (mean - overall_mean)**2 for size, mean in zip(group_sizes, group_means))
                within_var = sum((len(group)-1) * np.var(group, ddof=1) for group in valid_groups)
                eta_squared = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0
                
                return {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'eta_squared': float(eta_squared),
                    'group_count': len(valid_groups),
                    'total_samples': sum(group_sizes)
                }
        except Exception as e:
            self.logger.error(f"Prompt style comparison failed: {str(e)}")
        return {}
    
    def _compare_age_groups(self, df):
        """比较不同年龄组"""
        try:
            age_groups = df.groupby('age')['flesch_score'].apply(list).to_dict()
            if len(age_groups) >= 2:
                groups = [scores for scores in age_groups.values() if len(scores) > 1]
                if len(groups) >= 2:
                    f_stat, p_value = stats.f_oneway(*groups)
                    return {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
        except Exception as e:
            st.warning(f"Age group comparison failed: {str(e)}")
        return {}
    
    def _analyze_distributions(self, df):
        """分析数据分布特征"""
        results = {}
        numeric_cols = ['flesch_score', 'word_count', 'sentence_count']
        
        for col in numeric_cols:
            if col in df.columns and not df[col].empty:
                try:
                    data = df[col].dropna()
                    if len(data) > 3:
                        results[col] = {
                            'skewness': stats.skew(data),
                            'kurtosis': stats.kurtosis(data),
                            'normality_test': stats.shapiro(data) if len(data) <= 5000 else None
                        }
                except Exception as e:
                    continue
        return results
    
    def _detect_outliers(self, df, columns):
        """检测异常值"""
        outliers = {}
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        return outliers

# 评估指标体系
class EvaluationMetrics:
    """评估指标类，提供系统性能的多维度评估"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def comprehensive_evaluation(self, stories_df: pd.DataFrame) -> Dict[str, Any]:
        """综合评估系统性能
        
        Args:
            stories_df: 故事数据框
            
        Returns:
            包含各维度评估结果的字典
        """
        evaluation = {}
        
        try:
            # 内容质量评估
            if not stories_df.empty:
                evaluation['content_quality'] = self._evaluate_content_quality(stories_df)
            
            # 可读性评估
            if not stories_df.empty:
                evaluation['readability_metrics'] = self._evaluate_readability(stories_df)
            
            # Prompt效果评估
            if not stories_df.empty:
                evaluation['prompt_effectiveness'] = self._evaluate_prompt_effectiveness(stories_df)
            
            self.logger.info(f"Comprehensive evaluation completed with {len(evaluation)} metrics")
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive evaluation: {str(e)}")
            return {}
    
    def _evaluate_content_quality(self, df):
        """内容质量评估"""
        quality_metrics = {}
        
        if 'word_count' in df.columns:
            quality_metrics['avg_word_count'] = df['word_count'].mean()
            quality_metrics['word_count_std'] = df['word_count'].std()
        
        if 'sentence_count' in df.columns:
            quality_metrics['avg_sentence_count'] = df['sentence_count'].mean()
            quality_metrics['sentence_count_std'] = df['sentence_count'].std()
        
        # 词汇多样性（基于词数和句数的比例）
        if 'word_count' in df.columns and 'sentence_count' in df.columns:
            df_clean = df.dropna(subset=['word_count', 'sentence_count'])
            if not df_clean.empty:
                quality_metrics['avg_words_per_sentence'] = (df_clean['word_count'] / df_clean['sentence_count']).mean()
        
        return quality_metrics
    
    def _evaluate_readability(self, df):
        """可读性评估"""
        readability_metrics = {}
        
        if 'flesch_score' in df.columns:
            readability_metrics['flesch_reading_ease'] = df['flesch_score'].describe()
            readability_metrics['age_appropriateness'] = self._check_age_appropriateness(df)
        
        return readability_metrics
    
    def _evaluate_satisfaction(self, df):
        """用户满意度评估"""
        satisfaction_metrics = {}
        
        if 'rating' in df.columns:
            satisfaction_metrics['overall_rating'] = df['rating'].describe()
            satisfaction_metrics['satisfaction_distribution'] = df['rating'].value_counts().to_dict()
            satisfaction_metrics['positive_feedback_rate'] = (df['rating'] >= 4).mean()
        
        return satisfaction_metrics
    
    def _evaluate_prompt_effectiveness(self, stories_df):
        """Prompt效果评估"""
        effectiveness_metrics = {}
        
        if 'prompt_style' in stories_df.columns:
            # 不同风格的可读性差异
            style_readability = stories_df.groupby('prompt_style')['flesch_score'].agg(['mean', 'std', 'count'])
            effectiveness_metrics['style_readability'] = style_readability.to_dict()
            
            # 不同风格的词数差异
            if 'word_count' in stories_df.columns:
                style_wordcount = stories_df.groupby('prompt_style')['word_count'].agg(['mean', 'std', 'count'])
                effectiveness_metrics['style_wordcount'] = style_wordcount.to_dict()
        
        return effectiveness_metrics
    
    def _check_age_appropriateness(self, df):
        """检查年龄适宜性"""
        if 'flesch_score' in df.columns and 'age' in df.columns:
            age_readability = df.groupby('age')['flesch_score'].mean().to_dict()
            return age_readability
        return {}

# 页面标题
title_col, lang_col = st.columns([8, 1])
with title_col:
    st.markdown(
        f"<h1 style='font-size:2.3em;margin-bottom:0.2em;'>{T('数据分析与可视化','Data Analysis & Visualization')}</h1>",
        unsafe_allow_html=True
    )
with lang_col:
    lang_map = {"中文": "zh", "English": "en"}
    lang_display = st.selectbox(
        "", 
        options=list(lang_map.keys()),
        index=1 if st.session_state.get('lang', 'en') == 'en' else 0
    )
    st.session_state['lang'] = lang_map[lang_display]

# 数据加载函数
@st.cache_data(ttl=300)  # 5分钟缓存
def load_stories_data() -> List[Dict[str, Any]]:
    """加载故事数据
    
    Returns:
        故事数据列表
    """
    try:
        data_manager = DataManager()
        stories = data_manager.list_stories()
        logger.info(f"Successfully loaded {len(stories)} stories")
        return stories
    except Exception as e:
        error_msg = f'Failed to load stories data: {str(e)}'
        logger.error(error_msg)
        st.error(T(f'加载故事数据失败：{str(e)}', error_msg))
        return []

@st.cache_data(ttl=300)  # 5分钟缓存
def load_feedback_data() -> List[Dict[str, Any]]:
    """加载反馈数据
    
    Returns:
        反馈数据列表
    """
    try:
        from src.config import FEEDBACK_DIR
        feedback_dir = Path(FEEDBACK_DIR)
        feedback_data = []
        
        if feedback_dir.exists():
            json_files = list(feedback_dir.glob('*.json'))
            for filepath in json_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        feedback_data.append(data)
                except (json.JSONDecodeError, IOError) as file_error:
                    logger.warning(f"Failed to load feedback file {filepath}: {str(file_error)}")
                    continue
        
        logger.info(f"Successfully loaded {len(feedback_data)} feedback entries")
        return feedback_data
        
    except Exception as e:
        error_msg = f'Failed to load feedback data: {str(e)}'
        logger.error(error_msg)
        st.error(T(f'加载反馈数据失败：{str(e)}', error_msg))
        return []

# 数据处理函数
@st.cache_data
def process_stories_data(stories: List[Dict[str, Any]]) -> pd.DataFrame:
    """处理故事数据为DataFrame
    
    Args:
        stories: 故事数据列表
        
    Returns:
        处理后的DataFrame
    """
    if not stories:
        logger.warning("No stories provided for processing")
        return pd.DataFrame()
    
    processed_data = []
    analyzer = ReadabilityAnalyzer()
    failed_count = 0
    
    for i, story in enumerate(stories):
        try:
            # 基本信息验证
            story_info = {
                'story_id': story.get('story_id', f'story_{i}'),
                'content': story.get('content', ''),
                'created_at': story.get('created_at', datetime.now().isoformat())
            }
            
            # 参数信息
            params = story.get('parameters', {})
            story_info.update({
                'age': max(1, min(18, params.get('age', 6))),  # 年龄范围验证
                'character': params.get('character', 'unknown'),
                'theme': params.get('theme', 'unknown'),
                'prompt_style': params.get('prompt_style', 'unknown'),
                'word_limit': max(50, min(1000, params.get('word_limit', 300))),  # 词数限制验证
                'model': params.get('model', 'unknown')
            })
            
            # 可读性分析
            if story_info['content'].strip():
                try:
                    readability = analyzer.analyze(story_info['content'])
                    story_info.update({
                        'flesch_score': max(0, min(100, readability.get('Flesch Reading Ease', 0))),
                        'sentence_count': max(1, readability.get('Sentence Count', 1)),
                        'word_count': max(1, readability.get('Word Count', 1)),
                        'avg_sentence_length': readability.get('Average Sentence Length', 0),
                        'avg_word_length': readability.get('Average Word Length', 0),
                        'recommended_age': readability.get('Recommended Age Range', 'unknown')
                    })
                except Exception as readability_error:
                    logger.warning(f"Readability analysis failed for story {story_info['story_id']}: {str(readability_error)}")
                    # 使用简单的备用计算
                    content = story_info['content']
                    word_count = len(content.split())
                    sentence_count = max(1, content.count('.') + content.count('!') + content.count('?'))
                    
                    story_info.update({
                        'flesch_score': 50,  # 默认中等可读性
                        'sentence_count': sentence_count,
                        'word_count': word_count,
                        'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0,
                        'avg_word_length': sum(len(word) for word in content.split()) / word_count if word_count > 0 else 0,
                        'recommended_age': 'unknown'
                    })
            else:
                # 空内容处理
                story_info.update({
                    'flesch_score': 0,
                    'sentence_count': 0,
                    'word_count': 0,
                    'avg_sentence_length': 0,
                    'avg_word_length': 0,
                    'recommended_age': 'unknown'
                })
            
            processed_data.append(story_info)
            
        except Exception as e:
            failed_count += 1
            logger.error(f'Error processing story {i}: {str(e)}')
            if failed_count <= 3:  # 只显示前3个错误
                st.warning(T(f'处理故事数据时出错：{str(e)}', f'Error processing story data: {str(e)}'))
    
    if failed_count > 0:
        logger.warning(f"Failed to process {failed_count} out of {len(stories)} stories")
    
    df = pd.DataFrame(processed_data)
    logger.info(f"Successfully processed {len(df)} stories")
    return df

@st.cache_data
def process_feedback_data(feedback_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """处理反馈数据为DataFrame
    
    Args:
        feedback_data: 反馈数据列表
        
    Returns:
        处理后的DataFrame
    """
    if not feedback_data:
        logger.warning("No feedback data provided for processing")
        return pd.DataFrame()
    
    processed_data = []
    failed_count = 0
    
    for i, feedback in enumerate(feedback_data):
        try:
            # 数据验证和清理
            rating = feedback.get('rating', 0)
            if not isinstance(rating, (int, float)) or rating < 0 or rating > 5:
                rating = 0
            
            feedback_info = {
                'feedback_id': feedback.get('feedback_id', f'feedback_{i}'),
                'story_id': feedback.get('story_id', 'unknown'),
                'rating': rating,
                'comment': str(feedback.get('comment', '')).strip(),
                'timestamp': feedback.get('timestamp', datetime.now().isoformat())
            }
            
            # 额外信息验证
            extra = feedback.get('extra', {})
            age = extra.get('age', 6)
            if not isinstance(age, (int, float)) or age < 1 or age > 18:
                age = 6
            
            word_limit = extra.get('word_limit', 300)
            if not isinstance(word_limit, (int, float)) or word_limit < 50 or word_limit > 1000:
                word_limit = 300
            
            feedback_info.update({
                'theme': extra.get('theme', 'unknown'),
                'character': extra.get('character', 'unknown'),
                'age': age,
                'model': extra.get('model', 'unknown'),
                'prompt_style': extra.get('prompt_style', 'unknown'),
                'word_limit': word_limit
            })
            
            processed_data.append(feedback_info)
            
        except Exception as e:
            failed_count += 1
            logger.error(f'Error processing feedback {i}: {str(e)}')
            if failed_count <= 3:  # 只显示前3个错误
                st.warning(T(f'处理反馈数据时出错：{str(e)}', f'Error processing feedback data: {str(e)}'))
    
    if failed_count > 0:
        logger.warning(f"Failed to process {failed_count} out of {len(feedback_data)} feedback entries")
    
    df = pd.DataFrame(processed_data)
    logger.info(f"Successfully processed {len(df)} feedback entries")
    return df

# 操作按钮
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button(T("返回主页", "Back to Home")):
        st.switch_page("streamlit_app.py")
with col2:
    if st.button(T("刷新数据", "Refresh Data")):
        try:
            # 清除缓存
            st.cache_data.clear()
            logger.info("Data cache cleared by user")
            st.success(T('数据已刷新', 'Data refreshed successfully'))
            st.rerun()
        except Exception as e:
            logger.error(f"Error refreshing data: {str(e)}")
            st.error(T('刷新数据失败', 'Failed to refresh data'))

# 主要内容
st.markdown("---")

# 数据加载
try:
    with st.spinner(T('正在加载数据...', 'Loading data...')):
        stories_data = load_stories_data()
        
        if stories_data:
            stories_df = process_stories_data(stories_data)
        else:
            stories_df = pd.DataFrame()
            st.warning(T('未找到故事数据', 'No story data found'))
            
except Exception as e:
    logger.error(f"Critical error in data loading: {str(e)}")
    st.error(T(f'数据加载失败：{str(e)}', f'Data loading failed: {str(e)}'))
    st.stop()

# 显示最后更新时间
st.caption(T(f'数据最后更新时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
             f'Data last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'))

# 初始化分析模块
stat_analyzer = StatisticalAnalysis()
evaluator = EvaluationMetrics()

# 执行统计分析
if not stories_df.empty:
    statistical_results = stat_analyzer.analyze_stories(stories_df)
else:
    statistical_results = {}

# 执行综合评估
if not stories_df.empty:
    evaluation_results = evaluator.comprehensive_evaluation(stories_df)
else:
    evaluation_results = {}

# 数据概览
if not stories_df.empty:
    st.markdown(f"## {T('数据概览', 'Data Overview')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        story_count = len(stories_df) if not stories_df.empty else 0
        st.metric(
            T('总故事数', 'Total Stories'),
            story_count,
            delta=None if story_count == 0 else f"+{story_count}"
        )
    
    with col2:
        if not stories_df.empty and 'flesch_score' in stories_df.columns:
            valid_scores = stories_df['flesch_score'].dropna()
            if not valid_scores.empty:
                avg_readability = valid_scores.mean()
                readability_level = "Good" if avg_readability >= 60 else "Fair" if avg_readability >= 30 else "Poor"
                st.metric(
                    T('平均可读性', 'Average Readability'),
                    f"{avg_readability:.1f} {readability_level}",
                    delta=f"{avg_readability - 50:.1f}" if avg_readability != 50 else None
                )
            else:
                st.metric(T('平均可读性', 'Average Readability'), "N/A")
        else:
            st.metric(T('平均可读性', 'Average Readability'), "N/A")
else:
    st.warning(T('暂无数据可显示', 'No data available to display'))

# 统计分析概览
if statistical_results and not stories_df.empty:
    st.markdown("---")
    st.markdown(f"## {T('统计分析概览', 'Statistical Analysis Overview')}")
    
    try:
        # 描述性统计
        if 'descriptive_stats' in statistical_results and statistical_results['descriptive_stats']:
            with st.expander(T("描述性统计", "Descriptive Statistics"), expanded=True):
                desc_stats = statistical_results['descriptive_stats']
                
                if 'basic_stats' in desc_stats and not desc_stats['basic_stats'].empty:
                    st.write(T("**基本统计信息:**", "**Basic Statistics:**"))
                    st.dataframe(desc_stats['basic_stats'].round(3), use_container_width=True)
                
                # 缺失值信息
                if 'missing_values' in desc_stats:
                    missing_data = desc_stats['missing_values']
                    if any(count > 0 for count in missing_data.values()):
                        st.write(T("**数据质量:**", "**Data Quality:**"))
                        col1, col2 = st.columns(2)
                        with col1:
                            for col, count in missing_data.items():
                                if count > 0:
                                    st.warning(f"{col}: {count} {T('个缺失值', 'missing values')}")
                        with col2:
                            total_records = len(stories_df)
                            completeness = {col: (total_records - count) / total_records * 100 
                                          for col, count in missing_data.items()}
                            st.write(T("数据完整性:", "Data Completeness:"))
                            for col, pct in completeness.items():
                                st.write(f"{col}: {pct:.1f}%")
                
                # 异常值检测
                if 'outliers' in desc_stats:
                    outliers = desc_stats['outliers']
                    if outliers and any(count > 0 for count in outliers.values()):
                        st.write(T("**异常值检测:**", "**Outlier Detection:**"))
                        outlier_cols = st.columns(len(outliers))
                        for i, (col, count) in enumerate(outliers.items()):
                            if count > 0:
                                outlier_cols[i % len(outlier_cols)].metric(f"{col}", f"{count} {T('个', 'outliers')}")
    except Exception as e:
        logger.error(f"Error in descriptive statistics display: {str(e)}")
        st.error(T("显示描述性统计时出错", "Error displaying descriptive statistics"))
    
    # 相关性分析
    if 'correlation_analysis' in statistical_results and statistical_results['correlation_analysis']:
        with st.expander(T("相关性分析", "Correlation Analysis")):
            corr_analysis = statistical_results['correlation_analysis']
            sample_size = corr_analysis.get('sample_size', 0)
            
            if sample_size > 0:
                st.info(T(f"分析样本量: {sample_size}", f"Analysis sample size: {sample_size}"))
            
            if 'pearson_correlations' in corr_analysis:
                corr_matrix = corr_analysis['pearson_correlations']
                if not corr_matrix.empty:
                    st.write(T("**皮尔逊相关系数:**", "**Pearson Correlations:**"))
                    
                    # 创建相关性热力图
                    fig = px.imshow(
                        corr_matrix,
                        text_auto='.3f',
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title=T("相关性矩阵", "Correlation Matrix"),
                        zmin=-1, zmax=1
                    )
                    fig.update_layout(
                        height=400,
                        title_x=0.5,
                        font=dict(size=12)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示强相关性
                    strong_corrs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.5:  # 强相关性阈值
                                strong_corrs.append({
                                    'variables': f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                                    'correlation': corr_val,
                                    'strength': T('强正相关', 'Strong Positive') if corr_val > 0.5 else T('强负相关', 'Strong Negative')
                                })
                    
                    if strong_corrs:
                        st.write(T("**强相关性 (|r| > 0.5):**", "**Strong Correlations (|r| > 0.5):**"))
                        for corr in strong_corrs:
                            st.write(f"• {corr['variables']}: {corr['correlation']:.3f} ({corr['strength']})")
            
            if 'spearman_correlations' in corr_analysis:
                st.write(T("**斯皮尔曼等级相关:**", "**Spearman Rank Correlation:**"))
                spearman_matrix = corr_analysis['spearman_correlations']
                if not spearman_matrix.empty:
                    st.dataframe(spearman_matrix.round(3), use_container_width=True)
    
    # 假设检验结果
    if 'hypothesis_testing' in statistical_results and statistical_results['hypothesis_testing']:
        with st.expander(T("假设检验", "Hypothesis Testing")):
            hypothesis_results = statistical_results['hypothesis_testing']
            
            # Prompt风格比较
            if 'prompt_style_comparison' in hypothesis_results and hypothesis_results['prompt_style_comparison']:
                prompt_result = hypothesis_results['prompt_style_comparison']
                st.write(T("**Prompt风格效果比较 (ANOVA):**", "**Prompt Style Effect Comparison (ANOVA):**"))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    f_stat = prompt_result.get('f_statistic', 0)
                    st.metric("F统计量", f"{f_stat:.4f}")
                with col2:
                    p_val = prompt_result.get('p_value', 1)
                    st.metric("p值", f"{p_val:.4f}")
                with col3:
                    eta_sq = prompt_result.get('eta_squared', 0)
                    st.metric("效应量 (η²)", f"{eta_sq:.3f}")
                
                # 显示结果解释
                if prompt_result.get('significant', False):
                    st.success(T("结果显著 (p < 0.05) - 不同Prompt风格间存在显著差异", 
                               "Significant result (p < 0.05) - Significant differences between prompt styles"))
                    
                    # 效应量解释
                    if eta_sq >= 0.14:
                        effect_desc = T("大效应", "Large effect")
                    elif eta_sq >= 0.06:
                        effect_desc = T("中等效应", "Medium effect")
                    else:
                        effect_desc = T("小效应", "Small effect")
                    st.info(f"{T('效应量大小', 'Effect size')}: {effect_desc}")
                else:
                    st.info(T("结果不显著 (p ≥ 0.05) - 不同Prompt风格间无显著差异", 
                             "Non-significant result (p ≥ 0.05) - No significant differences between prompt styles"))
                
                # 显示组信息
                if 'group_count' in prompt_result and 'total_samples' in prompt_result:
                    st.caption(f"{T('分析组数', 'Groups analyzed')}: {prompt_result['group_count']}, "
                             f"{T('总样本量', 'Total samples')}: {prompt_result['total_samples']}")
            
            # 年龄组比较
            if 'age_group_comparison' in hypothesis_results and hypothesis_results['age_group_comparison']:
                age_result = hypothesis_results['age_group_comparison']
                st.write(T("**年龄组比较 (ANOVA):**", "**Age Group Comparison (ANOVA):**"))
                
                col1, col2 = st.columns(2)
                with col1:
                    f_stat = age_result.get('f_statistic', 0)
                    st.metric("F统计量", f"{f_stat:.4f}")
                with col2:
                    p_val = age_result.get('p_value', 1)
                    st.metric("p值", f"{p_val:.4f}")
                
                if age_result.get('significant', False):
                    st.success(T("结果显著 (p < 0.05) - 不同年龄组间存在显著差异", 
                               "Significant result (p < 0.05) - Significant differences between age groups"))
                else:
                    st.info(T("结果不显著 (p ≥ 0.05) - 不同年龄组间无显著差异", 
                             "Non-significant result (p ≥ 0.05) - No significant differences between age groups"))
            
            # 如果没有可用的假设检验结果
            if not any(key in hypothesis_results for key in ['prompt_style_comparison', 'age_group_comparison']):
                st.warning(T("暂无可用的假设检验结果", "No hypothesis testing results available"))

# 综合评估指标
if evaluation_results:
    st.markdown("---")
    st.markdown(f"## {T('综合评估指标', 'Comprehensive Evaluation Metrics')}")
    
    # 内容质量评估
    if 'content_quality' in evaluation_results:
        with st.expander(T("内容质量评估", "Content Quality Assessment")):
            quality = evaluation_results['content_quality']
            col1, col2 = st.columns(2)
            
            with col1:
                if 'avg_word_count' in quality:
                    st.metric(T("平均词数", "Average Word Count"), f"{quality['avg_word_count']:.1f}")
                if 'avg_sentence_count' in quality:
                    st.metric(T("平均句数", "Average Sentence Count"), f"{quality['avg_sentence_count']:.1f}")
            
            with col2:
                if 'avg_words_per_sentence' in quality:
                    st.metric(T("平均句长", "Average Sentence Length"), f"{quality['avg_words_per_sentence']:.1f}")
    
    # 用户满意度评估
    if 'user_satisfaction' in evaluation_results:
        with st.expander(T("用户满意度评估", "User Satisfaction Assessment")):
            satisfaction = evaluation_results['user_satisfaction']
            
            if 'positive_feedback_rate' in satisfaction:
                st.metric(
                    T("正面反馈率", "Positive Feedback Rate"),
                    f"{satisfaction['positive_feedback_rate']:.1%}"
                )
            
            if 'satisfaction_distribution' in satisfaction:
                dist_data = satisfaction['satisfaction_distribution']
                fig = px.bar(
                    x=list(dist_data.keys()),
                    y=list(dist_data.values()),
                    title=T("评分分布", "Rating Distribution"),
                    labels={'x': T('评分', 'Rating'), 'y': T('数量', 'Count')}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Prompt效果评估
    if 'prompt_effectiveness' in evaluation_results:
        with st.expander(T("Prompt效果评估", "Prompt Effectiveness Assessment")):
            effectiveness = evaluation_results['prompt_effectiveness']
            
            if 'style_readability' in effectiveness:
                st.write(T("不同风格的可读性对比:", "Readability Comparison by Style:"))
                style_data = effectiveness['style_readability']
                if 'mean' in style_data:
                    readability_df = pd.DataFrame(style_data)
                    fig = px.bar(
                        readability_df,
                        y=readability_df.index,
                        x='mean',
                        error_x='std',
                        orientation='h',
                        title=T("各风格平均可读性", "Average Readability by Style"),
                        labels={'mean': T('平均可读性分数', 'Average Readability Score'), 'y': T('Prompt风格', 'Prompt Style')}
                    )
                    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")



# 分析选项卡
tab1, tab3 = st.tabs([
    T('故事生成分析', 'Story Generation Analysis'),
    T('可读性分析', 'Readability Analysis')
])

with tab1:
    st.markdown(f"### {T('故事生成统计', 'Story Generation Statistics')}")
    
    if not stories_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # 按主题分布
            theme_counts = stories_df['theme'].value_counts()
            fig_theme = px.pie(
                values=theme_counts.values, 
                names=theme_counts.index,
                title=T('故事主题分布', 'Story Theme Distribution')
            )
            st.plotly_chart(fig_theme, use_container_width=True)
            
        with col2:
            # 按提示词风格分布
            prompt_counts = stories_df['prompt_style'].value_counts()
            
            # 提示词风格翻译映射
            prompt_style_translation = {
                'Structured': T('结构化', 'Structured'),
                'Template': T('模板式', 'Template'), 
                'Question': T('问答式', 'Question')
            }
            
            # 应用翻译到标签
            translated_names = [prompt_style_translation.get(name, name) for name in prompt_counts.index]
            
            fig_prompt = px.pie(
                values=prompt_counts.values,
                names=translated_names,
                title=T('提示词风格分布', 'Prompt Style Distribution')
            )
            st.plotly_chart(fig_prompt, use_container_width=True)
        
        # 按年龄段分布
        age_counts = stories_df['age'].value_counts().sort_index()
        fig_age = px.bar(
            x=age_counts.index, 
            y=age_counts.values,
            title=T('目标年龄分布', 'Target Age Distribution'),
            labels={'x': T('年龄', 'Age'), 'y': T('故事数量', 'Story Count')}
        )
        st.plotly_chart(fig_age, use_container_width=True)
        

        
    else:
        st.info(T('暂无故事数据可供分析', 'No story data available for analysis'))




with tab3:
    st.markdown(f"### {T('可读性指标分析', 'Readability Metrics Analysis')}")
    
    if not stories_df.empty and 'flesch_score' in stories_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Flesch分数分布
            if 'flesch_score' in stories_df.columns and stories_df['flesch_score'].notna().sum() > 0:
                # 计算合适的bins数量
                data_range = stories_df['flesch_score'].max() - stories_df['flesch_score'].min()
                bins = max(5, min(20, int(data_range / 5)))  # 动态调整bins数量
                
                fig_flesch = px.histogram(
                    stories_df, 
                    x='flesch_score',
                    nbins=bins,
                    title=T('Flesch可读性分数分布', 'Flesch Readability Score Distribution'),
                    labels={'flesch_score': 'Flesch Score', 'count': T('数量', 'Count')}
                )
                fig_flesch.update_layout(bargap=0.1)
                st.plotly_chart(fig_flesch, use_container_width=True)
            else:
                st.info(T('暂无Flesch分数数据', 'No Flesch score data available'))
            
        with col2:
            # 词数分布
            if 'word_count' in stories_df.columns and stories_df['word_count'].notna().sum() > 0:
                # 计算合适的bins数量
                data_range = stories_df['word_count'].max() - stories_df['word_count'].min()
                bins = max(5, min(15, int(data_range / 20)))  # 动态调整bins数量
                
                fig_words = px.histogram(
                    stories_df, 
                    x='word_count',
                    nbins=bins,
                    title=T('故事词数分布', 'Story Word Count Distribution'),
                    labels={'word_count': T('词数', 'Word Count'), 'count': T('数量', 'Count')}
                )
                fig_words.update_layout(bargap=0.1)
                st.plotly_chart(fig_words, use_container_width=True)
            else:
                st.info(T('暂无词数数据', 'No word count data available'))
        
        # 可读性与年龄的关系
        fig_age_readability = px.scatter(
            stories_df, 
            x='age', 
            y='flesch_score',
            color='model',
            title=T('年龄与可读性关系', 'Age vs Readability Relationship'),
            labels={'age': T('目标年龄', 'Target Age'), 'flesch_score': 'Flesch Score'}
        )
        st.plotly_chart(fig_age_readability, use_container_width=True)
        
        # 各提示词风格可读性对比
        if 'prompt_style' in stories_df.columns and len(stories_df['prompt_style'].unique()) > 1:
            prompt_readability = stories_df.groupby('prompt_style')['flesch_score'].agg(['mean', 'std', 'count']).reset_index()
            # 处理标准差为NaN的情况（单个数据点）
            prompt_readability['std'] = prompt_readability['std'].fillna(0)
            # 只显示有足够数据的组
            prompt_readability = prompt_readability[prompt_readability['count'] >= 1]
            
            if not prompt_readability.empty:
                fig_prompt_readability = px.bar(
                    prompt_readability, 
                    x='prompt_style', 
                    y='mean',
                    error_y='std',
                    title=T('各提示词风格可读性对比', 'Readability Comparison by Prompt Style'),
                    labels={'prompt_style': T('提示词风格', 'Prompt Style'), 'mean': T('平均Flesch分数', 'Average Flesch Score')}
                )
                st.plotly_chart(fig_prompt_readability, use_container_width=True)
            else:
                st.info(T('提示词风格数据不足，无法进行对比分析', 'Insufficient prompt style data for comparison analysis'))
        else:
            st.info(T('需要多种提示词风格数据才能进行对比分析', 'Multiple prompt styles needed for comparison analysis'))
        
    else:
        st.info(T('暂无可读性数据可供分析', 'No readability data available for analysis'))





# st.markdown("---")

# # 数据导出
# st.subheader(T("数据导出", "Data Export"))

# col1, col2 = st.columns(2)

# with col1:
#     if not stories_df.empty:
#         # 基础CSV导出
#         csv_stories = stories_df.to_csv(index=False)
#         st.download_button(
#             label=T("故事数据 (CSV)", "Stories Data (CSV)"),
#             data=csv_stories,
#             file_name=f"stories_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#             mime="text/csv"
#         )
#         

#     else:
#         st.info(T("暂无故事数据可导出", "No story data available for export"))

# with col2:
#     st.info(T("反馈数据将通过Google Form单独收集", "Feedback data will be collected separately via Google Form"))

# 页脚
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #666; margin-top: 1rem;'>{T('数据驱动的故事创作优化', 'Data-Driven Story Creation Optimization')}</div>",
    unsafe_allow_html=True
)