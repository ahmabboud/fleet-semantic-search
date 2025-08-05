#!/usr/bin/env python3
"""
Semantic Search Application with LLM Validation
A Streamlit web application for searching maintenance records using semantic search and optional LLM validation.
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import tempfile
import shutil
from datetime import datetime
import json
import re

# testing SQLite connection
import sqlite3
###
print("SQLite version in use:", sqlite3.sqlite_version)
# Load environment variables
load_dotenv()

# Import required libraries for vector database and LLM
try:
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.schema import Document
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
except ImportError as e:
    st.error(f"Required libraries not installed: {e}")
    st.stop()

# Configuration
COLLECTION_NAME = "seatbelt_comments"
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")
DEFAULT_SIMILARITY_THRESHOLD = 0.7

# Page configuration
st.set_page_config(
    page_title="Semantic Search for Maintenance Records",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_environment():
    """Load and validate environment variables"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o")
    llm_temperature = float(os.getenv("TEMPERATURE", "0.1"))
    
    if not openai_api_key or openai_api_key == "your_openai_api_key_here":
        st.error("⚠️ OpenAI API key not found! Please set your API key in the .env file.")
        st.stop()
    
    return {
        "api_key": openai_api_key,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "llm_temperature": llm_temperature
    }

@st.cache_resource
def initialize_search_system():
    """Initialize the vector database and search components"""
    env_config = load_environment()
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model=env_config["embedding_model"],
        openai_api_key=env_config["api_key"]
    )
    
    # Load vector database
    try:
        vectordb = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        
        # Check if database has documents
        collection_count = vectordb._collection.count()
        if collection_count == 0:
            st.error("❌ Vector database is empty! Please run the notebook first to populate the database.")
            st.stop()
        
        st.success(f"✅ Vector database loaded with {collection_count} documents")
        
    except Exception as e:
        st.error(f"❌ Failed to load vector database: {e}")
        st.stop()
    
    # Initialize LLM for validation
    validation_llm = ChatOpenAI(
        model=env_config["llm_model"],
        temperature=env_config["llm_temperature"],
        openai_api_key=env_config["api_key"]
    )
    
    # Create validation prompt
    validation_prompt = ChatPromptTemplate.from_template("""
You are an expert automotive maintenance analyst. Your task is to validate if search results match the user's diagnostic query.

USER QUERY: "{user_query}"

SEARCH RESULTS TO VALIDATE:
{search_results}

For each search result, analyze:
1. Does the maintenance comment describe the same or similar diagnostic issue as the user's query?
2. How relevant is this result to the user's specific problem?
3. What is the confidence level of this match?

Provide your analysis in the following JSON format:
{{
    "overall_assessment": "brief summary of how well results match the query",
    "validated_results": [
        {{
            "result_index": 1,
            "relevance_score": 0.85,
            "is_diagnostic_match": true,
            "explanation": "detailed explanation of why this result matches or doesn't match",
            "key_similarities": ["list", "of", "key", "matching", "elements"],
            "concerns": ["any", "concerns", "about", "the", "match"]
        }}
    ],
    "recommendations": "suggestions for the user based on the validated results"
}}

Be thorough but concise. Focus on diagnostic accuracy and practical relevance.
""")
    
    validation_chain = validation_prompt | validation_llm | StrOutputParser()
    
    return {
        "vectordb": vectordb,
        "embeddings": embeddings,
        "validation_chain": validation_chain,
        "collection_count": collection_count,
        "env_config": env_config
    }

def parse_date_from_metadata(metadata):
    """Extract and parse date from metadata for sorting"""
    date_fields = ['date', 'Date', 'created_date', 'timestamp', 'created_at', 'Completedate']
    
    for field in date_fields:
        if field in metadata and metadata[field]:
            try:
                date_str = str(metadata[field])
                if 'T' in date_str:
                    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
            except:
                continue
    
    return datetime(1900, 1, 1)

def semantic_search_with_ranking(vectordb, query, k=10, similarity_threshold=0.7):
    """Perform semantic search with date ranking and similarity threshold filtering"""
    
    # Perform similarity search with scores
    results_with_scores = vectordb.similarity_search_with_score(query, k=k*2)
    
    # Filter by similarity threshold
    filtered_results = [
        (doc, score) for doc, score in results_with_scores 
        if score >= similarity_threshold
    ]
    
    if not filtered_results:
        return []
    
    # Sort by date descending (most recent first)
    def get_sort_key(doc_score_tuple):
        doc, score = doc_score_tuple
        date = parse_date_from_metadata(doc.metadata)
        return (date, score)
    
    sorted_results = sorted(filtered_results, key=get_sort_key, reverse=True)
    return sorted_results[:k]

def validate_search_results_with_llm(validation_chain, user_query, search_results, max_results=5):
    """Validate search results using LLM"""
    if not search_results:
        return {
            "validated_results": [],
            "overall_assessment": "No search results to validate",
            "recommendations": "Try a different query or lower the similarity threshold"
        }
    
    # Prepare search results for LLM analysis
    results_for_validation = []
    for i, (doc, score) in enumerate(search_results[:max_results], 1):
        result_data = {
            "index": i,
            "similarity_score": score,
            "comment": doc.page_content,
            "metadata": {
                "date": doc.metadata.get('Completedate', 'No date'),
                "status": doc.metadata.get('Orderstatus', 'Unknown'),
                "cost": doc.metadata.get('Unitcst', 'No cost'),
                "compcode": doc.metadata.get('Compcode', 'No code')
            }
        }
        results_for_validation.append(result_data)
    
    # Format results for LLM prompt
    formatted_results = ""
    for result in results_for_validation:
        formatted_results += f"""
Result {result['index']} (Similarity: {result['similarity_score']:.3f}):
Comment: {result['comment'][:300]}{'...' if len(result['comment']) > 300 else ''}
Date: {result['metadata']['date']}
Status: {result['metadata']['status']}
Cost: {result['metadata']['cost']}
---
"""
    
    try:
        # Run validation
        validation_response = validation_chain.invoke({
            "user_query": user_query,
            "search_results": formatted_results
        })
        
        # Parse JSON response
        try:
            clean_response = validation_response.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:]
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            validation_data = json.loads(clean_response)
        except json.JSONDecodeError:
            validation_data = {
                "overall_assessment": validation_response[:200] + "..." if len(validation_response) > 200 else validation_response,
                "validated_results": [],
                "recommendations": "LLM validation completed but format needs review"
            }
        
        return validation_data
        
    except Exception as e:
        return {
            "overall_assessment": f"Validation failed: {str(e)}",
            "validated_results": [],
            "recommendations": "Validation system encountered an error"
        }

def perform_search(search_system, query, similarity_threshold, enable_llm_validation, max_results):
    """Perform complete search with optional LLM validation"""
    
    # Step 1: Semantic search
    semantic_results = semantic_search_with_ranking(
        search_system["vectordb"], 
        query, 
        k=max_results, 
        similarity_threshold=similarity_threshold
    )
    
    if not semantic_results:
        return {
            "semantic_results": [],
            "validation_results": None,
            "final_results": [],
            "recommendations": "No semantic matches found. Try lowering similarity threshold or different keywords."
        }
    
    # Step 2: LLM validation (if enabled)
    validation_results = None
    if enable_llm_validation:
        validation_results = validate_search_results_with_llm(
            search_system["validation_chain"], 
            query, 
            semantic_results, 
            max_results
        )
    
    # Step 3: Process results
    final_results = []
    
    if enable_llm_validation and validation_results and 'validated_results' in validation_results:
        # Use LLM validation to filter results
        for validated_item in validation_results['validated_results']:
            if validated_item.get('is_diagnostic_match', False) and validated_item.get('relevance_score', 0) > 0.5:
                result_index = validated_item['result_index'] - 1
                if result_index < len(semantic_results):
                    doc, original_score = semantic_results[result_index]
                    final_results.append({
                        'document': doc,
                        'semantic_score': original_score,
                        'llm_relevance': validated_item['relevance_score'],
                        'explanation': validated_item['explanation'],
                        'key_similarities': validated_item.get('key_similarities', []),
                        'concerns': validated_item.get('concerns', [])
                    })
    else:
        # Use semantic results only
        for doc, score in semantic_results:
            final_results.append({
                'document': doc,
                'semantic_score': score,
                'llm_relevance': None,
                'explanation': 'LLM validation not performed' if not enable_llm_validation else 'LLM validation enabled but no matches',
                'key_similarities': [],
                'concerns': []
            })
    
    return {
        "semantic_results": semantic_results,
        "validation_results": validation_results,
        "final_results": final_results,
        "recommendations": validation_results.get('recommendations', 'Search completed successfully') if validation_results else "Semantic search completed"
    }

def create_results_dataframe(results):
    """Create a pandas DataFrame from search results"""
    if not results['final_results']:
        return pd.DataFrame()
    
    table_data = []
    for i, result in enumerate(results['final_results'], 1):
        doc = result['document']
        metadata = doc.metadata
        
        row = {
            'Result #': i,
            'Similarity Score': f"{result['semantic_score']:.3f}",
            'LLM Relevance': f"{result['llm_relevance']:.3f}" if result['llm_relevance'] is not None else "N/A",
            'Comment Preview': doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
            'Date': metadata.get('Completedate', 'N/A'),
            'Status': metadata.get('Orderstatus', 'N/A'),
            'Cost': metadata.get('Unitcst', 'N/A'),
            'Code': metadata.get('Compcode', 'N/A'),
            'Order ID': metadata.get('orderid', 'N/A'),
            'Line Type': metadata.get('Linetype', 'N/A'),
            'Hours': metadata.get('Hours', 'N/A'),
            'Quantity': metadata.get('Qty', 'N/A'),
            'LLM Explanation': result['explanation'][:100] + "..." if len(result['explanation']) > 100 else result['explanation']
        }
        
        table_data.append(row)
    
    return pd.DataFrame(table_data)

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🔍 Semantic Search for Maintenance Records")
    st.markdown("Search through maintenance records using AI-powered semantic search with optional LLM validation")
    
    # Initialize search system
    with st.spinner("🚀 Loading search system..."):
        search_system = initialize_search_system()
    
    # Sidebar for settings
    st.sidebar.header("🔧 Search Settings")
    
    # Search query input
    query = st.text_input(
        "🔍 Enter your search query:",
        placeholder="e.g., broken seatbelt needs replacement",
        help="Describe the maintenance issue you're looking for"
    )
    
    # Similarity threshold slider
    similarity_threshold = st.sidebar.slider(
        "📊 Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_SIMILARITY_THRESHOLD,
        step=0.05,
        help="Higher values = more strict matching. Lower values = more results but potentially less relevant."
    )
    
    # LLM validation toggle
    enable_llm_validation = st.sidebar.checkbox(
        "🧠 Enable LLM Validation",
        value=True,
        help="Use AI to validate and filter search results for better accuracy"
    )
    
    # Maximum results
    max_results = st.sidebar.selectbox(
        "📋 Maximum Results",
        options=[5, 10, 15, 20],
        index=1,
        help="Maximum number of results to display"
    )
    
    # Database info
    st.sidebar.markdown("---")
    st.sidebar.markdown("📊 **Database Info**")
    st.sidebar.info(f"Documents: {search_system['collection_count']}")
    st.sidebar.info(f"Collection: {COLLECTION_NAME}")
    
    # Search button and results
    if st.button("🔍 Search", type="primary") and query:
        
        with st.spinner("🔍 Searching..."):
            results = perform_search(
                search_system,
                query,
                similarity_threshold,
                enable_llm_validation,
                max_results
            )
        
        # Display results
        if results['final_results']:
            
            # Summary
            st.success(f"✅ Found {len(results['final_results'])} matching results")
            
            # LLM Assessment (if available)
            if enable_llm_validation and results['validation_results']:
                assessment = results['validation_results'].get('overall_assessment', '')
                if assessment:
                    st.info(f"🧠 **LLM Assessment:** {assessment}")
            
            # Results table
            st.markdown("## 📊 Search Results")
            
            df_results = create_results_dataframe(results)
            
            # Display table with styling
            st.dataframe(
                df_results,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Similarity Score": st.column_config.NumberColumn("Similarity Score", format="%.3f"),
                    "LLM Relevance": st.column_config.TextColumn("LLM Relevance"),
                    "Cost": st.column_config.NumberColumn("Cost", format="$%.2f"),
                }
            )
            
            # Detailed results (expandable)
            with st.expander("📋 View Detailed Results", expanded=False):
                for i, result in enumerate(results['final_results'], 1):
                    doc = result['document']
                    metadata = doc.metadata
                    
                    st.markdown(f"### 📄 Result #{i}")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Full Comment:**")
                        st.text_area(
                            "Comment",
                            value=doc.page_content,
                            height=100,
                            key=f"comment_{i}",
                            label_visibility="collapsed"
                        )
                        
                        if result['llm_relevance'] is not None:
                            st.markdown("**LLM Analysis:**")
                            st.write(result['explanation'])
                            
                            if result['key_similarities']:
                                st.markdown("**Key Similarities:**")
                                for similarity in result['key_similarities']:
                                    st.markdown(f"• {similarity}")
                            
                            if result['concerns']:
                                st.markdown("**Concerns:**")
                                for concern in result['concerns']:
                                    st.markdown(f"⚠️ {concern}")
                    
                    with col2:
                        st.markdown("**Scores:**")
                        st.metric("Similarity", f"{result['semantic_score']:.3f}")
                        if result['llm_relevance'] is not None:
                            st.metric("LLM Relevance", f"{result['llm_relevance']:.3f}")
                        
                        st.markdown("**Metadata:**")
                        for key, value in metadata.items():
                            if value is not None and str(value) != 'nan':
                                clean_key = key.replace('_', ' ').title()
                                st.text(f"{clean_key}: {value}")
                    
                    st.markdown("---")
            
            # Recommendations
            if results['recommendations']:
                st.markdown("## 💡 Recommendations")
                st.info(results['recommendations'])
            
            # Export option
            st.markdown("## 📥 Export Results")
            csv_data = df_results.to_csv(index=False)
            st.download_button(
                label="📁 Download Results as CSV",
                data=csv_data,
                file_name=f"search_results_{query.replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("❌ No results found. Try:")
            st.markdown("• Lowering the similarity threshold")
            st.markdown("• Using different keywords")
            st.markdown("• Checking your spelling")
            
    elif query and not st.button:
        st.info("👆 Click the Search button to find results")
    
    # Footer
    st.markdown("---")
    st.markdown("**🔧 Powered by:** OpenAI Embeddings, ChromaDB, LangChain, and Streamlit")

if __name__ == "__main__":
    main()
