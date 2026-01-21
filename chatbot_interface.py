"""
WCCCD Virtual Assistant - Chat Interface
WCCCDbot with LARGE three-dot menu
"""

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from anthropic import Anthropic
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# ===== CONFIGURATION =====
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
TOP_K_RESULTS = 20

# ===== Initialize Components =====

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot components (runs once)"""
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    
    try:
        collection = client.get_collection(
            name="college_knowledge",
            embedding_function=embedding_function
        )
    except Exception as e:
        st.error(f"Error: Could not load knowledge base. Did you run build_knowledge_base.py? Error: {e}")
        return None, None
    
    if not ANTHROPIC_API_KEY:
        st.error("Error: ANTHROPIC_API_KEY not found. Please add it to your .env file")
        return None, None
    
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    
    return collection, anthropic_client


def search_knowledge_base(collection, query, top_k=5):
    """Search the knowledge base for relevant information"""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    relevant_chunks = []
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            relevant_chunks.append({
                'content': doc,
                'url': metadata['url'],
                'title': metadata['title']
            })
    
    return relevant_chunks


def generate_response(anthropic_client, query, relevant_chunks):
    """Generate a response using Claude - SHORT AND DIRECT"""
    context = "\n\n".join([
        f"Source: {chunk['title']} ({chunk['url']})\n{chunk['content']}"
        for chunk in relevant_chunks
    ])
    
    prompt = f"""You are WCCCDbot, a helpful academic assistant for Wayne County Community College District.

TODAY'S DATE: {datetime.now().strftime('%B %d, %Y')}
IMPORTANT: Only mention events happening TODAY or LATER. Skip any past events.

Context from WCCCD catalog and website:
{context}

Student's question: {query}

RESPONSE RULES:

For PROGRAM/ADVISING questions:
1. Provide specific course lists, sequences, and credit requirements from the catalog
2. Explain prerequisites clearly with course codes
3. Suggest realistic semester-by-semester plans when relevant
4. Compare programs when students are deciding between options
5. Keep responses 3-5 sentences with clear, actionable information
6. ALWAYS end with: "For personalized advising, contact an academic advisor at (313) 496-2634."

For GENERAL/FACTUAL questions:
1. Keep answer to 1-3 sentences maximum
2. Answer directly and concisely
3. ALWAYS include the relevant website link
4. Format: "Brief answer. Visit [URL] for more information."

For ALL responses:
- Use PLAIN TEXT ONLY - no bold, no italics, no special formatting
- Include actual URLs from the context sources
- Be helpful and encouraging
- If information is from the catalog, mention it ("According to the 2025-2026 catalog...")

EXAMPLES:

Program Question: "What courses do I need for Business Administration?"
Answer: "The Business Administration A.A.S. requires 62 credits including: ACC 201/202 (Accounting), BUS 221 (Business Law), BUS 251 (Management), MKT 201 (Marketing), and ECO 201/202 (Economics). First semester typically includes ENG 101, MTH 110, BUS 101, and CIS 105. For personalized advising, contact an academic advisor at (313) 496-2634."

General Question: "When can I register for classes?"
Answer: "Registration for Spring 2026 begins October 21, 2025. Visit https://www.wcccd.edu/registration for dates and complete information."

Now answer the student's question:"""
    
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return message.content[0].text


def get_transcript():
    """Generate downloadable transcript of conversation"""
    if not st.session_state.messages:
        return "No conversation yet."
    
    transcript = "WCCCD Virtual Assistant - Chat Transcript\n"
    transcript += "="*50 + "\n\n"
    
    for msg in st.session_state.messages:
        role = "Student" if msg["role"] == "user" else "Assistant"
        transcript += f"{role}:\n{msg['content']}\n\n"
        transcript += "-"*50 + "\n\n"
    
    return transcript


# ===== Streamlit UI =====

def main():
    # Page configuration
    st.set_page_config(
        page_title="WCCCDbot",
        page_icon="",
        layout="wide"
    )
    
    # Initialize session state FIRST
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "show_about" not in st.session_state:
        st.session_state.show_about = False
    
    # CSS Styling - VERY SPECIFIC for menu button
    st.markdown("""
    <style>
    /* Remove default Streamlit padding */
    .main > div {
        padding-top: 0rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Remove gaps between columns */
    [data-testid="column"] {
        padding: 0px !important;
    }
    
    .block-container {
        padding-left: 0rem !important;
        padding-right: 0rem !important;
    }
    
    /* Chat input styling */
    .stChatInputContainer {
        border: 3px solid #003DA5 !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 12px rgba(0, 61, 165, 0.3) !important;
    }
    
    .stChatInput > textarea {
        font-size: 18px !important;
        padding: 15px !important;
        min-height: 60px !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: white !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin: 8px 0 !important;
    }
    
    /* LARGE THREE-DOT MENU BUTTON - VERY SPECIFIC */
    button[data-testid="baseButton-secondary"] {
        background: transparent !important;
        border: none !important;
        color: white !important;
        font-size: 50px !important;
        padding: 0px 10px !important;
        margin: 0 !important;
        line-height: 1 !important;
        font-weight: 900 !important;
    }
    
    button[data-testid="baseButton-secondary"]:hover {
        background: rgba(255,255,255,0.2) !important;
        border-radius: 6px !important;
    }
    
    /* Also target the popover button directly */
    [data-testid="stPopover"] > button {
        font-size: 50px !important;
        color: white !important;
        background: transparent !important;
        border: none !important;
        font-weight: 900 !important;
        padding: 0px 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # BANNER - Two columns only (title and menu)
    with st.container():
        # Create two columns for banner
        col_title, col_menu = st.columns([10, 0.7])
        
        # Apply blue background to all columns
        st.markdown("""
        <style>
        /* Make all banner columns blue with same height */
        [data-testid="stHorizontalBlock"] [data-testid="column"] {
            background: linear-gradient(135deg, #003DA5 0%, #002870 100%);
            padding: 20px 0px !important;
            display: flex;
            align-items: center;
        }
        
        /* Remove any gaps */
        [data-testid="stHorizontalBlock"] {
            gap: 0px !important;
            background: linear-gradient(135deg, #003DA5 0%, #002870 100%);
            margin: -1rem -1rem 2rem -1rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }
        </style>
        """, unsafe_allow_html=True)
        
        with col_title:
            st.markdown("""
            <div style="padding-left: 24px; display: flex; align-items: center; height: 100%;">
                <h1 style="color: white; font-size: 22px; font-weight: 600; margin: 0; line-height: 1;">WCCCDbot</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col_menu:
            st.markdown("""
            <div style="padding-right: 20px; text-align: right; display: flex; align-items: center; justify-content: flex-end; height: 100%;">
            """, unsafe_allow_html=True)
            
            # Three-dot menu - EXTRA LARGE
            with st.popover("⋮", use_container_width=False):
                st.markdown("### Menu")
                
                # Clear Conversation
                if st.button("Clear Conversation", key="menu_clear", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()
                
                # Download Transcript
                if st.session_state.messages and len(st.session_state.messages) > 1:
                    transcript = get_transcript()
                    st.download_button(
                        label="Download Transcript",
                        data=transcript,
                        file_name="wcccd_chat_transcript.txt",
                        mime="text/plain",
                        key="menu_download",
                        use_container_width=True
                    )
                else:
                    st.button("Download Transcript", disabled=True, use_container_width=True, help="No conversation to download")
                
                # About
                if st.button("About", key="menu_about", use_container_width=True):
                    st.session_state.show_about = True
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Show About dialog if triggered
    if st.session_state.show_about:
        st.markdown("---")
        st.info("""
        ### About WCCCDbot
        
        **What is this?**
        Your 24/7 guide to Wayne County Community College District. Ask questions about:
        - Enrollment and registration
        - Academic calendar and important dates
        - Campus locations and hours
        - Financial aid and tuition
        - Programs and courses
        
        **How it works:**
        This WCCCDbot searches WCCCD's website and official documents to provide accurate, 
        up-to-date information to help you navigate your college experience.
        
        **Privacy Notice:**
        Please don't share sensitive personal information like student ID numbers, 
        Social Security numbers, or birthdates in this chat.
        
        **Need more help?**
        - Phone: (313) 496-2600
        - Website: wcccd.edu
        - Email: info@wcccd.edu
        
        ---
        *Powered by Maverick Tech Systems*
        """)
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("Close", use_container_width=True, key="close_about"):
                st.session_state.show_about = False
                st.rerun()
        
        st.markdown("---")
    
    # Initialize chatbot
    collection, anthropic_client = initialize_chatbot()
    
    if not collection or not anthropic_client:
        st.stop()
    
    # Add welcome message if chat is empty
    if len(st.session_state.messages) == 0:
        welcome_message = """**Hey, welcome!**

I am a chatbot here to help with your Academic Journey! Feel free to ask any questions related to WCCCD. You can try...

- When does registration start?
- How do I apply for admission?
- How do I apply for financial aid?
- How do I log into Blackboard?
- How do I log into my WCCCD Email?

Just a quick note—please don't share sensitive personal information like ID numbers, birthdates, or Social Security numbers to keep your data safe."""
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_message
        })
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="wildcat_logo.png"):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"], avatar=None):
                st.markdown(message["content"])
    
    # Chat input
    if query := st.chat_input("Type your question here and press Enter..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user", avatar=None):
            st.markdown(query)
        
        # Generate response
        with st.chat_message("assistant", avatar="wildcat_logo.png"):
            with st.spinner("Searching and thinking..."):
                # Search knowledge base
                relevant_chunks = search_knowledge_base(collection, query, TOP_K_RESULTS)
                
                if not relevant_chunks:
                    response = "I couldn't find any relevant information in the college website content. Please try rephrasing your question or contact WCCCD directly at (313) 496-2600 or visit wcccd.edu."
                else:
                    # Generate response with Claude
                    response = generate_response(anthropic_client, query, relevant_chunks)
                
                # Display response
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })


if __name__ == "__main__":
    main()