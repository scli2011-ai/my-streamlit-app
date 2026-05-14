import streamlit as st
from supabase import create_client, Client
from google import genai
from google.genai import types
import time

#st.title("HELLO WORLD")

# --- 1. INITIALIZE SUPABASE & GEMINI ---
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

# Initialize the new Gemini Client
@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

try:
    gemini_client = get_genai_client()
except Exception as e:
    st.error(f"Failed to initialize Gemini client: {e}")
    gemini_client = None

# --- AUTOMATICALLY FIND A VALID CHAT MODEL & SET PERSONA ---
@st.cache_resource
def get_valid_gemini_model():
    if not gemini_client:
        return None, None, None
        
    available_models = []
    for m in gemini_client.models.list():
        if m.supported_actions and 'generateContent' in m.supported_actions:
            name = m.name.replace('models/', '')
            available_models.append(name)
            
    if not available_models:
        raise ValueError("Your API key does not have access to any text generation models.")
        
    print("AVAILABLE MODELS:", available_models)   
    
    preferred_models = ['gemini-3.1-flash-lite','gemini-2.5-flash','gemini-2.5-pro', 'gemini-flash-latest','gemini-3.1-pro-preview',  'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
    
    selected_model_name = available_models[0]
    for pref in preferred_models:
        if pref in available_models:
            selected_model_name = pref
            break
            
    print(f"Auto-selected chat model: {selected_model_name}")
    
    # --- NEW/CHANGED: Facilitator Persona with Handoff Trigger ---
    facilitator_instruction = """
        Greeting:
        If someone greets you, try to give a warm response and briefly introduce yourself.
        
        Role:
        You are Einstein Junior, a primary school science teacher (Facilitator). 
        You teach the topic on aerodynamics for Grade 3 to Grade 6.
        
        Goal:
        Your goal is to facilitate users learning the concepts of aerodynamics confined in the knowledge base. 
        
        
        Behaviour:
        Guide and stimulate students to learn by posing questions to them and prompt them to answer. Do not give answers straightaway.
        When users have problems, rephrase questions or provide hints. 
        Redirect off-topic questions back to aerodynamics.
        
        CRITICAL HANDOFF INSTRUCTION:
        Through the dialogues, if you identify the user has a good understanding of the key concepts of aerodynamics, you MUST append the exact word [HANDOFF] at the very end of your response. This will signal the Assessment Bot to take over. Do NOT ask them if they want a quiz yourself; just append [HANDOFF] when they are ready.
        
        Personality:
        You are an inviting teacher. Give encouragement to students as much as possible.
    """

    # --- NEW/CHANGED: Assessment Bot Persona ---
    assessment_instruction = """
        Role:
        You are Einstein Junior's assistant. You have just taken over the conversation from Einstein Junior because the user is ready for a quiz.
        
        Behaviour:
        1. No need to introduce yourself.
        2. If the user agrees to take the quiz, generate 5 Multiple-choice questions based on the knowledge base.
        3. Ask ONE question at a time. Wait for the user to answer before moving on to the next.
        4. When the user answers, tell them if they are correct or incorrect, briefly explain why using the knowledge base, and then ask the next question.
        5. After all 5 questions have been answered, assess their overall performance with a grade (A for excellent, B for very good, C for developing, etc.) and provide an encouraging summary.
    """
    
    return selected_model_name, facilitator_instruction, assessment_instruction

try:
    chat_model_name, facilitator_instruction, assessment_instruction = get_valid_gemini_model()
except Exception as e:
    st.error(f"Failed to load Gemini chat model: {e}")
    chat_model_name, facilitator_instruction, assessment_instruction = None, None, None

# --- AUTOMATICALLY FIND A VALID EMBEDDING MODEL ---
@st.cache_resource
def get_valid_embedding_model():
    if not gemini_client:
        return None
        
    available_embedding_models = []
    
    for m in gemini_client.models.list():
        if m.supported_actions and 'embedContent' in m.supported_actions:
            name = m.name.replace('models/', '')
            available_embedding_models.append(name)
            
    if not available_embedding_models:
        available_embedding_models = ['text-embedding-004']
        
    working_models = [m for m in available_embedding_models if "text-embedding-004" not in m]
    
    if working_models:
        selected_model = working_models[-1]
    else:
        selected_model = available_embedding_models[0]
        
    print(f"Auto-selected embedding model: {selected_model}")
    return selected_model

# --- RAG HELPER FUNCTIONS ---
def get_embedding(text: str) -> list[float]:
    if not gemini_client:
        return []
        
    try:
        embedding_model_name = get_valid_embedding_model()
        
        response = gemini_client.models.embed_content(
            model=embedding_model_name, 
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return response.embeddings[0].values
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return []

def search_documents(query_embedding: list[float], match_threshold=0.7, match_count=2):
    try:
        response = supabase.rpc(
            'match_document_chunks',
            {
                'query_embedding': query_embedding,
                'match_threshold': match_threshold,
                'match_count': match_count
            }
        ).execute()
        return response.data
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return []

# --- 2. SESSION STATE SETUP ---
if "user" not in st.session_state:
    st.session_state.user = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- NEW/CHANGED: Track which bot is currently active ---
if "active_bot" not in st.session_state:
    st.session_state.active_bot = "facilitator"

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("My AI RAG App")
    if st.session_state.user:
        st.success(f"Logged in as: {st.session_state.user.email}")
        
        # --- NEW/CHANGED: Display current active bot status ---
        st.info(f"Current Mode: {'👨‍🏫 Facilitator' if st.session_state.active_bot == 'facilitator' else '📝 Assessment'}")
        
        if st.button("Log Out"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.active_bot = "facilitator" # Reset bot state on logout
            st.rerun()
    else:
        st.warning("You are not logged in.")

# --- 4. MAIN APP LOGIC ---
st.title("Welcome to the Future Science Classroom")

if not st.session_state.user:
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.header("Login")
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password", autocomplete="off")
        
        if st.button("Login"):
            try:
                response = supabase.auth.sign_in_with_password({
                    "email": login_email, 
                    "password": login_password
                })
                if response.user:
                    st.session_state.user = response.user
                    st.success("Login successful!")
                    st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")

    with tab2:
        st.header("Create an Account")
        
        signup_name = st.text_input("Full Name", key="signup_name")
        signup_age = st.number_input("Age", min_value=1, max_value=120, step=1, value=18, key="signup_age")
        signup_gender = st.selectbox("Gender", ["Select...", "Male", "Female", "Non-binary", "Prefer not to say"], key="signup_gender")
        
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password", autocomplete="off")
        
        if st.button("Sign Up"):
            if not signup_name.strip():
                st.error("Please enter your full name.")
            elif signup_gender == "Select...":
                st.error("Please select a gender.")
            elif not signup_email or not signup_password:
                st.error("Please enter an email and password.")
            else:
                try:
                    response = supabase.auth.sign_up({
                        "email": signup_email, 
                        "password": signup_password
                    })
                    
                    if response.user:
                        st.success("Auth account created successfully! Now saving profile data...")
                        try:
                            db_response = supabase.table("participants").insert({
                                "participant_id": response.user.id,
                                "email": signup_email,
                                "name": signup_name,
                                "age": int(signup_age),
                                "gender": signup_gender
                            }).execute()
                            
                            st.success("User successfully added to the participants table! You can now log in.")
                        except Exception as db_error:
                            st.error(f"Database Error: Could not save to participants table. Details: {db_error}")
                            
                except Exception as auth_error:
                    st.error(f"Auth Error: Could not create account. Details: {auth_error}")

# --- 5. LOGGED IN VIEW (GEMINI AI CHATBOT WITH RAG) ---
else:
    # Dynamic subheader based on active bot
    if st.session_state.active_bot == "facilitator":
        st.subheader("👨‍🏫 I am Einstein Junior, your science teacher!")
    else:
        st.subheader("📝 I am the Assessment Bot!")
    
    if not chat_model_name or not gemini_client:
        st.error("Cannot start chat because no compatible Gemini models were found for your API key.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Type your message here..."):
            
            with st.chat_message("user"):
                st.markdown(prompt)
                
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                query_embedding = get_embedding(prompt)
                retrieved_chunks = []
                current_rag_context = ""
                
                if query_embedding:
                    retrieved_chunks = search_documents(query_embedding)
                    
                if retrieved_chunks:
                    context_texts = [chunk['content'] for chunk in retrieved_chunks]
                    current_rag_context = "\n\n---\n\n".join(context_texts)
                else:
                    current_rag_context = "No specific context found in the Knowledge Base."

            gemini_history = []
            
            for msg in st.session_state.messages[:-1]:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append(
                    types.Content(
                        role=role, 
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )

            # --- NEW/CHANGED: Select the correct system instruction based on active bot ---
            current_instruction = facilitator_instruction if st.session_state.active_bot == "facilitator" else assessment_instruction
            bot_name_context = "Einstein Junior (Facilitator)" if st.session_state.active_bot == "facilitator" else "Assessment Bot"

            augmented_prompt = f"""
            You are {bot_name_context}. Use the following Knowledge Base to inform your response. 
            
            Knowledge Base Context:
            {current_rag_context}
            
            Student's Query:
            {prompt}
            """

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    chat_session = gemini_client.chats.create(
                        model=chat_model_name,
                        config=types.GenerateContentConfig(
                            system_instruction=current_instruction, # Uses the dynamic instruction
                        ),
                        history=gemini_history
                    )
                    
                    response_stream = chat_session.send_message_stream(augmented_prompt)
                    
                    full_response = ""
                    for chunk in response_stream:
                        full_response += chunk.text
                        # Hide the [HANDOFF] tag from the UI while streaming if it appears
                        display_text = full_response.replace("[HANDOFF]", "")
                        message_placeholder.markdown(display_text + "▌")
                    
                    # --- NEW/CHANGED: Check for Handoff Signal ---
                    if "[HANDOFF]" in full_response:
                        # Clean the response and save it
                        clean_response = full_response.replace("[HANDOFF]", "").strip()
                        message_placeholder.markdown(clean_response)
                        st.session_state.messages.append({"role": "assistant", "content": clean_response})
                        
                        # Switch state and inject the Assessment Bot's greeting
                        st.session_state.active_bot = "assessment"
                        handoff_greeting = "Hello! I am the Assessment Bot. Einstein Junior tells me you have a great understanding of aerodynamics! Would you like to take a 5-question quiz to test your knowledge?"
                        st.session_state.messages.append({"role": "assistant", "content": handoff_greeting})
                        
                        # Rerun to show the new bot's message immediately
                        st.rerun()
                    else:
                        # Normal response handling
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    try:
                        supabase.table("study_logs").insert({
                            "participant_id": st.session_state.user.id,
                            "user_query": prompt,
                            "bot_response": full_response,
                            "rag_context": current_rag_context
                        }).execute()
                    except Exception as db_log_error:
                        st.error(f"Failed to save log to database: {db_log_error}")
                    
                except Exception as e:
                    st.error(f"Error communicating with Gemini: {e}")
