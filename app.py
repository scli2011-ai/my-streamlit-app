import streamlit as st
from supabase import create_client, Client
from google import genai
from google.genai import types
import time

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
        return None, None
        
    available_models = []
    # Ask Google which models this API key is allowed to use
    # In the new SDK, we use .list() and check .supported_actions
    for m in gemini_client.models.list():
        if m.supported_actions and 'generateContent' in m.supported_actions:
            name = m.name.replace('models/', '')
            available_models.append(name)
            
    if not available_models:
        raise ValueError("Your API key does not have access to any text generation models.")
        
    # Try to prefer a standard model if available, otherwise just pick the first valid one
    preferred_models = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
    
    selected_model_name = available_models[0] # Default to the first one found
    for pref in preferred_models:
        if pref in available_models:
            selected_model_name = pref
            break
            
    print(f"Auto-selected chat model: {selected_model_name}")
    
    # --- DEFINE YOUR BOT'S PERSONA HERE ---
    my_system_instruction = """
        Greeting:
        If someone greets you, try to give a warm response and briefly introduce yourself.
        Role:
        You are Einstein Junior, a primary school science teacher. 
        You teach the topic on aerodynamics for Grade 3 to Grade 6.
        Goal:
        Your goal is to facilitate users learning the concepts of aerodynamics confined in the knowledge base.
        Behaviour:
        When users ask you questions, you normally do not give them answers straightaway.
        Instead, you guide and stimulate students to learn by posing questions to them and prompt them to answer. 
        When users have problems in answering your questions, you may rephrase your questions or provide hints for them. 
        When being asked questions beyond the scope of the knowledge base, try to redirect their interest to aerodynamics.
        Personality:
        You are an inviting teacher. Try to give encouragement to students as much as possible.
    """
    
    # Return the model name and the instruction string to be used later in chat creation
    return selected_model_name, my_system_instruction

try:
    chat_model_name, system_instruction = get_valid_gemini_model()
except Exception as e:
    st.error(f"Failed to load Gemini chat model: {e}")
    chat_model_name, system_instruction = None, None

# --- AUTOMATICALLY FIND A VALID EMBEDDING MODEL ---
@st.cache_resource
def get_valid_embedding_model():
    if not gemini_client:
        return None
        
    available_embedding_models = []
    
    # Ask Google which models support embeddings
    for m in gemini_client.models.list():
        if m.supported_actions and 'embedContent' in m.supported_actions:
            name = m.name.replace('models/', '')
            available_embedding_models.append(name)
            
    if not available_embedding_models:
        # Fallback if supported_actions check fails but models exist
        available_embedding_models = ['text-embedding-004']
        
    # Explicitly filter out the deprecated model that causes the 404 error
    working_models = [m for m in available_embedding_models if "text-embedding-004" not in m]
    
    # If we found newer models, use the most recent one. Otherwise, fallback.
    if working_models:
        selected_model = working_models[-1] # Grab the latest one in the list
    else:
        selected_model = available_embedding_models[0]
        
    print(f"Auto-selected embedding model: {selected_model}")
    return selected_model

# --- RAG HELPER FUNCTIONS ---
def get_embedding(text: str) -> list[float]:
    """Generates an embedding for the user's query using the latest Gemini model."""
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

def search_documents(query_embedding: list[float], match_threshold=0.7, match_count=3):
    """Searches Supabase for the most relevant document chunks."""
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

# Initialize UI chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("My AI RAG App")
    if st.session_state.user:
        st.success(f"Logged in as: {st.session_state.user.email}")
        if st.button("Log Out"):
            supabase.auth.sign_out()
            st.session_state.user = None
            # Reset chat histories on logout
            st.session_state.messages = []
            st.rerun()
    else:
        st.warning("You are not logged in.")

# --- 4. MAIN APP LOGIC ---
st.title("Welcome to the Future Science Classroom")

# If the user is NOT logged in, show the Login/Sign Up tabs
if not st.session_state.user:
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # --- LOGIN TAB ---
    with tab1:
        st.header("Login")
        login_email = st.text_input("Email", key="login_email")
        # CHANGED: Added autocomplete="new-password"
        login_password = st.text_input("Password", type="password", key="login_password", autocomplete="new-password")
        
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

    # --- SIGN UP TAB ---
    with tab2:
        st.header("Create an Account")
        
        # New profile fields
        signup_name = st.text_input("Full Name", key="signup_name")
        signup_age = st.number_input("Age", min_value=1, max_value=120, step=1, value=18, key="signup_age")
        signup_gender = st.selectbox("Gender", ["Select...", "Male", "Female", "Non-binary", "Prefer not to say"], key="signup_gender")
        
        # Standard auth fields
        signup_email = st.text_input("Email", key="signup_email")
        # CHANGED: Added autocomplete="new-password"
        signup_password = st.text_input("Password", type="password", key="signup_password", autocomplete="new-password")
        
        if st.button("Sign Up"):
            if not signup_name.strip():
                st.error("Please enter your full name.")
            elif signup_gender == "Select...":
                st.error("Please select a gender.")
            elif not signup_email or not signup_password:
                st.error("Please enter an email and password.")
            else:
                try:
                    # 1. Create the user in Auth
                    response = supabase.auth.sign_up({
                        "email": signup_email, 
                        "password": signup_password
                    })
                    
                    if response.user:
                        st.success("Auth account created successfully! Now saving profile data...")
                        
                        # 2. Try to insert into the database WITH the new fields
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
    st.subheader("I am your beloved science teacher")
    
    if not chat_model_name or not gemini_client:
        st.error("Cannot start chat because no compatible Gemini models were found for your API key.")
    else:
        st.write("Ask me anything about aerodynamics!")

        # Display chat messages from UI history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What would you like to know?"):
            
            # 1. Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # 2. Add user message to UI chat history BEFORE formatting for Gemini
            st.session_state.messages.append({"role": "user", "content": prompt})

            # 3. Retrieve context from Supabase (RAG)
            with st.spinner("Searching study materials..."):
                query_embedding = get_embedding(prompt)
                retrieved_chunks = []
                current_rag_context = ""
                
                if query_embedding:
                    retrieved_chunks = search_documents(query_embedding)
                    
                if retrieved_chunks:
                    # Combine the retrieved text chunks into a single context string
                    context_texts = [chunk['content'] for chunk in retrieved_chunks]
                    current_rag_context = "\n\n---\n\n".join(context_texts)
                else:
                    current_rag_context = "No specific context found in the study materials."

            # 4. Format the clean Chat History for Gemini using the new SDK types
            gemini_history = []
            
            # We loop through all messages EXCEPT the very last one (the current prompt)
            for msg in st.session_state.messages[:-1]:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append(
                    types.Content(
                        role=role, 
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )

            # 5. Augment the prompt with the retrieved context
            augmented_prompt = f"""
            You are Einstein Junior. Use the following study materials to inform your response. 
            Remember your persona: guide the student, ask questions, and don't just give away the answer immediately.
            
            Study Materials Context:
            {current_rag_context}
            
            Student's Query:
            {prompt}
            """

            # 6. Display AI response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    # Start a fresh chat session with the clean history and system instruction
                    chat_session = gemini_client.chats.create(
                        model=chat_model_name,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                        ),
                        history=gemini_history
                    )
                    
                    # Send augmented message to Gemini using the streaming method
                    response_stream = chat_session.send_message_stream(augmented_prompt)
                    
                    # Stream the response to the UI
                    full_response = ""
                    for chunk in response_stream:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                    
                    # Finalize the message without the cursor
                    message_placeholder.markdown(full_response)
                    
                    # 7. Add assistant response to UI chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # --- SAVE TO STUDY LOGS ---
                    try:
                        supabase.table("study_logs").insert({
                            "participant_id": st.session_state.user.id,
                            "user_query": prompt,
                            "bot_response": full_response,
                            "rag_context": current_rag_context
                        }).execute()
                    except Exception as db_log_error:
                        st.error(f"Failed to save log to database: {db_log_error}")
                    # --------------------------------
                    
                except Exception as e:
                    st.error(f"Error communicating with Gemini: {e}")
