
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OpenAI")

class Chatbot:
    def __init__(self, model_name="gpt-4o-mini"):
        # Generate a unique conversation ID to manage multiple user sessions.
        self.thread_id = uuid.uuid4()
        self.config = {"configurable": {"thread_id": self.thread_id}}
        
        # Initialize the chat model.
        self.model = ChatOpenAI(model=model_name)
        
        # Set up embeddings and load FAISS vector store.
        self.embeddings = OpenAIEmbeddings()
        self.vector_db = FAISS.load_local(
            r"C:\Users\Lisardo Iniesta\OneDrive\1_WBS\9. Generative AI\content\faiss_index", 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})
        
        # Define the prompt template.
        template = ("""Eres un chatbot experto en recetas Catalanas que está teniendo una conversación con un humano. Responde la pregunta basándote en el siguiente contexto y la conversación previa.

Conversación previa:
{chat_history}

Contexto para responder la pregunta:
{context}

Nueva pregunta del humano: {question}

Respuesta:""")
        self.prompt = PromptTemplate(template=template, input_variables=["chat_history", "context", "question"])
        
        # Build the state graph workflow.
        self.workflow = StateGraph(state_schema=MessagesState)
        self._build_graph()
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)

    def _build_graph(self):
        # Add a node for calling the model within the workflow.
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", self.call_model)

    def call_model(self, state: MessagesState):
        # Take the most recent message as the new query.
        question = state["messages"][-1].content
        
        # Retrieve relevant documents.
        docs = self.retriever.get_relevant_documents(question)
        context_str = "\n".join(doc.page_content for doc in docs)
        
        # Gather previous conversation history, if any.
        chat_history = "\n".join(msg.content for msg in state["messages"][:-1])
        
        # Format the prompt with context and history.
        formatted_prompt = self.prompt.format(chat_history=chat_history, context=context_str, question=question)
        
        # Generate a response with the model.
        response = self.model.invoke([HumanMessage(content=formatted_prompt)])
        
        # Return both the generated message and the retrieved documents.
        return {"messages": response, "retrieved_docs": docs}

    def send_message(self, message_content: str):
        # Prepare the input message.
        input_message = HumanMessage(content=message_content)
        events = []
        # Process the message through the state graph workflow.
        for event in self.app.stream({"messages": [input_message]}, self.config, stream_mode="values"):
            events.append(event)
        return events

# Create a helper function that wraps the chatbot's send_message method.
def chain(prompt):
    events = st.session_state.chatbot.send_message(prompt)
    final_event = events[-1]
    answer = final_event["messages"][-1].content
    source_documents = final_event.get("retrieved_docs", [])
    return {"answer": answer, "source_documents": source_documents}

# -------------------------
# Streamlit App Integration
# -------------------------
st.title("DeepPaladar del JP")

# Initialize the chatbot only once.
if "chatbot" not in st.session_state:
    st.session_state.chatbot = Chatbot()

# Initialize the chat history in session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# React to user input.
if prompt := st.chat_input("Curious minds wanted!"):
    # Display the user's message.
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Going down the rabbithole for answers..."):
        # Get the answer and the source documents via the chain function.
        answer_dict = chain(prompt)
        response = answer_dict["answer"]
        source_documents = answer_dict["source_documents"]

        # Display the assistant's response.
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Display the source documents within an expander.
        with st.expander("Source Documents"):
            for doc in source_documents:
                # Display document metadata if available.
                source = doc.metadata.get("source", "Unknown") if hasattr(doc, "metadata") else "Unknown"
                st.markdown(f"**Document: {source}**")
                st.markdown(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": response})
