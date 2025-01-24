In this comprehensive tutorial, you’ll discover:

The key concepts behind RAG and how to use LangChain to create sophisticated chatbots.
How to build both stateless and stateful (context-aware) chatbots using LangChain with step by step explanation of the code.
The steps to connect your chatbot to a knowledge repository like a PDF, empowering it to answer questions about the document’s content. 📖
Hidden Secrets: A bonus section awaits those who crave a deeper understanding. We’ll crack open some LangChain secrets and see how the magic works under the hood.
Spoiler Alert — In this tutorial, we’ll dive into building a RAG chatbot that can interact with a research paper (PDF format). The beauty is, you can easily adapt the code to work with any content — html files, csv, SQL databases, websites, and more! Get ready to unlock the knowledge within your documents.

To make this journey even smoother, you’ll find the complete code and data on my GitHub repository.

A Note for Experienced AI Adventurers — This article is packed with information! It starts with a thorough exploration of RAG and LangChain concepts and gradually guides you through building your chatbot. If you’re already well-versed in the theory, feel free to jump to the sections titled “Setting up Your Environment” or “Building Your RAG Chatbot (Step-by-Step)”.

However, even for seasoned AI enthusiasts, skimming the earlier sections might provide a helpful refresher. As the saying goes, “Knowledge is power”

Ready to take your AI skills to the next level? Let’s dive in and build the knowledge-powered chatbot of your dreams!


Prerequisites
Before diving into the world of RAG chatbot creation, let’s make sure you have the right tools and knowledge:

Basic Python Proficiency: While I will provide code examples, a fundamental understanding of Python concepts (variables, functions) will make the process much smoother.
Essential Libraries: You’ll need to install the following libraries using the ‘pip’ package manager in your terminal:
— langchain: The heart of our chatbot building process.
— openai: Lets us tap into powerful language models from OpenAI.
— pinecone-client: For setting up our vector database to store knowledge.
OpenAI and Pinecone Accounts: You’ll need API keys to use these services. Instructions for getting them are included below.
Your Knowledge Source: The beauty of RAG is that you provide the data your chatbot learns from! Have your research paper (PDF) or other content type (text files, website URLs, company documents) ready.
Don’t worry if you’re new to all of this! I’ll help guide you through setting up your environment along the way.

Ready to take the next step? Let’s uncover the fascinating theory behind rag and Langchain!

Understanding RAG, and LangChain
Ever had a conversation where someone seems to know everything? Maybe a friend who aced history class or a family member who can fix anything. That’s kind of the magic we’re aiming for with your PDF chatbot — a constant source of knowledge at your beck and call.

But how do we make a computer program that can tap into the vastness of the PDF and have a conversation? That’s where RAG and LangChain come in!

RAG (Retrieval-Augmented Generation): Your Chatbot’s Super-powered Search Engine
Imagine you’re at a giant library with countless books on every topic imaginable. RAG is like having a super-efficient assistant who can instantly find the most relevant books based on your questions. Here’s the gist of how it works:

You ask your chatbot a question. For instance, “What’s the capital of France?”
RAG swings into action! It scans your chosen data source for entries that match your question.
With the most relevant information retrieved, RAG hands it over to the language model.
The language model now uses this factual information to form an accurate and helpful response, telling you that the capital of France is Paris. ⛪️
At its core, Retrieval-Augmented Generation (RAG) leverages the strengths of two powerful AI techniques: information retrieval and large language models (LLMs). Let’s delve into the process:

Query Understanding: When a user interacts with your chatbot, RAG first employs natural language processing (NLP) techniques. This involves breaking down the user’s question into its constituent parts (tokens) and analyzing its semantic meaning and intent.
Retrieval from the Knowledge Base: Armed with an understanding of the user’s query, RAG interacts with a specialized database like FAISS, Weaviate, or Pinecone. These databases don’t store text directly. Instead, they store meaning-based mathematical representations of information called vectors. RAG generates a similar vector for the user’s query and finds the most closely matching vectors in the database. These matching vectors lead RAG to the specific sections of text most likely to contain the answer.
Enhancing the LLM with Retrieved Knowledge: Once RAG successfully retrieves the most pertinent information (text snippets, article summaries etc.), it feeds this data to a pre-trained LLM like GPT, Gemini etc. These LLMs are statistical models trained on massive amounts of text data, granting them the ability to process information and generate human-quality text.
Response Generation: Empowered by the retrieved knowledge from the provided text, the LLM steps in to craft the response to the user’s query. This response can take various forms depending on the prompt provided. It could be a concise answer to the question, a comprehensive summary of the article, or even a creatively formatted text response.

Source — https://www.elastic.co/search-labs/blog/retrieval-augmented-generation-rag
Summary — Traditionally, an LLM relies on its stored knowledge to answer questions. RAG enhances this process through three key steps: retrieval, augmentation, and generation. First, your question is converted into a “vector embedding”. RAG then performs retrieval, searching a database where content from your information source is also stored as vectors. It identifies the most relevant content, which is used for augmentation — combining it with your original question to create a richer input. Finally, the LLM uses this enhanced input for generation, producing a more accurate and helpful answer.

LangChain: The Modular Maestro
LangChain is a powerful 💪 Python-based framework designed to simplify the development of applications powered by large language models (LLMs). It does this by providing a modular and flexible structure that streamlines common NLP tasks and the integration of various AI components. Here’s why it’s a valuable tool for our chatbot project:

Abstraction: LangChain breaks down complex LLM interactions into reusable building blocks like chains, agents, and prompts. This abstraction layer eliminates much of the intricate coding, especially when dealing with the interplay of tools like RAG, making development more accessible.
Modularity: Its modular design promotes flexibility and experimentation. You can easily swap out different components or modify your chatbot’s structure without the need to completely rebuild from scratch.
Integration: LangChain seamlessly connects with various external data sources, retrieval models, and LLMs. This makes the process of integrating the any knowledge base and RAG a smooth experience.
Think of LangChain as a toolbox filled with different components to build your chatbot. Here are the key players:

Chains: These are essentially pipelines that connect various components within the chatbot architecture. In our case, we’ll construct a chain that seamlessly integrates the user’s query with the retrieval manager (RAG) and subsequently, the response generation stage.
Agents: These are like the workers in your chatbot factory. We will create a specialized agent that acts as a bridge between LangChain and RAG. It receives the user’s query from the chain, transmits it to RAG for information retrieval, and then feeds the retrieved data back into the chain for further processing.
Prompts: LangChain empowers you to design prompts that guide the LLM in crafting the most effective response. For example, a prompt might instruct the LLM to provide a succinct answer to the user’s question or generate a concise summary of a retrieved Wikipedia article.
The synergy between RAG’s information retrieval capabilities and LangChain’s modular structure lays the foundation for constructing a chatbot that leverages the vast and complicated content in any data source (in our case PDF) to deliver informative and creative responses to user queries.

If you’re wondering why we wouldn’t simply use standard Large Language Models (LLMs) to get answers to specific questions through a chatbot, then you’re on the right track. Let’s delve into why this approach might not be ideal.

Why RAG? The Limitations of Standard LLMs
Standard Large Language Models (LLMs) like GPT-3.5 are remarkable tools, able to generate impressively human-like text, translate languages, and more. However, when it comes to building a chatbot for specialized knowledge domains, they have shortcomings:

Open-Domain Limitations: While LLMs have learned from a massive amount of text data, they may struggle when presented with questions about very specific topics, such as the technical details of a product or the intricacies of a company policy.
Handling Large Documents: LLMs often have token limits, meaning they can only process a limited amount of text at once. This makes it difficult to use them directly on lengthy documents like research papers or complex manuals.
The Risk of Hallucination: LLMs can sometimes generate responses that are confidently worded but factually incorrect, particularly if the topic is outside the common knowledge they were trained on. This can be misleading for users relying on your chatbot as an information source.
Harnessing Your Knowledge: Your company likely has a wealth of information in documents, guides, and databases. Standard LLMs don’t tap directly into this knowledge; you need a way to channel this information for your chatbot to leverage.
How RAG Excels:
RAG addresses these limitations by:

Focused Knowledge: RAG allows you to train your chatbot on a specific knowledge base, such as your company’s manuals or research papers. This ensures responses are relevant and grounded in verified information.
Fact-Checking Potential: With RAG, your chatbot can cite relevant sections of your documents, increasing the trustworthiness of its responses.
Controlled Expertise: You gain control over what information the chatbot knows, allowing you to tailor its expertise to the exact domain you need.
Time to Gear Up! Let’s get your environment ready for the exciting journey of building a RAG chatbot.

Setting up Your Environment
Ready to make some magic happen with RAG and LangChain? Follow these steps to make sure you have everything in place to build our chatbot:

API Keys
OpenAI: To interact with powerful language models, you’ll need an API key from OpenAI.
— Visit https://platform.openai.com/signup/ to create an account if you don’t have one.
— Navigate to your account settings and find your existing API keys, or generate a new one. Keep this secure!
— Important Note: As far as I know, OpenAI no longer offers a free tier. You may need to add a payment method and a small amount of credit (around $5 should be more than plenty for a plethora of experimental projects) to your account.

Generating open-AI API key
Pinecone: For this tutorial, we’ll use Pinecone to create a vector database — a powerful way to store our knowledge base for fast and accurate information retrieval.
— It offers a generous free tier, perfect for getting started.
— Create an account at https://www.pinecone.io/
— Find your API key in your Pinecone console.
— You’ll also need to create an index in Pinecone. An index is like a container that will hold the vector embeddings of your knowledge documents. You can choose a suitable embedding model like ‘text-embedding-ada-002’ which generates embeddings of size 1536. I created an index named ‘pdf-vectorised’, but feel free to give it a different name. Remember, we’ll use this index name in our code to access and store vectors.

Create a Pinecone index (pdf-vectorised) and API Key to store vector embeddings
Install the Necessary Libraries
I highly recommend setting up a virtual environment for this project. Virtual environments help isolate your project’s dependencies from other Python projects on your system, preventing potential conflicts. Here’s how to install the libraries:

Create a file named requirements.txt in your project directory and add the libraries as shown below.
langchain
openai
pinecone-client
langchain-pinecone
langchain-openai
python-dotenv
pypdf
From your terminal, navigate to your project directory and run the following command pip install -r requirements.txt
Now that we have our foundation, let’s get to the interesting stuff and code up our first chatbot that interacts with a PDF using RAG and LangChain. I hope you guys are excited to see it in action 😎

Building Your RAG Chatbot (Step-by-Step)
Get ready to dive into the exciting world of coding! In this section, we’ll lay the groundwork for your RAG chatbot. We’ll be using a PDF of a research paper titled “The Impact, Advancements and Applications of Generative AI” to build a chatbot that can interact with its content. First, let’s set up some essential files and load the paper’s data into our Pinecone vector database.

Environment Files and Data Ingestion
First, let’s prepare the ground for successful development:

Environment Variables

Create a file named .env in your project directory. Remember to keep this file safe — it’ll hold sensitive information like your API keys. We’ll use a Python library called dotenv to access these keys securely.

OPENAI_API_KEY={openAI API key}
INDEX_NAME={Pinecone index name}
PINECONE_API_KEY={Pinecone API key}
Data Ingestion

Step 1 — Importing Libraries

Create a file named ingestion.py and import all the libraries we’ll need to read and ingest data.

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
Don’t worry if some of these libraries and their purposes seem unclear now. As we write our data ingestion code, you’ll gain a clearer understanding of how each one plays its part.

Step 2 — Loading the PDF Content

loader = PyPDFLoader("data/impact_of_generativeAI.pdf")
document = loader.load()
pdf_file_path: Make sure to replace the placeholder with the correct location of your research paper PDF on your computer.
loader: We create a PyPDFLoader object, giving it the path to your PDF file. This loader is specifically tailored to handle PDF documents within the LangChain framework.
documents: When we call loader.load(), it reads your PDF, extracts the text, and returns a list of LangChain documents, ready for further processing.
Step 3 — Splitting Documents into Chunks

Now that we have our loaded documents, let’s break them down into smaller pieces for a more efficient workflow. Here, we’ll use LangChain’s CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(document)
print(f"created {len(texts)} chunks")
Why Split Documents? There are three main reasons for splitting documents into smaller chunks:

Memory Efficiency: Large documents can be resource-intensive to process. Splitting them into smaller pieces reduces the memory footprint, making the process smoother, especially when dealing with hefty PDFs.
Token Limits: Language models can only process a limited amount of text at once. Splitting up our documents allows us to work within those limitations.
Context Preservation: While splitting, we want to ensure each chunk retains some context from its surrounding text. This helps the language model understand the meaning of each piece accurately.
Understanding Chunk Size and Overlap

The CharacterTextSplitter takes two important parameters:

chunk_size: This defines the maximum number of characters in each split chunk. A smaller chunk size creates more, but shorter, chunks. A larger chunk size creates fewer, but longer, chunks. It’s crucial to find a good balance when choosing the chunk_size. Extremely small chunks might not contain enough context for the language model to understand properly, whereas very large chunks might become processing bottlenecks.
chunk_overlap: This specifies the number of characters that overlap between consecutive chunks. Overlap helps the language model maintain context between neighboring pieces, especially important for longer sentences that might be split across chunks. We set chunk_overlap to 100 characters. This ensures some overlap between chunks, helping the language model understand the flow of information across splits.
Step 4 — Creating Embeddings and Storing in Pinecone

Now that we have our text chunks, let’s use OpenAI’s language models to create vector representations (embeddings) that capture their meaning. We’ll then store these embeddings in our Pinecone database for efficient retrieval later.

embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))
PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))
OpenAIEmbeddings: We import this class from LangChain to interact with OpenAI’s embedding models using OpenAI API key which we set in the .env file.
PineconeVectorStore: This class connects LangChain to your Pinecone vector database. By calling the from_documents method, we save the vector embeddings corresponding to every text chunk in your Pinecone database.
After running this code, if you go to your Pinecone UI and navigate to the specified index, you should see the vector embeddings being uploaded and stored (as shown in the image below), ready for your RAG chatbot to access!


Vector Embeddings updated in the Pinecode index
Building a Stateless RAG Chatbot with LangChain
It’s time to build the heart of your chatbot! Let’s start by creating a new Python file named stateless-bot.py. Within this file, begin by importing the necessary packages:

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
We’re calling this script stateless-bot.py because in its current form, the bot doesn’t keep track of previous conversations. Each question and answer interaction is independent, which is a great start, but shortly we will also explore adding a “memory” to the bot.

Step 1 — Loading Your Knowledge

Now that we have our imports in place, let’s set up a way to access the vector embeddings we stored in Pinecone earlier. This will allow our chatbot to retrieve relevant information from the PDF. Here’s the code:

embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
This code specifically establishes a connection to your existing Pinecone index where you’ve stored the vector representations (embeddings) of your research paper.

Step 2 — Building Your RAG Chains and Asking Questions

Let’s set up the LangChain workflows (chains) that will drive the logic behind your RAG chatbot:

chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

qa = RetrievalQA.from_chain_type(
    llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
)    

res = qa.invoke("What are the applications of generative AI according the the paper? Please number each application.")
print(res) 

res = qa.invoke("Can you please elaborate more on application number 2?")
print(res)
Initializing OpenAI: We initialize the ChatOpenAI LLM specifying temperature=0. This encourages the model to generate more focused and less creative responses.
Core RAG Chain: In LangChain, RetrievalQA.from_chain_type is a function used to create a RetrievalQA chain, a specific type of chain designed for question answering tasks.
Here is the output

{'query': 'What are the applications of generative AI according the the paper? Please number each application.',
 'result': 'Based on the provided context, the applications of generative AI mentioned in the paper are as follows:\n\n1. Creative Content Generation\n2. Data Augmentation\n3. Simulation and Modeling\n4. Scenario Generation and Planning\n5. Personalization and Recommendation Systems\n6. Design and Creativity Assistance\n7. Scientific Discovery and Exploration\n8. Bridging Gaps in Data\n\nThese applications cover a wide range of fields and industries where generative AI can be utilized.'}
{'query': 'Can you please elaborate more on application number 2?',
 'result': "I'm sorry, but the specific application numbers are not provided in the context you shared. If you can provide more details or context about application number 2, I would be happy to help elaborate on it."}
Based on the responses you received, you can clearly see the impact of the chatbot being stateless. While the first question, “What are the applications of generative AI according to the paper? Please number each application,” was answered successfully, the second question, “Can you please elaborate more on application number 2?”, resulted in the response “I do not have enough context.”

This happens because our current chatbot doesn’t retain information from previous interactions. Each question is treated independently. In the second query, the bot lacks the context of the specific application we refer to (number 2) because it doesn’t remember the answer or details provided in the first response

Congratulations! 🎉 You’ve successfully built your first chatbot with the remarkable power of RAG! It can tap into the knowledge within your PDF and respond thoughtfully to your questions.

The beauty of LangChain lies in how it simplifies the process of building such sophisticated chatbots. With a few lines of code, we’ve harnessed powerful language models and created a focused knowledge retrieval system.

For a more detailed breakdown of how RetrievalQA works, refer to the “Bonus — Breaking Down the LangChain Chain” section.

Enhancing Your Chatbot with Context (Stateful)
So far, our chatbot treated each question as an isolated event. While it could answer standalone queries, it struggled with follow-up questions or conversations that build upon previous interactions. This is where stateful chatbot excel.

Why Stateful Matters
Stateful chatbot maintain a memory of the conversation. This offers several advantages:

Understanding Context: They can understand how the current question relates to previous inquiries, providing more insightful and accurate answers.
Natural Conversation Flow: The conversation feels more fluid and less repetitive, as the chatbot can reference past interactions, making the experience more engaging for the user.
Tailored Responses: The ability to draw upon past context allows the chatbot to tailor its responses specifically to the ongoing conversation.
Let’s see how we can transform our chatbot into a stateful one using LangChain’s ConversationalRetrievalChain

import os
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

warnings.filterwarnings("ignore")

load_dotenv()

chat_history = []

if __name__ == "__main__":

    embeddings = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
    )  

    res = qa({"question": "What are the applications of generative AI according the the paper? Please number each application.", "chat_history": chat_history})
    print(res)

    history = (res["question"], res["answer"])
    chat_history.append(history)

    res = qa({"question": "Can you please elaborate more on application number 2?", "chat_history": chat_history})
    print(res)
Explanation: Tracking the Conversation

chat_history List: We’ve introduced the chat_history list, which acts as a memory bank for our chatbot.
Saving Interactions: After each question-answer exchange, we create a tuple containing both the user’s question and the model’s response. This tuple is then appended to the chat_history list, effectively storing the conversation history.
This is enabled by using ConversationalRetrievalChainthat facilitates context-aware retrieval with history. Let’s look at the output.

{'question': 'What are the applications of generative AI according the the paper? Please number each application.',
 'chat_history': [],
 'answer': 'Based on the paper, the applications of generative AI are as follows:\n\n1. Art, entertainment, design, and scientific research\n2. Healthcare\n3. Fashion\n4. Gaming\n5. Advertising\n6. Content creation\n\nThese are some of the industries where generative AI can be applied according to the paper.'}
{'question': 'Can you please elaborate more on application number 2?',
 'chat_history': [('What are the applications of generative AI according the the paper? Please number each application.', 'Based on the paper, the applications of generative AI are as follows:\n\n1. Art, entertainment, design, and scientific research\n2. Healthcare\n3. Fashion\n4. Gaming\n5. Advertising\n6. Content creation\n\nThese are some of the industries where generative AI can be applied according to the paper.')],
 'answer': 'Generative AI has various applications in healthcare. It can contribute by generating synthetic medical data for training AI models, simulating biological processes, and designing personalized treatment plans. Generative AI can also assist in drug discovery by generating new molecule structures and predicting their properties. Additionally, it can be used to simulate complex scenarios for research and training purposes in the medical field. Overall, generative AI in healthcare aims to improve personalized treatment options, enhance drug discovery processes, and simulate biological processes for better understanding and advancements in the medical field.'}
Such an amazing thing! 🤩 Observe how our chatbot is learning as the conversation progresses. Initially, the chat_history was empty during the first question. Our chatbot provided a solid general answer. Now, look at the output from the second query! The first interaction has become part of the chat_history, providing much-needed context. Because of this, the chatbot understands we’re specifically interested in healthcare applications of generative AI. This allows it to provide a focused and informative answer tailored to application number 2. Our chatbot is becoming truly conversational!

Therefore, by adding state awareness with chat_history management and leveraging the capabilities of ConversationalRetrievalChain, our chatbot gains the ability to understand context and craft responses that build upon previous interactions. This significantly enhances the user experience by creating a more natural and engaging conversation flow.

Before we conclude this article, let’s take a moment to peek under the hood and gain a deeper understanding of how those LangChain chains function. Let’s take a moment to peek under the hood and gain a deeper understanding of how those LangChain chains function.

Bonus — Breaking down the LangChain chain
LangChain chains are the building blocks of applications built with the LangChain framework. These chains connect various components, like Large Language Models (LLMs), retrieval methods, and data sources, to perform complex tasks. They essentially define the workflow for how information flows through your application.

RetrievalQA
RetrievalQA is a specialized type of Langchain chain designed specifically for question answering tasks. These chains leverage the power of LLMs while incorporating a retrieval step to improve the accuracy and context of the answers.

Here’s a breakdown of the key components within a RetrievalQA chain created using RetrievalQA.from_chain_type:

LLM (Large Language Model): This is the core AI component responsible for generating the answer. You can specify the LLM you want to use within the function’s arguments.
Retriever: This component plays a crucial role in the retrieval step. It identifies and fetches documents or information most relevant to the user’s question. The specific retrieval method used depends on the chosen retriever within the function.
Chain Type: While the default chain type is “stuff”, which combines all retrieved documents, RetrievalQA.from_chain_type allows you to specify different chain types. These chain types offer various strategies for handling retrieved information and interacting with the LLM for answer generation. Examples include MapReduce for processing large datasets or Refine for iterative refinement based on LLM feedback.
By combining these elements, RetrievalQA chains enable you to build effective question answering applications within the LangChain framework. The retrieved information provides valuable context for the LLM, leading to more comprehensive and informative answers.

ConversationalRetrievalChain
Remember how we transformed our chatbot to understand conversations? That magic happened thanks to LangChain’s ConversationalRetrievalChain. Let’s dive deeper into how it enables us to build stateful chatbots that maintain a conversation history. While RetrievalQA excels at answering individual questions based on retrieved information, ConversationalRetrievalChain takes it a level up. This specialized chain empowers you to build stateful chatbots that maintain a conversation history when passed explicitly.

To further understand how these chains work, let’s understand into the respective prompts used underneath both chain types.

Sneak Peek into Prompts Used
RetrievalQA Prompt: The emphasis here is on providing the relevant context and current question to the LLM for the best possible answer as shown in the prompt below.

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
ConversationalRetrievalChain Prompt: This prompt is more complex. It’s designed to work chat history, if pass explicitly. It cleverly leverages the LLM’s capability to rephrase a follow-up question into a standalone one, ensuring that it can be answered using the provided chat history as context.

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""