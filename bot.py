from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

checkpoint = "./chatbot_weights.ckpt"

template = """
تصرف كمساعدة شخصية متخصصة
تعليمات:
* اقرأ السؤال بعناية لفهم المقصود بدقة.
* حدد ما إذا كان السؤال بسيطًا أو معقدًا، وكن على استعداد لتقديم إجابات متعمقة إذا لزم الأمر.
* إذا كان السؤال يتطلب تفاصيل أو توضيحات، قدمها بطريقة منظمة وسهلة الفهم.
* للأسئلة التقنية: قدم شرحًا واضحًا واحترافيًا، واستخدم أمثلة عند الحاجة.
* للأسئلة العامة: قدم إجابات شاملة ولكن مختصرة، وكن موجزًا ومباشرًا.
* إذا كانت هناك حاجة لمزيد من التفاصيل، اعرض المساعدة الإضافية أو المصادر التي يمكن الرجوع إليها.
* كن صبورًا ومهذبًا في جميع الردود، وكن مستعدًا لتوضيح أي نقاط غير واضحة.
* حافظ على نبرة مهنية وودية، وكن على استعداد للتكيف مع أسلوب المستخدم.
* بعد تقديم الإجابة، تحقق من رضا المستخدم وعرض المساعدة الإضافية إذا لزم الأمر.

المعرفة التى تعرفها:
{context}

السؤال: {question}

الاجابة:
"""

def format_docs(docs):
    return "\n\n". join(doc.page_content for doc in docs)


class embedding:
    def __init__(self):
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
        
    def embed_query(self, query):
        return self.model.encode(query).tolist()


def get_api_key():
	# Load environment variables from .env file
	load_dotenv()
	
	# Retrieve the API key from environment variables
	api_key = os.getenv("GOOGLE_API_KEY")
	return api_key


def load_retriver(persist_dir, embed_model):
	vector_database = Chroma(embedding_function=embed_model, persist_directory=persist_dir)
	return vector_database.as_retriever(search_types='similarity', search_kwargs={'k':10})

def get_chain():
	return rag_chain, retriever


custom_rag_prompt = PromptTemplate.from_template(template)
embed = embedding()
retriever = load_retriver('./chroma_db', embed)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=get_api_key())
rag_chain = ({"context":retriever | format_docs , 'question': RunnablePassthrough()}| custom_rag_prompt| llm | StrOutputParser())


