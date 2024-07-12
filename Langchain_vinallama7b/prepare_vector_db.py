from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# Khai bao bien
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

# Ham 1. Tao ra vector DB tu 1 doan text
model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embeddings = GPT4AllEmbeddings(
    model_name=model_name,
    gpt4all_kwargs=gpt4all_kwargs
)

def create_db_from_text():
    raw_text = """Bạn có biết rằng KFC, thương hiệu gà rán nổi tiếng toàn cầu, vừa kỷ niệm 70 năm hoạt động chưa? Từ những ngày đầu tiên khi Colonel Harland Sanders mở cửa hàng đầu tiên tại Corbin, Kentucky, KFC đã trở thành một biểu tượng ẩm thực không thể thiếu trong cuộc sống của hàng triệu người trên thế giới.
Với công thức gia vị bí mật độc đáo và quy trình chế biến khắt khe, KFC mang đến những miếng gà rán giòn ngon, vàng ươm, thơm phức. Kết hợp với những phụ gia tươi ngon như khoai tây chiên, salad và bánh mì, KFC tạo nên những bữa ăn hoàn hảo, đầy đủ dinh dưỡng và hấp dẫn vị giác.
Không chỉ vậy, KFC còn không ngừng đổi mới và ra mắt các sản phẩm mới độc đáo, đáp ứng mọi khẩu vị của từng vùng miền. Từ gà rán truyền thống đến gà sốt các loại, từ món ăn nóng hổi đến món ăn lạnh mát, KFC luôn mang đến những trải nghiệm ẩm thực khó quên cho thực khách.
Hãy đến với KFC - nơi có những miếng gà rán ngon tuyệt, phục vụ chu đáo và không gian ấm cúng, để thưởng thức những bữa ăn đầy hương vị và khoảnh khắc gia đình sum họp bên nhau. Với KFC, bạn sẽ khó lòng quên được vị giác tuyệt vời mà họ mang lại!
Slogan của KFC là "Finger Lickin' Good" - Ngon đến phải liếm ngón tay.
"""

    # Chia nho van ban
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=20,
        chunk_overlap=2,
        length_function=len

    )

    chunks = text_splitter.split_text(raw_text)
    # Embeding
    embedding_model = GPT4AllEmbeddings(model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf", gpt4all_kwargs = gpt4all_kwargs)

    # Dua vao Faiss Vector DB
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db



def create_db_from_files():
    # Khai bao loader de quet toan bo thu muc dataa
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embeding
    embedding_model = GPT4AllEmbeddings(model_name ="all-MiniLM-L6-v2.gguf2.f16.gguf")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

create_db_from_text()
