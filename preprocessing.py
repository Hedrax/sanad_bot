import os
from langchain_chroma import Chroma
from embedding import embedding


def load_context_data(dir):
	# Path to the directory containing your .txt files
	main_folder_path = dir
	
	# List to store the contents of each file
	list_of_strings = []
	
	# Iterate through each file in the directory
	for folder in os.listdir(main_folder_path):
	    folder_path = os.path.join(main_folder_path, folder)
	    for filename in os.listdir(folder_path):
	        if filename.endswith('.txt'):
	            file_path = os.path.join(folder_path, filename)
	
	            # Open and read the file, then add its content to the list
	            with open(file_path, 'r', encoding='utf-8') as file:
	                file_content = file.read()
	                list_of_strings.append(file_content)
	return list_of_strings

list_of_strings = load_context_data('./context/')
embed_model = embedding()


vector_database = Chroma.from_texts(list_of_strings[:40000], embed_model, persist_directory="./chroma_db")
vector_database.add_texts(list_of_strings[40000:])