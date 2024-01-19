# app.py
import streamlit as st
import os
from pdf_to_qa_retrieve import pdf_to_qa_retrieve

def main():
    st.title("PDF to QA Retrieval")

    # Create a directory named 'uploads' if it doesn't exist
    os.makedirs("uploads", exist_ok=True)

    # File upload
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

    # Question input
    question = st.text_input("Enter your question:")

    # Check if both PDF and question are provided
    if pdf_file is not None and question:
        # Save the PDF file
        pdf_path = os.path.join("uploads", pdf_file.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())
        # print(pdf_path)
        # Perform QA retrieval
        answer = pdf_to_qa_retrieve(pdf_path, question)

        # Display question and answer
        st.header("Question:")
        st.write(question)

        st.header("Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()
