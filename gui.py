from PyQt5.QtWidgets import QApplication, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor
from bot import get_chain
import sys

class WorkerThread(QThread):
    # Define a signal to send data to the main thread
    update_text_signal = pyqtSignal(str)

    def __init__(self, user_input):
        super().__init__()
        self.user_input = user_input

    def run(self):
        rag_chain, retriever = get_chain()
        context = retriever.invoke(self.user_input)
        self.update_text_signal.emit("**المحتوى** \n")
        new_line= '\n'
        for i in range(len(context)):
            self.update_text_signal.emit(new_line)
            self.update_text_signal.emit(f'المقال {i+1} :')
            self.update_text_signal.emit(new_line)
            self.update_text_signal.emit(f'{context[i].page_content}\n')

        self.update_text_signal.emit("Gemini: \n")
        for chunk in rag_chain.stream(self.user_input):
            # Emit the signal with the chunk of text
            self.update_text_signal.emit(chunk)

class ChatInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("واجهة الدردشة")
        self.setGeometry(200, 200, 1400, 600)

        # إعداد التخطيط
        layout = QVBoxLayout()

        # مربع النص لعرض الرسائل
        self.text_box = QTextEdit()
        self.text_box.setFontPointSize(14)
        self.text_box.setReadOnly(True)
        layout.addWidget(self.text_box)

        # إعداد صندوق الإدخال
        self.entry_field = QLineEdit()
        layout.addWidget(self.entry_field)

        # إعداد زر الإرسال
        self.send_button = QPushButton("إرسال")
        self.send_button.clicked.connect(self.send_message)
        layout.addWidget(self.send_button)

        self.setLayout(layout)


                
    def send_message(self):
        user_input = self.entry_field.text()
        if user_input:
            self.display_message(user_input, "العميل")
            # Start the worker thread to fetch the reply
            self.worker = WorkerThread(user_input)
            self.worker.update_text_signal.connect(self.append_to_text_box)
            self.worker.start()
            self.entry_field.clear()

    def display_message(self, message, sender):
        self.text_box.append(f"{sender}: {message}\n")  

    def append_to_text_box(self, text):
        # Insert text into the text box
        cursor = self.text_box.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.text_box.setTextCursor(cursor)
        self.text_box.ensureCursorVisible()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ChatInterface()
    ex.show()
    sys.exit(app.exec_())
