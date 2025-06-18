import customtkinter as ctk

class JarvisGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("J.A.R.V.I.S. Core")
        self.geometry("400x150")
        self.resizable(False, False)

        # Настройка внешнего вида
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Контейнер
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Лейбл для отображения статуса
        self.status_label = ctk.CTkLabel(
            self,
            text="Инициализация...",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.status_label.grid(row=0, column=0, padx=20, pady=20)

    def update_status(self, new_status):
        """Метод для безопасного обновления текста лейбла из других потоков."""
        self.status_label.configure(text=new_status)

    def run_gui(self):
        """Этот метод больше не будет использоваться напрямую,
           управление циклом будет в main.py."""
        self.mainloop()