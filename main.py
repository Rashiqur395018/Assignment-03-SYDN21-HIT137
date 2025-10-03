import tkinter as tk
from gui import AIApp

def main():
    root = tk.Tk()
    root.title("Tkinter GUI using OOP")
    root.geometry("900x650")
    app = AIApp(master=root)
    app.pack(fill="both", expand=True)
    root.mainloop()

if __name__ == "__main__":
    main()
