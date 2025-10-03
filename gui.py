import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from models import TextSentimentModel, ImageClassificationModel
from utils import timed, logged
import threading
import os

class AIApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self._create_models()
        self._build_ui()

    def _create_models(self):
        self._text_model = TextSentimentModel()
        self._image_model = ImageClassificationModel()

    def _build_ui(self):
        controls = ttk.LabelFrame(self, text="Input & Model Selection", padding=10)
        controls.pack(fill="x", padx=10, pady=6)

        ttk.Label(controls, text="Input Type:").grid(row=0, column=0, sticky="w")
        self.input_type = tk.StringVar(value="text")
        input_menu = ttk.OptionMenu(controls, self.input_type, "text", "text", "image", command=self._on_input_type_change)
        input_menu.grid(row=0, column=1, sticky="w")

        self.text_frame = ttk.Frame(controls)
        ttk.Label(self.text_frame, text="Enter text:").grid(row=0, column=0, sticky="w")
        self.text_entry = tk.Text(self.text_frame, height=4, width=60)
        self.text_entry.grid(row=1, column=0, columnspan=3, pady=4)
        self.text_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=6)

        self.image_frame = ttk.Frame(controls)
        ttk.Label(self.image_frame, text="Selected image:").grid(row=0, column=0, sticky="w")
        self.image_path_var = tk.StringVar(value="")
        ttk.Label(self.image_frame, textvariable=self.image_path_var, width=50).grid(row=0, column=1, sticky="w")
        ttk.Button(self.image_frame, text="Browse...", command=self._browse_image).grid(row=0, column=2, padx=6)

        models_frame = ttk.Frame(controls)
        models_frame.grid(row=2, column=0, columnspan=4, pady=6, sticky="w")
        ttk.Label(models_frame, text="Run models:").grid(row=0, column=0, sticky="w")
        self.run_text_model = tk.BooleanVar(value=True)
        self.run_image_model = tk.BooleanVar(value=True)
        ttk.Checkbutton(models_frame, text="Text sentiment", variable=self.run_text_model).grid(row=0, column=1, padx=6)
        ttk.Checkbutton(models_frame, text="Image classification", variable=self.run_image_model).grid(row=0, column=2, padx=6)

        actions = ttk.Frame(controls)
        actions.grid(row=3, column=0, columnspan=4, sticky="w", pady=(6,0))
        ttk.Button(actions, text="Run Selected", command=self._on_run).grid(row=0, column=0, padx=6)
        ttk.Button(actions, text="Clear Outputs", command=self._clear_outputs).grid(row=0, column=1, padx=6)
        ttk.Button(actions, text="Model Info", command=self._show_model_info).grid(row=0, column=2, padx=6)

        outputs = ttk.LabelFrame(self, text="Model Outputs", padding=10)
        outputs.pack(fill="both", expand=False, padx=10, pady=6)

        ttk.Label(outputs, text="Text Model Output:").grid(row=0, column=0, sticky="w")
        self.text_output = tk.Text(outputs, height=6, width=60, state="disabled")
        self.text_output.grid(row=1, column=0, padx=6, pady=4)

        ttk.Label(outputs, text="Image Model Output:").grid(row=0, column=1, sticky="w")
        self.image_output = tk.Text(outputs, height=6, width=60, state="disabled")
        self.image_output.grid(row=1, column=1, padx=6, pady=4)

        bottom = ttk.Notebook(self)
        bottom.pack(fill="both", expand=True, padx=10, pady=6)

        self.oop_tab = ttk.Frame(bottom)
        bottom.add(self.oop_tab, text="OOP Concepts & Explanations")
        self._build_oop_tab(self.oop_tab)

        self.model_tab = ttk.Frame(bottom)
        bottom.add(self.model_tab, text="Selected Model Info")
        self._build_model_tab(self.model_tab)

        self._on_input_type_change(self.input_type.get())

    def _build_oop_tab(self, parent):
        txt = tk.Text(parent, wrap="word")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", self._oop_text())
        txt.config(state="disabled")

    def _build_model_tab(self, parent):
        self.model_info_text = tk.Text(parent, wrap="word")
        self.model_info_text.pack(fill="both", expand=True)
        self._fill_model_info()

    def _fill_model_info(self):
        info = self._compose_model_info()
        self.model_info_text.config(state="normal")
        self.model_info_text.delete("1.0", "end")
        self.model_info_text.insert("1.0", info)
        self.model_info_text.config(state="disabled")

    def _compose_model_info(self):
        info_parts = []
        info_parts.append("Text Model:\n" + self._text_model.get_info() + "\n")
        info_parts.append("Image Model:\n" + self._image_model.get_info() + "\n")
        return "\n".join(info_parts)

    def _oop_text(self):
        return (
            "This application demonstrates OOP concepts:\n\n"
            "1. Encapsulation\n"
            "- UI state and model instances are stored as attributes on AIApp with leading underscores.\n"
            "- Model classes hide internal pipeline instances and expose a simple predict() method.\n\n"
            "2. Multiple inheritance\n"
            "- Model classes inherit from ModelInterface and LoggingMixin (see models.py and utils.py).\n"
            "- This allows adding logging behaviour without changing base model implementations.\n\n"
            "3. Decorators (multiple)\n"
            "- We use @timed and @logged stacked on model predict calls (defined in utils.py).\n"
            "- The decorators add timing, caching or logging behaviour in a composable way.\n\n"
            "4. Polymorphism\n"
            "- TextSentimentModel and ImageClassificationModel implement predict(input) with the same signature.\n"
            "- The GUI calls predict(...) on any model interchangeably.\n\n"
            "5. Method overriding\n"
            "- Derived classes override the abstract predict() method from the ModelInterface.\n\n"
            "This design keeps UI code separate from model code and makes it easy to add new models or "
            "replace implementations without changing the GUI."
        )

    def _on_input_type_change(self, value):
        if value == "text":
            self.image_frame.grid_forget()
            self.text_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=6)
        elif value == "image":
            self.text_frame.grid_forget()
            self.image_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=6)

    def _browse_image(self):
        path = filedialog.askopenfilename(title="Select image",
                                          filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if path:
            self.image_path_var.set(path)

    @timed
    def _on_run(self):
        t = threading.Thread(target=self._run_models)
        t.daemon = True
        t.start()

    @logged("run_models")
    def _run_models(self):
        selected_text = self.input_type.get() == "text"
        results_text = ""
        results_image = ""
        if selected_text:
            text = self.text_entry.get("1.0", "end").strip()
            if not text:
                self._append_text_output("No text provided.\n")
            elif self.run_text_model.get():
                self._append_text_output("Running text model...\n")
                try:
                    out = self._text_model.predict(text)
                    self._append_text_output(str(out) + "\n")
                except Exception as e:
                    self._append_text_output(f"Text model error: {e}\n")
        else:
            image_path = self.image_path_var.get()
            if not image_path or not os.path.exists(image_path):
                self._append_image_output("No valid image selected.\n")
            else:
                if self.run_image_model.get():
                    self._append_image_output("Running image model...\n")
                    try:
                        out = self._image_model.predict(image_path)
                        self._append_image_output(str(out) + "\n")
                    except Exception as e:
                        self._append_image_output(f"Image model error: {e}\n")


    def _append_text_output(self, s):
        self.text_output.config(state="normal")
        self.text_output.insert("end", s)
        self.text_output.see("end")
        self.text_output.config(state="disabled")

    def _append_image_output(self, s):
        self.image_output.config(state="normal")
        self.image_output.insert("end", s)
        self.image_output.see("end")
        self.image_output.config(state="disabled")

    def _clear_outputs(self):
        self.text_output.config(state="normal")
        self.text_output.delete("1.0", "end")
        self.text_output.config(state="disabled")
        self.image_output.config(state="normal")
        self.image_output.delete("1.0", "end")
        self.image_output.config(state="disabled")

    def _show_model_info(self):
        self._fill_model_info()
        messagebox.showinfo("Model Info", "Model information updated in the 'Selected Model Info' tab.")
