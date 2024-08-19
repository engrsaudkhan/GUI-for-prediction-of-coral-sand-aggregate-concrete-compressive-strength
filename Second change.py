import tkinter as tk
from tkinter import ttk
from math import pow, sqrt
from PIL import Image, ImageTk
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import BaggingRegressor 
from sklearn.metrics import mean_squared_error

class RangeInputGUI:
    def __init__(self, master):
        self.master = master
        master.title("Graphical User Interface (GUI) for compressive strength prediction of coral sand aggregate concrete")
        master.configure(background="#FFFFFF")
        window_width = 690
        window_height = 790
        x_cord = 0  # Start from the left edge of the screen
        y_cord = 0  # Start from the top edge of the screen
        master.geometry(f"{window_width}x{window_height}+{x_cord}+{y_cord}")
        x_cord = 0
        y_cord = 0
        master.geometry(f"{window_width}x{window_height}+{x_cord}+{y_cord}")
        main_heading = tk.Label(master, text="Graphical User Interface (GUI) for: \n Compressive strength prediction of coral sand aggregate concrete",
                                bg="#C41E3A", fg="#FFFFFF", font=("Helvetica", 16, "bold"), pady=10)
        main_heading.pack(side=tk.TOP, fill=tk.X)
        self.content_frame = tk.Frame(master, bg="#E8E8E8")
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=50, pady=50, anchor=tk.CENTER)
        self.canvas = tk.Canvas(self.content_frame, bg="#E8E8E8")
        self.scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#FFFFFF")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.input_frame = tk.Frame(self.scrollable_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
        self.input_frame.pack(side=tk.TOP, fill="both", padx=10, pady=10, expand=False)
        heading = tk.Label(self.input_frame, text="Input Parameters", bg="#FFFFFF", fg="black", font=("Helvetica", 16, "bold"), padx=10, pady=10)
        heading.grid(row=0, column=0, columnspan=3, pady=10)
        self.output_frame = tk.Frame(self.scrollable_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
        self.output_frame.pack(side=tk.TOP, fill="both", padx=20, pady=20)
        heading = tk.Label(self.output_frame, text="Output Parameters", bg="#FFFFFF", fg="black", font=("Helvetica", 16, "bold"), pady=10)
        heading.grid(row=0, column=0, columnspan=2, pady=10)
        self.create_entry("Pressure:", 90.0, 1)
        self.create_entry("Particle Size:", 7.5, 3)
        self.create_entry("Particle Shape:", 4, 5)
        self.create_entry("CSA Content:", 60, 7)
        self.create_entry("Immersion Period:", 2, 9)
        heading.grid(row=0, column=0, columnspan=3, pady=10)
        self.calculate_button_mepx = tk.Button(self.output_frame, text="Multi Expression Programming (MEP)", command=self.calculate_y_mepx,
                                            bg="#743089", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.calculate_button_mepx.grid(row=2, column=0, pady=10, padx=10)
        self.gep_output_text_mepx = tk.Text(self.output_frame, height=1.5, width=20)
        self.gep_output_text_mepx.grid(row=2, column=1, padx=10, pady=10, sticky="sw")
        self.xgboost_button_XGBoost = tk.Button(self.output_frame, text="Bagging Regressor (BR)", command=self.calculate_Bagging_regressor,
                                        bg="#743089", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.xgboost_button_XGBoost.grid(row=3, column=0, pady=10, padx=10, sticky="sw")
        self.br_output_text_baggingregressor = tk.Text(self.output_frame, height=1.5, width=20)
        self.br_output_text_baggingregressor.grid(row=3, column=1, padx=10, pady=10)
        self.predict_button = tk.Button(self.output_frame, text="Predict", command=self.predict,
                                        bg="green", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.predict_button.grid(row=4, column=0, pady=10, padx=10)

        self.clear_button = tk.Button(self.output_frame, text="Clear", command=self.clear_fields,
                                      bg="red", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.clear_button.grid(row=4, column=1, pady=10, padx=10)

        developer_info = tk.Label(text="This GUI is developed by:\nMuhammad Saud Khan (khans28@myumanitoba.ca), University of Manitoba, Canada\n",
                                  bg="light blue", fg="purple", font=("Helvetica", 11, "bold"), pady=10)
        developer_info.pack()
    def create_entry(self, text, default_val, row):
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_columnconfigure(1, weight=1)
        label = tk.Label(self.input_frame, text=text, font=("Helvetica", 12, "bold"), fg="white", bg="#8968CD", anchor="w")
        label.grid(row=row*2, column=0, padx=10, pady=5, sticky="ew")
        entry = tk.Entry(self.input_frame, font=("Helvetica", 12, "bold italic"), fg="#4B0082", bg="#FFF0F5", width=10, bd=2, relief=tk.GROOVE)
        entry.insert(0, f"{default_val:.2f}")
        entry.grid(row=row*2, column=1, padx=10, pady=5, sticky="se")
        setattr(self, f'entry_{row}', entry)
    def get_entry_values(self):
        try:
            d1 = float(self.entry_1.get())
            d2 = float(self.entry_3.get())
            d3 = float(self.entry_5.get())
            d4 = float(self.entry_7.get())
            d5 = float(self.entry_9.get())
            return d1, d2, d3, d4, d5
        except ValueError as ve:
            print("Error: Invalid data format")
            print("Error:", ve)
            return None
    def calculate_y_mepx(self):
        values = self.get_entry_values()
        if values is None:
            self.xgboost_output_text_mepx.delete(1.0, tk.END)
            self.xgboost_output_text_mepx.insert(tk.END, "Error: Invalid input values")
            return
        d1, d2, d3, d4, d5 = values
        prg = [0] * 100
        prg[0] = d1
        prg[1] = math.sqrt(prg[0])
        prg[2] = prg[1] + prg[1]
        prg[3] = prg[2] / prg[0]
        prg[4] = prg[3] * prg[2]
        prg[5] = prg[0] + prg[4]
        prg[6] = prg[4] + prg[2]
        prg[7] = d5
        prg[8] = prg[7] / prg[1]
        prg[9] = prg[3] + prg[6]
        prg[10] = prg[2] / prg[9]
        prg[11] = prg[1] - prg[4]
        prg[12] = prg[6] + prg[9]
        prg[13] = prg[9] * prg[4]
        prg[14] = prg[12] * prg[3]
        prg[15] = prg[11] * prg[3]
        prg[16] = d4
        prg[17] = d2
        prg[18] = prg[12] + prg[14]
        prg[19] = prg[17] + prg[17]
        prg[20] = prg[11] * prg[19]
        prg[21] = prg[4] - prg[10]
        prg[22] = prg[7] - prg[16]
        prg[23] = prg[22] - prg[17]
        prg[24] = d3
        prg[25] = prg[17] - prg[24]
        prg[26] = d4
        prg[27] = d1
        prg[28] = prg[8] - prg[3]
        prg[29] = prg[14] / prg[3]
        prg[30] = prg[19] + prg[25]
        prg[31] = d4
        prg[32] = math.sqrt(prg[12])
        prg[33] = d1
        prg[34] = d3
        prg[35] = d5
        prg[36] = prg[5] / prg[20]
        prg[37] = prg[30] * prg[17]
        prg[38] = prg[19] / prg[21]
        prg[39] = d3
        prg[40] = d4
        prg[41] = d1
        prg[42] = d1
        prg[43] = d2
        prg[44] = prg[26] * prg[38]
        prg[45] = prg[18] - prg[7]
        prg[46] = d5
        prg[47] = d5
        prg[48] = prg[37] + prg[23]
        prg[49] = prg[38] / prg[23]
        prg[50] = d3
        prg[51] = d3
        prg[52] = d3
        prg[53] = math.sqrt(prg[51])
        prg[54] = d3
        prg[55] = d1
        prg[56] = prg[24] - prg[36]
        prg[57] = d3
        prg[58] = prg[3] - prg[43]
        prg[59] = d2
        prg[60] = d3
        prg[61] = prg[19] + prg[15]
        prg[62] = d4
        prg[63] = d3
        prg[64] = prg[11] + prg[37]
        prg[65] = d5
        prg[66] = d1
        prg[67] = d1
        prg[68] = d1
        prg[69] = prg[49] / prg[68]
        prg[70] = prg[56] / prg[48]
        prg[71] = d4
        prg[72] = prg[3] / prg[11]
        prg[73] = d3
        prg[74] = d4
        prg[75] = d1
        prg[76] = prg[48] * prg[27]
        prg[77] = d2
        prg[78] = prg[9] - prg[47]
        prg[79] = d1
        prg[80] = d2
        prg[81] = prg[79] * prg[17]
        prg[82] = d3
        prg[83] = d5
        prg[84] = d1
        prg[85] = prg[28] * prg[7]
        prg[86] = d3
        prg[87] = d5
        prg[88] = prg[45] - prg[36]
        prg[89] = prg[27] + prg[62]
        prg[90] = d4
        prg[91] = d5
        prg[92] = d3
        prg[93] = d1
        prg[94] = prg[49] + prg[88]
        prg[95] = d2
        prg[96] = prg[94] - prg[85]
        prg[97] = prg[71] * prg[8]
        prg[98] = prg[70] + prg[96]
        prg[99] = d3
        outputs = [0]   
        outputs[0] = prg[98]       
        self.gep_output_text_mepx.delete(1.0, tk.END)
        self.gep_output_text_mepx.insert(tk.END, f"{outputs[0]:.4f}")
        self.gep_output_text_mepx.config(font=("Helvetica", 12, "bold"), foreground="#E30B5D")
        return outputs
    def calculate_Bagging_regressor(self):
        values = self.get_entry_values()
        if values is None:
            self.br_output_text_baggingregressor.delete(1.0, tk.END)
            self.br_output_text_baggingregressor.insert(tk.END, "Error: Invalid input values")
            return
        d1, d2, d3, d4, d5 = values
        try:
            print("Starting XGBoost density calculation...")
            base_dir = r"C:\Users\MUHAMMAD SAUD KHAN\Documents\Waleed\software-and-prediction-main\Coral V2"
            filename = r"Raw Data.xlsx"
            df = pd.read_excel(f"{base_dir}/{filename}")
            print("Excel file loaded successfully.")
            print(f"Excel file head:\n{df.head()}")
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1].values.ravel()  # Convert y to a 1D array
            print(f"Data split into inputs and outputs:\nx shape: {x.shape}, y shape: {y.shape}")
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=500)
            model = BaggingRegressor(n_estimators=60, bootstrap=True)
            model.fit(x_train, y_train)
            input_data = np.array([d1, d2, d3, d4, d5]).reshape(1, -1)
            y_pred = model.predict(input_data)
            self.br_output_text_baggingregressor.delete(1.0, tk.END)
            self.br_output_text_baggingregressor.insert(tk.END, f"{y_pred[0]:.4f}")
            self.br_output_text_baggingregressor.config(font=("Helvetica", 12, "bold"), foreground="#E30B5D")

        except Exception as e:
            print(f"An error occurred: {e}")
            self.br_output_text_baggingregressor.delete(1.0, tk.END)
            self.br_output_text_baggingregressor.insert(tk.END, f"Error: {str(e)}")
    def predict(self):
        self.calculate_y_mepx()
        self.calculate_Bagging_regressor()
    def clear_fields(self):
        for i in range(1, 10, 2):
            entry = getattr(self, f'entry_{i}', None)
            if entry:
                entry.delete(0, tk.END)
        self.gep_output_text_mepx.delete(1.0, tk.END)
        self.br_output_text_baggingregressor.delete(1.0, tk.END)
if __name__ == "__main__":
    root = tk.Tk()
    gui = RangeInputGUI(root)
    root.mainloop()