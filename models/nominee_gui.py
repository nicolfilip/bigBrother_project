import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import pandas as pd
from models.training_model_Israel import run_model_on_nominees

df = pd.read_csv("../data/big_brother_israel_new_new.csv")
df["week eliminated"] = pd.to_numeric(df["week eliminated"], errors="coerce")
alive_contestants = df[df["week eliminated"].isna()]["שם מלא"].astype(str).tolist()

image_folder = os.path.join(os.path.dirname(__file__), "..", "images")
image_files = os.listdir(image_folder)

valid_contestants = []
for name in alive_contestants:
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        filename = f"{name}{ext}"
        if filename in image_files:
            valid_contestants.append((name, filename))
            break

# חלון ראשי
root = tk.Tk()
root.title("בחירת מועמדים להדחה")
root.configure(bg="#f5f5f5")

# כותרת
title_label = tk.Label(root, text="בחר את המועמדים להדחה", font=("Arial", 18, "bold"), bg="#f5f5f5")
title_label.pack(pady=(20, 10))

frame = tk.Frame(root, bg="#f5f5f5")
frame.pack()

columns = 4
row = 0
col = 0
buttons = {}
selected_nominees = set()

def toggle_selection(name, button):
    if name in selected_nominees:
        selected_nominees.remove(name)
        button.config(relief="raised", bg="SystemButtonFace")
    else:
        selected_nominees.add(name)
        button.config(relief="sunken", bg="#d0e0ff")

def confirm_selection():
    if not selected_nominees:
        messagebox.showwarning("שגיאה", "יש לבחור לפחות מועמד אחד.")
        return

    # הפעלת המודל עם בקשה להחזרת טופ 3
    result = run_model_on_nominees(list(selected_nominees), return_top=True)

    if result:
        eliminated = result["eliminated"]
        top_risk = result["top_risk"]  # רשימה של (שם, ציון)

        result_win = tk.Toplevel(root)
        result_win.title("תוצאה")
        result_win.configure(bg="white")

        tk.Label(result_win, text="המודח החזוי הבא:", font=("Arial", 16, "bold"), bg="white").pack(pady=10)

        # תמונת המודח
        try:
            for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                image_path = os.path.join(image_folder, f"{eliminated}{ext}")
                if os.path.exists(image_path):
                    img = Image.open(image_path).resize((150, 150))
                    photo = ImageTk.PhotoImage(img)
                    img_label = tk.Label(result_win, image=photo, bg="white")
                    img_label.image = photo
                    img_label.pack(pady=5)
                    break
        except Exception as e:
            print(f"שגיאה בטעינת תמונת מודח: {e}")

        tk.Label(result_win, text=eliminated, font=("Arial", 14), bg="white").pack(pady=5)

        # טופ 3
        tk.Label(result_win, text="\ntop 3 at risk:", font=("Arial", 12, "bold"), bg="white").pack(pady=(15, 5))

        for name, score in top_risk:
            line = f"{name} (grade: {score:.3f})"
            tk.Label(result_win, text=line, font=("Arial", 11), bg="white").pack()


# תמונות + כפתורים
for name, img_file in valid_contestants:
    try:
        path = os.path.join(image_folder, img_file)
        img = Image.open(path).resize((100, 100))
        photo = ImageTk.PhotoImage(img)

        container = tk.Frame(frame, bg="#f5f5f5", padx=10, pady=10)
        container.grid(row=row, column=col)

        btn = tk.Button(container, image=photo, text=name, compound="top",
                        wraplength=100, justify="center",
                        font=("Arial", 10),
                        command=lambda n=name: toggle_selection(n, buttons[n]))
        btn.image = photo
        btn.pack()
        buttons[name] = btn

        col += 1
        if col == columns:
            col = 0
            row += 1

    except Exception as e:
        print(f"שגיאה בטעינת {name}: {e}")

# כפתור חישוב
submit_btn = tk.Button(root, text="חשב הדחה", font=("Arial", 14, "bold"), bg="#4CAF50", fg="white",
                       activebackground="#45a049", padx=20, pady=10, command=confirm_selection)
submit_btn.pack(pady=20)

root.mainloop()
