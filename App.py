import gradio as gr
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import subprocess
import pandas as pd

def authenticate(username, password):
    df = pd.read_csv("user_database.csv")
    role = " "
    print("hi")
    for index, row in df.iterrows():
        if row["Username"] == username:
            if row["Password"] == password:
                if username[:5] == "ADMIN":
                    role = "ADMIN"
                else:
                    role = "USER"
                break
            else:
                role = "Not Defined" 
        else: 
            role = "Not Defined"  
    
    if role == "USER":
        gr.Info("User Signing In...")
        user_app = subprocess.run(['python3', 'User_UI.py'])
        # return user_app
    elif role == "ADMIN":
        gr.Info("Admin Signing In...")
        admin_app = subprocess.run(['python3', 'Admin_UI.py'])
        # return admin_app
    else:
        gr.Info("Username or Password is incorrect.")
        #return False

def sign_up_fn(username, password1, password2):
    df = pd.read_csv("user_database.csv")
    if (df["Username"] != username).any().any(): 
        if password1 == password2:
            new_row = pd.DataFrame({'Username': username, 'Password': password1}, index=[0])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv("user_database.csv", index=False)
            gr.Info("Your account has been successfully signed up!")
        else:
            gr.Info("Passwords does not match. Change Password!!!")
    else:
        gr.Info("Username does not exist. Change Username!!!")

with gr.Blocks() as demo:
    #gr.Image("../Documentation/Context Diagram.png", scale=2)
    #gr(title="Your Interface Title")
    gr.Markdown("""
                <center> 
                <span style='font-size: 50px; font-weight: Bold; font-family: "Graduate", serif'>                
                PatentProSum Sign-in
                </span>
                </center>
                """)
    with gr.Group():
        username = gr.Textbox(label="Username", max_lines = 1)
        password = gr.Textbox(label="Password", max_lines = 1, type = "password")

    with gr.Row():
        login_btn = gr.Button(value="Login")
    
    login_btn.click(authenticate, inputs=[username, password])

    with gr.Accordion("Don't have an account?", open=False):
        gr.Markdown("""
                <center> 
                <span style='font-size: 30px; font-weight: Bold; font-family: "Graduate", serif'>
                Sign Up
                </span>
                </center>
                """)
        with gr.Group():
            username = gr.Textbox(label="Username", max_lines = 1)
            password = gr.Textbox(label="Password", max_lines = 1, type = "password")
            confirm_password = gr.Textbox(label="Confim Password", max_lines = 1, type = "password")

        with gr.Row():
            sign_up_btn = gr.Button(value="Sign Up")

        sign_up_btn.click(sign_up_fn, inputs=[username, password, confirm_password])
   
# demo.launch(share = True, auth=authenticate)
demo.launch(share = True)

