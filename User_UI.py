import gradio as gr
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from PyPDF2 import PdfReader, PdfWriter
from fpdf import FPDF




def preprocess_text(text):
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
    return tokenized_sentences

def sentence_similarity(sentence1, sentence2):
    stop_words = set(stopwords.words('english'))
    filtered_sentence1 = [w for w in sentence1 if w not in stop_words]
    filtered_sentence2 = [w for w in sentence2 if w not in stop_words]
    all_words = list(set(filtered_sentence1 + filtered_sentence2))
    vector1 = [filtered_sentence1.count(word) for word in all_words]
    vector2 = [filtered_sentence2.count(word) for word in all_words]
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
    return similarity_matrix

def apply_lexrank(similarity_matrix, damping=0.85, threshold=0.2, max_iter=100):
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph, alpha=damping, tol=threshold, max_iter=max_iter)
    return scores

def get_top_sentences(sentences, scores):
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
    top_sentences = [sentence for score, sentence in ranked_sentences]
    return top_sentences

def extract_important_sentences(text):
    preprocessed_sentences = preprocess_text(text)
    similarity_matrix = build_similarity_matrix(preprocessed_sentences)
    scores = apply_lexrank(similarity_matrix)
    top_sentences = get_top_sentences(preprocessed_sentences, scores)
    paragraph = ' '.join([' '.join(sentence) for sentence in top_sentences])
    return paragraph

def summarize(text, max_tokens, model_type):

    if model_type == 'Generalized':
        peft_model_path = "../Adapter_Tuning/LoRA+BART/M2L_LR_S2_EXT4_EXP17_model"
    elif model_type == 'Specialized (Textiles and Paper)':
        peft_model_path = "../Adapter_Tuning/LoRA+BART/M2L_LR_S2_EXT4_EXP16_model"

    config = PeftConfig.from_pretrained(peft_model_path)

    # load base LLM model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_path)

    sorted_text = extract_important_sentences(text)

    input_ids = tokenizer(sorted_text, return_tensors="pt", truncation=True).input_ids
    # with torch.inference_mode():
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_tokens, do_sample=True, top_p=0.9)
    summary = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    return summary

def document_to_text(doc, max_tokens, model_type):

    reader = PdfReader(doc) 
    text = ""
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        text += page.extract_text() 
    #print(text) 
    summary = summarize(text, max_tokens, model_type)
    # pdf_writer = PdfWriter()
    
    # pdf_writer.add_blank_page(width = 800, height = 800)
    
    # annotation = PyPDF2.generic.AnnotationBuilder.free_text(
    #     summary,
    #     rect=(50, 550, 200, 650),
    #     font="Arial",
    #     bold=True,
    #     italic=True,
    #     font_size="20pt",
    #     font_color="00ff00",
    #     border_color="0000ff",
    #     background_color="cdcdcd",
    # )
    # pdf_writer.add_annotation(page_number=0, annotation=annotation)
    
    # with open('summary.pdf', "wb") as f:
    #     pdf_writer.write(f)

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size = 15)

    pdf.multi_cell(200, 10, txt = summary, align = 'C')

    pdf.output("summary.pdf")   

    return 'summary.pdf'

     
with gr.Blocks() as demo:
    # gr.Image("../Documentation/Context Diagram.png", scale=2)
    # gr(title="Your Interface Title")
    gr.Markdown("""
                <center> 
                <span style='font-size: 50px; font-weight: Bold; font-family: "Graduate", serif'>
                PatentProSum 
                </span>
                </center>
                """)
    with gr.Row():
        slider = gr.Slider(128, 512, value=256, step=2, label="Maximum Summary Length", info="Limitation for summary length in words", scale=1)  
        
        with gr.Column():
            select_model = gr.Radio(["Generalized", "Specialized (Textiles and Paper)"], value= "Generalized", label="Select Model", info="Model to perform summarization", scale=1)

    with gr.Row():
        # with gr.Column():
        description = gr.Textbox(label="Patent Document Text")
            
        # with gr.Column():
        summary = gr.Textbox(label="Summary Text")
    
    summarize_btn = gr.Button(value="Summarize Text", size = 'sm')

    with gr.Row():
        with gr.Column():
            patent_doc = gr.File(label="Original Patent Document", scale = 1)
        
        with gr.Column():
            summary_doc = gr.File(label="Summarized Patent Document", scale = 1)

    gr.Markdown("""
                <div style="text-align: center;">
                    <a style="text-decoration: none; color: white;" href="https://github.com/NevJay/PatentProSum/blob/main/README.md">About Us</a>
                </div>
                """)
    
    patent_doc.upload(document_to_text, inputs = [patent_doc, slider, select_model], outputs=summary_doc)
    summarize_btn.click(summarize, inputs=[description, slider, select_model], outputs=summary)


demo.launch(inbrowser=True, share=True)
