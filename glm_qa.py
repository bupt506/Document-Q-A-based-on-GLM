from pypdf import PdfReader
import docx
import os
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import gradio as gr
import shutil


def get_data(root_path):
    all_content = []

    files = os.listdir(root_path)

    for file in files:
        path = os.path.join(root_path, file)
        if path.endswith(".docx"):
            doc = docx.Document(path)
            paragraphs = doc.paragraphs
            content = [i.text for i in paragraphs]

            texts = ""
            for text in content:
                if len(text) <= 1:
                    continue
                if len(texts) > 150:
                    all_content.append(texts)
                    texts = ""

                texts += text

            # all_content.append("\n".join(content))
        elif path.endswith(".pdf"):
            # content = ""
            with open(path, "rb") as f:
                pdf_reader = PdfReader(f)

                pages_info = pdf_reader.pages

                for page_info in pages_info:
                    text = page_info.extract_text()
                    # content += text
                    all_content.append(text)

        elif path.endswith(".txt"):
            with open(path, encoding="utf-8") as f:
                content = f.read()
            all_content.append(content)

    return all_content


class DFaiss:
    def __init__(self):
        # self.sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2",cache_folder="sent-model")
        self.index = faiss.IndexFlatL2(4096)

        self.text_str_list = []

    # def add_content(self,text_str_list):
    #     self.text_str_list = text_str_list
    #     text_emb = self.get_text_emb(text_str_list)
    #     self.index.add(text_emb)
    #     print("")
    #
    # def get_text_emb(self,text_str_list):
    #     text_emb = self.sentence_model.encode(text_str_list)
    #     return text_emb

    def search(self, emb):
        # text_emb = self.get_text_emb([text])

        D, I = self.index.search(emb, 3)

        if D[0][0] > 35000:
            content = ""
        else:
            content = self.text_str_list[I[0][0]]
        return content


class Dprompt:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "G:\\all_models\\huggingface\\hub\\hub\\models--THUDM--chatglm-6b-int4\\snapshots\\dac03c3ac833dab2845a569a9b7f6ac4e8c5dc9b",
            trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            "G:\\all_models\\huggingface\\hub\\hub\\models--THUDM--chatglm-6b-int4\\snapshots\\dac03c3ac833dab2845a569a9b7f6ac4e8c5dc9b",
            trust_remote_code=True).half().cuda()

        self.myfaiss = DFaiss()

        # self.load_data("datas")

        # all_content = get_data("datas")
        # self.myfaiss.add_content(all_content)

    def answer(self, text):
        emb = self.get_sentence_emb(text, is_numpy=True)
        prompt = self.myfaiss.search(emb)
        if prompt:
            prompt_content = f"请根据内容回答问题，如果内容和问题不相关，请拒绝回答，内容是：\n{prompt}\n问题是：{text}"
        else:
            prompt_content = text

        response, history = self.model.chat(self.tokenizer, prompt_content, history=[])

        return response

    def load_data(self, path):
        all_content = get_data(path)

        for content in all_content:
            self.myfaiss.text_str_list.append(content)
            emb = self.get_sentence_emb(content, is_numpy=True)
            self.myfaiss.index.add(emb)

    def get_sentence_emb(self, text, is_numpy=False):
        idx = self.tokenizer([text], return_tensors="pt")
        idx = idx["input_ids"].to("cuda")
        emb = self.model.transformer(idx, return_dict=False)[0]
        emb = emb.transpose(0, 1)
        emb = emb[:, -1]

        if is_numpy:
            emb = emb.detach().cpu().numpy()

        return emb


def load_file(files):
    global prompt_model
    print("hello")

    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.mkdir("temp")

    for file in files:
        n = os.path.basename(file.orig_name)
        p = os.path.join("temp", n)
        shutil.move(file.name, p)
        print("ok")

    prompt_model.myfaiss.index.reset()
    prompt_model.load_data("temp")

    return [[None, "文件加载成功"]]


def ans_stream(query, his):
    global prompt_model

    result = his + [[query, ""]]

    emb = prompt_model.get_sentence_emb(query, is_numpy=True)
    prompt = prompt_model.myfaiss.search(emb)
    if prompt:
        prompt_content = f"请根据内容回答问题，内容是：{prompt}，问题是：{query}"
    else:
        prompt_content = query

    for res, his in prompt_model.model.stream_chat(prompt_model.tokenizer, prompt_content, history=[]):
        result[-1] = [query, res]
        yield result


def ans(query, his):
    global prompt_model
    res = prompt_model.answer(query)

    return his + [[query, res]]


if __name__ == "__main__":
    # prompt_model = Dprompt()
    # gr.close_all()
    with gr.Blocks() as Robot:
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    [[None, "这里是聊天机器人"], [None, "模型加载ok"]],
                    show_label=False
                ).style(height=600)
                query = gr.Textbox(placeholder="输入问题，回车进行提问", show_label=False).style(container=False)

            with gr.Column(scale=1):
                file = gr.File(file_count="multiple")
                button = gr.Button("加载文件")

        button.click(load_file, inputs=file, outputs=chatbot, show_progress=True)
        query.submit(ans_stream, inputs=[query, chatbot], outputs=chatbot, show_progress=True)
        # with gr.Row():
        #     button2 = gr.Button("加载文件2")

    Robot.queue(concurrency_count=3).launch(server_name="127.0.0.1", server_port=9999, share=False)

    prompt_model = Dprompt()

    while True:
        text = input("请输入：")

        ans = prompt_model.answer(text)

        print(ans)
