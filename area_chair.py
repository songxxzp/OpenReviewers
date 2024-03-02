import time
import json
import copy
import os
import requests as rq
import random
import os
import gradio as gr
import argparse

import colorama
from colorama import Fore
from transformers import AutoTokenizer
from rich import print
from termcolor import colored
from concurrent.futures import ThreadPoolExecutor

from utils import *


colorama.init()

chapters = {
    "Abstract": ["abstract"],
    "Introduction": ["introduction", "intro"],
    "Related Work": [
        "related work", "literature review", "background", 
        "prior art", "state of the art"
    ],
    "Methodology": [
        "methodology", "methods", "approach", 
        "algorithm", "framework", "model"
    ],
    "Experiments": [
        "experiments", "experimental setup", "experimentation", 
        "experimental study", "evaluation", "validation"
    ],
    "Results": [
        "results", "findings", "outcomes", 
        "data analysis"
    ],
    "Discussion": [
        "discussion", "analysis", "discussion and analysis", 
        "interpretation", "insights"
    ],
    "Conclusion": [
        "conclusion", "concluding remarks", "summary", 
        "final thoughts", "wrap-up"
    ],
    "Future Work": [
        "future work", "directions for future research", 
        "prospects"
    ],
    "Acknowledgments": [
        "acknowledgments", "acknowledgements", "thanks"
    ],
    "References": [
        "references", "bibliography", "works cited"
    ],
    "Appendices": [
        "appendices", "appendix", "supplementary material", 
        "additional information"
    ]
}
sections = ['ABSTRACT', 'INTRODUCTION', 'EXPERIMENTS', 'RESULTS', 'CONCLUSION']
max_length = {
    "ABSTRACT": 1024,
    "INTRODUCTION": 4096,
    "EXPERIMENTS": 3072,
    "RESULTS": 3072,
    "CONCLUSION": 2048, 
}
assessments = ['summaries', 'strengths', 'weaknesses', 'questions']


tokenizer_path = "vicuna"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

def add_inference_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group()
    group.add_argument("--re-path", type=str, default='vicuna-7b-v1.5-re-awq')
    group.add_argument("--ac-path", type=str, default='vicuna-7b-v1.5-ac-awq')
    group.add_argument("--re-port", type=str, default='39174')
    group.add_argument("--ac-port", type=str, default='8001')
    group.add_argument("--server-port", type=str, default='10730')
    return parser


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_inference_args(parser)

    args, unknown = parser.parse_known_args()
    
    return args


def generate(input_str: str, model_path='vicuna-7b-v1.5-re-awq', port=39174, max_tokens=1024, temperature=0.7, index=0):
    if isinstance(port, str) and ',' in port:
        ports = port.split(',')
        port = ports[index % len(ports)]
    else:
        ports = [port]

    cnt = 0
    text = ""
    while text == "":
        try:
            query = {
                "model": model_path,
                "prompt": input_str,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            api_link = f"http://127.0.0.1:{port}/v1/completions"
            output = rq.post(api_link, json.dumps(query), headers={'Content-Type': 'application/json'})
            text: str = json.loads(output.text)["choices"][0]['text']
        except:
            print(f"Retry: {cnt}, {api_link}")
            cnt += 1
            index -= 1
            port = ports[index % len(ports)]
            print(f"Switch to backup port: {port}")
            time.sleep(0.2)
        if cnt > 10:
            print("Failed too many times.")
            break
    return text.strip(' ')


def build_input(elements):
    global tokenizer
    start_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:"
    input_str = start_message
    input_ids = tokenizer.encode(start_message, add_special_tokens=True)
    loss_mask = [0] * len(input_ids)

    last_turn = 0

    for element in elements:
        io_mark, string = element
        if io_mark != last_turn:
            if io_mark == 1:
                assistant_token = tokenizer.encode("ASSISTANT:", add_special_tokens=False)
                input_ids += assistant_token
                input_str += "ASSISTANT:"
                loss_mask += [0] * len(assistant_token)
                # string = "ASSISTANT: " + string
            elif io_mark == 0:
                string = "USER: " + string
            else:
                raise AssertionError(f"Unkown io_mark {io_mark}")
        tokens = tokenizer.encode(string, add_special_tokens=False)
        if io_mark == 0:
            loss_mask += [0] * len(tokens)
        elif io_mark == 1:
            string += tokenizer.eos_token
            tokens.append(tokenizer.eos_token_id)
            loss_mask += [1] * len(tokens)
        input_str += string
        input_ids += tokens
        last_turn = io_mark
    
    assistant_token = tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    input_ids += assistant_token
    input_str += "ASSISTANT:"
    loss_mask += [0] * len(assistant_token)

    print(Fore.CYAN)
    return input_ids, loss_mask, input_str


def ends_with_punctuation(s):
    punctuation = ".!?;,()"  # 定义标点符号
    if s and s[-1] in punctuation:
        return True
    else:
        return False


def remove_duplicate(review: str) -> str:
    if review.count('\n') < 2:
        return review
    points = [s.strip(' ') for s in review.split('\n')]
    filtered_points = []
    for point in points:
        if point not in filtered_points or len(point) < 1:
            filtered_points.append(point)

    if len(points) > 8 and (not ends_with_punctuation(filtered_points[-1])) and ends_with_punctuation(filtered_points[-2]):
        filtered_points = filtered_points[:-1]

    filtered_review = '\n'.join(filtered_points)

    if filtered_points.count('\n') < points.count('\n'):
        print(Fore.GREEN)
        print(f"{len(points)} -> {len(filtered_points)}")
        print('================================')
        print(f"{review}")
        print('--------------------------------')
        print(Fore.CYAN, end=None)
        print(f"{filtered_review}")
        print('================================')
        print(Fore.CYAN)
    return filtered_review


class Reviewer:
    def __init__(self):
        self.seeds = []
        self.logs_dir = "logs"

    def load_pdf(self, seed, title, keywords, file_name):
        seed = seed if seed else random.randint(0, 65535)
        self.seeds.append((seed, time.time()))

        paper = ""
        split_text = {}
        
        if os.path.exists(file_name):
            extracted_text = extract_chapter(file_name)
            
            if len(extracted_text.split('\n')) > 2:
                
                extracted_text = extracted_text.replace("I NTRODUCTION", "INTRODUCTION")
                extracted_text = extracted_text.replace("C ONCLUSION", "CONCLUSION")
            split_text: Dict = {}
            for k, v in filter_iclr_pdf(extracted_text, chapters).items():
                if k.upper() in sections:
                    tokens = tokenizer.encode(v, add_special_tokens=False)
                    if len(tokens) > max_length[k.upper()]:
                        cut_tokens = tokens[:max_length[k.upper()]]
                        split_text[k.upper()] = tokenizer.decode(cut_tokens, skip_special_tokens=False)
                    else:
                        split_text[k.upper()] = v
            

        for section_name, content in split_text.items():
            paper += f"{section_name}\n\n{content}\n\n"

        try:
            os.makedirs(self.logs_dir, exist_ok=True)
            with open(os.path.join(self.logs_dir, 'paper.jsonl'), 'a', encoding='utf-8') as f:
                f.write(
                    json.dumps(
                        {
                            'file': file_name,
                            'title': title,
                            'keywords': keywords,
                            'seed': seed
                        }
                    ) + '\n'
                )
        except Exception as e:
            print(e)
        return seed, title, keywords, paper, split_text

    def generate_notes(self, seed, title, keywords: List, split_text: Dict, index=0):
        global args
        Seed = random.randint(0, 65535)
        Title = title
        Keywords = keywords

        Seed = seed
        Title = title
        Keywords = keywords

        elements = [
            (0, f"You are the No.{Seed} reviewer of openreivew. You are reviewing the paper titled {Title}. The keywords are {Keywords}. You will read this paper and write a review for it.\n\n")
        ]

        note_dict = {}
        assessment_dict = {}
        for assessment in assessments:
            assessment_dict[assessment] = []

        for section in sections:
            elements.append((0, f"Read the {section} of this paper, and write down your reading notes, such as summaries, questions, weaknesses, or strengths:\n\n"))
            elements.append((0, split_text[section] + '\n\n'))
            elements.append((0, f"Now write down your note for the {section} of this paper, such as summaries, questions, weaknesses, or strengths:\n\n"))
            input_ids, _, input_str = build_input(elements)
            local_text = remove_duplicate(generate(input_str, model_path=args.re_path, port=args.re_port, index=index))
            elements.append((1, local_text))
            print(Fore.BLUE + local_text)
            note_dict[section] = local_text

            for assessment in assessments:
                find_assessments = [
                    f"{assessment.upper()}:\n",
                    f"{assessment.upper()}\n",
                    f"{assessment.lower().capitalize()}:\n",
                    f"{assessment.lower().capitalize()}\n",
                    f"{assessment.lower()}:\n",
                    f"{assessment.lower()}\n",
                ]
                if assessment == 'summaries':
                    find_assessments.extend([
                        f"{'summary'.upper()}:\n",
                        f"{'summary'.upper()}\n",
                        f"{'summary'.lower().capitalize()}:\n",
                        f"{'summary'.lower().capitalize()}\n",
                        f"{'summary'.lower()}:\n",
                        f"{'summary'.lower()}\n",
                    ])

                flag = False
                for find_assessment in find_assessments:
                    if find_assessment in local_text:
                        flag = True
                        start_index = local_text.find(find_assessment) + len(find_assessment)
                        break
                if flag:
                    end_index = local_text.find("\n\n", start_index)
                    splited_assessment = local_text[start_index : end_index]
                    if splited_assessment not in assessment_dict[assessment]:
                        assessment_dict[assessment].append(splited_assessment)

        review_dict = {}

        for assessment in assessments:
            elements.append((0, f"You will read through your note about {assessment.upper()}, then write down the {assessment.upper()} for the paper. Your note about {assessment.upper()}:\n\n"))
            used_splited_assessment = []
            for splited_assessment in assessment_dict[assessment]:
                if splited_assessment not in used_splited_assessment:
                    elements.append((0, f"{splited_assessment}\n"))
                    used_splited_assessment.append(splited_assessment)
            elements.append((0, f"\n"))
            elements.append((0, f"Your final {assessment.upper()}:\n\n"))
            input_ids, _, input_str = build_input(elements)
            local_text = remove_duplicate(generate(input_str, max_tokens=512, model_path=args.re_path, port=args.re_port, index=index))
            elements.append((1, local_text))
            print(Fore.BLUE + local_text)
            review_dict[assessment.upper()] = local_text

        elements.append((0, f"Now give this article a score from multiple dimensions:\n\n"))
        for metric in ["soundness", "presentation", "contribution", "rating", "confidence"]:
            elements.append((0, f"{metric}: "))
            input_ids, _, input_str = build_input(elements)
            local_text = generate(input_str, model_path=args.re_path, port=args.re_port, temperature=0.01, index=index)
            elements.append((1, local_text))
            print(Fore.BLUE + local_text)
            review_dict[metric] = local_text

        try:
            os.makedirs(self.logs_dir, exist_ok=True)
            with open(os.path.join(self.logs_dir, 'review.jsonl'), 'a', encoding='utf-8') as f:
                f.write(
                    json.dumps(
                        {
                            'elements': elements,
                            'review_dict': review_dict
                        }
                    ) + '\n'
                )
        except Exception as e:
            print(e)

        return review_dict
    
    def generate_decision(self, title, keywords, split_text, review_dict_1, review_dict_2, review_dict_3, review_dict_4, index=0):
        content_str_list = []
        for i,comment in enumerate([review_dict_1, review_dict_2, review_dict_3, review_dict_4]):
            content_str = f"Reviewer {i + 1}:\n"
            for key, value in comment.items():
                if key == "responses":
                    continue
                content_str += f"{key.lower()}:\n{value}\n"
            content_str_list.append(content_str)
        final_str = "\n".join(content_str_list)

        input_prompt = "Here are some reviews of the paper '" + str(title) + "'. And this is the abstract of the paper: " + str(split_text["ABSTRACT"]) + ".\n\nHere are some official reviews fellow researchers.\n " + final_str[:12000] + "\n\nNow, as an Area Chair, you should directly give the FINAL DECISION and overall explanation for that."
        
        final_input_prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {input_prompt} ASSISTANT:"
        
        print(final_input_prompt)
        
        result = generate(final_input_prompt, model_path=args.ac_path, port=args.ac_port, index=index)
        print(result)

        result_dict = {}

        try:
            decision, comments = result.strip().split('. ', maxsplit=1)
        except:
            decision = ''
            comments = result.strip()
        result_dict['Area Chair'] = decision
        result_dict['Review'] = comments
        
        return result_dict



description = '''        
<h1>📝 Openreviewers: Multi Agent Academic Review Simulation System</h1>
<p style="font-size: 16px;letter-spacing: 1.5px;">|<a href="https://github.com/songxxzp/Openreviewers">GitHub</a>|<a href="https://github.com/songxxzp/Openreviewers">Dataset</a>|<a href="http://116.204.104.227:5173/">ArxivReviewers</a>|</p><p style="font-size: 16px;letter-spacing: 1.5px;">|<a href="http://116.204.104.227:10729/">OpenReviewer</a>|<a href="http://116.204.104.227:10730/">OpenReviewers</a>|</p>
<h2>📜 Rules</h2>
<ul>
    <li style="font-size: 16px;">Please upload your paper in PDF format for review.</li>
    <li style="font-size: 16px;">The title of your paper will be filled in automatically.</li>
    <li style="font-size: 16px;">Our current PDF parser only supports specific types of papers. There may be parsing errors that could affect the final results.</li>
    <li style="font-size: 16px;">The results are for reference only.</li>
</ul>
<h2>👇 Upload your paper now! </h2>
'''



def build_review_text(review_dict: Dict):
    review_text = f'''Summaries:\n{review_dict['SUMMARIES']}\nStrengths:\n{review_dict['STRENGTHS']}\nWeaknesses:\n{review_dict['WEAKNESSES']}\nQuestions:\n{review_dict['QUESTIONS']}\n'''

    return review_text


def generate_4_notes(seed, title, keywords: List, split_text: Dict, debug=False):
    reviews = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(1 if debug else 4):
            seed = random.randint(0, 65535)
            futures.append(executor.submit(reviewer.generate_notes, seed, title, keywords, split_text, index=i))

        for future in futures:
            review = future.result()
            reviews.append(review)

    if debug:
        review_dict_1 = review_dict_2 = review_dict_3 = review_dict_4 = copy.deepcopy(reviews[0])
    else:
        review_dict_1, review_dict_2, review_dict_3, review_dict_4 = copy.deepcopy(reviews)
    
    for i in range(4):
        for assessment in assessments:
            reviews[i].pop(assessment.upper())
    review_dict_10, review_dict_20, review_dict_30, review_dict_40 = reviews

    return build_review_text(review_dict_1), review_dict_10, review_dict_1, build_review_text(review_dict_2), review_dict_20, review_dict_2, build_review_text(review_dict_3), review_dict_30, review_dict_3, build_review_text(review_dict_4), review_dict_40, review_dict_4


global args
args = get_args()

with gr.Blocks() as demo:
    reviewer = Reviewer()
    gr.HTML(description)
    with gr.Row():
        seed_output_box = gr.Textbox(label="Seed(Optional)", visible=False)
        file_input_box = gr.File(label="pdf", value='pdfs/AttentionIsAllYouNeed.pdf')
        with gr.Column():

            title_box = gr.Textbox(label="Title", info='write the title of your paper', value="Attention Is All You Need")
            keywords_box = gr.Textbox(label="Keywords(Optional)", value="Language Modeling, Deep Learning", info='write the keywords of your paper')
    upload_pdf_btn = gr.Button("Upload")
    section_output_box = gr.Textbox(label="Paper", visible=False)
    split_text_box = gr.JSON(label="Splitted text", render=True, visible=False)
    
    review_btn = gr.Button("Generate Reviews", visible=True)
    with gr.Tab("Reviewer 1"):      
        with gr.Column():
            review_box_10 = gr.JSON(label="reviewer 1", render=True)
            review_box_1 = gr.Textbox(label="review", render=True, lines=24)
            review_box1 = gr.JSON(label="reviewer 1", render=True, visible=False)
    with gr.Tab("Reviewer 2"):
        with gr.Column():
            review_box_20 = gr.JSON(label="reviewer 2", render=True)
            review_box_2 = gr.Textbox(label="review", render=True, lines=24)
            review_box2 = gr.JSON(label="reviewer 2", render=True, visible=False)
    with gr.Tab("Reviewer 3"):
        with gr.Column():
            review_box_30 = gr.JSON(label="reviewer 3", render=True)
            review_box_3 = gr.Textbox(label="review", render=True, lines=24)
            review_box3 = gr.JSON(label="reviewer 3", render=True, visible=False)
    with gr.Tab("Reviewer 4"):
        with gr.Column():
            review_box_40 = gr.JSON(label="reviewer 4", render=True)
            review_box_4 = gr.Textbox(label="review", render=True, lines=24)
            review_box4 = gr.JSON(label="reviewer 4", render=True, visible=False)
    
    
    ac_btn = gr.Button("Area Chair")
    decison_box = gr.JSON(label="Area Chair", render=True)

    
    upload_pdf_btn.click(fn=reviewer.load_pdf, inputs=[seed_output_box, title_box, keywords_box, file_input_box], outputs=[seed_output_box, title_box, keywords_box, section_output_box, split_text_box])
    review_btn.click(fn=generate_4_notes, inputs=[seed_output_box, title_box, keywords_box, split_text_box], outputs=[review_box_1, review_box_10, review_box1, review_box_2, review_box_20, review_box2, review_box_3, review_box_30, review_box3, review_box_4, review_box_40, review_box4])
    
    ac_btn.click(fn=reviewer.generate_decision, inputs=[title_box, keywords_box, split_text_box, review_box1, review_box2, review_box3, review_box4], outputs=[decison_box])

demo.launch(share=True, server_port=int(args.server_port), server_name='0.0.0.0')

