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
max_length = {
    "ABSTRACT": 1024,
    "INTRODUCTION": 4096,
    "EXPERIMENTS": 3072,
    "RESULTS": 3072,
    "CONCLUSION": 2048, 
}
sections = ['ABSTRACT', 'INTRODUCTION', 'EXPERIMENTS', 'RESULTS', 'CONCLUSION']
assessments = ['summaries', 'strengths', 'weaknesses', 'questions']
assessments_ = ['SUMMARIES', 'STRENGTHS', 'WEAKNESSES', 'QUESTIONS']


model_path = "0101-v2-full-injected"
tokenizer_path = "vicuna"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def add_inference_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group()
    group.add_argument("--re-path", type=str, default='0101-v2-full-injected')
    group.add_argument("--ac-path", type=str, default='vicuna-7b-v1.5-ac-awq')
    group.add_argument("--re-port", type=str, default='39174')
    group.add_argument("--ac-port", type=str, default='8001')
    group.add_argument("--server-port", type=str, default='10729')
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
                title = title if title else extracted_text.split('\n')[1]
                extracted_text = extracted_text.replace("I NTRODUCTION", "INTRODUCTION")
                extracted_text = extracted_text.replace("C ONCLUSION", "CONCLUSION")
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

    def generate_notes(self, seed, title, keywords, split_text):
        global args
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
            local_text = remove_duplicate(generate(input_str, model_path=args.re_path, port=args.re_port))
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

        return note_dict, assessment_dict, elements

    def generate_review(self, note_dict, assessment_dict, elements):
        global args
        review_dict = {}

        for assessment in assessments:
            elements.append((0, f"You will read through your note about {assessment.upper()}, then write down the {assessment.upper()} for the paper. Your note about {assessment.upper()}:\n\n"))
            for splited_assessment in assessment_dict[assessment]:
                elements.append((0, f"{splited_assessment}\n"))
            elements.append((0, f"\n"))
            elements.append((0, f"Your final {assessment.upper()}:\n\n"))
            input_ids, _, input_str = build_input(elements)
            local_text = remove_duplicate(generate(input_str, max_tokens=512, model_path=args.re_path, port=args.re_port))
            elements.append((1, local_text))
            print(Fore.BLUE + local_text)
            review_dict[assessment.upper()] = local_text

        elements.append((0, f"Now give this article a score from multiple dimensions:\n\n"))
        for metric in ["soundness", "presentation", "contribution", "rating", "confidence"]:
            elements.append((0, f"{metric}: "))
            input_ids, _, input_str = build_input(elements)
            local_text = generate(input_str, model_path=args.re_path, port=args.re_port, temperature=0.01)
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

        review_text = f'''Summaries:\n{review_dict['SUMMARIES']}\nStrengths:\n{review_dict['STRENGTHS']}\nWeaknesses:\n{review_dict['WEAKNESSES']}\nQuestions:\n{review_dict['QUESTIONS']}\n'''
        for assessment in assessments_:
            del review_dict[assessment]
        
        return review_dict, review_text


description = '''        
<h1>📝 Openreviewer: Simulated Academic Review</h1>
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

global args
args = get_args()


with gr.Blocks() as demo:
    reviewer = Reviewer()
    gr.HTML(description)
   
    with gr.Column():
        with gr.Row():
            file_input_box = gr.File(label="PDF", value='pdfs/AttentionIsAllYouNeed.pdf')
            with gr.Column():  
                seed_output_box = gr.Textbox(label="Seed(Optional)")
                title_box = gr.Textbox(label="Title", value="Attention Is All You Need")
                keywords_box = gr.Textbox(label="Keywords(Optional)", value="Language Modeling, Deep Learning")

        upload_pdf_btn = gr.Button("Upload pdf")

    section_output_box = gr.Textbox(label="Paper", visible=False)
    split_text_box = gr.JSON(label="Splitted text", render=True, visible=False)

    generate_notes_btn = gr.Button("Generate notes")

    generated_notes_box = gr.JSON(label="Notes", render=True)
    splited_generated_notes_box = gr.JSON(label="Notes(reorder)", render=True, visible=True)
    elements = gr.List(label="elements", visible=False)

    generate_review_btn = gr.Button("Generate review")
    with gr.Column():
        review_box = gr.JSON(label="score", render=True)
        review2_box = gr.Textbox(label="review", render=True, lines=20, interactive=False)


    upload_pdf_btn.click(fn=reviewer.load_pdf, inputs=[seed_output_box, title_box, keywords_box, file_input_box], outputs=[seed_output_box, title_box, keywords_box, section_output_box, split_text_box])
    generate_notes_btn.click(fn=reviewer.generate_notes, inputs=[seed_output_box, title_box, keywords_box, split_text_box], outputs=[generated_notes_box, splited_generated_notes_box, elements])
    generate_review_btn.click(fn=reviewer.generate_review, inputs=[generated_notes_box, splited_generated_notes_box, elements], outputs=[review_box, review2_box])

demo.launch(share=True, server_port=int(args.server_port), server_name='0.0.0.0')

