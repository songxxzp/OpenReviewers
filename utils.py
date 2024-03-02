import re, json
import PyPDF2
import os

from typing import List, Dict, Optional

translations = {
    "abstract": ["中文摘要", "摘要", "概要", "要旨"],
    "introduction": ["引言", "简介", "导论"],
    "related work": ["相关工作", "相关研究", "文献回顾", "研究背景"],
    "methodology": ["方法", "研究方法"],
    "experiments": ["实验", "实验研究", "实验设计", "实验部分"],
    "results": ["结果", "研究成果", "实验结果", "结果分析"],
    "discussion": ["讨论", "结果讨论", "讨论部分", "讨论与分析"],
    "conclusion": ["结论", "研究结论", "结论部分", "总结"],
    "future work": ["未来工作", "后续工作", "未来研究方向", "未来研究计划"],
    "acknowledgments": ["致谢", "感谢", "致谢部分", "谢辞"],
    "references": ["参考文献", "参考书目", "文献引用", "引用文献"],
    "appendices": ["附录", "附表", "附加材料", "补充材料", "其他内容"]
}


def translate(content: str):
    cnt = 0
    for k, vs in translations.items():
        for v in vs:
            if v in content:
                cnt += 1
    if cnt < 3:
        return content

    print("Chinese detected, translating...")

    for v in ["表格索引", "插图索引"]:
        if v in content[len(content) // 2:]:
            content = content[:content.rindex(v, len(content) // 2)]

    for v in translations["acknowledgments"]:
        if v in content[len(content) // 2:]:
            content = content[:content.rindex(v, len(content) // 2)]
            break
    for v in translations["appendices"]:
        if v in content[len(content) // 2:]:
            content = content[:content.rindex(v, len(content) // 2)]
            break
    for v in translations["references"]:
        if v in content[len(content) // 2:]:
            content = content[:content.rindex(v, len(content) // 2)]
            break

    for k, vs in translations.items():
        for v in vs:
            if v in content:
                content = content.replace(v, '\n' + k + '\n')
                break
    return content


# 章节别名定义
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


def normalize_title(title):
    return re.sub(r'[^a-zA-Z0-9\s]', '', title).lower().strip()


def extract_section_number(title):
    # 尝试从标题中提取章节编号
    match = re.match(r'(\d+(?:\.\d+)*)\b', title)
    return match.group(1) if match else ""


def find_nearest_section_start(text, start_pos, section_titles, current_section_number):
    nearest_pos = len(text)
    for titles in section_titles:
        for title in titles:
            # 确保对整个章节标题的匹配考虑了前后非单词字符的界定
            pattern = re.compile(r'\b' + re.escape(normalize_title(title)) + r'\b', re.IGNORECASE)
            match = pattern.search(text, start_pos)
            if match:
                match_section_number = extract_section_number(text[:match.start()].split('\n')[-1])
                # 仅当找到的章节编号不是当前章节的子章节时，才考虑它作为结束标志
                if match_section_number and not match_section_number.startswith(current_section_number + '.'):
                    nearest_pos = match.start()
                    return nearest_pos
    return nearest_pos if nearest_pos != len(text) else None


def filter_iclr_pdf(content, keep_chapters: Dict):
    content = translate(content)
    normalized_chapters = {k: [normalize_title(t) for t in v] for k, v in keep_chapters.items()}
    section_titles = [v for v in normalized_chapters.values()]
    paper_dict = {}

    for section, titles in normalized_chapters.items():
        for title in titles:
            start_pattern = re.compile(r'\b' + re.escape(title) + r'\b', re.IGNORECASE)
            start_match = start_pattern.finditer(content)
            for match in start_match:
                start_pos = match.start()
                # 提取当前章节的编号
                current_section_number = extract_section_number(content[:start_pos].split('\n')[-1])
                end_pos = find_nearest_section_start(content, start_pos + len(title), section_titles, current_section_number)
                if section not in paper_dict:
                    paper_dict[section] = content[start_pos:end_pos].strip() if end_pos else content[start_pos:].strip()
                break  # 匹配到一个标题后即跳出

    return paper_dict


def extract_chapter(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    # 获取PDF的总页数
    num_pages = len(pdf_reader.pages)
    # 初始化提取状态和提取文本
    extraction_started = False
    extracted_text = ""
    # 遍历PDF中的每一页
    for page_number in range(num_pages):
        page = pdf_reader.pages[page_number]
        page_text = page.extract_text()

        # 开始提取
        extraction_started = True
        page_number_start = page_number
        # 如果提取已开始，将页面文本添加到提取文本中
        if extraction_started:
            extracted_text += page_text
            # 停止提取
            if page_number_start + 1 < page_number:
                break
    return extracted_text


if __name__ == "__main__":
    pdf_dir = 'arxiv/pdf/'

    for file_name in os.listdir(pdf_dir):
        file_path = os.path.join(pdf_dir, file_name)
        if os.path.exists(file_path):
            extracted_text = extract_chapter(file_path)
            print(translate(extracted_text))
            if len(extracted_text.split('\n')) > 2:
                extracted_text = extracted_text.replace("I NTRODUCTION", "INTRODUCTION")
                extracted_text = extracted_text.replace("C ONCLUSION", "CONCLUSION")
            split_text: Dict = filter_iclr_pdf(extracted_text, chapters)
        print(split_text.keys())
        for section, text in split_text.items():
            print(f"{section}, len = {len(text)}")
        # for section, text in split_text.items():
            # print(f"{section}, {text}")
        print()