# coding=utf-8

import requests
from lxml import etree
import re
import os
import json

def get_qa(page_url):
    response = requests.get(url= page_url) # 返回结果
    wb_data = response.text # 文本展示返回结果
    html = etree.HTML(wb_data) # 将页面转换成文档树
    question = html.xpath('//*[@id="container"]/div[2]/div[2]/h1/span')[0]
    question = question.text.replace('\r', '').replace('\n', '')
    anses = html.xpath('//*[@class="answer-item atl-item"]') # [@class="answer-item atl-item"]')
    num_ans = len(anses)

    answers = []
    for i in range(1, num_ans + 1):
        a = html.xpath('//*[@id="container"]/div[3]/div/div[%d]/div[@class="content"]' % i)[0]
        if a.text == None:
            continue
        a = a.text.strip()
        a = re.sub('[\r\n]+', ' ', a)
        answers.append(a)

    return question, answers




def get_qa_and_next_page_url(start_url):
    response = requests.get(url= start_url) # 返回结果
    wb_data = response.text # 文本展示返回结果
    html = etree.HTML(wb_data) # 将页面转换成文档树
    blocks = html.xpath('//*[@id="main"]/div[6]/table/tbody')

    num_block = len(blocks)

    qa_page_urls = []
    for i in range(2, num_block + 1):
        ans = html.xpath('//*[@id="main"]/div[6]/table/tbody[%d]/tr' % i)
        num_ans = len(ans)
        for j in range(1, num_ans + 1):
            q = html.xpath('//*[@id="main"]/div[6]/table/tbody[%d]/tr[%d]/td[1]/a' % (i, j))[0]
            qa_page_url = 'http://bbs.tianya.cn' + q.attrib['href']
            qa_page_urls.append(qa_page_url)

    qa_pairs = []
    for url in qa_page_urls:
        question, answers = get_qa(url)
        if answers != []:
            qa_pairs.append((question, answers))
            # print(question, answers)

    next_page_url = html.xpath('//*[@id="main"]/div[7]/div/a[last()]')[0]
    next_page_url = 'http://bbs.tianya.cn' + next_page_url.attrib['href']


    return qa_pairs, next_page_url


script_abs_path = os.path.dirname(__file__)
ROOT_DIR = os.path.join(script_abs_path, '../../')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA = os.path.join(DATA_DIR, 'raw_data')

if __name__ == '__main__':
    data_dir = os.path.join(RAW_DATA, 'house_qa_data')
    next_page_url = "http://bbs.tianya.cn/list.jsp?item=house&sub=11"  # 请求地址

    num_sample = 0
    num_data_file = 0
    while True:
        print('Num sample %d, Num file %d Now reading page %s ...' % (num_sample, num_data_file, next_page_url))
        qa_pairs, next_page_url = get_qa_and_next_page_url(next_page_url)
        data_file = os.path.join(data_dir, 'part_%d.json' % num_data_file)
        for question, answers in qa_pairs:
            data = {
                'question':question,
                'answers':answers
            }
            json_data = json.dumps(data, ensure_ascii=False)

            num_sample += 1
            if num_sample % 1000 == 0: # 每1000个question写入一个文件
                num_data_file += 1
            with open(data_file, 'a', encoding='utf-8') as f:
                f.write(json_data + '\n')
        print('Done !!!')








