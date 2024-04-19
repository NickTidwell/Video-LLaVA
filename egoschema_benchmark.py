import torch
import json
import os
from collections import defaultdict
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

disable_torch_init()
model_path = 'LanguageBind/Video-LLaVA-7B'
cache_dir = 'cache_dir'
device = 'cuda'
load_4bit, load_8bit = True, False
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)

def generate(video_path, query):
    video_processor = processor['video']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()

    video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)

    input = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + query
    conv.append_message(conv.roles[0], input)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensor,
            do_sample=True,
            temperature=0.1,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return outputs

def parse_pred(response):
    numeric_value = ""
    found_numeric = False

    for char in response:
        if char.isdigit():
            numeric_value += char
            found_numeric = True
        elif found_numeric:
            break

    if numeric_value:
        return int(numeric_value)
    else:
        return 999 # Number not in voting
def benchmark():
    video_dir = "/datasets/EgoSchema/videos/videos/"
    answer_path = "/datasets/EgoSchema/subset_answers.json"
    question_path = "/datasets/EgoSchema/questions.json"
    answers = json.load(open(answer_path, "r"))
    questions = json.load(open(question_path, "r"))
    indexed_questions = {entry['q_uid']: entry for entry in questions}
    pred_freq1 = defaultdict(int)
    pred_freq2 = defaultdict(int)
    pred_freq3 = defaultdict(int)

    ans_freq = defaultdict(int)


    total = 0
    query1_correct = 0
    query2_correct = 0
    query3_correct = 0
    for id, answer in answers.items():
        video_path = video_dir + id + ".mp4"
        if not os.path.exists(video_path):
            continue
        question = indexed_questions[id]['question']
        response_1 = indexed_questions[id]['option 0']
        response_2 = indexed_questions[id]['option 1']
        response_3 = indexed_questions[id]['option 2']
        response_4 = indexed_questions[id]['option 3']
        response_5 = indexed_questions[id]['option 4']

        query_1 = f"""Select on of the 5 options to answer the following Question: {question}
                    1.) { response_1} 2.) {response_2} 3.) {response_3} 4.) {response_4} 5.) {response_5}
                    """
        query_2 = f"""Respond with a number (1 - 5 ) to the response that best answers the following question: {question}
                    1.) { response_1} 2.) {response_2} 3.) {response_3} 4.) {response_4} 5.) {response_5}
                    """
        query_3 = f"""Question: {question} Response 1: {response_1} Response 2: {response_2} Response 3: {response_3} Response 4: {response_4} Response 5: {response_5}
                    Which of these responses best answer the question. Respond 1,2,3,4, or 5
        """
        
        raw_out = generate(video_path, query_1)
        output_1 = parse_pred(raw_out)
        output_2 = parse_pred(generate(video_path, query_2))
        output_3 = parse_pred(generate(video_path, query_3))

        if output_1 == answer: query1_correct +=1
        if output_2 == answer: query2_correct +=1
        if output_3 == answer: query3_correct +=1
        total += 1

        pred_freq1[output_1] += 1
        pred_freq2[output_2] += 1
        pred_freq3[output_3] += 1
        ans_freq[answer] += 1

        print(f"query: {query_1}")
        print(f"response: {raw_out}")

    accuracy1 = query1_correct / total
    accuracy2 = query2_correct / total
    accuracy3 = query3_correct / total
    print(f"Accuracy 1: {accuracy1 * 100:.4f}%")
    print(f"Accuracy 2: {accuracy2 * 100:.4f}%")
    print(f"Accuracy 3: {accuracy3 * 100:.4f}%")

    print("Answer Distribution:")
    for choice, freq in ans_freq.items():
        print(f"choice: {choice}, Freq: {freq}")

    print("Prediction Distribution ( Query 1 ):")
    for choice, freq in pred_freq1.items():
        print(f"choice: {choice}, Freq: {freq}")

    print("Prediction Distribution ( Query 2 ):")
    for choice, freq in pred_freq2.items():
        print(f"choice: {choice}, Freq: {freq}")

    print("Prediction Distribution ( Query 3 ):")
    for choice, freq in pred_freq3.items():
        print(f"choice: {choice}, Freq: {freq}")

if __name__ == '__main__':
    benchmark()