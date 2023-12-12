
!pip install youtube_transcript_api



from youtube_transcript_api import YouTubeTranscriptApi
while True:
    url=input("Enter url: ")
    try:
        url=url.split('/')
        code=url[3]

        from youtube_transcript_api import YouTubeTranscriptApi
        srt = YouTubeTranscriptApi.get_transcript(code)

        global subtitle
        subtitle = " ".join([x['text'] for x in srt])
        #print(subtitle)
        print(code)
        break
    except:
        print("ERROR! Enter a valid url")

video_code=url[3]
srt = YouTubeTranscriptApi.get_transcript(video_code)
global captions
captions= " ".join([x['text'] for x in srt])
captions

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

import torch

text = subtitle


max_token_limit = 500


input_tokens = tokenizer.tokenize(text)
input_ids = tokenizer.convert_tokens_to_ids(input_tokens)


batched_input_ids = [input_ids[i:i + max_token_limit] for i in range(0, len(input_ids), max_token_limit)]


for batch_ids in batched_input_ids:

    input_ids_batch = torch.tensor(batch_ids).unsqueeze(0)


    summary_ids = model.generate(
        input_ids=input_ids_batch,
        max_length=150,
        num_beams=5,
        temperature=0.9,
        length_penalty=2.0,
        no_repeat_ngram_size=2,
        top_p=0.92,
        early_stopping=True
    )


    decoded_summary = tokenizer.decode(summary_ids[0].cpu(), skip_special_tokens=True)
    bullet_points = [point.strip() for point in decoded_summary.split(".")]

    for idx, point in enumerate(bullet_points, start=1):
        print(f"^ {point}")


tokenized_subtitle = tokenizer.encode(subtitle, return_tensors="pt")



max_tokens_per_batch = 512
max_summary_length = 120


batch_ranges = range(0, tokenized_subtitle.size(1), max_tokens_per_batch)
batches = [tokenized_subtitle[0, i:i+max_tokens_per_batch] for i in batch_ranges]


for batch in batches:
    summary_ids = model.generate(batch.unsqueeze(0), max_length=80, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0].cpu(), skip_special_tokens=True)  # Move the generated summary to CPU for decoding
    print(summary)