from deeppavlov import build_model, configs
import os
import torch
print(torch.cuda.is_available()) 
# import nltk
# nltk.download('punkt_tab')
os.environ["CUDA_VISIBLE_DEVICES"]="3"
# model = build_model('ner_bert_base', install=True, download=True)
# model = build_model('entity_detection_en', install=True, download=True)
model = build_model('few_shot_roberta', install=True, download=True)

texts = [
    "what expression would i use to say i love you if i were an italian",
    "what's the currency conversion between krones and yen",
    "i'd like to reserve a high-end car",
    'What does this word mean?'
]

dataset = [
    ["please help me book a rental car for nashville",                       "car_rental"],
    ["how can i rent a car in boston",                                       "car_rental"],
    ["help me get a rental car for march 2 to 6th",                          "car_rental"],

    ["how many pesos can i get for one dollar",                              "exchange_rate"],
    ["tell me the exchange rate between rubles and dollars",                 "exchange_rate"],
    ["what is the exchange rate in pesos for 100 dollars",                   "exchange_rate"],

    ["can you tell me how to say 'i do not speak much spanish', in spanish", "translate"],
    ["please tell me how to ask for a taxi in french",                       "translate"],
    ["how would i say thank you if i were russian",                          "translate"]
]

a = model(texts, dataset)

# get predictions for 'input_text1', 'input_text2'
# a = model(['Video provides a powerful way to help you prove your point. When you click Online Video, you can paste in the embed code for the video you want to add. You can also type a keyword to search online for the video that best fits your document.'])

print("output \n", a)