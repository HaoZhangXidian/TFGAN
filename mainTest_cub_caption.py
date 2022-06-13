import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time

def Prepare_test_data(path):
    with open(path, "r") as f:
        data = json.loads(f.read())

    sen_index_minibatch_list = []
    for sen in tqdm(data[0:10000]):
        sen_index_minibatch_list.append((sen+'.').lower().split())

    return sen_index_minibatch_list

generate_model = 'LSTM_gpt_reg'


if generate_model == 'fine_tuned gpt2':
    model = torch.load('./trained_model/cub_caption/gpt2_117M_params_cub.pkl')
elif generate_model == 'gpt2':
    model = GPT2LMHeadModel.from_pretrained('gpt2')
elif generate_model == 'pure_LSTM':
    model = torch.load('./trained_model/cub_caption/Pure_LSTM_cub.pkl')
    #model = torch.load('./trained_model/coco_caption/Pure_LSTM_texy_coco.pkl')
elif generate_model == 'LSTM_gpt_reg_gan':
    model = torch.load('./trained_model/cub_caption/LSTM_GPT_L2reg_gan_cub.pkl')
elif generate_model == 'LSTM_gpt_reg':
    model = torch.load('./trained_model/cub_caption/LSTM_GPT_L2reg_cub.pkl')

model.eval()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model.to(device)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
start = time.time()
Length = 0
for i in range(50):
    outputs = model.generate(max_length=50, bos_token_id=tokenizer.bos_token_id,
                             eos_token_ids=tokenizer.eos_token_id,
                             do_sample=True, num_beams=None, num_return_sequences=1, temperature=1)
    print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))
    Length += len(outputs[0])

end = time.time()
Time = (end-start) / Length
######################### calculate blue
num_for_BLUE = 256

print('Generate the sentence')
generate_sent = []
for i in tqdm(range(num_for_BLUE)):
    outputs = model.generate(max_length=30, bos_token_id=tokenizer.bos_token_id,
                             eos_token_ids=tokenizer.eos_token_id,
                             do_sample=True, num_beams=10, num_return_sequences=1, temperature=1)
    sen = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generate_sent.append(sen.lower().split())
# Train_data = Prepare_test_data("./dataset/texygen_coco/train.json")
# ref_index = np.random.choice(len(Train_data), num_for_BLUE)
# generate_sent = [Train_data[t] for t in ref_index]

#####################################################################################################################
print('\n')
print('Calculate self-Blue Score')
self_bleu1 = 0
self_bleu2 = 0
self_bleu3 = 0
self_bleu4 = 0

generate_sent_copy = generate_sent[:]
for i in tqdm(range(len(generate_sent_copy))):
    sen = generate_sent[i]
    del generate_sent[i]
    ref_sents = generate_sent

    self_bleu1 += sentence_bleu(ref_sents, sen, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
    self_bleu2 += sentence_bleu(ref_sents, sen, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1)
    self_bleu3 += sentence_bleu(ref_sents, sen, weights=(0.33, 0.33, 0.33, 0), smoothing_function=SmoothingFunction().method1)
    self_bleu4 += sentence_bleu(ref_sents, sen, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)

    generate_sent = generate_sent_copy[:]

print('\n')
print('Cumulative self 1-gram: %f' % (self_bleu1 / num_for_BLUE))
print('Cumulative self 2-gram: %f' % (self_bleu2 / num_for_BLUE))
print('Cumulative self 3-gram: %f' % (self_bleu3 / num_for_BLUE))
print('Cumulative self 4-gram: %f' % (self_bleu4 / num_for_BLUE))


########################################################################################
Test_data = Prepare_test_data("./dataset/cub_captions/test.json")
ref_sents = Test_data

bleu1 = 0
bleu2 = 0
bleu3 = 0
bleu4 = 0


print('\n')
print('Calculate Blue Score')
for sen in tqdm(generate_sent):
    bleu1 += sentence_bleu(ref_sents, sen, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
    bleu2 += sentence_bleu(ref_sents, sen, weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1)
    bleu3 += sentence_bleu(ref_sents, sen, weights=(0.33, 0.33, 0.33, 0), smoothing_function=SmoothingFunction().method1)
    bleu4 += sentence_bleu(ref_sents, sen, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)

print('\n')
print('Cumulative 1-gram: %f' % (bleu1/num_for_BLUE))
print('Cumulative 2-gram: %f' % (bleu2/num_for_BLUE))
print('Cumulative 3-gram: %f' % (bleu3/num_for_BLUE))
print('Cumulative 4-gram: %f' % (bleu4/num_for_BLUE))



