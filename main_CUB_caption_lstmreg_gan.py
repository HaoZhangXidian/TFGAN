import json
import random
import torch
from torch.nn import MSELoss
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch import optim
from tqdm import tqdm
from model.Build_model import Build_LSTM_model, Build_cnn_discriminator
from torch.nn import functional as F

def toogle_grad(model, requires_grad):
    for name, p in model.named_parameters():
        #p.requires_grad_(requires_grad)
        if name == 'lm_head.weight':
            p.requires_grad_(False)
        else:
            p.requires_grad_(requires_grad)

def sample_sequence(model, length, context, device='cuda'):

    context = torch.tensor(context, dtype=torch.long, device=device)
    generated = context
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][-1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            if next_token == 50256:
                break
            else:
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
        return generated

def sample_LSTM_sequence(model, length, context, device='cuda'):

    context = torch.tensor(context, dtype=torch.long, device=device)
    generated = context.unsqueeze(0)
    with torch.no_grad():
        for _ in range(length):
            inputs = {'x': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if next_token == 50256:
                break
            else:
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
        return generated.squeeze()

def Prepare_test_data(path):
    with open(path, "r") as f:
        data = json.loads(f.read())

    maxlength = 0
    sen_index_minibatch_list = []
    for sen in tqdm(data[0:10000]):
        sen = '<|endoftext|> ' + sen + ' <|endoftext|>'
        sen_index_tensor = tokenizer.encode(sen, return_tensors='pt')
        sen_index_minibatch_list.append(sen_index_tensor)
        maxlength = max(sen_index_tensor.size()[1], maxlength)

    sen_minibatch_index_tensor = 50256 * torch.ones([len(data[0:10000]), maxlength], dtype=int)
    label_minibatch_index_tensor = -100 * torch.ones([len(data[0:10000]), maxlength], dtype=int)
    reg_index = torch.zeros([len(data[0:10000]), maxlength]).to(torch.float32)

    for num in tqdm(range(len(sen_index_minibatch_list))):
        sen_index_tensor = sen_index_minibatch_list[num]
        length = sen_index_tensor.size()[1]
        sen_minibatch_index_tensor[num, 0:length] = sen_index_tensor
        label_minibatch_index_tensor[num, 1:length] = sen_index_tensor[0, 1:]
        reg_index[num, 0:length] = 1
    reg_index = reg_index.unsqueeze(2)

    return[sen_minibatch_index_tensor, label_minibatch_index_tensor, reg_index]

def compute_loss(d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)
    loss = F.binary_cross_entropy_with_logits(d_out, targets)

    return loss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## Load GPT2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
load_fine_tune_GPT = True
if not load_fine_tune_GPT:
    model = GPT2LMHeadModel.from_pretrained('gpt2')
else:
    model = torch.load('./trained_model/cub_caption/gpt2_117M_params_cub.pkl')
model.eval()

## Build LSTM
pre_trained_wordE = model.transformer.wte
vocab_size = pre_trained_wordE.num_embeddings
dim_wordE = pre_trained_wordE.embedding_dim
dim_LSTMh = 768
lm_head = model.lm_head
LSTM = Build_LSTM_model(vocab_size, dim_wordE, dim_LSTMh, lm_head)
LSTM.train()

model.to(device)
LSTM.to(device)

############################## build discriminator
discriminator = Build_cnn_discriminator(dim_LSTMh)
discriminator.to(device)


## Load data
with open("./dataset/cub_captions/train.json","r") as f:
    data_train = json.loads(f.read())

Test_data = Prepare_test_data("./dataset/cub_captions/test.json")

Train_num = len(data_train)
batchsize = 32


## construct optimizer

LSTM_params = LSTM.parameters()
Total_lr_decay = 0
learning_rate = 1e-4
LSTM_optimizer = optim.Adam(LSTM_params, lr=learning_rate)

discriminator_params = discriminator.parameters()
discriminator_optimizer = optim.Adam(discriminator_params, lr=1e-4)

##
best_loss = 1000000
not_improved = 0

for epoch in range(1000):
    print('Start epoch %d...' % epoch)

    random.shuffle(data_train)

    for iteration in range(Train_num // batchsize):
        data_minibatch = data_train[iteration * batchsize: (iteration + 1) * batchsize]

        Loss = 0
        maxlength = 0
        sen_index_minibatch_list = []
        for sen in data_minibatch:
            sen = '<|endoftext|> ' + sen + ' <|endoftext|>'
            sen_index_tensor = tokenizer.encode(sen, return_tensors='pt')
            sen_index_minibatch_list.append(sen_index_tensor)
            maxlength = max(sen_index_tensor.size()[1], maxlength)

        if maxlength * batchsize > 1200:
            continue

        sen_minibatch_index_tensor = 50256 * torch.ones([batchsize, maxlength], dtype=int).to(device)
        label_minibatch_index_tensor = -100 * torch.ones([batchsize, maxlength], dtype=int).to(device)
        reg_index = torch.zeros([batchsize, maxlength]).to(torch.float32).to(device)

        for num in range(len(sen_index_minibatch_list)):
            sen_index_tensor = sen_index_minibatch_list[num]
            length = sen_index_tensor.size()[1]
            sen_minibatch_index_tensor[num, 0:length] = sen_index_tensor
            label_minibatch_index_tensor[num, 1:length] = sen_index_tensor[0, 1:]
            reg_index[num, 0:length] = 1
        reg_index = reg_index.unsqueeze(2)

        with torch.no_grad():
            gpt_hidden_output = model(sen_minibatch_index_tensor)[2]

        d_real = discriminator(gpt_hidden_output)
        output = LSTM(sen_minibatch_index_tensor, labels=label_minibatch_index_tensor)
        lstm_hidden_output = output[2]
        d_fake = discriminator(lstm_hidden_output)

        ## update D
        toogle_grad(LSTM, False)
        toogle_grad(discriminator, True)
        discriminator_optimizer.zero_grad()

        dloss_real = compute_loss(d_real, 1)
        dloss_fake = compute_loss(d_fake, 0)

        dloss = dloss_real + dloss_fake
        dloss.backward()

        discriminator_optimizer.step()

        ## update LSTM
        toogle_grad(LSTM, True)
        toogle_grad(discriminator, False)
        LSTM_optimizer.zero_grad()

        output = LSTM(sen_minibatch_index_tensor, labels=label_minibatch_index_tensor)
        loss = output[0]
        lstm_hidden_output = output[2]

        d_fake = discriminator(lstm_hidden_output)
        gloss = compute_loss(d_fake, 1)

        loss_reg = MSELoss()
        reg = loss_reg(lstm_hidden_output * reg_index, gpt_hidden_output * reg_index)
        Loss = loss + 100 * reg + gloss
        Loss.backward()
        LSTM_optimizer.step()

        if (iteration + 1) % 10 == 0:
            print(
                'At epoch %d iteration %d, the training loss is %4f, the reg loss is %4f, the d loss is %4f, the g loss is %4f, Total lr decay is %d' % (
                    epoch + 1, iteration + 1, loss, reg.item(), dloss.item(), gloss.item(), Total_lr_decay))

        if (iteration + 1) % 100 == 0:
            prompt_text = "<|endoftext|>"
            context_tokens = tokenizer.encode(prompt_text)

            out = sample_LSTM_sequence(
                model=LSTM,
                length=30,
                context=context_tokens,
                device=device
            )

            generated_list = out[1:].tolist()
            text = tokenizer.decode(generated_list)
            print('\n')
            print('The generated sentence at epoch %d iteration %d is:' % (epoch + 1, iteration + 1))
            print(text + '\n')

        if (iteration + 1) % 100 == 0:
            print('Calculate loss on validation dataset')
            with torch.no_grad():
                test_loss = 0
                test_reg = 0
                for it in range(len(Test_data[0]) // 20):
                    test_ids = Test_data[0][it * 20: (it + 1) * 20].to(device)
                    test_label = Test_data[1][it * 20: (it + 1) * 20].to(device)
                    reg_index = Test_data[2][it * 20: (it + 1) * 20].to(device)

                    gpt_hidden_output = model(test_ids)[2]

                    output = LSTM(test_ids, labels=test_label)
                    loss = output[0]
                    lstm_hidden_output = output[2]
                    reg = loss_reg(lstm_hidden_output * reg_index, gpt_hidden_output * reg_index)

                    test_loss += loss
                    test_reg += reg

                test_loss = test_loss / (it + 1)
                test_reg = test_reg / (it + 1)

            if test_loss < best_loss:
                best_loss = test_loss
                print('\n')
                torch.save(LSTM, './trained_model/cub_caption/LSTM_GPT_L2reg_gan_cub.pkl')
                not_improved = 0
            else:
                not_improved += 1

                if not_improved == 5:
                    Total_lr_decay += 1
                    not_improved = 0
                    learning_rate = learning_rate * 0.5
                    LSTM_optimizer = optim.Adam(LSTM.parameters(), lr=learning_rate)

            print('At epoch %d iteration %d, the validation loss is %4f, the reg loss is %4f, the best loss is %4f' % (
            epoch + 1, iteration + 1, test_loss, test_reg, best_loss))
            print('\n')

