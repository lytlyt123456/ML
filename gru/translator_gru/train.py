import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from model.encoder import Encoder
from model.decoder import Decoder
import config as cfg
import os
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dataprep.dataprep import LanguageWordList, load_and_preprocess_data, train_test_split

def train(train_set, input_wl_size: int, output_wl_size: int, embed_dim: int,
          learning_rate: float, num_epochs: int = 10):
    # 模型创建
    encoder = Encoder(input_wl_size, embed_dim).to(cfg.DEVICE)
    decoder = Decoder(output_wl_size, embed_dim).to(cfg.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer_encoder = optim.SGD(encoder.parameters(), lr=learning_rate)
    optimizer_decoder = optim.SGD(decoder.parameters(), lr=learning_rate)
    # scheduler_encoder = lr_scheduler.StepLR(optimizer_encoder, step_size=1, gamma=0.95)
    # scheduler_decoder = lr_scheduler.StepLR(optimizer_decoder, step_size=1, gamma=0.95)
    cnt = 0
    history = {'loss': []}

    # 训练
    for epoch in range(num_epochs):
        avg_loss = 0.0
        for sample_idx, [chinese_sentence, english_sentence] in enumerate(train_set):
            cnt = cnt + 1

            n_ch = len(chinese_sentence)
            n_en = len(english_sentence)

            encoder_output = []
            output, hidden = encoder(chinese_sentence[0], None)
            encoder_output.append(output[0])
            for i in range(1, n_ch):
                output, hidden = encoder(chinese_sentence[i], hidden)
                encoder_output.append(output[0])
            encoder_output = torch.cat(encoder_output, dim=0).to(cfg.DEVICE)
            if encoder_output.shape[0] < cfg.MAX_SEQ_LENGTH:
                suffix = torch.zeros([cfg.MAX_SEQ_LENGTH - encoder_output.shape[0], embed_dim]).to(cfg.DEVICE)
                encoder_output = torch.cat([encoder_output, suffix], dim=0).to(cfg.DEVICE)

            output, hidden, _ = decoder(cfg.SOS, hidden, encoder_output)
            loss = criterion(output, torch.tensor([english_sentence[0]]).to(cfg.DEVICE))
            for i in range(1, n_en):
                if epoch + 1 > 50:
                    output, hidden, _ = decoder(torch.argmax(output, dim=1).item(), hidden, encoder_output)
                else:
                    output, hidden, _ = decoder(english_sentence[i - 1], hidden, encoder_output)
                loss += criterion(output, torch.tensor([english_sentence[i]]).to(cfg.DEVICE))

            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

            avg_loss += loss.item() / n_en

        if epoch + 1 == 50:
            optimizer_encoder = optim.SGD(encoder.parameters(), lr=learning_rate / 5)
            optimizer_decoder = optim.SGD(decoder.parameters(), lr=learning_rate / 5)

        avg_loss /= len(train_set)
        history['loss'].append(avg_loss)
        print(f'[epoch {epoch + 1}] loss = {avg_loss}')

    torch.save(encoder, os.path.join(cfg.OUTPUT_DIR, 'encoder.pt'))
    torch.save(decoder, os.path.join(cfg.OUTPUT_DIR, 'decoder.pt'))

    plt.figure(figsize=(8, 8))
    plt.xlabel('Check Number')
    plt.ylabel('Loss')
    plt.plot([i for i in range(1, len(history['loss']) + 1)], history['loss'])
    plt.savefig(f'{cfg.OUTPUT_DIR}/train_loss.svg')

@torch.no_grad()
def test(test_set, chinese_word_list, english_word_list, num_test_samples: int = 50):
    encoder = torch.load(os.path.join(cfg.OUTPUT_DIR, 'encoder.pt'),
                         map_location='cpu', weights_only=False).to(cfg.DEVICE)
    decoder = torch.load(os.path.join(cfg.OUTPUT_DIR, 'decoder.pt'),
                         map_location='cpu', weights_only=False).to(cfg.DEVICE)

    test_set = [pair for pair in test_set if len(pair[0]) <= cfg.MAX_SEQ_LENGTH
                and len(pair[1]) <= cfg.MAX_SEQ_LENGTH]

    test_samples = test_set[:num_test_samples]
    for chinese_sentence, english_sentence in test_samples:
        n_ch = len(chinese_sentence)
        n_en = len(english_sentence)

        encoder_output = []
        output, hidden = encoder(chinese_sentence[0], None)
        encoder_output.append(output[0])
        for i in range(1, n_ch):
            output, hidden = encoder(chinese_sentence[i], hidden)
            encoder_output.append(output[0])
        encoder_output = torch.cat(encoder_output, dim=0).to(cfg.DEVICE)
        if encoder_output.shape[0] < cfg.MAX_SEQ_LENGTH:
            suffix = torch.zeros([cfg.MAX_SEQ_LENGTH - encoder_output.shape[0], encoder.embed_dim]).to(cfg.DEVICE)
            encoder_output = torch.cat([encoder_output, suffix], dim=0).to(cfg.DEVICE)

        output_sentence = []
        output, hidden, _ = decoder(cfg.SOS, hidden, encoder_output)
        predict = torch.argmax(output, dim=1).item()
        output_sentence.append(predict)
        if predict != cfg.EOS:
            for i in range(1, cfg.MAX_SEQ_LENGTH):
                output, hidden, _ = decoder(predict, hidden, encoder_output)
                predict = torch.argmax(output, dim=1)[0].item()
                output_sentence.append(predict)
                if predict == cfg.EOS:
                    break

        str_ch_sentence = ' '.join([chinese_word_list.index_to_word[i] for i in chinese_sentence])
        str_en_sentence = ' '.join([english_word_list.index_to_word[i] for i in english_sentence])
        str_output_sentence = ' '.join([english_word_list.index_to_word[i] for i in output_sentence])

        print(f'中文原文: {str_ch_sentence}')
        print(f'英文参考译文: {str_en_sentence}')
        print(f'模型翻译结果: {str_output_sentence}\n')

@torch.no_grad()
def show_attn(dataset, chinese_word_list, english_word_list, num_test_samples=10):
    encoder = torch.load(os.path.join(cfg.OUTPUT_DIR, 'encoder.pt'),
                         map_location='cpu', weights_only=False).to(cfg.DEVICE)
    decoder = torch.load(os.path.join(cfg.OUTPUT_DIR, 'decoder.pt'),
                         map_location='cpu', weights_only=False).to(cfg.DEVICE)

    dataset = [pair for pair in dataset if len(pair[0]) <= cfg.MAX_SEQ_LENGTH
               and len(pair[1]) <= cfg.MAX_SEQ_LENGTH]

    test_samples = dataset[:num_test_samples]

    cnt = 0
    for chinese_sentence, english_sentence in test_samples:
        cnt += 1

        n_ch = len(chinese_sentence)
        n_en = len(english_sentence)

        encoder_output = []
        output, hidden = encoder(chinese_sentence[0], None)
        encoder_output.append(output[0])
        for i in range(1, n_ch):
            output, hidden = encoder(chinese_sentence[i], hidden)
            encoder_output.append(output[0])
        encoder_output = torch.cat(encoder_output, dim=0).to(cfg.DEVICE)
        if encoder_output.shape[0] < cfg.MAX_SEQ_LENGTH:
            suffix = torch.zeros([cfg.MAX_SEQ_LENGTH - encoder_output.shape[0], encoder.embed_dim]).to(cfg.DEVICE)
            encoder_output = torch.cat([encoder_output, suffix], dim=0).to(cfg.DEVICE)

        attn_weights = []
        output_sentence = []
        output, hidden, attn_weight = decoder(cfg.SOS, hidden, encoder_output)
        attn_weights.append(attn_weight)
        predict = torch.argmax(output, dim=1).item()
        output_sentence.append(predict)
        if predict != cfg.EOS:
            for i in range(1, cfg.MAX_SEQ_LENGTH):
                output, hidden, attn_weight = decoder(predict, hidden, encoder_output)
                attn_weights.append(attn_weight)
                predict = torch.argmax(output, dim=1)[0].item()
                output_sentence.append(predict)
                if predict == cfg.EOS:
                    break

        attn_weights = torch.cat(attn_weights, dim=0)

        str_ch_sentence = ' '.join([chinese_word_list.index_to_word[i] for i in chinese_sentence])
        str_output_sentence = ' '.join([english_word_list.index_to_word[i] for i in output_sentence])

        print(f'中文原文: {str_ch_sentence}')
        print(f'模型翻译结果: {str_output_sentence}\n')

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(8, 8))
        cax = plt.matshow(attn_weights.numpy()[:, :len(chinese_sentence)], cmap='bone')
        fig.colorbar(cax)
        plt.xticks(ticks=[i for i in range(len(chinese_sentence))], labels=str_ch_sentence.split(' '))
        plt.yticks(ticks=[i for i in range(len(output_sentence))], labels=str_output_sentence.split(' '))
        plt.savefig(f'{cfg.OUTPUT_DIR}/show_attn_{cnt}.svg')


@torch.no_grad()
def compute_bleu_score(test_set, chinese_word_list, english_word_list, num_test_samples = 100):
    encoder = torch.load(os.path.join(cfg.OUTPUT_DIR, 'encoder.pt'),
                         map_location='cpu', weights_only=False).to(cfg.DEVICE)
    decoder = torch.load(os.path.join(cfg.OUTPUT_DIR, 'decoder.pt'),
                         map_location='cpu', weights_only=False).to(cfg.DEVICE)

    test_set = [pair for pair in test_set if len(pair[0]) <= cfg.MAX_SEQ_LENGTH
                and len(pair[1]) <= cfg.MAX_SEQ_LENGTH]
    test_set = test_set[:num_test_samples]

    sum_bleu_score = 0.0

    for chinese_sentence, english_sentence in test_set:
        n_ch = len(chinese_sentence)
        n_en = len(english_sentence)

        encoder_output = []
        output, hidden = encoder(chinese_sentence[0], None)
        encoder_output.append(output[0])
        for i in range(1, n_ch):
            output, hidden = encoder(chinese_sentence[i], hidden)
            encoder_output.append(output[0])
        encoder_output = torch.cat(encoder_output, dim=0).to(cfg.DEVICE)
        if encoder_output.shape[0] < cfg.MAX_SEQ_LENGTH:
            suffix = torch.zeros([cfg.MAX_SEQ_LENGTH - encoder_output.shape[0], encoder.embed_dim]).to(cfg.DEVICE)
            encoder_output = torch.cat([encoder_output, suffix], dim=0).to(cfg.DEVICE)

        output_sentence = []
        output, hidden, _ = decoder(cfg.SOS, hidden, encoder_output)
        predict = torch.argmax(output, dim=1).item()
        output_sentence.append(predict)
        if predict != cfg.EOS:
            for i in range(1, cfg.MAX_SEQ_LENGTH):
                output, hidden, _ = decoder(predict, hidden, encoder_output)
                predict = torch.argmax(output, dim=1)[0].item()
                output_sentence.append(predict)
                if predict == cfg.EOS:
                    break

        english_sentence = [english_word_list.index_to_word[i] for i in english_sentence]
        output_sentence = [english_word_list.index_to_word[i] for i in output_sentence]

        smoother = SmoothingFunction()
        bleu_score = sentence_bleu([english_sentence], output_sentence, smoothing_function=smoother.method1)
        sum_bleu_score += bleu_score

    avg_bleu_score = sum_bleu_score / len(test_set)
    return avg_bleu_score

def main():
    chinese_word_list, english_word_list, language_pairs, lp_to_num_seq = load_and_preprocess_data(cfg.DATASET_PATH)
    input_wl_size = chinese_word_list.num_words
    output_wl_size = english_word_list.num_words
    train_set, test_set = train_test_split(lp_to_num_seq)

    train(train_set, input_wl_size, output_wl_size, cfg.EMBED_DIM, cfg.LEARNING_RATE, num_epochs=cfg.NUM_EPOCHS)

    test(test_set, chinese_word_list, english_word_list)

    avg_bleu_score = compute_bleu_score(test_set, chinese_word_list, english_word_list)
    print(f'BLEU分数：{avg_bleu_score}\n')

    show_attn(test_set, chinese_word_list, english_word_list)

if __name__ == '__main__':
    main()