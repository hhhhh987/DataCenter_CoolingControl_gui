'''
预测脚本
'''
import jieba
import torch
import config
from model import create_model


def predict(text,model,word2index,index2word):
    #为防止每次调用功能都加载一遍词表和模型 将1，2，3放入循环外部

    #4处理输入
    tokens=jieba.cut(text)
    indexs=[word2index.get(token,0) for token in tokens]
    input_tensor=torch.tensor([indexs],dtype=torch.long)

    #5预测逻辑
    model.eval()
    output=model(input_tensor)
    #output.shape:[batch_size,vocab_size]
    print(output)
    top5index=torch.topk(output,5).indices
    #top5index.shape:[batch_size,5]
    print(top5index)

    top5list=top5index.tolist()
    print(top5list)
    top5tokens=[index2word[index] for index in top5list[0]]

    return top5tokens

def run_predict():
    #准备资源
    # 1设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2词表
    with open(config.MODELS_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        # vocab_list=f.read().splitlines()
        vocab_list = [line.strip() for line in f.readlines()]
    word2index = {word: index for index, word in enumerate(vocab_list)}
    index2word = {index: word for index, word in enumerate(vocab_list)}

    # 3加载模型
    model = create_model(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model1.pth'))

    print('欢迎使用')
    input_history=''
    while True:
        user_input=input('请输入：')
        if user_input in ['q','quit']:
            print('结束')
            break
        if user_input.strip()=='':
            print('请输入正确内容')
            continue
        input_history+=user_input
        print('输入历史：',input_history)
        top5token=predict(input_history,model,word2index, index2word)
        print('预测结果',top5token)


if __name__ == '__main__':
    run_predict()