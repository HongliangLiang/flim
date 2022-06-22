import torch
from tqdm import tqdm, trange
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AdamW
import transformers
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random
from sklearn import metrics
import torch.nn as nn
import argparse
import torch.distributed as dist
import torch.utils.data.distributed
import os
import numpy as np
from sklearn.metrics import f1_score
from utils import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer)
from CodeBertModel import TextCNNClassifer_pair
import os
# for i in range(58):
#     os.system('python3 run_classifier.py --model_type roberta --model_name_or_path microsoft/codebert-base --task_name codesearch --do_predict --output_dir ../data/codesearch/test/ --data_dir ../data/codesearch/test/ --max_seq_length 512 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --learning_rate 1e-5 --num_train_epochs 8 --test_file aspectj_'+str(i)+'.txt  --pred_model_dir ./models/java/checkpoint-best/ --test_result_dir ./results/java/batch_result_'+str(i)+'.txt')
from more_itertools import chunked
def calculate_same_value(labels_sorted, test_p_sorted, start_pos):
    i = start_pos
    num_same = 0
    num_p = 0
    while test_p_sorted[start_pos] == test_p_sorted[i]:
        num_same = num_same + 1
        if labels_sorted[i] ==1 : num_p = num_p + 1
        i = i + 1
        if i == len(labels_sorted ): break
    return num_p, num_same
def eval_mrr(test_p, labels):#在第二维相似度得分，真实标签
    test_p_sorted = test_p
    test_p_index = sorted(range(len(test_p_sorted)), key=lambda k: test_p_sorted[k], reverse=True)  # 降序排序
    test_p_sorted = sorted(test_p, reverse=True)

    labels_sorted = []
    for index in test_p_index:
        labels_sorted.append(labels[index])

    top_num = 10
    top10rank = 0
    for i in range(top_num):
        if (labels_sorted[i] == 1):
            '''
            num_p, num_s = calculate_same_value(labels_sorted, test_p_sorted, i)
            num_r = top_num - i
            if num_p > (num_s-num_r):
                top10rank = 1
                break
            v1 = perm(num_s-num_r, num_p)*perm(num_s-num_p,num_s-num_p)
            v2 = perm(num_s,num_s)
            top10rank = 1-(float)((float)(v1)/(float)(v2))
            if top10rank > 1: top10rank=1
            if top10rank!=top10rank: top10rank=1
            break

    return top10rank

    '''
            top10rank = 1
            break
    num_p, num_s = calculate_same_value(labels_sorted, test_p_sorted, 10)
    if (num_p >= 1): top10rank = 1  # 统计在第十位并列排名相同的文件中，是否含有相关文件

    MRRrank = 0.0
    for i in range(len(labels_sorted)):
        if (labels_sorted[i] == 1):
            MRRrank = MRRrank + float(1 / (i + 1))

    MAPrank = 0.0
    pos_num = 0
    for i in range(len(labels_sorted)):
        if (labels_sorted[i] == 1):
            pos_num = pos_num + 1
            MAPrank = MAPrank + float(pos_num / (i + 1))
    if pos_num==0:
        print('出现不存在pos的例子')
        pos_num=1
    MAPrank = float(MAPrank / pos_num)
    MRRrank = float(MRRrank / pos_num)

    return top10rank, MRRrank, MAPrank

def simple_accuracy(preds, labels):
#     print(type(preds),type(labels))
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)

    return acc_and_f1(preds, labels)
#设置种子，为了结果复现
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

class args(object):
    """A single set of features of data."""

    def __init__(self):
        self.model_type = 'roberta'
        self.output_dir='/data/hdj/data/CodeBERT/codesearch/models/java'
        self.test_result_dir='/data/hdj/tracking_buggy_files/joblib_memmap_swt/notebook_0531_test_3hard_2_data.txt'
        self.start_epoch=0
        self.config_name=''
        self.model_name_or_path=None
        self.task_name='codesearch'
        self.tokenizer_name=''
        self.model_name_or_path='microsoft/codebert-base'
        self.do_lower_case=True
        self.seed=42
        self.gradient_accumulation_steps=1
        self.weight_decay=0.0
        self.max_grad_norm=1.0
        self.learning_rate=5e-5#1e-6
        self.adam_epsilon=1e-8
        self.warmup_steps=0
        self.max_steps=-1
        self.num_train_epochs=4
# class args(object):
#     """A single set of features of data."""

#     def __init__(self):
#         self.model_type = 'codesearch'
#         self.output_dir='/data/hdj/data/CodeBERT/codesearch/models/java'
#         self.test_result_dir='/data/hdj/data/CodeBERT/codesearch/results/java/test_v1_0414.txt'
#         self.start_epoch=0
#         self.num_train_epochs=1
#         self.model_type='roberta'
args=args()
#带权重的交叉熵
# weights=torch.tensor([0.4,0.6]).cuda()
# loss_fun=CrossEntropyLoss(weight=weights)
#不带权重的交叉熵
# loss_fun=CrossEntropyLoss()
#设置种子 复现结果
set_seed(args)


# 加载数据并保存在cache里
def load_and_cache_examples(file_prefix, data_dir, train_file, dev_file, test_file, task, tokenizer, max_seq_length,
                            ttype='train'):
    '''
        data_dir:数据目录
        train_file:训练文件
        dev_file:验证文件
        test_file:测试文件
        task:任务名 固定为codesearch
        tokenizer:分词器
        max_seq_length:最大序列长度 convert_examples_to_features使用到了
        ttype:类型 train dev test

        return：包装好的数据集 [all_input_ids, all_input_mask, all_segment_ids, all_label_ids]
    '''
    processor = processors[task]()  # 拿到CodesearchProcessor
    output_mode = output_modes[task]  # 输出模式 固定为 classification
    # if os.path.exists(cached_features_file):

    label_list = processor.get_labels()  # 固定为 ["0","1"]

    '''
        example结构：[InputExample(uid 'test-1',report,code,label)]
    '''
    if ttype == 'train':
        file_name = train_file.split('.')[0]
    elif ttype == 'dev':
        file_name = dev_file.split('.')[0]
    elif ttype == 'test':
        file_name = test_file.split('.')[0]
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        ttype,
        file_name,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(max_seq_length),
        str(task)))
    try:
        features = torch.load(cached_features_file)
        if ttype == 'test':
            examples, instances = processor.get_test_examples(data_dir, test_file, file_prefix)
    except:
        examples = None
        if ttype == 'train':
            examples = processor.get_train_examples(data_dir, train_file, file_prefix)
        elif ttype == 'dev':
            examples = processor.get_dev_examples(data_dir=data_dir, dev_file=dev_file, prefix=file_prefix)
        elif ttype == 'test':
            # 如果是test的话，instances就是[每一行是一条代测试的数据]
            examples, instances = processor.get_test_examples(data_dir, test_file, file_prefix)

        '''
            example:[[uid,report,code,label],[]...]
            label_list:['0','1']
            max_seq_length:200
            tokenizer:分词器
            output_mode:输出模式 固定为 classification
            cls_token:用来分类的token     '<s>', 0, 
            sep_token:用来分割语句的token '</s>', 2

            return：[  [       ( input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id)
                        ]
                    ]
                    instances：[["0","5222","i am a noy","public class"],[],,]
        '''
        #     features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode,
        #                                             cls_token=tokenizer.cls_token,
        #                                             sep_token=tokenizer.sep_token,
        #                                            )
        features = convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    feature = torch.tensor([f.feature for f in features], dtype=torch.float16)
    print('human_feature :', feature)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, feature)
    if (ttype == 'test'):
        return dataset, instances
    else:
        return dataset


tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

#数据集的配置参数
file_prefix='/data/hdj/tracking_buggy_files/swt/swt'
data_dir = "/data/hdj/tracking_buggy_files/joblib_memmap_swt/data/"
# test_dir="/data/hdj/tracking_buggy_files/joblib_memmap_swt/data/" #注意目前测试集的目录是单独的，更换测试数据需要更改
# train_file = "aspectj_train_small.txt"
# dev_file = "aspectj_val_small.txt"
test_file = "test_3hard_2_data.txt"#目前只使用一个例子来进行测试，后期增加数据量
# train_file='aspectj_train_oversample.txt'
train_file='train_3hard_1_train_data.txt'
# dev_file = "aspectj_val_oversample.txt"
dev_file = "train_3hard_1_valid_data.txt"
task_name = "codesearch"
max_seq_length = 256 #关键参数设置 整个序列的最大长度
#加载数据集
#这里load_and_cache_examples 是没有将数据进行缓存的，和codeBert里的同名函数意义不一样，所以统一按照每次都读取数据这样的模式来处理
train_set = load_and_cache_examples(file_prefix,data_dir,train_file,dev_file,test_file,task_name,tokenizer,max_seq_length,'train')
val_set = load_and_cache_examples(file_prefix,data_dir,train_file,dev_file,test_file,task_name,tokenizer,max_seq_length,'dev')
# # test_set, instances=load_and_cache_examples(test_dir,train_file,dev_file,test_file,task_name,tokenizer,max_seq_length,'test')
# #将数据进行batch_size设置
batch_size = 64 #关键参数
training_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
#测试集顺序不应该打乱，不然后期就对不上了 shuffle的作用是在枚举的时候才会改变顺序
# testing_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

#模型1的实现
device = 'cuda' if torch.cuda.is_available() else 'cpu'
google_v=False #当为true时，使用RobertClassification模型
num_labels=2

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                     num_labels=num_labels, finetuning_task=args.task_name)
if google_v:
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                    config=config)
    print('使用RobertClassification 模型')
else:
    model = TextCNNClassifer_pair()
    print('使用TextCNNClassifier模型')
_ = model.to(device)



if google_v or True:
    #Google: 使用的优化器和迭代器
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #Google
else:
#     hdj: textCNNClassifier 使用的优化器和计划器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-05, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
#     hdj
# 开始验证模型的有效性 把模型预测结果连同数据一起保存起来 然后再去判断top mrr mAP
# 流程 1.训练过程中 每个10轮去在val集上验证 比较f1值是否优于之前，是就保存结果到best，2.每次保存上一次验证的结果 3.最后再在test集上进行测试

# check 词表是否与之前model有很大的不同 如果有那么该怎么改善，之前embedding矩阵是否就作废了
def evaluate(args, model, eval_loader, instances, tokenizer, checkpoint=None, prefix="", mode='dev'):
    '''
        args:模型训练时关键参数
        model：待验证的模型
        eval_loader:待验证的数据集 DataLoader 类型的参数
        instances: 若为test模式，则将原始数据输入进来，然后将预测结果一起保存，等待后面计算TOP MAP MRR使用
        tokenizer:分词器
        checkpoint: 检查点
        prefix:前缀
        mode: 模式 分为 dev 和 test（需要保存数据）

        return: dict{'acc':,'f1':,'acc_f1'} 主要关注f1的值 越大说明正例的预测越准
    '''
    # 待返回的验证结果
    results = {}
    eval_loss = 0.0  # 验证损失累加
    nb_eval_steps = 0  # 验证批次
    preds = None  # 保存预测的值
    out_label_ids = None  # 保存真实标签target
    print('开始验证。。。')
    for _, data in enumerate(eval_loader, 0):  # start=0 默认就是0
        with torch.no_grad():
            ids = data[0].to(device, dtype=torch.long).cuda(non_blocking=True)
            mask = data[1].to(device, dtype=torch.long).cuda(non_blocking=True)
            token_type_ids = None  # token_type_ids = data[2].to(device, dtype=torch.long).cuda(non_blocking=True) 因为模型不支持2维设置 所以不再输入这个参数
            targets = data[3].to(device, dtype=torch.long).cuda(non_blocking=True)
            feature = data[4].to(device, dtype=torch.long).cuda(non_blocking=True)
            if google_v:
                pred = model(input_ids=ids, attention_mask=mask, token_type_ids=None, labels=targets)
                loss = pred[0]
            else:
                pred = model(ids, mask,feature)  # 输出是[batch_size,2]
                # print(pred, target)
                # loss = loss_fun(pred, targets)
                loss=floss(pred,targets)
            # pred_choice = pred.max(1)[1]
            #             correct = pred_choice.eq(targets).cpu().sum()
            #             metrics = compute_metrics(pred_choice.cpu().numpy(), targets.cpu().numpy())
            #             print('[',epoch,': ',_,'/',num_batch,'] ',blue('val')," loss :%.4f" % loss.item(),' , accuracy: ',correct.item() / float(batch_size) ,'true nums:',sum(targets.cpu().numpy()),' ration :',sum(targets.cpu().numpy())/len(targets),"acc :",metrics['acc']," f1 :",metrics['f1'])
            #             loss_list_val.append(loss.item())

            eval_loss += loss.item()
            if (_ % 100 == 0):
                print(_, '/', len(eval_loader),loss)
        nb_eval_steps += 1
        # 将preds值和标签进行保存
        if preds is None:
            if google_v:
                preds = pred[1].detach().cpu()
            else:
                preds = pred.detach().cpu()
            out_label_ids = targets.detach().cpu()
        else:
            if google_v:
                preds = np.append(preds, pred[1].detach().cpu().numpy(), axis=0)
            else:
                preds = np.append(preds, pred.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, targets.detach().cpu().numpy(), axis=0)
    print('验证结束，开始计算指标',eval_loss)
    # 计算指标，并更新结果

    # ndarra转换为tensor类型
    preds = torch.tensor(preds)
    out_label_ids = torch.tensor(out_label_ids)
    #     print(type(preds),preds.shape,preds)
    #     print(type(out_label_ids),out_label_ids.shape,out_label_ids)
    pred_choice = preds.max(1)[1]  # [batch_size,2]
    correct = pred_choice.eq(out_label_ids).cpu().sum()  # eq需要tensor类型的数据
    # 但是compute_metrics 需要numpy的数据类型 真烦人
    metrics = compute_metrics(pred_choice.numpy(), out_label_ids.numpy())
    results.update(metrics)
    # 计算完成

    # 下面分dev 和 test 模式来完成数据模型的保存工作
    if (mode == 'dev'):
        # 这里只是把result字典里acc和f1给保存到文件
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a+") as writer:
            print("***** Eval results {} *****".format(prefix))
            writer.write('evaluate %s\n' % checkpoint)
            for key in sorted(results.keys()):
                writer.write("%s = %s\n" % (key, str(results[key])))
    elif (mode == 'test'):
        # test结果目录 将test数据连同预测结果进行保存
        if (instances == None or len(instances) == 0):
            print('输入的测试数据有问题 为None 或者 长度为0')
        # 如果目录不存在就创建
        output_test_file = args.test_result_dir
        output_dir = os.path.dirname(output_test_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 检验创建目录完成
        with open(output_test_file, "w") as writer:
            print("***** Output test results *****")
            preds = F.softmax(torch.tensor(preds))
            # print('检查是否softmax了 : ',preds)
            all_logits = preds.tolist()
            for i, logit in tqdm(enumerate(all_logits), desc='Testing'):  # desc是进度条的标题
                # instances即代表test集里每一行数据的值
                instance_rep = '<CODESPLIT>'.join(
                    [item.encode('ascii', 'ignore').decode('ascii') for item in instances[i]])
                writer.write(instance_rep + '<CODESPLIT>' + '<CODESPLIT>'.join([str(l) for l in logit]) + '\n')
            # 打印验证的结果
            for key in sorted(results.keys()):
                print("%s = %s" % (key, str(results[key])))
    return results
from torch import nn
import torch
from torch.nn import functional as F

class focal_loss(nn.Module):
    def __init__(self, alpha=[0.25,0.75], gamma=2, num_classes = 2, size_average=False):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha: 阿尔法α,类别权重.
                    当α是列表时,为各类别权重；
                    当α为常数时,类别权重为[α, 1-α, 1-α, ....],
                    常用于目标检测算法中抑制背景类,
                    retainnet中设置为0.25
        :param gamma: 伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes: 类别数量
        :param size_average: 损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes
            # α可以以list方式输入,
            # size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1  #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print("Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用.".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds: 预测类别. size:[B,N,C] or [B,C]    分
                别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
#         print('preds :',preds)
        self.alpha = self.alpha.to(preds.device)
        # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然也可以使用log_softmax,然后进行exp操作)
        preds_softmax = F.softmax(preds, dim=1)
#         print('preds_softmax :',preds_softmax)
        preds_logsoft = torch.log(preds_softmax)
#         print('preds_logsoft :',preds_logsoft)
        # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))
#         print('preds_softmax :',preds_softmax)
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
#         print('preds_logsoft :',preds_logsoft)
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
floss=focal_loss()

# 模型应该保存效果最好的那个，而不是训练到最后的那个模型
print("training!!!")
blue = lambda x: '\033[94m' + x + '\033[0m'
epoch = 1
best_f1 = 0.0  # 保存最好的f1的模型
model.zero_grad()
set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
# 这里就是在指定 一共训练多少Epoch [start_epoch,num_train_epcchs]
t_total = len(training_loader) // args.gradient_accumulation_steps * args.num_train_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

train_iterator = trange(args.start_epoch, int(args.num_train_epochs), desc="Epoch")
model.train()
for idx, p in enumerate(train_iterator):
    loss_list = []
    acc_list = []
    loss_list_val = []
    acc_list_val = []
    num_batch = len(train_set) / batch_size  # 总共训练次数
    for _, data in enumerate(training_loader, 0):  # start=0 默认就是0
        ids = data[0].to(device, dtype=torch.long).cuda(non_blocking=True)
        mask = data[1].to(device, dtype=torch.long).cuda(non_blocking=True)
        token_type_ids = None
        #     token_type_ids = data[2].to(device, dtype=torch.long).cuda(non_blocking=True) 因为模型不支持2维设置 所以不再输入这个参数
        targets = data[3].to(device, dtype=torch.long).cuda(non_blocking=True)
        feature = data[4].to(device, dtype=torch.long).cuda(non_blocking=True)
        #         print('feature shape: ',feature.shape)
        #         print(feature)
        if google_v:
            #             outputs = model(ids, mask, token_type_ids)
            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=None, labels=targets)
            loss = outputs[0]  # google loss
            #             print(outputs)
            #             loss_hdj = loss_fun(outputs[1], targets)
            #             print('google loss ',loss,' loss_hdj :',loss_hdj)
            pred_choice = outputs[1].max(1)[1]
        else:
            outputs = model(ids, mask, feature)
            # 使用floss
            loss = floss(outputs, targets)
            # 使用floss
            # 使用ghmloss
            #             ghm_target = targets.view(-1, 1)
            #             ghm_target=ghm_target.cpu()
            #             ghm_target = torch.LongTensor(ghm_target)
            #             # print("before target :",targets)
            #             ghm_target = torch.zeros(len(ghm_target), 2).scatter_(1, ghm_target, 1)
            #             outputs=outputs.cpu()
            #             loss=ghm(outputs,ghm_target)
            # 使用ghmloss
            # print(outputs)

            # 交叉熵loss
            #             loss = loss_fun(outputs, targets)
            # 交叉熵loss

            #         print('查看target和ouput的类型 检查max函数返回的东西是否正确',type(outputs),type(targets))
            #     print(outputs)
            # 取outputs二维中最大维的下标
            pred_choice = outputs.max(1)[1]
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if (_ + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        # targets=targets.cpu()
        correct = pred_choice.eq(targets).cpu().sum()
        metrics = compute_metrics(pred_choice.cpu().numpy(), targets.cpu().numpy())
        if _ % 100 == 0:
            print('[', epoch, ':', _, '/', num_batch, ']', "loss :%.4f" % loss.item(), ' , accuracy: ',
                  correct.item() / float(batch_size), 'true nums:', sum(targets.cpu().numpy()), ' ration :',
                  sum(targets.cpu().numpy()) / len(targets), "acc :", metrics['acc'], " f1 :", metrics['f1'])
        loss_list.append(loss.item())
        acc_list.append(correct.item() / float(batch_size))

    # 下面是在训练过程中进行验证 每一轮都会验证一次，保存模型最好的 和最近一次的模型
    results = evaluate(args, model, val_loader, None, tokenizer, checkpoint=str(args.start_epoch + idx), mode='dev')
    print('验证结果 ：', results)
    # torch.save(model, '%s/cls_model_%d.pth' % ('models', epoch))
    # ./models/java/checkpoint-last
    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
    if not os.path.exists(last_output_dir):
        os.makedirs(last_output_dir)
    # model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(last_output_dir)
    torch.save(model, '%s/cls_model_%d.pth' % (last_output_dir, epoch))
    print("Saving model checkpoint to %s", last_output_dir)
    # 保存idx_file 文件 第几轮训练的记录
    idx_file = os.path.join(last_output_dir, 'idx_file.txt')
    with open(idx_file, 'w', encoding='utf-8') as idxf:
        idxf.write(str(args.start_epoch + idx) + '\n')
    # 保存idx_file完成
    # 保存优化器和计划器
    torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
    print("Saving optimizer and scheduler states to %s", last_output_dir)

    if (results['f1'] > best_f1):
        best_f1 = results['f1']
        # ./models/checkpoint-best 检查目录是否存在 否则创建
        output_dir = os.path.join(args.output_dir, 'checkpoint-best')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            # 保存model
        #         model_to_save = model.module if hasattr(model,'module') else model  # Take care of distributed/parallel training
        #         model_to_save.save_pretrained(output_dir)
        torch.save(model, '%s/cls_model_%d.pth' % (output_dir, epoch))
        #         torch.save(args, os.path.join(output_dir, 'training_{}.bin'.format(idx)))
        print("Saving model checkpoint to %s", output_dir)
        # 保存优化器的计划器
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        print("Saving optimizer and scheduler states to %s", output_dir)
# results = evaluate(args, model,val_loader,None ,tokenizer, checkpoint=str(args.start_epoch + idx),mode='dev')
#下面开启test模型性能
#加载模型进行测试
# del model
import gc
gc.collect()
test_dir="/data/hdj/tracking_buggy_files/joblib_memmap_swt/data/" #注意目前测试集的目录是单独的，更换测试数据需要更改
# test_dir='/data/hdj/data/CodeBERT/data/codesearch/test/zxing_test/zxing/'
# test_file = "test_all.txt"#目前只使用一个例子来进行测试，后期增加数据量
# test_dir="/data/hdj/data/CodeBERT/data/codesearch/test/swt_test/swt/"
test_file = "test_3hard_2_data.txt"#
# train_file = "aspectj_train_small.txt"
# dev_file = "aspectj_val_small.txt"

test_set, instances=load_and_cache_examples(file_prefix,test_dir,train_file,dev_file,test_file,task_name,tokenizer,max_seq_length,'test')
testing_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=4, drop_last=False)
#先不删除模型直接进行预测 之后再删除模型 加载保存下来的模型进行预测
test_results = evaluate(args, model,testing_loader,instances ,tokenizer, checkpoint=str(args.start_epoch + idx),mode='test')
print(test_results)

#加载模型 进行验证 保存结果 22;43
# del model
# import gc
# gc.collect()
# epoch=1
# idx=1
# output_dir = os.path.join(args.output_dir, 'checkpoint-best')
# model=torch.load('%s/cls_model_%d.pth' % (output_dir, epoch))
# test_results = evaluate(args, model,testing_loader,instances ,tokenizer, checkpoint=str(args.start_epoch + idx),mode='test')
# print(test_results)