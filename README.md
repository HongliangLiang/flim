# buglocalization_code
use git to manage different version code

背景及当前别的工具存在的问题
    
    挑战1: 源代码文件通常包含大量令牌。如表1-1所示，六个广泛使用的软件项目中的源代码包含 1585个令牌（在 Eclipse UI 中）到 5958 个令牌（在 SWT 中）。 当面对如此长的代码时，CNN网络无法捕捉到长距离代码的交互特征，而RNN网络（例如LSTM、GRU）容易出现梯度消失或爆炸，导致语义信息丢失。此外，RNN 网络由于其顺序相关的特性而具有很高的计算复杂性。

    挑战2: 训练一个深度学习模型需要大量的语料库，尽管现有的软件项目可用的bug报告通常以数千计。例如，上述项目中只有 593 个（对于 AspectJ）到 6495 个（对于 Eclipse UI）bug报告。但是这样的数据集不足以训练深度学习模型，而构建合适的数据集需要专业知识和大量的时间。

    挑战 3: 虽然基于信息检索的方法简单易懂，但它们很难跨越bug报告和程序之间的词法鸿沟；虽然基于深度学习的方法可以利用程序和bug报告之间丰富的语义信息，但它们容易受到噪声数据的影响并且无法对元数据特征（例如缺陷修复的新近度和频率）。 因此，本工作探索了一种将两种技术的优点结合起来的方法。

本工具的创新、贡献和解决的问题

    1）	本工作提出了一个新的名为 FLIM 的bug定位框架，它使用微调的语言模型在函数级提取代码语义特征，然后利用learning-to-rank的方法融合IR特征和聚合后的函数级语义交互特征来计算自然语言和编程语言之间的相似性。本工作将源代码文件分解为函数，并从细粒度的角度推断bug报告和函数之间的关系，这使本工作能够利用源代码文件中的完整语义信息，而不是截断或总结的语义信息。
    2）	本工作将bug定位视为代码搜索任务并提出一个函数粒度的数据集构建策略，并使用该策略构建的小规模数据集微调语言模型（CodeBERT[16]）。
    3）	在广泛使用的六个软件项目上的实验结果表明，FLIM 在三个指标（即 TOP-K、MRR、MAP）上定位缺陷文件方面明显优于六种最先进的方法。


使用方法及验证方法

    微调语言模型
        
        lang=java
        aspectj **************
        python3 run.py     --output_dir=./saved_models/$lang     --config_name=microsoft/codetive.py aspectj/aspectj  aspectj > log_220120_aspectj_crossproject_1-19-37-38 2>&1 &bert-base     --model_name_or_path=microsoft/codebert-base     --tokenizer_name=microsoft/codebert-base     --do_train     --train_data_file=dataset/$lang/aspectj/train_cross.jsonl     --eval_data_file=dataset/$lang/aspectj/valid.jsonl     --test_data_file=dataset/$lang/aspectj/test.jsonl     --codebase_file=dataset/$lang/aspectj/codebase_cross.jsonl     --num_train_epochs 10     --code_length 256     --nl_length 128     --train_batch_size 32     --eval_batch_size 64     --learning_rate 2e-5     --seed 123456 2>&1| tee saved_models/$lang/aspectj/train.log
    特征融合
        nohup python3 cal_vec.py > 1018_log_tomcat_calvec_tar 2>&1 &
        
        nohup python3 save_birt.py > 1018_log_tomcat_merge_feature_tar 2>&1 &

        ./load_data_to_joblib_memmap.py tomcat/tomcat
        
        nohup python3 train_adaptive.py tomcat/tomcat  > 1018_log_tomcat_add_3738_tar_max_mean 2>&1 &
        

目前存在的问题和未来的工作

    1）FLIM系统虽然利用细粒度的函数级语义交互特征和IR特征，不过程序的抽象语法树里包含更多程序执行时的信息，这对于理解程序的意义也有很大帮助。因此作为未来的一个研究方向，本工具可以利用基于树的深度学习模型进一步建模抽象法语树中含有的语义信息，来增强对程序的理解。
    2）目前的bug定位研究主要集中于源文件级别的bug定位，这对于开发者来说仍然需要花费很多精力去进一步定位到函数或者语句块级别，而本工作已经在函数粒度去建模bug报告和源码间的相关性，但是由于缺少函数粒度的标签数据，本工作暂时未在函数粒度去定位bug。未来一个研究方向是构建合适的函数粒度的数据集，这样可以直接用本工作方法在更为精确的函数粒度去定位bug，进一步帮助开发者提高bug定位的效率。
    3）本文是基于静态分析技术去解决bug定位问题，而基于动态技术由于其定位粒度较细，可以达到元素级别，因此未来可以在两个方面尝试结合这两种方法。第一种是在特征维度方面，对源文件提取动态执行的特征并融合到语义特征和IR特征中，提升bug定位的准确性。另一种则在执行流程上进行结合，对基于静态分析技术检索得到的可疑源文件进一步执行基于动态分析技术，按照可疑值倒序返回可疑元素列表，这样也可以得到细粒度的bug定位结果，方便开发者定位问题。

其他