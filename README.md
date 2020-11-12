# House-Property-QA
运行src/run_model.py 即可开始训练（注：参数 data_type 目前只能使用 qa_data3, 其它的还有BUG）

## TODO

* 使用其他损失函数（pair-wise, list-wise, triplet loss）
* 先通过 CMRC 微调，再通过 HouseQA 二次微调，控制 CMRC 的正负样本比例（接近 HouseQA 的正负比例）
* LCQMC!!!!!
* 增加交叉验证的 k 值
* 爬取房产论坛数据，扩大数据集
* 使用两阶段微调，point-wise (CrossEntropyLoss) + pair-wise (HingeLoss)