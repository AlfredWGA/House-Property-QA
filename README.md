# House-Property-QA
运行src/run_model.py 即可开始训练（注：参数 data_type 目前只能使用 qa_data3, 其它的还有BUG）

## TODO

* 使用其他损失函数（pair-wise, list-wise, triplet loss）
* 先通过 CMRC 微调，再通过 HouseQA 二次微调，控制 CMRC 的正负样本比例（接近 HouseQA 的正负比例）
* LCQMC!!!!!
* 增加交叉验证的 k 值
* 爬取房产论坛数据，扩大数据集
* 使用两阶段微调，point-wise (CrossEntropyLoss) + pair-wise (HingeLoss)

## 训练 Regression 模型需要修改的位置

```python
all_pred = np.zeros(shape=(len(df_train)))

train_loader = DataLoader(train_set,
                        batch_size=kwargs['batch_size'],
                        # shuffle=True  # 如果使用分类训练，建议True -->
                        )

# criterion = torch.nn.MarginRankingLoss(margin=1.0)
# criterion = torch.nn.MSELoss()
criterion = torch.nn.CrossEntropyLoss()

# Train
# CrossEntropy
# loss = criterion(model_outputs, y)
# MSE
# loss = criterion(model_outputs, y.float().unsqueeze(-1))

# 使用 HingeLoss
train_qa_id_sub = sample[2].numpy()
loss = get_hinge_loss(model_outputs, train_qa_id_sub, criterion)

# Valid
# MSELoss
# loss = criterion(model_outputs, y_true.float().unsqueeze(-1)).cpu().detach().item()
# HingeLoss
# valid_qa_id_sub = sample[2].numpy()
# loss = get_hinge_loss(model_outputs, valid_qa_id_sub, criterion)
# y_pred = model_outputs.cpu().detach().squeeze(-1).numpy()
# CrossEntropy
loss = criterion(model_outputs, y_true).cpu().detach().item()
y_pred = F.softmax(model_outputs.cpu().detach(), dim=1).numpy()

# 如果使用回归模型
# valid_f1, thr = search_f1(valid_true, valid_pred)
# logger.info("Epoch {}, valid loss {:.5f}, valid f1 {:.4f}".format(epoch, valid_loss, valid_f1)))

torch.save(
            {
                "model_name": "BertForHouseQA",
                "epoch": epoch,
                "valid_loss": valid_loss,
                "valid_f1": valid_f1,
                "model_state_dict": model.state_dict(),
                # "optimizer_state_dict": optimizer.state_dict(),
                # "thr": thr
                # 'scheduler_state_dict': scheduler.state_dict()
            },

# MSELoss
# all_f1, all_thr = search_f1(all_true, all_pred)
# logger.info("All f1 {:.4f}, all thr {:.4f}".format(all_f1, all_thr))


all_f1, checkpoint_path = train_pytorch(batch_size=128, valid_batch_size=512, epoch=15, lr=2e-5, weight_decay=1e-3, 
                                        n_splits=10, patience=8, device=1, inputs=inputs, 
                                        outputs=outputs, test_inputs=test_inputs)
```

## 切换早停指标需要修改的代码

```python
stopper = EarlyStopping(patience=kwargs['patience'], mode='max')    # 'max' for F1, 'min' for loss

stop_flag, best_flag = stopper.step(valid_f1)
```
