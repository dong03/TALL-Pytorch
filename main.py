from utils import *
import dataset

writer = SummaryWriter(logdir = tensoboard_logdir)
main_model = model.MAIN_model()
random.seed(int(time.time()))

try:
    main_model.load_state_dict(torch.load(model_pth))
    print("load finish!")
except:
    pdb.set_trace()

main_model = main_model.to(device)
main_model.train()
criterion_res = torch.nn.SmoothL1Loss().to(device)
criterion_ali = Loss_ali(1.0/(batch_size)).to(device)
optimizer = torch.optim.Adam(main_model.parameters(),lr=lr)

board_num = 0

train_set = dataset.TrainingDataSet(train_visual_feature_dir,train_csv_path,batch_size,pth_clip_sentence_pairs_iou)
test_set = dataset.TestingDataSet(test_visual_feature_dir, test_csv_path,batch_size)

for epoch in range(EPOCH):
    fpnp, sentence, offset = train_set.next_batch_iou()
    optimizer.zero_grad()
    fpnp = torch.from_numpy(fpnp).to(device).type(torch.float32)
    sentence = torch.from_numpy(sentence).to(device).type(torch.float32)
    offset = torch.from_numpy(offset).to(device).type(torch.float32)

    triple_score,mask1,mask2 = main_model(fpnp,sentence)
    mask1 = mask1.to(device)
    mask2 = mask2.to(device)
    score_mat = triple_score[:,:,0]
    shift = triple_score[:,:,1:]
    shift_ = torch.zeros_like(offset).to(device)
    for i in range(batch_size):
        shift_[i] = shift[i][i]

    loss_res = criterion_res(shift_,offset)
    loss_ali = criterion_ali(score_mat,mask1,mask2)
    loss = loss_ali + 0.01*loss_res

    if loss<0 or torch.isinf(loss) or torch.isnan(loss):
        pdb.set_trace()
    loss.backward()
    optimizer.step()

    train_acc = 1.0 *torch.sum(torch.argmax(score_mat,dim=1) == torch.argmin(mask1,dim=1))/batch_size
    writer.add_scalars('tall:%s'%nname,{"train loss":loss.item(),'train acc':train_acc},board_num)
    board_num += 1

    if epoch%1000 == 0:

        torch.save(main_model.state_dict(), model_pth)
        main_model.eval()
        do_eval(main_model,test_set)
        main_model.train()
