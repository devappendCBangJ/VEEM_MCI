import time

import numpy as np

import torch
from torchmetrics.aggregation import MeanMetric

# Define training loop
def train(loader, learn_type, input_size, model, optimizer, scheduler, loss_fn, metric_fn, device):
    # 모델 학습
    model.train()
    
    # 변수 초기화
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()

    for key, value in loader:
        """
        print("key : ", key)
        print("value : ", value)
        print("value[0] : ", value[0])
        print("value[0][0] : ", value[0][0])
        print("value[0][1] : ", value[0][1])
        print("value[0][2] : ", value[0][2])
        """
        # 데이터셋 정보 unzip
        df_drop8_torch = value[0][0]
        df_sequence_length = value[0][1]
        target_label = value[0][2]
        """
        print("target_label : ", target_label)
        print("target_label.shape : ", target_label.shape)
        """
        
        # input, target 불러오기
        df_drop8_torch = df_drop8_torch.reshape(-1, df_sequence_length, input_size).to(device) # 형식 통일화!!!
        if learn_type == "regression":
            target_label = target_label.to(device, dtype=torch.float32) # 자료형 통일화!!!
        else:
            target_label = target_label.to(device, dtype=torch.int64) # 자료형 통일화!!!

        # loss, acc 계산
        output = model(df_drop8_torch)
        """
        print("output : ", output)
        print("output.shape : ", output.shape)
        """
        
        loss = loss_fn(output, target_label)
        if learn_type == "regression":
            pass
        else:
            metric = metric_fn(output, target_label)
        
        """
        if learn_type == "regression":
            print("ㅡㅡㅡㅡㅡshapeㅡㅡㅡㅡㅡ")
            print("output.shape : ", output.shape)
            print("target_label.shape : ", target_label.shape)
            print("ㅡㅡㅡㅡㅡoutput vs labelㅡㅡㅡㅡㅡ")
            print("output : ", output)
            print("target_label : ", target_label)
        """
#         else:
#             # 최대확률 예측 라벨 추출
#             _, predicted = torch.max(output, 1)
#             print("predicted : ", predicted)
            
#             # 비교
#             correct += (predicted == labels).sum().item()
#             print("correct : ", correct)
#             print(f"Accuracy : {100 * correct / total} %") 

        # loss, optimizer 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_mean.update(loss.to('cpu'))
        if learn_type == "regression":
            pass
        else:
            metric_mean.update(metric.to('cpu'))

        # scheduler 업데이트
        scheduler.step()

    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}

    return summary

# Define evaluation loop
def evaluate(loader, learn_type, input_size, model, loss_fn, metric_fn, device):
    # 모델 평가 # Dropout, Batchnorm 등 실행 x
    model.eval()
    
    # 변수 초기화
    loss_mean = MeanMetric()
    metric_mean = MeanMetric()
    # print(loss_mean) # 확인용 코드
    
#     # 변수 선언
#     if(args.learn_type == "regression"):
#         pass
#     else:
#         correct = 0
#         total = len(test_rpt_all_dataset)
    with torch.no_grad():
        for key, value in loader:
            """
            print("key : ", key)
            print("value : ", value)
            print("value[0] : ", value[0])
            print("value[0][0] : ", value[0][0])
            print("value[0][1] : ", value[0][1])
            print("value[0][2] : ", value[0][2])
            """
            # 데이터셋 정보 unzip
            df_drop8_torch = value[0][0]
            df_sequence_length = value[0][1]
            target_label = value[0][2]
            """
            print("target_label : ", target_label)
            print("target_label.shape : ", target_label.shape)
            """
            
            # input, target 불러오기
            df_drop8_torch = df_drop8_torch.reshape(-1, df_sequence_length, input_size).to(device) # 형식 통일화!!!
            if learn_type == "regression":
                target_label = target_label.to(device, dtype=torch.float32) # 자료형 통일화!!!
            else:
                target_label = target_label.to(device, dtype=torch.int64) # 자료형 통일화!!!

            # loss, acc 계산
            output = model(df_drop8_torch)
            loss = loss_fn(output, target_label)
            if(learn_type == "regression"):
                pass
            else:
                metric = metric_fn(output, target_label)
            
            loss_mean.update(loss.to('cpu'))
            if learn_type == "regression":
                pass
            else:
                metric_mean.update(metric.to('cpu'))

            if learn_type == "regression":
                """
                print("ㅡㅡㅡㅡㅡshapeㅡㅡㅡㅡㅡ")
                print("output.shape : ", output.shape)
                print("target_label.shape : ", target_label.shape)
                """
                print("ㅡㅡㅡㅡㅡoutput vs labelㅡㅡㅡㅡㅡ")
                print("output : ", output)
                print("target_label : ", target_label)
#             else:
#                 # 최대확률 예측 라벨 추출
#                 _, predicted = torch.max(output, 1)
#                 print("ㅡㅡㅡㅡㅡpredicted vs labelㅡㅡㅡㅡㅡ")
#                 print("predicted : ", predicted)
#                 print("target_label : ", target_label)

#                 # 비교
#                 correct += (predicted == target_label).sum().item()
#                 """
#                 print("correct : ", correct)
#                 """
    
    summary = {'loss': loss_mean.compute(), 'metric': metric_mean.compute()}
    # print(summary) # 확인용 코드
    
    return summary

# 모델 학습 소요시간
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    
    return elapsed_mins, elapsed_secs