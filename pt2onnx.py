from models import *
from test_model import load_model, board_normalization, get_encoded_state
import torch
import numpy as np
import onnx
from onnx import shape_inference
import onnxruntime

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

model_path = 'files/model/'
easy_model = load_model('easy')
normal_model, value_model = load_model('normal')
hard_model = load_model('hard')
easy_model.eval()
normal_model.eval()
value_model.eval()
hard_model.eval()

state = [
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,2,0,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,2,1,0,0,0]
]
state = np.array(state)
state = board_normalization(state, 'CNN', 2)
encoded_state = torch.tensor(get_encoded_state(state).squeeze()).unsqueeze(0)
# print(state)
print("prev easy:",easy_model(encoded_state))
print("prev normal:",normal_model(encoded_state))
print("prev value:",value_model(state))
print("prev hard:",hard_model(encoded_state))


batch_size = 1    # 임의의 수
# 모델에 대한 입력값
x = encoded_state
y = easy_model(x)

# 모델 변환
torch.onnx.export(easy_model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  model_path+"easy/"+"easy_model.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})

onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model_path+"easy/"+"easy_model.onnx")), model_path+"easy/"+"easy_model.onnx")
onnx_model = onnx.load(model_path+"easy/"+"easy_model.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession( model_path+"easy/"+"easy_model.onnx")

# ONNX 런타임에서 계산된 결과값
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# ONNX 런타임과 PyTorch에서 연산된 결과값 비교
np.testing.assert_allclose(to_numpy(y), ort_outs[0], rtol=1e-03, atol=1e-05)
print("prev easy:",easy_model(encoded_state))
print("after easy:", ort_outs[0])
print("easy pass")

# 모델에 대한 입력값
x = encoded_state
y = normal_model(x)

# normal 모델 변환
torch.onnx.export(normal_model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  model_path+"normal/"+"normal_model.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})

onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model_path+"normal/"+"normal_model.onnx")), model_path+"normal/"+"normal_model.onnx")
onnx_model = onnx.load(model_path+"normal/"+"normal_model.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession( model_path+"normal/"+"normal_model.onnx")

# ONNX 런타임에서 계산된 결과값
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# ONNX 런타임과 PyTorch에서 연산된 결과값 비교
np.testing.assert_allclose(to_numpy(y), ort_outs[0], rtol=1e-03, atol=1e-05)
print("prev normal:",normal_model(encoded_state))
print("after normal:", ort_outs[0])
print("normal pass")



# 모델에 대한 입력값
x = encoded_state
y = hard_model(x)

# hard 모델 변환
torch.onnx.export(hard_model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  model_path+"hard/"+"hard_model.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})

onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model_path+"hard/"+"hard_model.onnx")), model_path+"hard/"+"hard_model.onnx")
onnx_model = onnx.load(model_path+"hard/"+"hard_model.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession( model_path+"hard/"+"hard_model.onnx")

# ONNX 런타임에서 계산된 결과값
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# ONNX 런타임과 PyTorch에서 연산된 결과값 비교
print("prev hard:",hard_model(encoded_state))
print("after hard:", ort_outs)
np.testing.assert_allclose(to_numpy(y[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
np.testing.assert_allclose(to_numpy(y[1]), ort_outs[1], rtol=1e-03, atol=1e-05)
print("hard pass")


# 모델에 대한 입력값
x = state
y = value_model(x)

# hard 모델 변환
torch.onnx.export(value_model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  model_path+"normal/"+"value_model.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})

onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model_path+"normal/"+"value_model.onnx")), model_path+"normal/"+"value_model.onnx")

onnx_model = onnx.load(model_path+"normal/"+"value_model.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(model_path+"normal/"+"value_model.onnx")

# ONNX 런타임에서 계산된 결과값
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# ONNX 런타임과 PyTorch에서 연산된 결과값 비교
print("prev value:",y)
print("after value:", ort_outs)
np.testing.assert_allclose(to_numpy(y), ort_outs[0], rtol=1e-03, atol=1e-05)
print("value pass")