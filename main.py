from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import numpy as np

def load_data():
  data = load_iris();

  # print(data['target_names'])
  # print(data);
  
  input = data['data'];
  target = data['target'];
  
  return input, target;

def pretreatment(input):
  poly = PolynomialFeatures();
  poly.fit(input)
  
  input = poly.transform(input);
  
  return input;

def build_model(max_iter):
  model = LogisticRegression(max_iter=max_iter)
  return model

def train_model(model, X_train, y_train):
  model.fit(X_train, y_train);

def test():
  predict_target = [];

  for i in range(0, len(test_input)):
    predict_target.append(model.predict(test_input[i].reshape(1, -1))[0]);

  print("test target :",[', '.join([str(x) for x in test_target])]);
  print("predict target :",predict_target);

if __name__ == "__main__":
  input_data, target_data = load_data();
  train_input, test_input, train_target, test_target = train_test_split(input_data, target_data, stratify=target_data);

  train_input = pretreatment(train_input);
  test_input = pretreatment(test_input);

  model = build_model(1000);
  train_model(model, train_input, train_target);

  # test();

  print("-----------------------------------");
  print("훈련 점수 :",model.score(train_input, train_target));
  print("테스트 점수 :",model.score(test_input, test_target));
  print("-----------------------------------");

  while True:
    sepal_length = float(input("꽃받침 가로길이 : "));
    if(sepal_length == 0): break;
    
    sepal_width = float(input("꽃받침 세로길이 : "));
    if(sepal_width == 0): break;
    
    petal_length = float(input("꽃잎 가로길이 : "));
    if(petal_length == 0): break;
    
    petal_width = float(input("꽃잎 세로길이 : "));
    if(petal_width == 0): break;
    
    predict = model.predict(pretreatment(np.array([[sepal_length, sepal_width, petal_length, petal_width]])));
    print("붓꽃 유형은 ","setosa" if(predict == 0) else "versicolor" if(predict == 1) else "virginica");
    print("-----------------------------------");
